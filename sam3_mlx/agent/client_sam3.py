"""SAM3 agent client compatibility surface."""

from __future__ import annotations

import json
import os
from collections.abc import Sized
from pathlib import Path
from typing import SupportsFloat, cast

import mlx.core as mx
import numpy as np
from PIL import Image

from sam3_mlx.agent.contracts import AgentOutput, ImageProcessor
from sam3_mlx.agent.helpers.mask_overlap_removal import remove_overlapping_masks
from sam3_mlx.agent.viz import visualize
from sam3_mlx.mlx_runtime import to_numpy as _to_numpy
from sam3_mlx.model.box_ops import box_xyxy_to_xywh
from sam3_mlx.train.masks_ops import rle_encode


def _mask_count_length(value: object) -> int:
    if not isinstance(value, Sized):
        raise TypeError("Serialized masks must provide a length.")
    return len(value)


def sam3_inference(
    processor: ImageProcessor, image_path: str, text_prompt: str
) -> AgentOutput:
    """Run SAM3 image inference with a text prompt and JSON-safe outputs."""

    image = Image.open(image_path).convert("RGB")
    orig_img_w, orig_img_h = image.size

    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(
        state=inference_state,
        prompt=text_prompt,
    )

    boxes_xyxy = _to_numpy(inference_state["boxes"]).astype(np.float32)
    pred_boxes_xywh: list[list[float]]
    if boxes_xyxy.size == 0:
        pred_boxes_xywh = []
    else:
        normalizer = np.array(
            [orig_img_w, orig_img_h, orig_img_w, orig_img_h],
            dtype=np.float32,
        )
        boxes_norm = mx.array(boxes_xyxy / normalizer)
        boxes_list = cast(
            list[list[object]],
            _to_numpy(box_xyxy_to_xywh(boxes_norm)).tolist(),
        )
        pred_boxes_xywh = [
            [float(cast(SupportsFloat, value)) for value in row] for row in boxes_list
        ]

    masks = _to_numpy(inference_state["masks"]).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (N, H, W), got {masks.shape}.")
    pred_masks: list[object] = [mask["counts"] for mask in rle_encode(masks)]

    scores = _to_numpy(inference_state["scores"]).reshape(-1)
    return {
        "orig_img_h": orig_img_h,
        "orig_img_w": orig_img_w,
        "pred_boxes": pred_boxes_xywh,
        "pred_masks": pred_masks,
        "pred_scores": [float(value) for value in scores],
    }


def call_sam_service(
    sam3_processor: ImageProcessor,
    image_path: str,
    text_prompt: str,
    output_folder_path: str = "sam3_output",
) -> str:
    """Run local SAM3 inference, save JSON output, and render visualization."""

    text_prompt_for_save_path = text_prompt.replace("/", "_")
    image_key = image_path.replace("/", "-")
    output_folder = Path(output_folder_path) / image_key
    output_folder.mkdir(parents=True, exist_ok=True)
    output_json_path = output_folder / f"{text_prompt_for_save_path}.json"
    output_image_path = output_folder / f"{text_prompt_for_save_path}.png"

    inference_response = sam3_inference(sam3_processor, image_path, text_prompt)
    inference_response = remove_overlapping_masks(inference_response)
    serialized_response: AgentOutput = {
        "original_image_path": image_path,
        "output_image_path": os.fspath(output_image_path),
        **inference_response,
    }

    if serialized_response["pred_scores"]:
        score_indices = sorted(
            range(len(serialized_response["pred_scores"])),
            key=lambda index: serialized_response["pred_scores"][index],
            reverse=True,
        )
        serialized_response["pred_scores"] = [
            serialized_response["pred_scores"][index] for index in score_indices
        ]
        serialized_response["pred_boxes"] = [
            serialized_response["pred_boxes"][index] for index in score_indices
        ]
        serialized_response["pred_masks"] = [
            serialized_response["pred_masks"][index] for index in score_indices
        ]

    valid_indices = [
        index
        for index, rle in enumerate(serialized_response["pred_masks"])
        if _mask_count_length(rle) > 4
    ]
    serialized_response["pred_masks"] = [
        serialized_response["pred_masks"][index] for index in valid_indices
    ]
    serialized_response["pred_boxes"] = [
        serialized_response["pred_boxes"][index] for index in valid_indices
    ]
    serialized_response["pred_scores"] = [
        serialized_response["pred_scores"][index] for index in valid_indices
    ]

    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized_response, handle, indent=4)

    viz_image = visualize(serialized_response)
    viz_image.save(output_image_path)
    return os.fspath(output_json_path)
