"""SAM3 agent client compatibility surface."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from sam3_mlx.agent.helpers.mask_overlap_removal import remove_overlapping_masks
from sam3_mlx.agent.viz import visualize
from sam3_mlx.mlx_runtime import to_numpy as _to_numpy
from sam3_mlx.model.box_ops import box_xyxy_to_xywh
from sam3_mlx.train.masks_ops import rle_encode


def sam3_inference(processor, image_path, text_prompt):
    """Run SAM3 image inference with a text prompt and JSON-safe outputs."""

    image = Image.open(image_path).convert("RGB")
    orig_img_w, orig_img_h = image.size

    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(
        state=inference_state,
        prompt=text_prompt,
    )

    boxes_xyxy = _to_numpy(inference_state["boxes"]).astype(np.float32)
    if boxes_xyxy.size == 0:
        pred_boxes_xywh = []
    else:
        normalizer = np.array(
            [orig_img_w, orig_img_h, orig_img_w, orig_img_h],
            dtype=np.float32,
        )
        boxes_norm = mx.array(boxes_xyxy / normalizer)
        pred_boxes_xywh = _to_numpy(box_xyxy_to_xywh(boxes_norm)).tolist()

    masks = _to_numpy(inference_state["masks"]).astype(bool)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (N, H, W), got {masks.shape}.")
    pred_masks = [mask["counts"] for mask in rle_encode(masks)]

    return {
        "orig_img_h": orig_img_h,
        "orig_img_w": orig_img_w,
        "pred_boxes": pred_boxes_xywh,
        "pred_masks": pred_masks,
        "pred_scores": _to_numpy(inference_state["scores"]).reshape(-1).tolist(),
    }


def call_sam_service(
    sam3_processor,
    image_path: str,
    text_prompt: str,
    output_folder_path: str = "sam3_output",
):
    """Run local SAM3 inference, save JSON output, and render visualization."""

    text_prompt_for_save_path = text_prompt.replace("/", "_")
    image_key = image_path.replace("/", "-")
    output_folder = Path(output_folder_path) / image_key
    output_folder.mkdir(parents=True, exist_ok=True)
    output_json_path = output_folder / f"{text_prompt_for_save_path}.json"
    output_image_path = output_folder / f"{text_prompt_for_save_path}.png"

    serialized_response = sam3_inference(sam3_processor, image_path, text_prompt)
    serialized_response = remove_overlapping_masks(serialized_response)
    serialized_response = {
        "original_image_path": image_path,
        "output_image_path": os.fspath(output_image_path),
        **serialized_response,
    }

    if serialized_response.get("pred_scores"):
        score_indices = sorted(
            range(len(serialized_response["pred_scores"])),
            key=lambda index: serialized_response["pred_scores"][index],
            reverse=True,
        )
        for key in ("pred_scores", "pred_boxes", "pred_masks"):
            serialized_response[key] = [
                serialized_response[key][index] for index in score_indices
            ]

    valid_indices = [
        index
        for index, rle in enumerate(serialized_response["pred_masks"])
        if len(rle) > 4
    ]
    for key in ("pred_masks", "pred_boxes", "pred_scores"):
        serialized_response[key] = [
            serialized_response[key][index] for index in valid_indices
        ]

    with output_json_path.open("w", encoding="utf-8") as handle:
        json.dump(serialized_response, handle, indent=4)

    viz_image = visualize(serialized_response)
    viz_image.save(output_image_path)
    return os.fspath(output_json_path)
