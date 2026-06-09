"""Agent visualization entrypoint implemented with PIL/NumPy."""

from __future__ import annotations

import numpy as np
from PIL import Image

from sam3_mlx.agent.helpers.rle import rle_decode
from sam3_mlx.agent.helpers.visualizer import Visualizer
from sam3_mlx.agent.helpers.zoom_in import render_zoom_in


def visualize(
    input_json: dict,
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
):
    """Render agent JSON predictions onto the original image."""
    orig_h = int(input_json["orig_img_h"])
    orig_w = int(input_json["orig_img_w"])
    img_path = input_json["original_image_path"]
    image = Image.open(img_path).convert("RGB")

    if zoom_in_index is None:
        boxes = np.asarray(input_json.get("pred_boxes", []), dtype=float)
        rle_masks = [
            {"size": [orig_h, orig_w], "counts": rle}
            for rle in input_json.get("pred_masks", [])
        ]
        binary_masks = [rle_decode(rle) for rle in rle_masks]
        viz = Visualizer(
            np.asarray(image),
            font_size_multiplier=font_size_multiplier,
            boarder_width_multiplier=boarder_width_multiplier,
        )
        viz.overlay_instances(
            boxes=boxes,
            masks=rle_masks,
            binary_masks=binary_masks,
            alpha=mask_alpha,
            label_mode=label_mode,
        )
        return Image.fromarray(viz.output.get_image())

    idx = int(zoom_in_index)
    num_masks = len(input_json.get("pred_masks", []))
    if idx < 0 or idx >= num_masks:
        raise ValueError(f"zoom_in_index {idx} is out of range (0..{num_masks - 1}).")

    object_data = {
        "labels": [{"noun_phrase": f"mask_{idx}"}],
        "segmentation": {
            "counts": input_json["pred_masks"][idx],
            "size": [orig_h, orig_w],
        },
    }
    pil_mask_i_zoomed, color_hex = render_zoom_in(
        object_data, image, mask_alpha=mask_alpha
    )

    boxes_i = np.asarray([input_json["pred_boxes"][idx]], dtype=float)
    rle_i = {"size": [orig_h, orig_w], "counts": input_json["pred_masks"][idx]}
    viz_i = Visualizer(
        np.asarray(image),
        font_size_multiplier=font_size_multiplier,
        boarder_width_multiplier=boarder_width_multiplier,
    )
    viz_i.overlay_instances(
        boxes=boxes_i,
        masks=[rle_i],
        binary_masks=[rle_decode(rle_i)],
        assigned_colors=[color_hex],
        alpha=mask_alpha,
        label_mode=label_mode,
    )
    return Image.fromarray(viz_i.output.get_image()), pil_mask_i_zoomed
