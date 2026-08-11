"""Zoom-in visualization helper implemented with PIL."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import IO, cast

import numpy as np
from PIL import Image, ImageDraw

from sam3_mlx.agent.helpers.rle import rle_decode, rle_to_bbox
from sam3_mlx.agent.helpers.som_utils import ColorPalette


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError(f"{name} must use string keys.")
    return cast(Mapping[str, object], mapping)


def _label_text(object_data: Mapping[str, object]) -> str:
    labels_value = object_data.get("labels")
    if not labels_value:
        return ""
    if isinstance(labels_value, str | bytes) or not isinstance(labels_value, Sequence):
        raise TypeError("labels must be a sequence of mappings.")
    labels = cast(Sequence[object], labels_value)
    label = _mapping(labels[0], "labels[0]")
    noun_phrase = label.get("noun_phrase", "")
    if not isinstance(noun_phrase, str):
        raise TypeError("labels[0].noun_phrase must be a string.")
    return noun_phrase


def render_zoom_in(
    object_data: Mapping[str, object],
    image_file: Image.Image | str | os.PathLike[str] | IO[bytes],
    show_box: bool = True,
    show_text: bool = False,
    show_holes: bool = True,
    mask_alpha: float = 0.15,
) -> tuple[Image.Image, str]:
    """Render a crop panel and a zoomed mask panel; return ``(PIL.Image, color_hex)``."""
    img = (
        image_file.convert("RGB")
        if isinstance(image_file, Image.Image)
        else Image.open(image_file).convert("RGB")
    )
    segmentation = _mapping(object_data["segmentation"], "segmentation")
    mask = rle_decode(segmentation)
    bbox_xywh = rle_to_bbox(segmentation)
    x, y, w, h = bbox_xywh
    x0, y0 = max(0, int(math.floor(x))), max(0, int(math.floor(y)))
    x1 = min(img.width, int(math.ceil(x + w)))
    y1 = min(img.height, int(math.ceil(y + h)))
    pad = max(8, int(max(w, h) * 0.2))
    crop_box = (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(img.width, x1 + pad),
        min(img.height, y1 + pad),
    )

    crop = img.crop(crop_box)
    palette = ColorPalette.default()
    color_obj, _ = palette.find_farthest_color(np.asarray(crop))
    color = color_obj.as_rgb()
    color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

    crop_draw = ImageDraw.Draw(crop)
    if show_box:
        rel_box = [
            x0 - crop_box[0],
            y0 - crop_box[1],
            x1 - crop_box[0],
            y1 - crop_box[1],
        ]
        crop_draw.rectangle(rel_box, outline=color, width=2)
    if show_text:
        crop_draw.text((2, 2), _label_text(object_data), fill=color)

    zoom = img.crop(crop_box).convert("RGBA")
    crop_mask = mask[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]]
    overlay = Image.new("RGBA", zoom.size, color + (0,))
    overlay.putalpha(
        Image.fromarray((crop_mask.astype(np.uint8) * int(255 * mask_alpha)), mode="L")
    )
    zoom = Image.alpha_composite(zoom, overlay).convert("RGB")

    panel_w = crop.width + zoom.width
    panel_h = max(crop.height, zoom.height)
    out = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
    out.paste(crop, (0, 0))
    out.paste(zoom, (crop.width, 0))
    return out, color_hex
