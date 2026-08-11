"""Agent visualization entrypoint implemented with PIL/NumPy."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Protocol, cast, overload

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from sam3_mlx.agent.helpers.rle import rle_decode


class _VisualizerOutput(Protocol):
    def get_image(self) -> NDArray[np.uint8]: ...


class _Visualizer(Protocol):
    output: _VisualizerOutput

    def overlay_instances(self, **kwargs: object) -> object: ...


class _VisualizerFactory(Protocol):
    def __call__(self, image: object, **kwargs: object) -> _Visualizer: ...


class _RenderZoom(Protocol):
    def __call__(
        self,
        object_data: Mapping[str, object],
        image: Image.Image,
        *,
        mask_alpha: float,
    ) -> tuple[Image.Image, str]: ...


_visualizer = cast(
    _VisualizerFactory,
    getattr(import_module("sam3_mlx.agent.helpers.visualizer"), "Visualizer"),
)
_render_zoom_in = cast(
    _RenderZoom,
    getattr(import_module("sam3_mlx.agent.helpers.zoom_in"), "render_zoom_in"),
)


def _integer_field(input_json: Mapping[str, object], key: str) -> int:
    value = input_json[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer.")
    return value


def _string_field(input_json: Mapping[str, object], key: str) -> str:
    value = input_json[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string.")
    return value


def _list_field(input_json: Mapping[str, object], key: str) -> list[object]:
    value = input_json.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{key} must be a list.")
    return cast(list[object], value)


@overload
def visualize(
    input_json: Mapping[str, object],
    zoom_in_index: None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
) -> Image.Image: ...


@overload
def visualize(
    input_json: Mapping[str, object],
    zoom_in_index: int,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
) -> tuple[Image.Image, Image.Image]: ...


def visualize(
    input_json: Mapping[str, object],
    zoom_in_index: int | None = None,
    mask_alpha: float = 0.15,
    label_mode: str = "1",
    font_size_multiplier: float = 1.2,
    boarder_width_multiplier: float = 0,
) -> Image.Image | tuple[Image.Image, Image.Image]:
    """Render agent JSON predictions onto the original image."""
    orig_h = _integer_field(input_json, "orig_img_h")
    orig_w = _integer_field(input_json, "orig_img_w")
    img_path = _string_field(input_json, "original_image_path")
    image = Image.open(img_path).convert("RGB")

    if zoom_in_index is None:
        boxes = np.asarray(_list_field(input_json, "pred_boxes"), dtype=float)
        rle_masks = [
            {"size": [orig_h, orig_w], "counts": rle}
            for rle in _list_field(input_json, "pred_masks")
        ]
        binary_masks = [rle_decode(rle) for rle in rle_masks]
        viz = _visualizer(
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
    pred_masks = _list_field(input_json, "pred_masks")
    pred_boxes = _list_field(input_json, "pred_boxes")
    num_masks = len(pred_masks)
    if idx < 0 or idx >= num_masks:
        raise ValueError(f"zoom_in_index {idx} is out of range (0..{num_masks - 1}).")

    object_data = {
        "labels": [{"noun_phrase": f"mask_{idx}"}],
        "segmentation": {
            "counts": pred_masks[idx],
            "size": [orig_h, orig_w],
        },
    }
    pil_mask_i_zoomed, color_hex = _render_zoom_in(
        object_data, image, mask_alpha=mask_alpha
    )

    boxes_i = np.asarray([pred_boxes[idx]], dtype=float)
    rle_i = {"size": [orig_h, orig_w], "counts": pred_masks[idx]}
    viz_i = _visualizer(
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
