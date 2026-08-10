# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of ``sam3.model.utils.sam1_utils`` from the official SAM3 tree."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias, cast

import numpy as np
import sam3_mlx.perflib.connected_components as connected_components_module
from numpy.typing import NDArray
from PIL import Image

import mlx.core as mx

from sam3_mlx.model import data_misc


ImageInput: TypeAlias = Image.Image | NDArray[np.generic]
ImageSize: TypeAlias = tuple[int, int]
TensorInput: TypeAlias = (
    mx.array | NDArray[np.generic] | list[object] | tuple[object, ...]
)


class _MxArrayCtor(Protocol):
    def __call__(self, val: TensorInput, dtype: object | None = None) -> mx.array: ...


class _TransposeFn(Protocol):
    def __call__(
        self,
        axis0: int,
        axis1: int,
        axis2: int,
        /,
        stream: object | None = None,
    ) -> mx.array: ...


class _InterpolateFn(Protocol):
    def __call__(
        self,
        input: mx.array,
        size: ImageSize | None = None,
        scale_factor: float | tuple[float, float] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        antialias: bool = False,
    ) -> mx.array: ...


class _ConnectedComponentsFn(Protocol):
    def __call__(self, input_tensor: mx.array) -> tuple[mx.array, mx.array]: ...


_mx_array = cast(_MxArrayCtor, mx.array)
_interpolate = cast(_InterpolateFn, getattr(data_misc, "interpolate"))


def _transpose_chw(image: mx.array) -> mx.array:
    transpose = cast(_TransposeFn, getattr(image, "transpose"))
    return transpose(2, 0, 1)


class SAM2Transforms:
    """MLX version of the SAM2 image/coordinate transforms used by SAM1 helpers."""

    def __init__(
        self,
        resolution: int,
        mask_threshold: float,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
    ) -> None:
        self.resolution = int(resolution)
        self.mask_threshold = float(mask_threshold)
        self.max_hole_area = float(max_hole_area)
        self.max_sprinkle_area = float(max_sprinkle_area)

    def __call__(self, image: ImageInput) -> mx.array:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image if image.mode == "RGB" else image.convert("RGB")
        image = image.resize(
            (self.resolution, self.resolution),
            resample=Image.Resampling.BILINEAR,
        )
        image_mx = _mx_array(np.asarray(image), dtype=mx.float32) / 255.0
        image_mx = (image_mx - 0.5) / 0.5
        return _transpose_chw(image_mx)

    def forward_batch(self, img_list: Sequence[ImageInput]) -> mx.array:
        return mx.stack([self(image) for image in img_list], axis=0)

    def transform_coords(
        self,
        coords: object,
        normalize: bool = False,
        orig_hw: ImageSize | None = None,
    ) -> mx.array:
        coords_mx = _mx_array(cast(TensorInput, coords), dtype=mx.float32)
        if normalize:
            if orig_hw is None:
                raise ValueError("orig_hw is required when normalize=True.")
            h, w = orig_hw
            scale = mx.array([w, h], dtype=mx.float32)
            coords_mx = coords_mx / scale
        return coords_mx * self.resolution

    def transform_boxes(
        self,
        boxes: object,
        normalize: bool = False,
        orig_hw: ImageSize | None = None,
    ) -> mx.array:
        boxes_mx = _mx_array(cast(TensorInput, boxes), dtype=mx.float32)
        return self.transform_coords(
            mx.reshape(boxes_mx, (-1, 2, 2)),
            normalize=normalize,
            orig_hw=orig_hw,
        )

    def postprocess_masks(self, masks: object, orig_hw: ImageSize) -> mx.array:
        masks_mx = _mx_array(cast(TensorInput, masks), dtype=mx.float32)
        if masks_mx.ndim < 4:
            raise ValueError(
                f"postprocess_masks expects shape (..., C, H, W), got {masks_mx.shape}."
            )
        if self.max_hole_area > 0 or self.max_sprinkle_area > 0:
            connected_components_fn = cast(
                _ConnectedComponentsFn,
                getattr(connected_components_module, "connected_components"),
            )
            mask_flat = mx.reshape(
                masks_mx,
                (-1, 1, masks_mx.shape[-2], masks_mx.shape[-1]),
            )
            if self.max_hole_area > 0:
                labels, areas = connected_components_fn(
                    (mask_flat <= self.mask_threshold).astype(mx.uint8)
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                masks_mx = mx.where(
                    mx.reshape(is_hole, masks_mx.shape),
                    self.mask_threshold + 10.0,
                    masks_mx,
                )
            if self.max_sprinkle_area > 0:
                mask_flat = mx.reshape(
                    masks_mx,
                    (-1, 1, masks_mx.shape[-2], masks_mx.shape[-1]),
                )
                labels, areas = connected_components_fn(
                    (mask_flat > self.mask_threshold).astype(mx.uint8)
                )
                is_sprinkle = (labels > 0) & (areas <= self.max_sprinkle_area)
                masks_mx = mx.where(
                    mx.reshape(is_sprinkle, masks_mx.shape),
                    self.mask_threshold - 10.0,
                    masks_mx,
                )
        return _interpolate(
            masks_mx,
            size=orig_hw,
            mode="bilinear",
            align_corners=False,
        )
