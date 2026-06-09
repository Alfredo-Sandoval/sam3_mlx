# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of ``sam3.model.utils.sam1_utils`` from the official SAM3 tree."""

from __future__ import annotations

import numpy as np
from PIL import Image

import mlx.core as mx

from sam3_mlx.model.data_misc import interpolate


class SAM2Transforms:
    """MLX version of the SAM2 image/coordinate transforms used by SAM1 helpers."""

    def __init__(
        self,
        resolution,
        mask_threshold,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    ) -> None:
        self.resolution = int(resolution)
        self.mask_threshold = float(mask_threshold)
        self.max_hole_area = float(max_hole_area)
        self.max_sprinkle_area = float(max_sprinkle_area)

    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image if image.mode == "RGB" else image.convert("RGB")
        image = image.resize(
            (self.resolution, self.resolution),
            resample=Image.Resampling.BILINEAR,
        )
        image_mx = mx.array(np.asarray(image), dtype=mx.float32) / 255.0
        image_mx = (image_mx - 0.5) / 0.5
        return image_mx.transpose(2, 0, 1)

    def forward_batch(self, img_list):
        return mx.stack([self(image) for image in img_list], axis=0)

    def transform_coords(self, coords, normalize=False, orig_hw=None):
        coords = mx.array(coords, dtype=mx.float32)
        if normalize:
            if orig_hw is None:
                raise ValueError("orig_hw is required when normalize=True.")
            h, w = orig_hw
            scale = mx.array([w, h], dtype=mx.float32)
            coords = coords / scale
        return coords * self.resolution

    def transform_boxes(self, boxes, normalize=False, orig_hw=None):
        boxes = mx.array(boxes, dtype=mx.float32)
        return self.transform_coords(
            boxes.reshape(-1, 2, 2),
            normalize=normalize,
            orig_hw=orig_hw,
        )

    def postprocess_masks(self, masks, orig_hw):
        masks = mx.array(masks, dtype=mx.float32)
        if masks.ndim < 4:
            raise ValueError(
                f"postprocess_masks expects shape (..., C, H, W), got {masks.shape}."
            )
        if self.max_hole_area > 0 or self.max_sprinkle_area > 0:
            from sam3_mlx.perflib.connected_components import connected_components

            mask_flat = masks.reshape(-1, 1, masks.shape[-2], masks.shape[-1])
            if self.max_hole_area > 0:
                labels, areas = connected_components(
                    (mask_flat <= self.mask_threshold).astype(mx.uint8)
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                masks = mx.where(
                    is_hole.reshape(masks.shape),
                    self.mask_threshold + 10.0,
                    masks,
                )
            if self.max_sprinkle_area > 0:
                mask_flat = masks.reshape(-1, 1, masks.shape[-2], masks.shape[-1])
                labels, areas = connected_components(
                    (mask_flat > self.mask_threshold).astype(mx.uint8)
                )
                is_sprinkle = (labels > 0) & (areas <= self.max_sprinkle_area)
                masks = mx.where(
                    is_sprinkle.reshape(masks.shape),
                    self.mask_threshold - 10.0,
                    masks,
                )
        return interpolate(masks, size=orig_hw, mode="bilinear", align_corners=False)
