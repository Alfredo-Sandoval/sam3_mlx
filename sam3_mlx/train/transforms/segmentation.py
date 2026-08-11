# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Segmentation transforms ported from official SAM3 to MLX."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from sam3_mlx.mlx_runtime import evaluate_boundary, is_mlx_array, to_numpy
from sam3_mlx.model.box_ops import masks_to_boxes
from sam3_mlx.rle import rle_decode, rle_encode
from sam3_mlx.train.data.sam3_image_dataset import Datapoint
from sam3_mlx.train.data.sam3_image_dataset import Image
from sam3_mlx.train.transforms._array_contracts import mx_array, mx_ops, mx_shape


MLX_SEGMENTATION_BASE_COMMIT = "dc33741d86020f34c73f9534deabff1007cdd886"
_UINT8 = getattr(mx, "uint8", mx.bool_)


def _resize_mask(
    mask_np: NDArray[np.generic], size: tuple[int, int]
) -> NDArray[np.uint8]:
    height, width = size
    image = PILImage.fromarray(mask_np.astype(np.uint8, copy=False))
    resized = image.resize((width, height), resample=PILImage.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def _image_hw(image: Image) -> tuple[int, int]:
    data = image.data
    if isinstance(data, PILImage.Image):
        width, height = data.size
        return height, width
    if is_mlx_array(data):
        shape = mx_shape(data)
        return shape[-2], shape[-1]
    raise RuntimeError(f"Unexpected image type {type(data)!r}")


class InstanceToSemantic:
    """Convert instance segmentation masks to per-query semantic masks."""

    def __init__(self, delete_instance: bool = True, use_rle: bool = False) -> None:
        self.delete_instance = delete_instance
        self.use_rle = use_rle

    def __call__(self, datapoint: Datapoint, **kwargs: object) -> Datapoint:
        del kwargs
        for query in datapoint.find_queries:
            height, width = datapoint.images[query.image_id].size

            if self.use_rle:
                all_segs = [
                    datapoint.images[query.image_id].objects[obj_id].segment
                    for obj_id in query.object_ids_output
                ]
                if len(all_segs) > 0:
                    merged = np.zeros((height, width), dtype=np.uint8)
                    for seg in all_segs:
                        if seg is None:
                            raise ValueError(
                                "InstanceToSemantic requires object segments."
                            )
                        decoded = rle_decode(seg).astype(np.uint8, copy=False)
                        if decoded.shape != merged.shape:
                            raise AssertionError(
                                "Instance segments have inconsistent RLE sizes."
                            )
                        merged |= decoded
                    query.semantic_target = rle_encode(
                        merged.astype(bool, copy=False)[None, :, :]
                    )[0]
                else:
                    query.semantic_target = rle_encode(
                        np.zeros((1, height, width), dtype=bool)
                    )[0]
            else:
                semantic = np.zeros((height, width), dtype=np.uint8)
                for obj_id in query.object_ids_output:
                    segment = datapoint.images[query.image_id].objects[obj_id].segment
                    if segment is not None:
                        if not is_mlx_array(segment):
                            raise TypeError(
                                "InstanceToSemantic requires decoded MLX segments "
                                "when use_rle=False."
                            )
                        semantic |= to_numpy(segment).astype(np.uint8, copy=False)
                query.semantic_target = mx.array(semantic, dtype=_UINT8)

        if self.delete_instance:
            for image in datapoint.images:
                for obj in image.objects:
                    obj.segment = None
        return datapoint


class RecomputeBoxesFromMasks:
    """Recompute object boxes and areas from binary masks."""

    def __call__(self, datapoint: Datapoint, **kwargs: object) -> Datapoint:
        del kwargs
        for image in datapoint.images:
            for obj in image.objects:
                if obj.segment is None:
                    raise ValueError("RecomputeBoxesFromMasks requires obj.segment.")
                if not is_mlx_array(obj.segment):
                    raise TypeError(
                        "RecomputeBoxesFromMasks requires decoded MLX segments."
                    )
                mask = mx_ops(obj.segment).astype(mx.bool_)
                if mx_ops(mask).ndim == 2:
                    mask = mask[None, :, :]
                obj.bbox = masks_to_boxes(mask)[0]
                area = mx.sum(mx_ops(mask).astype(mx.float32))
                evaluate_boundary(area)
                obj.area = float(np.asarray(area))
        return datapoint


class DecodeRle:
    """Decode object and semantic COCO RLE masks into MLX uint8 masks."""

    def __call__(self, datapoint: Datapoint, **kwargs: object) -> Datapoint:
        del kwargs
        img_id_to_size: dict[int, tuple[int, int]] = {}
        warning_shown = False

        for img_id, image in enumerate(datapoint.images):
            img_h, img_w = _image_hw(image)
            img_id_to_size[img_id] = (img_h, img_w)

            for obj in image.objects:
                if obj.segment is None or is_mlx_array(obj.segment):
                    continue
                segment = rle_decode(obj.segment).astype(np.uint8, copy=False)
                if segment.sum() == 0:
                    print("Warning, empty mask found, approximating from box")
                    segment = np.zeros((img_h, img_w), dtype=np.uint8)
                    x1, y1, x2, y2 = to_numpy(obj.bbox).astype(int).tolist()
                    segment[y1 : max(y2, y1 + 1), x1 : max(x1 + 1, x2)] = 1

                if list(segment.shape) != [img_h, img_w]:
                    if not warning_shown:
                        print(
                            "Warning expected instance segmentation size to be "
                            f"{[img_h, img_w]} but found {list(segment.shape)}"
                        )
                        warning_shown = True
                    segment = _resize_mask(segment, (img_h, img_w))
                if list(segment.shape) != [img_h, img_w]:
                    raise AssertionError("Decoded instance segment has invalid size.")
                obj.segment = mx_array(segment, dtype=_UINT8)

        warning_shown = False
        for query in datapoint.find_queries:
            if query.semantic_target is None or is_mlx_array(query.semantic_target):
                continue
            semantic = rle_decode(query.semantic_target).astype(np.uint8)
            expected_size = img_id_to_size[query.image_id]
            if tuple(semantic.shape) != expected_size:
                if not warning_shown:
                    print(
                        "Warning expected semantic segmentation size to be "
                        f"{expected_size} but found {tuple(semantic.shape)}"
                    )
                    warning_shown = True
                semantic = _resize_mask(semantic, expected_size)
            if tuple(semantic.shape) != expected_size:
                raise AssertionError("Decoded semantic segment has invalid size.")
            query.semantic_target = mx_array(semantic, dtype=_UINT8)

        return datapoint


__all__ = [
    "DecodeRle",
    "InstanceToSemantic",
    "MLX_SEGMENTATION_BASE_COMMIT",
    "RecomputeBoxesFromMasks",
]
