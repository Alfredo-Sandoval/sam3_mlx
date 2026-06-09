# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Segmentation transforms ported from official SAM3 to MLX."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from PIL import Image as PILImage

from sam3_mlx.model.box_ops import masks_to_boxes
from sam3_mlx.train.data.coco_json_loaders import (
    _encode_binary_mask_to_uncompressed_rle,
)
from sam3_mlx.train.data.sam3_image_dataset import Datapoint
from sam3_mlx.train.transforms.point_sampling import _decode_coco_rle


MLX_SEGMENTATION_BASE_COMMIT = "dc33741d86020f34c73f9534deabff1007cdd886"
_UINT8 = getattr(mx, "uint8", mx.bool_)


def _is_array(value) -> bool:
    return isinstance(value, (mx.array, np.ndarray))


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, mx.array):
        mx.eval(value)
    return np.asarray(value)


def _to_mask_array(value) -> mx.array:
    return mx.array(_to_numpy(value).astype(np.uint8, copy=False), dtype=_UINT8)


def _resize_mask(mask_np: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = size
    image = PILImage.fromarray(mask_np.astype(np.uint8, copy=False))
    resized = image.resize((width, height), resample=PILImage.Resampling.NEAREST)
    return np.asarray(resized, dtype=np.uint8)


def _image_hw(image) -> tuple[int, int]:
    data = image.data
    if isinstance(data, PILImage.Image):
        width, height = data.size
        return height, width
    if isinstance(data, mx.array):
        return data.shape[-2:]
    if isinstance(data, np.ndarray):
        if data.ndim == 3 and data.shape[0] in (1, 3, 4):
            return data.shape[-2:]
        return data.shape[:2]
    raise RuntimeError(f"Unexpected image type {type(data)!r}")


class InstanceToSemantic:
    """Convert instance segmentation masks to per-query semantic masks."""

    def __init__(self, delete_instance=True, use_rle=False):
        self.delete_instance = delete_instance
        self.use_rle = use_rle

    def __call__(self, datapoint: Datapoint, **kwargs):
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
                        if seg["size"] != all_segs[0]["size"]:
                            raise AssertionError(
                                "Instance segments have inconsistent RLE sizes."
                            )
                        merged |= _decode_coco_rle(seg).astype(np.uint8, copy=False)
                    query.semantic_target = _encode_binary_mask_to_uncompressed_rle(
                        merged
                    )
                else:
                    query.semantic_target = _encode_binary_mask_to_uncompressed_rle(
                        np.zeros((height, width), dtype=np.uint8)
                    )
            else:
                semantic = np.zeros((height, width), dtype=np.uint8)
                for obj_id in query.object_ids_output:
                    segment = datapoint.images[query.image_id].objects[obj_id].segment
                    if segment is not None:
                        semantic |= _to_numpy(segment).astype(np.uint8, copy=False)
                query.semantic_target = mx.array(semantic, dtype=_UINT8)

        if self.delete_instance:
            for image in datapoint.images:
                for obj in image.objects:
                    obj.segment = None
        return datapoint


class RecomputeBoxesFromMasks:
    """Recompute object boxes and areas from binary masks."""

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        for image in datapoint.images:
            for obj in image.objects:
                if obj.segment is None:
                    raise ValueError("RecomputeBoxesFromMasks requires obj.segment.")
                mask = mx.array(obj.segment, dtype=mx.bool_)
                if mask.ndim == 2:
                    mask = mask[None, :, :]
                obj.bbox = masks_to_boxes(mask)[0]
                area = mx.sum(mask.astype(mx.float32))
                mx.eval(area)
                obj.area = float(np.asarray(area))
        return datapoint


class DecodeRle:
    """Decode object and semantic COCO RLE masks into MLX uint8 masks."""

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        img_id_to_size = {}
        warning_shown = False

        for img_id, image in enumerate(datapoint.images):
            img_h, img_w = _image_hw(image)
            img_id_to_size[img_id] = (img_h, img_w)

            for obj in image.objects:
                if obj.segment is None or _is_array(obj.segment):
                    continue
                segment = _decode_coco_rle(obj.segment).astype(np.uint8, copy=False)
                if segment.sum() == 0:
                    print("Warning, empty mask found, approximating from box")
                    segment = np.zeros((img_h, img_w), dtype=np.uint8)
                    x1, y1, x2, y2 = _to_numpy(obj.bbox).astype(int).tolist()
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
                obj.segment = mx.array(segment, dtype=_UINT8)

        warning_shown = False
        for query in datapoint.find_queries:
            if query.semantic_target is None or _is_array(query.semantic_target):
                continue
            semantic = _decode_coco_rle(query.semantic_target).astype(np.uint8)
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
            query.semantic_target = mx.array(semantic, dtype=_UINT8)

        return datapoint


__all__ = [
    "DecodeRle",
    "InstanceToSemantic",
    "MLX_SEGMENTATION_BASE_COMMIT",
    "RecomputeBoxesFromMasks",
]
