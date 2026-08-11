from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, SupportsInt, TypeGuard, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import mlx.core as mx

from sam3_mlx.perflib.masks_ops import mask_iom as _mask_iom
from sam3_mlx.rle import (
    CocoRle,
    ann_to_rle as _ann_to_rle,
    normalize_rle_counts as _normalize_rle_counts,
    rle_area as _rle_area,
    rle_decode,
    rle_encode as _rle_encode,
    robust_rle_encode as _robust_rle_encode,
)

MLX_TRAIN_MASKS_OPS_BASE_COMMIT = "c5c10874844917434cff889be1d64d008a79035d"


type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]


def _is_mlx_array(value: object) -> TypeGuard["mx.array"]:
    return type(value).__module__.startswith("mlx.")


def _from_numpy(value: npt.NDArray[np.generic], like: object) -> object:
    if _is_mlx_array(like):
        import mlx.core as mx

        return mx.array(value)
    return value


def normalize_rle_counts(counts: object) -> list[int]:
    return _normalize_rle_counts(counts)


def decode_coco_rle(rle: object) -> BoolArray:
    return rle_decode(rle)


def rle_encode(orig_mask: object, return_areas: bool = False) -> list[CocoRle]:
    return _rle_encode(orig_mask, return_areas=return_areas)


def robust_rle_encode(masks: object) -> list[CocoRle]:
    return _robust_rle_encode(masks)


def ann_to_rle(segm: object, im_info: Mapping[str, SupportsInt]) -> CocoRle:
    if isinstance(segm, dict):
        payload = cast(dict[object, object], segm)
        counts = payload.get("counts")
        if isinstance(counts, (str, bytes)):
            if "size" not in payload:
                payload = {
                    **payload,
                    "size": [int(im_info["height"]), int(im_info["width"])],
                }
            return cast(CocoRle, payload)
        return _ann_to_rle(payload, im_info)
    return _ann_to_rle(segm, im_info)


def _rle_intersection_area(rle1: object, rle2: object) -> int:
    mask1 = decode_coco_rle(rle1)
    mask2 = decode_coco_rle(rle2)
    if mask1.shape != mask2.shape:
        raise ValueError("COCO RLE masks must have matching sizes.")
    return int(np.logical_and(mask1, mask2).sum())


def instance_masks_to_semantic_masks(
    instance_masks: object, num_instances: object
) -> object:
    masks = cast(BoolArray, np.asarray(instance_masks).astype(bool, copy=False))
    counts = cast(IntArray, np.asarray(num_instances, dtype=np.int64).reshape(-1))
    if masks.ndim != 3:
        raise ValueError("instance_masks must have shape (N, H, W).")
    outputs: list[BoolArray] = []
    start = 0
    for count in counts:
        chunk = masks[start : start + count]
        if chunk.size == 0:
            outputs.append(np.zeros(masks.shape[1:], dtype=bool))
        else:
            outputs.append(np.any(chunk, axis=0))
        start += int(count)
    return _from_numpy(np.stack(outputs, axis=0), instance_masks)


def mask_intersection_vectorized(masks1: object, masks2: object) -> object:
    m1 = cast(BoolArray, np.asarray(masks1).astype(bool, copy=False))
    m2 = cast(BoolArray, np.asarray(masks2).astype(bool, copy=False))
    if m1.shape[1:] != m2.shape[1:]:
        raise ValueError("masks must have matching spatial shapes.")
    out = (
        m1.reshape(m1.shape[0], -1).astype(np.int64)
        @ m2.reshape(m2.shape[0], -1).astype(np.int64).T
    )
    return _from_numpy(out, masks1)


def mask_intersection(masks1: object, masks2: object, block_size: int = 16) -> object:
    del block_size
    return mask_intersection_vectorized(masks1, masks2)


def mask_iom(masks1: object, masks2: object) -> object:
    return _mask_iom(masks1, masks2)


def compute_boundary(seg: object) -> object:
    seg_np = cast(BoolArray, np.asarray(seg).astype(bool, copy=False))
    boundary = np.zeros_like(seg_np, dtype=bool)
    boundary[..., :, :-1] |= seg_np[..., :, :-1] ^ seg_np[..., :, 1:]
    boundary[..., :-1, :] |= seg_np[..., :-1, :] ^ seg_np[..., 1:, :]
    boundary[..., :-1, :-1] |= seg_np[..., :-1, :-1] ^ seg_np[..., 1:, 1:]
    return _from_numpy(boundary, seg)


def dilation(mask: object, kernel_size: int) -> object:
    mask_np = cast(BoolArray, np.asarray(mask).astype(bool, copy=False))
    kernel_size = int(kernel_size)
    if kernel_size % 2 != 1:
        raise ValueError("dilation expects an odd kernel size.")
    pad = kernel_size // 2
    padded = np.pad(mask_np, [(0, 0), (pad, pad), (pad, pad)], mode="constant")
    out = np.zeros_like(mask_np, dtype=bool)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            out |= padded[:, dy : dy + mask_np.shape[1], dx : dx + mask_np.shape[2]]
    return _from_numpy(out, mask)


def compute_F_measure(
    gt_boundary_rle: object,
    gt_dilated_boundary_rle: object,
    dt_boundary_rle: object,
    dt_dilated_boundary_rle: object,
) -> float:
    """Compute the boundary F-measure from precomputed COCO RLE boundaries."""

    gt_match = _rle_intersection_area(gt_boundary_rle, dt_dilated_boundary_rle)
    dt_match = _rle_intersection_area(dt_boundary_rle, gt_dilated_boundary_rle)

    n_dt = _rle_area(dt_boundary_rle)
    n_gt = _rle_area(gt_boundary_rle)
    if n_dt == 0 and n_gt > 0:
        precision = 1.0
        recall = 0.0
    elif n_dt > 0 and n_gt == 0:
        precision = 0.0
        recall = 1.0
    elif n_dt == 0 and n_gt == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = dt_match / float(n_dt)
        recall = gt_match / float(n_gt)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
