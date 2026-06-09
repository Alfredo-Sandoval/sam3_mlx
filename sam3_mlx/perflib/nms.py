"""NMS helpers with explicit NumPy and MLX host-boundary handling.

The greedy selection loop is Python/NumPy. For MLX mask inputs, mask overlaps
are computed in MLX and only the score vector plus threshold-filtered IoU
submatrix are exported to host memory for selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sam3_mlx.perflib.masks_ops import mask_iou


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


def _mlx():
    import mlx.core as mx

    return mx


def _from_numpy(value: np.ndarray, *templates):
    if any(_is_mlx_array(template) for template in templates):
        mx = _mlx()

        return mx.array(value)
    return value


def _as_mlx_array(value: Any):
    mx = _mlx()
    if _is_mlx_array(value):
        return value
    return mx.array(value)


def _host_array(value: Any, *, dtype=None) -> np.ndarray:
    if _is_mlx_array(value):
        from sam3_mlx.mlx_runtime import to_numpy

        # host-postprocess-boundary: greedy NMS selection is still NumPy/Python.
        return to_numpy(value, dtype=dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _is_numpy_mask_dtype(dtype: np.dtype) -> bool:
    return (
        dtype == np.bool_
        or np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def _is_mlx_mask_dtype(dtype) -> bool:
    mx = _mlx()
    return (
        dtype == mx.bool_
        or mx.issubdtype(dtype, mx.integer)
        or mx.issubdtype(dtype, mx.floating)
    )


def _validate_mlx_masks(masks, name: str):
    if masks.ndim != 3:
        raise ValueError(f"{name} must have shape (N, H, W), got {masks.shape}.")
    if not _is_mlx_mask_dtype(masks.dtype):
        raise TypeError(f"{name} must be boolean or numeric masks.")


def _validate_numpy_masks(masks: np.ndarray, name: str) -> None:
    if masks.ndim != 3:
        raise ValueError(f"{name} must have shape (N, H, W), got {masks.shape}.")
    if not _is_numpy_mask_dtype(masks.dtype):
        raise TypeError(f"{name} must be boolean or numeric masks.")


def generic_nms(ious, scores, iou_threshold=0.5):
    """Run greedy NMS on host arrays and return kept indices like ``scores``."""

    # host-postprocess-boundary: greedy NMS ordering is currently host-side.
    ious_np = _host_array(ious, dtype=np.float32)
    scores_np = _host_array(scores, dtype=np.float32).reshape(-1)
    if ious_np.ndim != 2 or ious_np.shape[0] != ious_np.shape[1]:
        raise ValueError("ious must be a square matrix.")
    if scores_np.shape != (ious_np.shape[0],):
        raise ValueError("scores must have one entry per IoU row.")
    order = scores_np.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        idx = int(order[0])
        kept.append(idx)
        order = order[1:][ious_np[idx, order[1:]] <= iou_threshold]
    return _from_numpy(np.asarray(kept, dtype=np.int64), scores)


def generic_nms_cpu(ious, scores, iou_threshold=0.5):
    """Alias for the explicit host-side greedy NMS implementation."""

    return generic_nms(ious, scores, iou_threshold=iou_threshold)


def nms_masks(pred_probs, pred_masks, prob_threshold, iou_threshold):
    """Run mask NMS and return a boolean keep mask."""

    if _is_mlx_array(pred_probs) or _is_mlx_array(pred_masks):
        return _nms_masks_mlx(pred_probs, pred_masks, prob_threshold, iou_threshold)

    probs_np = np.asarray(pred_probs, dtype=np.float32).reshape(-1)
    masks_np = np.asarray(pred_masks)
    _validate_numpy_masks(masks_np, "pred_masks")
    if probs_np.shape != (masks_np.shape[0],):
        raise ValueError("pred_probs must have one score per mask.")
    is_valid = probs_np > prob_threshold
    if not is_valid.any():
        return _from_numpy(is_valid, pred_probs)
    valid_masks = masks_np[is_valid] > 0
    ious = np.asarray(mask_iou(valid_masks, valid_masks), dtype=np.float32)
    kept_local = np.asarray(generic_nms(ious, probs_np[is_valid], iou_threshold))
    keep = np.zeros_like(is_valid)
    valid_indices = np.flatnonzero(is_valid)
    keep[valid_indices[kept_local]] = True
    return _from_numpy(keep, pred_probs)


def _nms_masks_mlx(pred_probs, pred_masks, prob_threshold, iou_threshold):
    mx = _mlx()
    probs = _as_mlx_array(pred_probs).reshape(-1)
    masks = _as_mlx_array(pred_masks)
    _validate_mlx_masks(masks, "pred_masks")
    if probs.shape != (masks.shape[0],):
        raise ValueError("pred_probs must have one score per mask.")
    if probs.dtype == mx.bool_ or not (
        mx.issubdtype(probs.dtype, mx.integer)
        or mx.issubdtype(probs.dtype, mx.floating)
    ):
        raise TypeError("pred_probs must be numeric scores.")

    # host-postprocess-boundary: thresholding chooses the compact NMS subproblem.
    probs_np = _host_array(probs, dtype=np.float32).reshape(-1)
    is_valid = probs_np > prob_threshold
    if not is_valid.any():
        return mx.array(is_valid)

    valid_indices = np.flatnonzero(is_valid)
    valid_indices_mx = mx.array(valid_indices, dtype=mx.int64)
    valid_masks = mx.take(masks, valid_indices_mx, axis=0) > 0
    valid_overlaps = mask_iou(valid_masks, valid_masks)
    # host-postprocess-boundary: only the valid overlap submatrix crosses host.
    ious_np = _host_array(valid_overlaps, dtype=np.float32)
    kept_local = np.asarray(
        generic_nms(ious_np, probs_np[valid_indices], iou_threshold)
    )
    keep = np.zeros_like(is_valid, dtype=bool)
    keep[valid_indices[kept_local]] = True
    return mx.array(keep)
