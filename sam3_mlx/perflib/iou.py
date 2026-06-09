"""Pairwise mask-overlap helpers with NumPy and MLX-native paths.

NumPy inputs keep the historical NumPy implementation. MLX inputs stay in MLX
until the caller explicitly exports the result.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


def _mlx():
    import mlx.core as mx

    return mx


def _from_numpy(value: np.ndarray, like):
    if _is_mlx_array(like):
        import mlx.core as mx

        return mx.array(value)
    return value


def _as_mlx_array(value: Any):
    mx = _mlx()
    if _is_mlx_array(value):
        return value
    return mx.array(value)


def _is_numpy_mask_dtype(dtype: np.dtype) -> bool:
    return (
        dtype == np.bool_
        or np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    )


def _ensure_numeric_numpy_masks(pred_masks, gt_masks) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_masks)
    gt = np.asarray(gt_masks)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError("pairwise mask metrics expect shapes (N, H, W) and (M, H, W).")
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError("pred_masks and gt_masks must have matching spatial shapes.")
    if not _is_numpy_mask_dtype(pred.dtype) or not _is_numpy_mask_dtype(gt.dtype):
        raise TypeError("pairwise mask metrics expect boolean or numeric masks.")
    return pred, gt


def _flatten_numpy_masks(pred_masks, gt_masks) -> tuple[np.ndarray, np.ndarray]:
    pred, gt = _ensure_numeric_numpy_masks(pred_masks, gt_masks)
    return (
        pred.reshape(pred.shape[0], pred.shape[1] * pred.shape[2]).astype(
            np.float32, copy=False
        ),
        gt.reshape(gt.shape[0], gt.shape[1] * gt.shape[2]).astype(
            np.float32, copy=False
        ),
    )


def _is_mlx_mask_dtype(dtype) -> bool:
    mx = _mlx()
    return (
        dtype == mx.bool_
        or mx.issubdtype(dtype, mx.integer)
        or mx.issubdtype(dtype, mx.floating)
    )


def _ensure_numeric_mlx_masks(pred_masks, gt_masks):
    pred = _as_mlx_array(pred_masks)
    gt = _as_mlx_array(gt_masks)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError("pairwise mask metrics expect shapes (N, H, W) and (M, H, W).")
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError("pred_masks and gt_masks must have matching spatial shapes.")
    if not _is_mlx_mask_dtype(pred.dtype) or not _is_mlx_mask_dtype(gt.dtype):
        raise TypeError("pairwise mask metrics expect boolean or numeric masks.")
    return pred, gt


def _flatten_mlx_masks(pred_masks, gt_masks):
    mx = _mlx()
    pred, gt = _ensure_numeric_mlx_masks(pred_masks, gt_masks)
    spatial_size = pred.shape[1] * pred.shape[2]
    return (
        pred.reshape(pred.shape[0], spatial_size).astype(mx.float32),
        gt.reshape(gt.shape[0], spatial_size).astype(mx.float32),
    )


def _divide_with_eps_mlx(numerator, denominator, eps):
    mx = _mlx()
    if eps is None:
        return numerator / mx.maximum(denominator, 1.0)
    return numerator / (denominator + eps)


def pairwise_iou(pred_masks, gt_masks, eps=1e-6):
    """Compute pairwise IoU, preserving NumPy or MLX execution."""

    if _is_mlx_array(pred_masks) or _is_mlx_array(gt_masks):
        pred_flat, gt_flat = _flatten_mlx_masks(pred_masks, gt_masks)
        intersection = pred_flat @ gt_flat.T
        area_pred = pred_flat.sum(axis=1, keepdims=True)
        area_gt = gt_flat.sum(axis=1, keepdims=True)
        union = area_pred + area_gt.T - intersection
        return _divide_with_eps_mlx(intersection, union, eps)

    pred_flat, gt_flat = _flatten_numpy_masks(pred_masks, gt_masks)
    intersection = pred_flat @ gt_flat.T
    area_pred = pred_flat.sum(axis=1, keepdims=True)
    area_gt = gt_flat.sum(axis=1, keepdims=True)
    union = area_pred + area_gt.T - intersection
    if eps is None:
        out = intersection / np.clip(union, 1.0, None)
    else:
        out = intersection / (union + eps)
    return _from_numpy(out.astype(np.float32, copy=False), pred_masks)


def pairwise_iom(pred_masks, gt_masks, eps=1e-8):
    """Compute pairwise intersection-over-min-area."""

    if _is_mlx_array(pred_masks) or _is_mlx_array(gt_masks):
        pred_flat, gt_flat = _flatten_mlx_masks(pred_masks, gt_masks)
        intersection = pred_flat @ gt_flat.T
        area_pred = pred_flat.sum(axis=1, keepdims=True)
        area_gt = gt_flat.sum(axis=1, keepdims=True)
        min_area = _mlx().minimum(area_pred, area_gt.T)
        return _divide_with_eps_mlx(intersection, min_area, eps)

    pred_flat, gt_flat = _flatten_numpy_masks(pred_masks, gt_masks)
    intersection = pred_flat @ gt_flat.T
    area_pred = pred_flat.sum(axis=1, keepdims=True)
    area_gt = gt_flat.sum(axis=1, keepdims=True)
    min_area = np.minimum(area_pred, area_gt.T)
    if eps is None:
        out = intersection / np.clip(min_area, 1.0, None)
    else:
        out = intersection / (min_area + eps)
    return _from_numpy(out.astype(np.float32, copy=False), pred_masks)
