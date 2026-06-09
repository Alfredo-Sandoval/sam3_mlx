"""Mask geometry helpers with NumPy and MLX-native overlap paths.

NumPy inputs stay on NumPy. MLX inputs keep box extraction and pairwise mask
overlaps on MLX until the caller explicitly exports results.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


def _mlx():
    import mlx.core as mx

    return mx


def _to_numpy(value) -> np.ndarray:
    if _is_mlx_array(value):
        from sam3_mlx.mlx_runtime import to_numpy

        return to_numpy(value, copy=False)
    return np.asarray(value)


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


def _masks_to_boxes_mlx(masks):
    mx = _mlx()
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}.")
    if masks.size == 0:
        return mx.zeros((0, 4), dtype=mx.float32)

    _, height, width = masks.shape
    mask = masks != 0
    x = mx.arange(width, dtype=mx.float32)[None, :]
    y = mx.arange(height, dtype=mx.float32)[None, :]
    has_x = mx.any(mask, axis=1)
    has_y = mx.any(mask, axis=2)

    x_min = mx.min(mx.where(has_x, x, float(width)), axis=1)
    y_min = mx.min(mx.where(has_y, y, float(height)), axis=1)
    x_max = mx.max(mx.where(has_x, x, 0.0), axis=1)
    y_max = mx.max(mx.where(has_y, y, 0.0), axis=1)
    return mx.stack([x_min, y_min, x_max, y_max], axis=1).astype(mx.float32)


def masks_to_boxes(masks, obj_ids: list[int]):
    """Compute upstream-style inclusive ``xyxy`` boxes for ``(N, H, W)`` masks."""
    if _is_mlx_array(masks):
        if masks.ndim != 3:
            raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}.")
        if masks.shape[0] != len(obj_ids):
            raise ValueError("masks and obj_ids must have the same length.")
        return _masks_to_boxes_mlx(masks)

    masks_np = _to_numpy(masks)
    if masks_np.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks_np.shape}.")
    if masks_np.shape[0] != len(obj_ids):
        raise ValueError("masks and obj_ids must have the same length.")

    if masks_np.size == 0:
        return _from_numpy(np.zeros((0, 4), dtype=np.float32), masks)

    num_masks, height, width = masks_np.shape
    y = np.arange(height, dtype=np.int64).reshape(1, height)
    x = np.arange(width, dtype=np.int64).reshape(1, width)

    masks_with_obj = masks_np != 0
    masks_with_obj_x = masks_with_obj.max(axis=1)
    masks_with_obj_y = masks_with_obj.max(axis=2)
    masks_without_obj_x = ~masks_with_obj_x
    masks_without_obj_y = ~masks_with_obj_y

    boxes = np.stack(
        [
            np.amin((masks_without_obj_x * width) + (masks_with_obj_x * x), axis=1),
            np.amin((masks_without_obj_y * height) + (masks_with_obj_y * y), axis=1),
            np.amax(masks_with_obj_x * x, axis=1),
            np.amax(masks_with_obj_y * y, axis=1),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if boxes.shape != (num_masks, 4):
        raise AssertionError(f"unexpected boxes shape {boxes.shape}")
    return _from_numpy(boxes, masks)


def _ensure_numpy_bool_masks(
    pred_masks, gt_masks, name: str
) -> tuple[np.ndarray, np.ndarray]:
    pred = _to_numpy(pred_masks)
    gt = _to_numpy(gt_masks)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"{name} expects masks with shape (N, H, W) and (M, H, W).")
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError("pred_masks and gt_masks must have matching spatial shapes.")
    if pred.dtype != np.bool_ or gt.dtype != np.bool_:
        raise TypeError(f"{name} expects boolean masks.")
    return pred, gt


def _ensure_mlx_bool_masks(pred_masks, gt_masks, name: str):
    mx = _mlx()
    pred = _as_mlx_array(pred_masks)
    gt = _as_mlx_array(gt_masks)
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"{name} expects masks with shape (N, H, W) and (M, H, W).")
    if pred.shape[1:] != gt.shape[1:]:
        raise ValueError("pred_masks and gt_masks must have matching spatial shapes.")
    if pred.dtype != mx.bool_ or gt.dtype != mx.bool_:
        raise TypeError(f"{name} expects boolean masks.")
    return pred, gt


def _flatten_mlx_masks(masks):
    spatial_size = masks.shape[1] * masks.shape[2]
    return masks.reshape(masks.shape[0], spatial_size).astype(_mlx().float32)


def _mask_iou_mlx(pred, gt):
    mx = _mlx()
    pred_flat = _flatten_mlx_masks(pred)
    gt_flat = _flatten_mlx_masks(gt)
    intersection = pred_flat @ gt_flat.T
    area_pred = pred_flat.sum(axis=1)
    area_gt = gt_flat.sum(axis=1)
    union = area_pred[:, None] + area_gt[None, :] - intersection
    return intersection / mx.maximum(union, 1.0)


def _mask_iom_mlx(pred, gt):
    mx = _mlx()
    pred_flat = _flatten_mlx_masks(pred)
    gt_flat = _flatten_mlx_masks(gt)
    intersection = pred_flat @ gt_flat.T
    min_area = mx.minimum(
        pred_flat.sum(axis=1)[:, None],
        gt_flat.sum(axis=1)[None, :],
    )
    return intersection / mx.maximum(min_area, 1.0)


def mask_iou(pred_masks, gt_masks):
    """Compute boolean mask IoU, preserving NumPy or MLX execution."""

    if _is_mlx_array(pred_masks) or _is_mlx_array(gt_masks):
        pred, gt = _ensure_mlx_bool_masks(pred_masks, gt_masks, "mask_iou")
        return _mask_iou_mlx(pred, gt)

    pred, gt = _ensure_numpy_bool_masks(pred_masks, gt_masks, "mask_iou")
    pred_flat = pred.reshape(pred.shape[0], pred.shape[1] * pred.shape[2])
    gt_flat = gt.reshape(gt.shape[0], gt.shape[1] * gt.shape[2])
    intersection = pred_flat.astype(np.float32) @ gt_flat.astype(np.float32).T
    area_pred = pred_flat.sum(axis=1, dtype=np.float32)
    area_gt = gt_flat.sum(axis=1, dtype=np.float32)
    union = area_pred[:, None] + area_gt[None, :] - intersection
    out = intersection / np.clip(union, 1.0, None)
    return _from_numpy(out.astype(np.float32, copy=False), pred_masks)


def mask_iom(pred_masks, gt_masks):
    """Compute boolean mask IoM, preserving NumPy or MLX execution."""

    if _is_mlx_array(pred_masks) or _is_mlx_array(gt_masks):
        pred, gt = _ensure_mlx_bool_masks(pred_masks, gt_masks, "mask_iom")
        return _mask_iom_mlx(pred, gt)

    pred, gt = _ensure_numpy_bool_masks(pred_masks, gt_masks, "mask_iom")
    pred_flat = pred.reshape(pred.shape[0], pred.shape[1] * pred.shape[2])
    gt_flat = gt.reshape(gt.shape[0], gt.shape[1] * gt.shape[2])
    intersection = pred_flat.astype(np.float32) @ gt_flat.astype(np.float32).T
    min_area = np.minimum(
        pred_flat.sum(axis=1, dtype=np.float32)[:, None],
        gt_flat.sum(axis=1, dtype=np.float32)[None, :],
    )
    out = intersection / np.clip(min_area, 1.0, None)
    return _from_numpy(out.astype(np.float32, copy=False), pred_masks)
