from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from sam3_mlx.mlx_runtime import to_numpy as _runtime_to_numpy

_IOU_EPS = 1e-8
_IOM_EPS = 1e-8


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


def _has_mlx_array(*values: Any) -> bool:
    return any(_is_mlx_array(value) for value in values)


def _to_mlx(value: Any) -> mx.array:
    if _is_mlx_array(value):
        return value
    return mx.array(value)


def _to_numpy(value: Any) -> np.ndarray:
    return _runtime_to_numpy(value, copy=False)


def _to_host_postprocess_numpy(value: Any) -> np.ndarray:
    # host-postprocess-boundary: greedy NMS selection is still NumPy/Python.
    return _to_numpy(value)


def _from_numpy(value: np.ndarray, *templates: Any) -> Any:
    if any(_is_mlx_array(template) for template in templates):
        return mx.array(value)
    return value


def _divide_with_eps_mlx(
    numerator: mx.array, denominator: mx.array, eps: float
) -> mx.array:
    return numerator / (denominator + eps)


def _validate_pairwise_mask_shapes(
    pred_shape: tuple[int, ...],
    gt_shape: tuple[int, ...],
) -> None:
    pred_shape = tuple(pred_shape)
    gt_shape = tuple(gt_shape)
    if len(pred_shape) != 3 or len(gt_shape) != 3:
        raise ValueError(
            "mask overlap inputs must have shape (N, H, W) and (M, H, W); "
            f"got {pred_shape} and {gt_shape}"
        )
    if pred_shape[1:] != gt_shape[1:]:
        raise ValueError(
            "mask overlap spatial dimensions must match; "
            f"got {pred_shape[1:]} and {gt_shape[1:]}"
        )


def _validate_batched_mask_shape(mask_shape: tuple[int, ...]) -> None:
    mask_shape = tuple(mask_shape)
    if len(mask_shape) != 4:
        raise ValueError(
            f"batched masks must have shape (B, N, H, W), got {mask_shape}"
        )


def _pairwise_mask_iou_np(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    _validate_pairwise_mask_shapes(pred_masks.shape, gt_masks.shape)
    spatial_size = pred_masks.shape[1] * pred_masks.shape[2]
    pred_flat = pred_masks.reshape(pred_masks.shape[0], spatial_size).astype(np.float32)
    gt_flat = gt_masks.reshape(gt_masks.shape[0], spatial_size).astype(np.float32)
    intersection = pred_flat @ gt_flat.T
    pred_area = pred_flat.sum(axis=-1)[:, None]
    gt_area = gt_flat.sum(axis=-1)[None, :]
    union = pred_area + gt_area - intersection
    return intersection / np.maximum(union, 1.0)


def _pairwise_mask_iou_mlx(pred_masks: Any, gt_masks: Any) -> mx.array:
    pred_masks_mlx = _to_mlx(pred_masks)
    gt_masks_mlx = _to_mlx(gt_masks)
    _validate_pairwise_mask_shapes(pred_masks_mlx.shape, gt_masks_mlx.shape)
    spatial_size = pred_masks_mlx.shape[1] * pred_masks_mlx.shape[2]
    pred_flat = pred_masks_mlx.reshape(pred_masks_mlx.shape[0], spatial_size).astype(
        mx.float32
    )
    gt_flat = gt_masks_mlx.reshape(gt_masks_mlx.shape[0], spatial_size).astype(
        mx.float32
    )
    intersection = mx.matmul(pred_flat, mx.swapaxes(gt_flat, 0, 1))
    pred_area = mx.sum(pred_flat, axis=-1)[:, None]
    gt_area = mx.sum(gt_flat, axis=-1)[None, :]
    union = pred_area + gt_area - intersection
    return intersection / mx.maximum(union, mx.ones_like(union))


def _pairwise_mask_iom_np(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    _validate_pairwise_mask_shapes(pred_masks.shape, gt_masks.shape)
    spatial_size = pred_masks.shape[1] * pred_masks.shape[2]
    pred_flat = pred_masks.reshape(pred_masks.shape[0], spatial_size).astype(np.float32)
    gt_flat = gt_masks.reshape(gt_masks.shape[0], spatial_size).astype(np.float32)
    intersection = pred_flat @ gt_flat.T
    pred_area = pred_flat.sum(axis=-1)[:, None]
    gt_area = gt_flat.sum(axis=-1)[None, :]
    min_area = np.minimum(pred_area, gt_area)
    return intersection / (min_area + _IOM_EPS)


def _pairwise_mask_iom_mlx(pred_masks: Any, gt_masks: Any) -> mx.array:
    pred_masks_mlx = _to_mlx(pred_masks)
    gt_masks_mlx = _to_mlx(gt_masks)
    _validate_pairwise_mask_shapes(pred_masks_mlx.shape, gt_masks_mlx.shape)
    spatial_size = pred_masks_mlx.shape[1] * pred_masks_mlx.shape[2]
    pred_flat = pred_masks_mlx.reshape(pred_masks_mlx.shape[0], spatial_size).astype(
        mx.float32
    )
    gt_flat = gt_masks_mlx.reshape(gt_masks_mlx.shape[0], spatial_size).astype(
        mx.float32
    )
    intersection = mx.matmul(pred_flat, mx.swapaxes(gt_flat, 0, 1))
    pred_area = mx.sum(pred_flat, axis=-1)[:, None]
    gt_area = mx.sum(gt_flat, axis=-1)[None, :]
    min_area = mx.minimum(pred_area, gt_area)
    return _divide_with_eps_mlx(intersection, min_area, _IOM_EPS)


def _self_mask_iom_source_area_np(masks: np.ndarray) -> np.ndarray:
    _validate_pairwise_mask_shapes(masks.shape, masks.shape)
    spatial_size = masks.shape[1] * masks.shape[2]
    masks_flat = masks.reshape(masks.shape[0], spatial_size).astype(np.float32)
    intersection = masks_flat @ masks_flat.T
    area = masks_flat.sum(axis=-1)[:, None]
    return intersection / (area + _IOM_EPS)


def _self_mask_iom_source_area_mlx(masks: Any) -> mx.array:
    masks_mlx = _to_mlx(masks)
    _validate_pairwise_mask_shapes(masks_mlx.shape, masks_mlx.shape)
    spatial_size = masks_mlx.shape[1] * masks_mlx.shape[2]
    masks_flat = masks_mlx.reshape(masks_mlx.shape[0], spatial_size).astype(mx.float32)
    intersection = mx.matmul(masks_flat, mx.swapaxes(masks_flat, 0, 1))
    area = mx.sum(masks_flat, axis=-1)[:, None]
    return _divide_with_eps_mlx(intersection, area, _IOM_EPS)


def _generic_nms_keep_np(
    overlaps: np.ndarray,
    scores: np.ndarray,
    is_valid: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    keep = np.zeros(scores.shape[0], dtype=bool)
    suppressed = np.zeros(scores.shape[0], dtype=bool)

    for idx in order:
        if suppressed[idx] or not is_valid[idx]:
            continue
        keep[idx] = True
        suppressed |= overlaps[idx] > iou_threshold
        suppressed[idx] = False

    return keep


def _generic_nms_mask_np(
    overlaps: np.ndarray,
    scores: np.ndarray,
    is_valid: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    keep_sorted = is_valid[order].astype(bool, copy=True)
    overlaps_sorted = overlaps[order][:, order]

    # Mirrors upstream generic_nms_mask in the active detector perflib path.
    for index in range(scores.shape[0]):
        suppress = overlaps_sorted[index] > iou_threshold
        suppress[: index + 1] = False
        keep_sorted[suppress] = False

    keep = np.zeros(scores.shape[0], dtype=bool)
    keep[order] = keep_sorted
    return keep


def nms_masks(
    pred_probs: Any,
    pred_masks: Any,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
    do_compile: bool = False,
    running_in_prod: bool = False,
) -> Any:
    """
    Score-threshold and mask-NMS detections.

    NumPy inputs stay on the NumPy path. MLX inputs compute mask overlaps with
    MLX primitives and only materialize scores/valid flags/overlap matrices for
    the greedy NMS selection.
    """
    del running_in_prod
    if do_compile:
        raise NotImplementedError(
            "do_compile=True requests the upstream torch.compile NMS path; "
            "the MLX port implements only the eager NumPy/MLX-safe helper."
        )
    return _nms_masks_core(
        pred_probs, pred_masks, prob_threshold, iou_threshold, nms_use_iom
    )


def _nms_masks_core(
    pred_probs: Any,
    pred_masks: Any,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> Any:
    if _has_mlx_array(pred_probs, pred_masks):
        return _nms_masks_core_mlx(
            pred_probs, pred_masks, prob_threshold, iou_threshold, nms_use_iom
        )

    probs_np = _to_numpy(pred_probs)
    masks_np = _to_numpy(pred_masks)
    if probs_np.ndim == 2:
        keep = _nms_masks_core_batched_np(
            probs_np, masks_np, prob_threshold, iou_threshold, nms_use_iom
        )
    elif probs_np.ndim == 1:
        keep = _nms_masks_core_single_np(
            probs_np, masks_np, prob_threshold, iou_threshold, nms_use_iom
        )
    else:
        raise ValueError(
            "pred_probs must have shape (num_det,) or (B, num_det); "
            f"got {probs_np.shape}"
        )
    return _from_numpy(keep, pred_probs, pred_masks)


def _nms_masks_core_mlx(
    pred_probs: Any,
    pred_masks: Any,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> mx.array:
    probs_mlx = _to_mlx(pred_probs)
    masks_mlx = _to_mlx(pred_masks)
    if len(probs_mlx.shape) == 2:
        return _nms_masks_core_batched_mlx(
            probs_mlx, masks_mlx, prob_threshold, iou_threshold, nms_use_iom
        )
    if len(probs_mlx.shape) == 1:
        return _nms_masks_core_single_mlx(
            probs_mlx, masks_mlx, prob_threshold, iou_threshold, nms_use_iom
        )
    raise ValueError(
        f"pred_probs must have shape (num_det,) or (B, num_det); got {probs_mlx.shape}"
    )


def _nms_masks_core_batched(
    pred_probs: Any,
    pred_masks: Any,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> Any:
    if _has_mlx_array(pred_probs, pred_masks):
        return _nms_masks_core_batched_mlx(
            _to_mlx(pred_probs),
            _to_mlx(pred_masks),
            prob_threshold,
            iou_threshold,
            nms_use_iom,
        )

    probs_np = _to_numpy(pred_probs)
    masks_np = _to_numpy(pred_masks)
    keep = _nms_masks_core_batched_np(
        probs_np, masks_np, prob_threshold, iou_threshold, nms_use_iom
    )
    return _from_numpy(keep, pred_probs, pred_masks)


def _nms_masks_core_batched_mlx(
    pred_probs: mx.array,
    pred_masks: mx.array,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> mx.array:
    if len(pred_masks.shape) != 4:
        raise ValueError(
            f"batched pred_masks must have shape (B, N, H, W), got {pred_masks.shape}"
        )
    batch_size, num_det = pred_probs.shape
    if pred_masks.shape[:2] != (batch_size, num_det):
        raise ValueError(
            "pred_probs and pred_masks leading dimensions must match; "
            f"got {pred_probs.shape} and {pred_masks.shape}"
        )

    # host-postprocess-boundary: scores and overlaps feed greedy NMS.
    scores_np = _to_host_postprocess_numpy(pred_probs)
    is_valid = scores_np > prob_threshold
    keep = np.zeros_like(is_valid, dtype=bool)
    if not np.any(is_valid):
        return mx.array(keep)

    for batch_idx in range(batch_size):
        valid_indices = np.flatnonzero(is_valid[batch_idx])
        if valid_indices.size == 0:
            continue

        valid_indices_mx = mx.array(valid_indices, dtype=mx.int64)
        masks_binary = mx.take(pred_masks[batch_idx], valid_indices_mx, axis=0) > 0
        overlaps = (
            _pairwise_mask_iom_mlx(masks_binary, masks_binary)
            if nms_use_iom
            else _pairwise_mask_iou_mlx(masks_binary, masks_binary)
        )
        overlaps_np = _to_host_postprocess_numpy(overlaps)
        kept_local = generic_nms_cpu_np(
            overlaps_np,
            scores_np[batch_idx, valid_indices],
            iou_threshold,
        )
        keep[batch_idx, valid_indices[kept_local]] = True
    return mx.array(keep)


def _nms_masks_core_batched_np(
    pred_probs: np.ndarray,
    pred_masks: np.ndarray,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> np.ndarray:
    if pred_masks.ndim != 4:
        raise ValueError(
            f"batched pred_masks must have shape (B, N, H, W), got {pred_masks.shape}"
        )
    batch_size, num_det = pred_probs.shape
    if pred_masks.shape[:2] != (batch_size, num_det):
        raise ValueError(
            "pred_probs and pred_masks leading dimensions must match; "
            f"got {pred_probs.shape} and {pred_masks.shape}"
        )

    is_valid = pred_probs > prob_threshold
    keep = np.zeros_like(is_valid, dtype=bool)
    for batch_idx in range(batch_size):
        valid_indices = np.nonzero(is_valid[batch_idx])[0]
        if valid_indices.size == 0:
            continue
        masks_binary = pred_masks[batch_idx, valid_indices] > 0
        overlaps = (
            _pairwise_mask_iom_np(masks_binary, masks_binary)
            if nms_use_iom
            else _pairwise_mask_iou_np(masks_binary, masks_binary)
        )
        kept_local = generic_nms_cpu_np(
            overlaps,
            pred_probs[batch_idx, valid_indices],
            iou_threshold,
        )
        keep[batch_idx, valid_indices[kept_local]] = True
    return keep


def _batched_mask_iou(masks: Any) -> Any:
    if _is_mlx_array(masks):
        return _batched_mask_iou_mlx(masks)

    masks_np = _to_numpy(masks)
    return _from_numpy(_batched_mask_iou_np(masks_np), masks)


def _batched_mask_iou_mlx(masks: Any) -> mx.array:
    masks_mlx = _to_mlx(masks)
    _validate_batched_mask_shape(masks_mlx.shape)
    batch_size, num_masks = masks_mlx.shape[:2]
    spatial_size = masks_mlx.shape[2] * masks_mlx.shape[3]
    masks_flat = masks_mlx.reshape(batch_size, num_masks, spatial_size).astype(
        mx.float32
    )
    intersection = mx.matmul(masks_flat, mx.swapaxes(masks_flat, 1, 2))
    areas = mx.sum(masks_flat, axis=-1)
    union = areas[:, :, None] + areas[:, None, :] - intersection
    return _divide_with_eps_mlx(intersection, union, _IOU_EPS)


def _batched_mask_iou_np(masks: np.ndarray) -> np.ndarray:
    _validate_batched_mask_shape(masks.shape)
    batch_size, num_masks = masks.shape[:2]
    spatial_size = masks.shape[2] * masks.shape[3]
    masks_flat = masks.reshape(batch_size, num_masks, spatial_size).astype(np.float32)
    intersection = np.matmul(masks_flat, np.swapaxes(masks_flat, 1, 2))
    areas = masks_flat.sum(axis=-1)
    union = areas[:, :, None] + areas[:, None, :] - intersection
    return intersection / (union + _IOU_EPS)


def _batched_mask_iom(masks: Any) -> Any:
    if _is_mlx_array(masks):
        return _batched_mask_iom_mlx(masks)

    masks_np = _to_numpy(masks)
    return _from_numpy(_batched_mask_iom_np(masks_np), masks)


def _batched_mask_iom_mlx(masks: Any) -> mx.array:
    masks_mlx = _to_mlx(masks)
    _validate_batched_mask_shape(masks_mlx.shape)
    batch_size, num_masks = masks_mlx.shape[:2]
    spatial_size = masks_mlx.shape[2] * masks_mlx.shape[3]
    masks_flat = masks_mlx.reshape(batch_size, num_masks, spatial_size).astype(
        mx.float32
    )
    intersection = mx.matmul(masks_flat, mx.swapaxes(masks_flat, 1, 2))
    areas = mx.sum(masks_flat, axis=-1)
    min_area = mx.minimum(areas[:, :, None], areas[:, None, :])
    return _divide_with_eps_mlx(intersection, min_area, _IOM_EPS)


def _batched_mask_iom_np(masks: np.ndarray) -> np.ndarray:
    _validate_batched_mask_shape(masks.shape)
    batch_size, num_masks = masks.shape[:2]
    spatial_size = masks.shape[2] * masks.shape[3]
    masks_flat = masks.reshape(batch_size, num_masks, spatial_size).astype(np.float32)
    intersection = np.matmul(masks_flat, np.swapaxes(masks_flat, 1, 2))
    areas = masks_flat.sum(axis=-1)
    min_area = np.minimum(areas[:, :, None], areas[:, None, :])
    return intersection / (min_area + _IOM_EPS)


def _batched_generic_nms_mask(
    ious: Any,
    scores: Any,
    is_valid: Any,
    iou_threshold: float,
) -> Any:
    # host-postprocess-boundary: batched greedy NMS is host-side.
    ious_np = _to_host_postprocess_numpy(ious)
    scores_np = _to_host_postprocess_numpy(scores)
    is_valid_np = _to_host_postprocess_numpy(is_valid).astype(bool)
    keep = np.zeros_like(is_valid_np, dtype=bool)
    for batch_idx in range(scores_np.shape[0]):
        keep[batch_idx] = _generic_nms_keep_np(
            ious_np[batch_idx],
            scores_np[batch_idx],
            is_valid_np[batch_idx],
            iou_threshold,
        )
    return _from_numpy(keep, ious, scores, is_valid)


def _nms_masks_core_single(
    pred_probs: Any,
    pred_masks: Any,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> Any:
    if _has_mlx_array(pred_probs, pred_masks):
        return _nms_masks_core_single_mlx(
            _to_mlx(pred_probs),
            _to_mlx(pred_masks),
            prob_threshold,
            iou_threshold,
            nms_use_iom,
        )

    probs_np = _to_numpy(pred_probs)
    masks_np = _to_numpy(pred_masks)
    keep = _nms_masks_core_single_np(
        probs_np, masks_np, prob_threshold, iou_threshold, nms_use_iom
    )
    return _from_numpy(keep, pred_probs, pred_masks)


def _nms_masks_core_single_mlx(
    pred_probs: mx.array,
    pred_masks: mx.array,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> mx.array:
    if len(pred_masks.shape) != 3:
        raise ValueError(
            f"pred_masks must have shape (N, H, W), got {pred_masks.shape}"
        )
    if pred_masks.shape[0] != pred_probs.shape[0]:
        raise ValueError(
            "pred_probs and pred_masks must contain the same number of detections; "
            f"got {pred_probs.shape[0]} and {pred_masks.shape[0]}"
        )

    # host-postprocess-boundary: scores and overlaps feed NMS ordering.
    scores_np = _to_host_postprocess_numpy(pred_probs)
    is_valid = scores_np > prob_threshold
    keep = np.zeros_like(is_valid, dtype=bool)
    if not np.any(is_valid):
        return mx.array(keep)

    masks_binary = pred_masks > 0
    overlaps = (
        _self_mask_iom_source_area_mlx(masks_binary)
        if nms_use_iom
        else _pairwise_mask_iou_mlx(masks_binary, masks_binary)
    )
    overlaps_np = _to_host_postprocess_numpy(overlaps)
    keep = _generic_nms_mask_np(
        overlaps_np,
        scores_np,
        is_valid,
        iou_threshold,
    )
    return mx.array(keep)


def _nms_masks_core_single_np(
    pred_probs: np.ndarray,
    pred_masks: np.ndarray,
    prob_threshold: float,
    iou_threshold: float,
    nms_use_iom: bool = False,
) -> np.ndarray:
    if pred_masks.ndim != 3:
        raise ValueError(
            f"pred_masks must have shape (N, H, W), got {pred_masks.shape}"
        )
    if pred_masks.shape[0] != pred_probs.shape[0]:
        raise ValueError(
            "pred_probs and pred_masks must contain the same number of detections; "
            f"got {pred_probs.shape[0]} and {pred_masks.shape[0]}"
        )

    is_valid = pred_probs > prob_threshold
    keep = np.zeros_like(is_valid, dtype=bool)
    if not np.any(is_valid):
        return keep

    masks_binary = pred_masks > 0
    overlaps = (
        _self_mask_iom_source_area_np(masks_binary)
        if nms_use_iom
        else _pairwise_mask_iou_np(masks_binary, masks_binary)
    )
    return _generic_nms_mask_np(
        overlaps,
        pred_probs,
        is_valid,
        iou_threshold,
    )


def generic_nms_cpu(ious: Any, scores: Any, iou_threshold: float = 0.5) -> Any:
    # host-postprocess-boundary: public generic NMS is host-side.
    kept = generic_nms_cpu_np(
        _to_host_postprocess_numpy(ious),
        _to_host_postprocess_numpy(scores),
        iou_threshold,
    )
    return _from_numpy(kept, ious, scores)


def generic_nms_cpu_np(
    ious: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    order = np.argsort(scores)[::-1]
    kept_inds: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        kept_inds.append(idx)
        remaining = np.where(ious[idx, order[1:]] <= iou_threshold)[0]
        order = order[remaining + 1]
    return np.asarray(kept_inds, dtype=np.int64)


def generic_nms_mask(
    ious: Any,
    scores: Any,
    is_valid: Any,
    iou_threshold: float = 0.5,
) -> Any:
    # host-postprocess-boundary: public generic mask NMS is host-side.
    keep = _generic_nms_mask_np(
        _to_host_postprocess_numpy(ious),
        _to_host_postprocess_numpy(scores),
        _to_host_postprocess_numpy(is_valid).astype(bool),
        iou_threshold,
    )
    return _from_numpy(keep, ious, scores, is_valid)


def perf_mask_iou(pred_masks: Any, gt_masks: Any) -> Any:
    if _has_mlx_array(pred_masks, gt_masks):
        ious = _pairwise_mask_iou_mlx(
            _to_mlx(pred_masks).astype(mx.bool_),
            _to_mlx(gt_masks).astype(mx.bool_),
        )
        return ious

    ious = _pairwise_mask_iou_np(
        _to_numpy(pred_masks).astype(bool),
        _to_numpy(gt_masks).astype(bool),
    )
    return _from_numpy(ious, pred_masks, gt_masks)


def perf_mask_iom(pred_masks: Any, gt_masks: Any) -> Any:
    if _has_mlx_array(pred_masks, gt_masks):
        ioms = _pairwise_mask_iom_mlx(
            _to_mlx(pred_masks).astype(mx.bool_),
            _to_mlx(gt_masks).astype(mx.bool_),
        )
        return ioms

    ioms = _pairwise_mask_iom_np(
        _to_numpy(pred_masks).astype(bool),
        _to_numpy(gt_masks).astype(bool),
    )
    return _from_numpy(ioms, pred_masks, gt_masks)


mask_iou = perf_mask_iou
mask_iom = perf_mask_iom
