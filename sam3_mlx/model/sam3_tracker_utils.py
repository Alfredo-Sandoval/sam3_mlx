# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX/NumPy tracker utility ports from the official SAM3 tree."""

from __future__ import annotations

from collections import deque
from typing import Protocol, TypeAlias, TypeGuard, TypeVar, cast, overload

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from sam3_mlx.model.edt import edt_triton
from sam3_mlx.mlx_runtime import to_numpy


NumpyArray: TypeAlias = NDArray[np.generic]
BoolNumpyArray: TypeAlias = NDArray[np.bool_]
FloatingNumpyArray: TypeAlias = (
    NDArray[np.float16] | NDArray[np.float32] | NDArray[np.float64]
)
TrackerArray: TypeAlias = mx.array | NumpyArray
_FrameOutputT = TypeVar("_FrameOutputT")


class _MlxPad(Protocol):
    def __call__(
        self,
        array: mx.array,
        pad_width: list[tuple[int, int]],
        mode: str = "constant",
        constant_values: int | float | bool | mx.array = 0,
    ) -> mx.array: ...


_mlx_pad = cast(_MlxPad, getattr(mx, "pad"))


def _is_mlx_array(value: object) -> TypeGuard[mx.array]:
    return type(value).__module__.startswith("mlx.")


TRACKER_CPU_BOUNDARIES = {
    "fill_holes_in_mask_scores": "CPU connected-components cleanup for holes and sprinkles.",
    "_get_connected_components_with_padding": "CPU 4-connected component labeling.",
    "sample_one_point_from_error_center": "CPU EDT boundary through edt_triton.",
}


def _to_host_tracker_input(value: TrackerArray, *, copy: bool = False) -> NumpyArray:
    """Synchronize and export tracker tensors at an explicit CPU cleanup boundary."""

    return cast(NumpyArray, to_numpy(value, copy=copy))


@overload
def _zeros_like_backend(value: mx.array) -> mx.array: ...


@overload
def _zeros_like_backend(value: NumpyArray) -> NumpyArray: ...


def _zeros_like_backend(value: TrackerArray) -> TrackerArray:
    return mx.zeros_like(value) if _is_mlx_array(value) else np.zeros_like(value)


@overload
def sample_box_points(
    masks: mx.array,
    noise: float = 0.1,
    noise_bound: int = 20,
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> tuple[mx.array, mx.array]: ...


@overload
def sample_box_points(
    masks: NumpyArray,
    noise: float = 0.1,
    noise_bound: int = 20,
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> tuple[NumpyArray, NumpyArray]: ...


def sample_box_points(
    masks: TrackerArray,
    noise: float = 0.1,
    noise_bound: int = 20,
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> tuple[TrackerArray, TrackerArray]:
    """
    Sample a noised version of the top-left and bottom-right mask box corners.

    Inputs follow the official shape ``[B, 1, H, W]``. Coordinates are returned
    as ``(x, y)`` pairs with shape ``[B, 2, 2]``.
    """

    bsz, _, height, width = masks.shape

    if _is_mlx_array(masks):
        box_coords = mask_to_box(masks)
        box_coords = box_coords.astype(mx.float32)
        box_labels = mx.tile(
            mx.array([top_left_label, bottom_right_label], dtype=mx.int32),
            (bsz,),
        )
        if noise > 0.0:
            bbox_w = box_coords[..., 2] - box_coords[..., 0]
            bbox_h = box_coords[..., 3] - box_coords[..., 1]
            noise_bound_arr = mx.array(noise_bound, dtype=box_coords.dtype)
            max_dx = mx.minimum(bbox_w * noise, noise_bound_arr)
            max_dy = mx.minimum(bbox_h * noise, noise_bound_arr)
            box_noise = 2 * mx.random.uniform(shape=(bsz, 1, 4)) - 1
            box_noise = box_noise * mx.stack([max_dx, max_dy, max_dx, max_dy], axis=-1)
            img_bounds = (
                mx.array([width, height, width, height], dtype=box_coords.dtype) - 1
            )
            box_coords = mx.minimum(mx.maximum(box_coords + box_noise, 0), img_bounds)
        return mx.reshape(box_coords, (-1, 2, 2)), mx.reshape(box_labels, (-1, 2))

    masks_np = cast(NumpyArray, masks)
    box_coords = np.asarray(mask_to_box(masks_np), dtype=np.float32)
    box_labels = np.tile(
        np.array([top_left_label, bottom_right_label], dtype=np.int32),
        bsz,
    )
    if noise > 0.0:
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = np.minimum(bbox_w * noise, noise_bound)
        max_dy = np.minimum(bbox_h * noise, noise_bound)
        box_noise = 2 * np.random.random((bsz, 1, 4)).astype(np.float32) - 1
        box_noise = box_noise * np.stack((max_dx, max_dy, max_dx, max_dy), axis=-1)
        img_bounds = np.array([width, height, width, height], dtype=np.float32) - 1
        box_coords = np.clip(box_coords + box_noise, 0, img_bounds)
    return cast(NumpyArray, box_coords.reshape(-1, 2, 2)), box_labels.reshape(-1, 2)


@overload
def mask_to_box(masks: mx.array) -> mx.array: ...


@overload
def mask_to_box(masks: NumpyArray) -> NumpyArray: ...


def mask_to_box(masks: TrackerArray) -> TrackerArray:
    """
    Compute bounding boxes for masks with shape ``[B, 1, H, W]``.

    Returns ``[B, 1, 4]`` boxes in ``(x_min, y_min, x_max, y_max)`` order.
    Empty masks receive all-zero boxes, matching the upstream behavior.
    """

    bsz, _, height, width = masks.shape
    if _is_mlx_array(masks):
        mask_bool = masks.astype(mx.bool_)
        mask_area = mx.sum(mask_bool, axis=(-1, -2))
        xs = mx.reshape(mx.arange(width, dtype=mx.int32), (1, 1, 1, width))
        ys = mx.reshape(mx.arange(height, dtype=mx.int32), (1, 1, height, 1))
        grid_xs = mx.broadcast_to(xs, (bsz, 1, height, width))
        grid_ys = mx.broadcast_to(ys, (bsz, 1, height, width))
        min_xs = mx.min(mx.where(mask_bool, grid_xs, width), axis=(-1, -2))
        max_xs = mx.max(mx.where(mask_bool, grid_xs, -1), axis=(-1, -2))
        min_ys = mx.min(mx.where(mask_bool, grid_ys, height), axis=(-1, -2))
        max_ys = mx.max(mx.where(mask_bool, grid_ys, -1), axis=(-1, -2))
        bbox_coords = mx.stack([min_xs, min_ys, max_xs, max_ys], axis=-1)
        return mx.where(
            mask_area[..., None] > 0, bbox_coords, mx.zeros_like(bbox_coords)
        )

    masks_np = cast(NumpyArray, masks)
    mask_bool = np.asarray(masks_np, dtype=np.bool_)
    mask_area = np.sum(mask_bool, axis=(-1, -2))
    xs = np.arange(width, dtype=np.int32).reshape(1, 1, 1, width)
    ys = np.arange(height, dtype=np.int32).reshape(1, 1, height, 1)
    grid_xs = np.broadcast_to(xs, (bsz, 1, height, width))
    grid_ys = np.broadcast_to(ys, (bsz, 1, height, width))
    min_xs = np.min(np.where(mask_bool, grid_xs, width), axis=(-1, -2))
    max_xs = np.max(np.where(mask_bool, grid_xs, -1), axis=(-1, -2))
    min_ys = np.min(np.where(mask_bool, grid_ys, height), axis=(-1, -2))
    max_ys = np.max(np.where(mask_bool, grid_ys, -1), axis=(-1, -2))
    bbox_coords = np.stack((min_xs, min_ys, max_xs, max_ys), axis=-1)
    return np.where(mask_area[..., None] > 0, bbox_coords, np.zeros_like(bbox_coords))


@overload
def sample_random_points_from_errors(
    gt_masks: mx.array, pred_masks: mx.array | None, num_pt: int = 1
) -> tuple[mx.array, mx.array]: ...


@overload
def sample_random_points_from_errors(
    gt_masks: NumpyArray, pred_masks: NumpyArray | None, num_pt: int = 1
) -> tuple[NumpyArray, NumpyArray]: ...


def sample_random_points_from_errors(
    gt_masks: TrackerArray,
    pred_masks: TrackerArray | None,
    num_pt: int = 1,
) -> tuple[TrackerArray, TrackerArray]:
    """Sample correction points from false-positive and false-negative regions."""

    if pred_masks is None:
        pred_masks = _zeros_like_backend(gt_masks)
    assert num_pt >= 0
    bsz, _, height, width = gt_masks.shape

    if _is_mlx_array(gt_masks):
        pred_masks = cast(mx.array, pred_masks)
        gt_masks = gt_masks.astype(mx.bool_)
        pred_masks = pred_masks.astype(mx.bool_)
        fp_masks = (~gt_masks) & pred_masks
        fn_masks = gt_masks & (~pred_masks)
        all_correct = mx.all(
            mx.reshape(mx.equal(gt_masks, pred_masks), (bsz, 1, -1)), axis=2
        )
        all_correct = all_correct[..., None, None]
        pts_noise = mx.random.uniform(shape=(bsz, num_pt, height, width, 2))
        neg_map = fp_masks | (all_correct & (~gt_masks))
        pos_map = fn_masks
        pts_noise = mx.stack(
            [pts_noise[..., 0] * neg_map, pts_noise[..., 1] * pos_map],
            axis=-1,
        )
        pts_idx = mx.argmax(mx.reshape(pts_noise, (bsz, num_pt, -1)), axis=2)
        labels = (pts_idx % 2).astype(mx.int32)
        pts_idx = pts_idx // 2
        pts_x = pts_idx % width
        pts_y = pts_idx // width
        points = mx.stack([pts_x, pts_y], axis=2).astype(mx.float32)
        return points, labels

    gt_masks_np = cast(NumpyArray, gt_masks)
    pred_masks_np = cast(NumpyArray, pred_masks)
    gt_bool = np.asarray(gt_masks_np, dtype=np.bool_)
    pred_bool = np.asarray(pred_masks_np, dtype=np.bool_)
    fp_masks = ~gt_bool & pred_bool
    fn_masks = gt_bool & ~pred_bool
    all_correct = np.all((gt_bool == pred_bool).reshape(bsz, 1, -1), axis=2)
    all_correct = all_correct[..., None, None]
    pts_noise = np.random.random((bsz, num_pt, height, width, 2)).astype(np.float32)
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_bool)
    pts_noise[..., 1] *= fn_masks
    pts_idx = np.argmax(pts_noise.reshape(bsz, num_pt, -1), axis=2)
    labels = (pts_idx % 2).astype(np.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % width
    pts_y = pts_idx // width
    points = np.stack([pts_x, pts_y], axis=2).astype(np.float32)
    return points, labels


@overload
def sample_one_point_from_error_center(
    gt_masks: mx.array, pred_masks: mx.array | None, padding: bool = True
) -> tuple[mx.array, mx.array]: ...


@overload
def sample_one_point_from_error_center(
    gt_masks: NumpyArray, pred_masks: NumpyArray | None, padding: bool = True
) -> tuple[NumpyArray, NumpyArray]: ...


def sample_one_point_from_error_center(
    gt_masks: TrackerArray,
    pred_masks: TrackerArray | None,
    padding: bool = True,
) -> tuple[TrackerArray, TrackerArray]:
    """Sample one correction point from the largest EDT center of an error region."""

    if pred_masks is None:
        pred_masks = _zeros_like_backend(gt_masks)
    bsz, _, _, width = gt_masks.shape

    if _is_mlx_array(gt_masks):
        pred_masks = cast(mx.array, pred_masks)
        gt_masks = gt_masks.astype(mx.bool_)
        pred_masks = pred_masks.astype(mx.bool_)
        fp_masks = mx.squeeze((~gt_masks) & pred_masks, axis=1)
        fn_masks = mx.squeeze(gt_masks & (~pred_masks), axis=1)
        if padding:
            fp_masks = _mlx_pad(fp_masks, [(0, 0), (1, 1), (1, 1)])
            fn_masks = _mlx_pad(fn_masks, [(0, 0), (1, 1), (1, 1)])
        fn_mask_dt = edt_triton(fn_masks)
        fp_mask_dt = edt_triton(fp_masks)
        if padding:
            fn_mask_dt = fn_mask_dt[:, 1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[:, 1:-1, 1:-1]
        fn_flat = mx.reshape(fn_mask_dt, (bsz, -1))
        fp_flat = mx.reshape(fp_mask_dt, (bsz, -1))
        fn_argmax = mx.argmax(fn_flat, axis=-1)
        fp_argmax = mx.argmax(fp_flat, axis=-1)
        fn_max = mx.max(fn_flat, axis=-1)
        fp_max = mx.max(fp_flat, axis=-1)
        is_positive = fn_max > fp_max
        chosen = mx.where(is_positive, fn_argmax, fp_argmax)
        points = mx.stack([chosen % width, chosen // width], axis=-1)
        labels = is_positive.astype(mx.int64)
        return mx.expand_dims(points, axis=1), mx.expand_dims(labels, axis=1)

    gt_masks_np = cast(NumpyArray, gt_masks)
    pred_masks_np = cast(NumpyArray, pred_masks)
    gt_bool = np.asarray(gt_masks_np, dtype=np.bool_)
    pred_bool = np.asarray(pred_masks_np, dtype=np.bool_)
    fp_masks = (~gt_bool & pred_bool).squeeze(1)
    fn_masks = (gt_bool & ~pred_bool).squeeze(1)
    if padding:
        fp_masks = np.pad(fp_masks, ((0, 0), (1, 1), (1, 1)), "constant")
        fn_masks = np.pad(fn_masks, ((0, 0), (1, 1), (1, 1)), "constant")
    fn_mask_dt = edt_triton(fn_masks)
    fp_mask_dt = edt_triton(fp_masks)
    if padding:
        fn_mask_dt = fn_mask_dt[:, 1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[:, 1:-1, 1:-1]
    fn_flat = fn_mask_dt.reshape(bsz, -1)
    fp_flat = fp_mask_dt.reshape(bsz, -1)
    fn_argmax = np.argmax(fn_flat, axis=-1)
    fp_argmax = np.argmax(fp_flat, axis=-1)
    is_positive = np.max(fn_flat, axis=-1) > np.max(fp_flat, axis=-1)
    chosen = np.where(is_positive, fn_argmax, fp_argmax)
    points = np.stack([chosen % width, chosen // width], axis=-1)
    labels = is_positive.astype(np.int64)
    return np.expand_dims(points, axis=1), np.expand_dims(labels, axis=1)


@overload
def sample_one_point_from_error_center_slow(
    gt_masks: mx.array, pred_masks: mx.array | None, padding: bool = True
) -> tuple[mx.array, mx.array]: ...


@overload
def sample_one_point_from_error_center_slow(
    gt_masks: NumpyArray, pred_masks: NumpyArray | None, padding: bool = True
) -> tuple[NumpyArray, NumpyArray]: ...


def sample_one_point_from_error_center_slow(
    gt_masks: TrackerArray,
    pred_masks: TrackerArray | None,
    padding: bool = True,
) -> tuple[TrackerArray, TrackerArray]:
    """
    Compatibility alias for the upstream OpenCV-based slow path.

    OpenCV is not a runtime dependency of ``sam3_mlx``; the MLX/NumPy EDT path is
    used instead.
    """

    if _is_mlx_array(gt_masks):
        return sample_one_point_from_error_center(
            gt_masks,
            cast(mx.array | None, pred_masks),
            padding=padding,
        )
    return sample_one_point_from_error_center(
        cast(NumpyArray, gt_masks),
        cast(NumpyArray | None, pred_masks),
        padding=padding,
    )


@overload
def get_next_point(
    gt_masks: mx.array, pred_masks: mx.array | None, method: str
) -> tuple[mx.array, mx.array]: ...


@overload
def get_next_point(
    gt_masks: NumpyArray, pred_masks: NumpyArray | None, method: str
) -> tuple[NumpyArray, NumpyArray]: ...


def get_next_point(
    gt_masks: TrackerArray,
    pred_masks: TrackerArray | None,
    method: str,
) -> tuple[TrackerArray, TrackerArray]:
    if _is_mlx_array(gt_masks):
        mlx_pred_masks = cast(mx.array | None, pred_masks)
        if method == "uniform":
            return sample_random_points_from_errors(gt_masks, mlx_pred_masks)
        if method == "center":
            return sample_one_point_from_error_center(gt_masks, mlx_pred_masks)
        raise ValueError(f"unknown sampling method {method}")

    numpy_gt_masks = cast(NumpyArray, gt_masks)
    numpy_pred_masks = cast(NumpyArray | None, pred_masks)
    if method == "uniform":
        return sample_random_points_from_errors(numpy_gt_masks, numpy_pred_masks)
    if method == "center":
        return sample_one_point_from_error_center(numpy_gt_masks, numpy_pred_masks)
    raise ValueError(f"unknown sampling method {method}")


def select_closest_cond_frames(
    frame_idx: int,
    cond_frame_outputs: dict[int, _FrameOutputT],
    max_cond_frame_num: int,
    keep_first_cond_frame: bool = False,
) -> tuple[dict[int, _FrameOutputT], dict[int, _FrameOutputT]]:
    """
    Select temporally closest conditioning frames, matching the official helper.
    """

    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs: dict[int, _FrameOutputT] = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs: dict[int, _FrameOutputT] = {}
        if keep_first_cond_frame:
            idx_first = min(
                (t for t in cond_frame_outputs if t < frame_idx), default=None
            )
            if idx_first is None:
                idx_first = max(
                    (t for t in cond_frame_outputs if t > frame_idx), default=None
                )
            if idx_first is not None:
                selected_outputs[idx_first] = cond_frame_outputs[idx_first]
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs


@overload
def get_1d_sine_pe(
    pos_inds: mx.array, dim: int, temperature: int = 10000
) -> mx.array: ...


@overload
def get_1d_sine_pe(
    pos_inds: NumpyArray, dim: int, temperature: int = 10000
) -> NumpyArray: ...


def get_1d_sine_pe(
    pos_inds: TrackerArray, dim: int, temperature: int = 10000
) -> TrackerArray:
    """Get 1D sine positional embedding as in the original Transformer paper."""

    pe_dim = dim // 2
    if _is_mlx_array(pos_inds):
        dim_t = mx.arange(pe_dim, dtype=mx.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
        pos_embed = mx.expand_dims(pos_inds, axis=-1) / dim_t
        return mx.concat([mx.sin(pos_embed), mx.cos(pos_embed)], axis=-1)

    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = np.expand_dims(pos_inds, axis=-1) / dim_t
    return np.concatenate([np.sin(pos_embed), np.cos(pos_embed)], axis=-1)


@overload
def get_best_gt_match_from_multimasks(
    pred_multimasks: mx.array,
    gt_masks: mx.array,
    pred_scores: mx.array | None = None,
) -> mx.array: ...


@overload
def get_best_gt_match_from_multimasks(
    pred_multimasks: NumpyArray,
    gt_masks: NumpyArray,
    pred_scores: NumpyArray | None = None,
) -> NumpyArray: ...


def get_best_gt_match_from_multimasks(
    pred_multimasks: TrackerArray,
    gt_masks: TrackerArray,
    pred_scores: TrackerArray | None = None,
) -> TrackerArray:
    """Select the predicted multimask with best IoU against GT masks."""

    assert pred_multimasks.ndim == 4 and gt_masks.ndim == 4
    if pred_multimasks.shape[1] == 1:
        return pred_multimasks

    if _is_mlx_array(pred_multimasks):
        gt_masks = cast(mx.array, gt_masks)
        pred_scores = cast(mx.array | None, pred_scores)
        pred_binary = pred_multimasks > 0
        area_i = mx.sum(pred_binary & gt_masks.astype(mx.bool_), axis=(2, 3)).astype(
            mx.float32
        )
        area_u = mx.sum(pred_binary | gt_masks.astype(mx.bool_), axis=(2, 3)).astype(
            mx.float32
        )
        ious = area_i / mx.clip(area_u, 1.0, None)
        if pred_scores is not None:
            has_nonzero_ious = mx.broadcast_to(mx.any(ious > 0), ious.shape)
            scores = mx.where(has_nonzero_ious, ious, pred_scores)
        else:
            scores = ious
        best_scores_inds = mx.argmax(scores, axis=-1)
        batch_inds = mx.arange(scores.shape[0], dtype=mx.int64)
        return pred_multimasks[batch_inds, best_scores_inds][:, None]

    pred_multimasks_np = cast(NumpyArray, pred_multimasks)
    gt_masks_np = cast(NumpyArray, gt_masks)
    pred_scores_np = cast(NumpyArray | None, pred_scores)
    pred_binary = np.greater(pred_multimasks_np, 0)
    gt_binary = np.asarray(gt_masks_np, dtype=np.bool_)
    area_i = np.sum(pred_binary & gt_binary, axis=(2, 3)).astype(np.float32)
    area_u = np.sum(pred_binary | gt_binary, axis=(2, 3)).astype(np.float32)
    ious = area_i / np.clip(area_u, 1.0, None)
    if pred_scores_np is not None:
        has_nonzero_ious = np.broadcast_to(np.any(ious > 0), ious.shape)
        scores = np.where(has_nonzero_ious, ious, pred_scores_np)
    else:
        scores = ious
    best_scores_inds = np.argmax(scores, axis=-1)
    scores_np = np.asarray(scores)
    batch_inds = np.arange(int(scores_np.shape[0]), dtype=np.int64)
    return pred_multimasks_np[batch_inds, best_scores_inds][:, None]


@overload
def fill_holes_in_mask_scores(
    mask: mx.array,
    max_area: int | None = None,
    fill_holes: bool = True,
    remove_sprinkles: bool = True,
    fill_hole_area: int | None = None,
    sprinkle_removal_area: int | None = None,
) -> mx.array: ...


@overload
def fill_holes_in_mask_scores(
    mask: NumpyArray,
    max_area: int | None = None,
    fill_holes: bool = True,
    remove_sprinkles: bool = True,
    fill_hole_area: int | None = None,
    sprinkle_removal_area: int | None = None,
) -> NumpyArray: ...


def fill_holes_in_mask_scores(
    mask: TrackerArray,
    max_area: int | None = None,
    fill_holes: bool = True,
    remove_sprinkles: bool = True,
    fill_hole_area: int | None = None,
    sprinkle_removal_area: int | None = None,
) -> TrackerArray:
    """Fill small holes and remove small foreground sprinkles in mask scores."""

    if fill_hole_area is not None and max_area is None:
        max_area = fill_hole_area
    if sprinkle_removal_area is not None and max_area is None:
        max_area = sprinkle_removal_area
    if max_area is None:
        raise ValueError("max_area or fill_hole_area must be provided.")
    if max_area <= 0:
        return mask

    mlx_dtype = mask.dtype if _is_mlx_array(mask) else None
    mask_np = cast(FloatingNumpyArray, _to_host_tracker_input(mask, copy=True))

    if fill_holes:
        mask_bg = mask_np <= 0
        _, areas_bg = _get_connected_components_with_padding(mask_bg)
        small_components_bg = mask_bg & (areas_bg <= max_area)
        mask_np = np.where(small_components_bg, np.array(0.1, mask_np.dtype), mask_np)

    if remove_sprinkles:
        mask_fg = mask_np > 0
        fg_area_thresh = np.sum(mask_fg, axis=(2, 3), keepdims=True, dtype=np.int32)
        fg_area_thresh = np.minimum(fg_area_thresh // 2, max_area)
        _, areas_fg = _get_connected_components_with_padding(mask_fg)
        small_components_fg = mask_fg & (areas_fg <= fg_area_thresh)
        mask_np = np.where(small_components_fg, np.array(-0.1, mask_np.dtype), mask_np)

    return mx.array(mask_np, dtype=mlx_dtype) if mlx_dtype is not None else mask_np


@overload
def _get_connected_components_with_padding(
    mask: mx.array,
) -> tuple[mx.array, mx.array]: ...


@overload
def _get_connected_components_with_padding(
    mask: NumpyArray,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]: ...


def _get_connected_components_with_padding(
    mask: TrackerArray,
) -> tuple[mx.array, mx.array] | tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Get 4-connected component labels and per-pixel component areas."""

    mask_np = _to_host_tracker_input(mask).astype(bool)
    if mask_np.ndim != 4:
        raise AssertionError("connected components expects shape (B, C, H, W).")
    labels = np.zeros(mask_np.shape, dtype=np.int32)
    counts = np.zeros(mask_np.shape, dtype=np.int32)

    bsz, channels, height, width = mask_np.shape
    component_id = 1
    for b_idx in range(bsz):
        for c_idx in range(channels):
            visited = np.zeros((height, width), dtype=bool)
            plane = mask_np[b_idx, c_idx]
            for y in range(height):
                for x in range(width):
                    if visited[y, x] or not plane[y, x]:
                        continue
                    queue: deque[tuple[int, int]] = deque([(y, x)])
                    visited[y, x] = True
                    coords: list[tuple[int, int]] = []
                    while queue:
                        cur_y, cur_x = queue.popleft()
                        coords.append((cur_y, cur_x))
                        for next_y, next_x in (
                            (cur_y - 1, cur_x),
                            (cur_y + 1, cur_x),
                            (cur_y, cur_x - 1),
                            (cur_y, cur_x + 1),
                        ):
                            if (
                                0 <= next_y < height
                                and 0 <= next_x < width
                                and not visited[next_y, next_x]
                                and plane[next_y, next_x]
                            ):
                                visited[next_y, next_x] = True
                                queue.append((next_y, next_x))
                    area = len(coords)
                    for cur_y, cur_x in coords:
                        labels[b_idx, c_idx, cur_y, cur_x] = component_id
                        counts[b_idx, c_idx, cur_y, cur_x] = area
                    component_id += 1

    if _is_mlx_array(mask):
        return mx.array(labels, dtype=mx.int32), mx.array(counts, dtype=mx.int32)
    return labels, counts


__all__ = [
    "fill_holes_in_mask_scores",
    "get_1d_sine_pe",
    "get_best_gt_match_from_multimasks",
    "get_next_point",
    "mask_to_box",
    "sample_box_points",
    "sample_one_point_from_error_center",
    "sample_one_point_from_error_center_slow",
    "sample_random_points_from_errors",
    "select_closest_cond_frames",
]
