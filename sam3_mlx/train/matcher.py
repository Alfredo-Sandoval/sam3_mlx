# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""SAM3 training matchers ported away from PyTorch-only dependencies.

The official implementation computes matching costs under ``torch.no_grad`` and
then moves the cost matrix to CPU for SciPy's linear-sum assignment solver. This
MLX port keeps that no-gradient boundary explicit: inputs are accepted as MLX
arrays, matching costs are computed as NumPy arrays, and match indices are
returned as MLX ``int64`` arrays.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn


MLX_MATCHER_BASE_COMMIT = "e85b2531ccc93307a936e08656fb19b2c2f75baf"
TRAINING_MATCHER_CPU_BOUNDARY = (
    "Hungarian training matchers materialize matching costs on the host because "
    "the assignment solver is CPU-bound in this port."
)


def _to_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        if isinstance(value, mx.array):
            mx.eval(value)
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _to_mx_int(values) -> mx.array:
    return mx.array(np.asarray(values, dtype=np.int64), dtype=mx.int64)


def _empty_int() -> mx.array:
    return mx.array([], dtype=mx.int64)


def _concat_int(parts) -> np.ndarray:
    parts = [np.asarray(part, dtype=np.int64).reshape(-1) for part in parts]
    parts = [part for part in parts if part.size > 0]
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts).astype(np.int64, copy=False)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _logsigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return -np.logaddexp(0, -x)


def _box_cxcywh_to_xyxy_np(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    x_c, y_c, w, h = np.moveaxis(boxes, -1, 0)
    return np.stack(
        [x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h],
        axis=-1,
    )


def _box_area_np(boxes: np.ndarray) -> np.ndarray:
    return np.clip(boxes[..., 2] - boxes[..., 0], 0, None) * np.clip(
        boxes[..., 3] - boxes[..., 1], 0, None
    )


def _box_iou_np(
    boxes1: np.ndarray, boxes2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    area1 = _box_area_np(boxes1)
    area2 = _box_area_np(boxes2)
    lt = np.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = np.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2[..., None, :] - inter
    return inter / np.maximum(union, 1e-12), union


def _generalized_box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    iou, union = _box_iou_np(boxes1, boxes2)
    lt = np.minimum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = np.maximum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    area = wh[..., 0] * wh[..., 1]
    return iou - (area - union) / np.maximum(area, 1e-12)


def _l1_cost_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    return np.abs(boxes1[..., :, None, :] - boxes2[..., None, :, :]).sum(axis=-1)


def _hungarian_rows(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign every row to a unique column for a matrix with rows <= columns."""

    n_rows, n_cols = cost.shape
    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for row in range(1, n_rows + 1):
        p[0] = row
        col0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        way.fill(0)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = np.inf
            col1 = 0
            for col in range(1, n_cols + 1):
                if used[col]:
                    continue
                cur = cost[row0 - 1, col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col] = cur
                    way[col] = col0
                if minv[col] < delta:
                    delta = minv[col]
                    col1 = col
            if not np.isfinite(delta):
                raise ValueError("cost matrix does not contain a finite assignment")
            for col in range(n_cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    row_to_col = np.empty(n_rows, dtype=np.int64)
    for col in range(1, n_cols + 1):
        if p[col] != 0:
            row_to_col[p[col] - 1] = col - 1
    return np.arange(n_rows, dtype=np.int64), row_to_col


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be a 2-D matrix, got shape {cost.shape}")
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    if n_rows <= n_cols:
        row_ind, col_ind = _hungarian_rows(cost)
    else:
        col_ind, row_ind = _hungarian_rows(cost.T)

    order = np.argsort(row_ind)
    return row_ind[order].astype(np.int64), col_ind[order].astype(np.int64)


def _do_matching(
    cost,
    repeats: int = 1,
    return_tgt_indices: bool = False,
    do_filtering: bool = False,
):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be a 2-D matrix, got shape {cost.shape}")
    if repeats < 0:
        raise ValueError(f"repeats must be >= 0, got {repeats}")

    if repeats > 1:
        cost = np.tile(cost, (1, repeats))

    src_idx, tgt_idx = _linear_sum_assignment(cost)
    if do_filtering:
        keep = cost[src_idx, tgt_idx] < 1e8
        src_idx = src_idx[keep]
        tgt_idx = tgt_idx[keep]
    if return_tgt_indices:
        return src_idx.astype(np.int64), tgt_idx.astype(np.int64)
    order = np.argsort(tgt_idx)
    return src_idx[order].astype(np.int64)


def _num_boxes_np(batched_targets) -> np.ndarray:
    return _to_numpy(batched_targets["num_boxes"], dtype=np.int64).reshape(-1)


def _split_flat_costs(cost: np.ndarray, num_boxes: np.ndarray) -> list[np.ndarray]:
    split_at = np.cumsum(num_boxes)[:-1]
    return [chunk[i] for i, chunk in enumerate(np.split(cost, split_at, axis=-1))]


def _batch_indices(indices, batch_ids=None) -> np.ndarray:
    parts = []
    for i, src in enumerate(indices):
        src = np.asarray(src, dtype=np.int64).reshape(-1)
        if src.size == 0:
            continue
        batch_id = i if batch_ids is None else int(batch_ids[i])
        parts.append(np.full(src.shape, batch_id, dtype=np.int64))
    return _concat_int(parts)


def _offset_target_indices(tgt_indices, num_boxes: np.ndarray) -> np.ndarray:
    parts = []
    offset = 0
    for indices, count in zip(tgt_indices, num_boxes):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size > 0:
            parts.append(indices + offset)
        offset += int(count)
    return _concat_int(parts)


def _return_indices(indices, tgt_indices=None, batch_ids=None):
    batch_idx = _to_mx_int(_batch_indices(indices, batch_ids=batch_ids))
    src_idx = _to_mx_int(_concat_int(indices))
    if tgt_indices is None:
        return batch_idx, src_idx
    return batch_idx, src_idx, _to_mx_int(tgt_indices)


class _MatcherModule(nn.Module):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HungarianMatcher(_MatcherModule):
    """Compute official SAM3 multi-class bipartite matching in MLX form."""

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2,
    ):
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise AssertionError("all costs cant be 0")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, outputs, batched_targets):
        pred_logits = _to_numpy(outputs["pred_logits"])
        pred_boxes = _to_numpy(outputs["pred_boxes"])
        batch_size, num_queries = pred_logits.shape[:2]

        flat_logits = pred_logits.reshape(batch_size * num_queries, -1)
        out_prob = (
            _sigmoid_np(flat_logits) if self.focal_loss else _softmax_np(flat_logits)
        )
        out_bbox = pred_boxes.reshape(batch_size * num_queries, 4)
        tgt_bbox = _to_numpy(batched_targets["boxes"])

        if "positive_map" in batched_targets:
            positive_map = _to_numpy(batched_targets["positive_map"])
            if len(tgt_bbox) != len(positive_map):
                raise AssertionError("positive_map and boxes length mismatch.")
            if self.focal_loss:
                positive_map = positive_map > 1e-4
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                neg = (1 - alpha) * (out_prob**gamma) * -np.log(1 - out_prob + 1e-8)
                pos = alpha * ((1 - out_prob) ** gamma) * -np.log(out_prob + 1e-8)
                cost_class = ((pos - neg)[:, None, :] * positive_map[None]).sum(-1)
            else:
                cost_class = -(out_prob[:, None, :] * positive_map[None]).sum(-1)
        else:
            tgt_ids = _to_numpy(batched_targets["labels"], dtype=np.int64).reshape(-1)
            if len(tgt_bbox) != len(tgt_ids):
                raise AssertionError("labels and boxes length mismatch.")
            if self.focal_loss:
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                neg = (1 - alpha) * (out_prob**gamma) * -np.log(1 - out_prob + 1e-8)
                pos = alpha * ((1 - out_prob) ** gamma) * -np.log(out_prob + 1e-8)
                cost_class = pos[:, tgt_ids] - neg[:, tgt_ids]
            else:
                cost_class = -out_prob[:, tgt_ids]

        cost_bbox = _l1_cost_np(out_bbox, tgt_bbox)
        cost_giou = -_generalized_box_iou_np(
            _box_cxcywh_to_xyxy_np(out_bbox),
            _box_cxcywh_to_xyxy_np(tgt_bbox),
        )
        if cost_class.shape != cost_bbox.shape:
            raise AssertionError("classification and bbox costs have different shapes.")

        cost = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        cost = cost.reshape(batch_size, num_queries, -1)
        costs = _split_flat_costs(cost, _num_boxes_np(batched_targets))
        indices = [_do_matching(chunk) for chunk in costs]
        return _return_indices(indices)


class BinaryHungarianMatcher(_MatcherModule):
    """Compute official binary SAM3 bipartite matching in MLX form."""

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
    ):
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise AssertionError("all costs cant be 0")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, batched_targets, repeats=0, repeat_batch=1):
        if repeat_batch != 1:
            raise NotImplementedError("please use BinaryHungarianMatcherV2 instead")

        pred_logits = _to_numpy(outputs["pred_logits"])
        pred_boxes = _to_numpy(outputs["pred_boxes"])
        batch_size, num_queries = pred_logits.shape[:2]
        out_prob = _sigmoid_np(
            pred_logits.reshape(batch_size * num_queries, -1)
        ).squeeze(-1)
        out_bbox = pred_boxes.reshape(batch_size * num_queries, 4)
        tgt_bbox = _to_numpy(batched_targets["boxes"])

        cost_bbox = _l1_cost_np(out_bbox, tgt_bbox)
        cost_class = -np.broadcast_to(out_prob[:, None], cost_bbox.shape)
        cost_giou = -_generalized_box_iou_np(
            _box_cxcywh_to_xyxy_np(out_bbox),
            _box_cxcywh_to_xyxy_np(tgt_bbox),
        )
        if cost_class.shape != cost_bbox.shape:
            raise AssertionError("classification and bbox costs have different shapes.")

        cost = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        cost = cost.reshape(batch_size, num_queries, -1)
        num_boxes = _num_boxes_np(batched_targets)
        costs = _split_flat_costs(cost, num_boxes)
        return_tgt_indices = any(
            chunk.shape[0] < chunk.shape[1] * max(int(repeats), 1) for chunk in costs
        )

        if return_tgt_indices:
            matched = [
                _do_matching(chunk, repeats=repeats, return_tgt_indices=True)
                for chunk in costs
            ]
            indices = [src for src, _tgt in matched]
            tgt_indices = _offset_target_indices(
                [tgt for _src, tgt in matched], num_boxes
            )
            return _return_indices(indices, tgt_indices=tgt_indices)

        indices = [_do_matching(chunk, repeats=repeats) for chunk in costs]
        batch_idx, src_idx = _return_indices(indices)
        return batch_idx, src_idx, None


class BinaryFocalHungarianMatcher(_MatcherModule):
    """Binary matcher using the official focal classification matching cost."""

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
    ):
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise AssertionError("all costs cant be 0")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma
        self.stable = stable

    def forward(self, outputs, batched_targets, repeats=1, repeat_batch=1):
        if repeat_batch != 1:
            raise NotImplementedError("please use BinaryHungarianMatcherV2 instead")

        pred_logits = _to_numpy(outputs["pred_logits"])
        pred_boxes = _to_numpy(outputs["pred_boxes"])
        batch_size, num_queries = pred_logits.shape[:2]
        out_score = pred_logits.reshape(batch_size * num_queries, -1).squeeze(-1)
        out_prob = _sigmoid_np(out_score)
        out_bbox = pred_boxes.reshape(batch_size * num_queries, 4)
        tgt_bbox = _to_numpy(batched_targets["boxes"])

        cost_bbox = _l1_cost_np(out_bbox, tgt_bbox)
        cost_giou = -_generalized_box_iou_np(
            _box_cxcywh_to_xyxy_np(out_bbox),
            _box_cxcywh_to_xyxy_np(tgt_bbox),
        )
        cost_class = _binary_focal_cost(
            out_score,
            out_prob,
            cost_bbox.shape,
            cost_giou,
            self.alpha,
            self.gamma,
            self.stable,
        )

        cost = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        cost = cost.reshape(batch_size, num_queries, -1)
        num_boxes = _num_boxes_np(batched_targets)
        costs = _split_flat_costs(cost, num_boxes)
        return_tgt_indices = any(
            chunk.shape[0] < chunk.shape[1] * max(int(repeats), 1) for chunk in costs
        )

        if return_tgt_indices:
            matched = [
                _do_matching(chunk, repeats=repeats, return_tgt_indices=True)
                for chunk in costs
            ]
            indices = [src for src, _tgt in matched]
            tgt_indices = _offset_target_indices(
                [tgt for _src, tgt in matched], num_boxes
            )
            return _return_indices(indices, tgt_indices=tgt_indices)

        indices = [_do_matching(chunk, repeats=repeats) for chunk in costs]
        batch_idx, src_idx = _return_indices(indices)
        return batch_idx, src_idx, None


def _binary_focal_cost(
    out_score: np.ndarray,
    out_prob: np.ndarray,
    cost_shape: tuple[int, ...],
    cost_giou: np.ndarray,
    alpha: float,
    gamma: float,
    stable: bool,
) -> np.ndarray:
    if stable:
        rescaled_giou = (-cost_giou + 1) / 2
        prob = np.clip(out_prob[:, None] * rescaled_giou, 1e-8, 1 - 1e-8)
        return -alpha * (1 - prob) ** gamma * np.log(prob) + (
            1 - alpha
        ) * prob**gamma * np.log(1 - prob)

    log_out_prob = _logsigmoid_np(out_score)
    log_one_minus_out_prob = _logsigmoid_np(-out_score)
    cost_class = (
        -alpha * (1 - out_prob) ** gamma * log_out_prob
        + (1 - alpha) * out_prob**gamma * log_one_minus_out_prob
    )
    return np.broadcast_to(cost_class[:, None], cost_shape)


class BinaryHungarianMatcherV2(_MatcherModule):
    """Efficient padded binary Hungarian matcher from official SAM3."""

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
        stable: bool = False,
        remove_samples_with_0_gt: bool = True,
    ):
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise AssertionError("all costs cant be 0")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal = focal
        self.alpha = alpha
        self.gamma = gamma
        self.stable = stable
        self.remove_samples_with_0_gt = remove_samples_with_0_gt

    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):
        pred_logits = _to_numpy(outputs["pred_logits"])
        pred_boxes = _to_numpy(outputs["pred_boxes"])
        _, num_queries = pred_logits.shape[:2]
        out_score = pred_logits.squeeze(-1)
        out_bbox = pred_boxes
        num_boxes = _num_boxes_np(batched_targets)
        tgt_bbox = _to_numpy(batched_targets["boxes_padded"])
        batch_keep = np.ones(num_boxes.shape, dtype=bool)

        if self.remove_samples_with_0_gt:
            batch_keep = num_boxes > 0
            num_boxes = num_boxes[batch_keep]
            tgt_bbox = tgt_bbox[batch_keep]
            if target_is_valid_padded is not None:
                target_is_valid_padded = _to_numpy(target_is_valid_padded, dtype=bool)[
                    batch_keep
                ]
        elif target_is_valid_padded is not None:
            target_is_valid_padded = _to_numpy(target_is_valid_padded, dtype=bool)

        if repeat_batch > 1:
            num_boxes = np.tile(num_boxes, int(repeat_batch))
            tgt_bbox = np.tile(tgt_bbox, (int(repeat_batch), 1, 1))
            batch_keep = np.tile(batch_keep, int(repeat_batch))
            if target_is_valid_padded is not None:
                target_is_valid_padded = np.tile(
                    target_is_valid_padded, (int(repeat_batch), 1)
                )

        if self.remove_samples_with_0_gt:
            out_score = out_score[batch_keep]
            out_bbox = out_bbox[batch_keep]
            if out_is_valid is not None:
                out_is_valid = _to_numpy(out_is_valid, dtype=bool)[batch_keep]
        elif out_is_valid is not None:
            out_is_valid = _to_numpy(out_is_valid, dtype=bool)

        if (
            out_bbox.shape[0] != tgt_bbox.shape[0]
            or out_bbox.shape[0] != num_boxes.shape[0]
        ):
            raise AssertionError("matcher batch dimensions do not align.")

        if out_bbox.shape[0] == 0:
            return_tgt = False
            if out_is_valid is not None or target_is_valid_padded is not None:
                return_tgt = True
            if return_tgt:
                return _empty_int(), _empty_int(), _empty_int()
            return _empty_int(), _empty_int(), None

        cost_bbox = _l1_cost_np(out_bbox, tgt_bbox)
        cost_giou = -_generalized_box_iou_np(
            _box_cxcywh_to_xyxy_np(out_bbox),
            _box_cxcywh_to_xyxy_np(tgt_bbox),
        )
        out_prob = _sigmoid_np(out_score)
        if not self.focal:
            cost_class = -np.broadcast_to(out_prob[:, :, None], cost_bbox.shape)
        else:
            cost_class = _binary_focal_cost(
                out_score.reshape(-1),
                out_prob.reshape(-1),
                (out_prob.size, cost_bbox.shape[-1]),
                cost_giou.reshape(out_prob.size, cost_bbox.shape[-1]),
                self.alpha,
                self.gamma,
                self.stable,
            ).reshape(cost_bbox.shape)

        cost = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        do_filtering = out_is_valid is not None or target_is_valid_padded is not None
        if out_is_valid is not None:
            cost = np.where(out_is_valid[:, :, None], cost, 1e9)
        if target_is_valid_padded is not None:
            cost = np.where(target_is_valid_padded[:, None, :], cost, 1e9)

        costs = [cost[i, :, : int(size)] for i, size in enumerate(num_boxes.tolist())]
        return_tgt_indices = do_filtering or bool(
            np.any(num_queries < num_boxes * max(int(repeats), 1))
        )

        if return_tgt_indices:
            matched = [
                _do_matching(
                    chunk,
                    repeats=repeats,
                    return_tgt_indices=True,
                    do_filtering=do_filtering,
                )
                for chunk in costs
            ]
            indices = [src for src, _tgt in matched]
            tgt_indices = _offset_target_indices(
                [tgt for _src, tgt in matched], num_boxes
            )
        else:
            indices = [
                _do_matching(chunk, repeats=repeats, do_filtering=do_filtering)
                for chunk in costs
            ]
            tgt_indices = None

        batch_ids = np.nonzero(batch_keep)[0] if self.remove_samples_with_0_gt else None
        if tgt_indices is None:
            batch_idx, src_idx = _return_indices(indices, batch_ids=batch_ids)
            return batch_idx, src_idx, None
        return _return_indices(indices, tgt_indices=tgt_indices, batch_ids=batch_ids)


class BinaryOneToManyMatcher(_MatcherModule):
    """Greedy one-to-many assignment used by the official training stack."""

    def __init__(
        self,
        alpha: float = 0.3,
        threshold: float = 0.4,
        topk: int = 6,
    ):
        super().__init__()
        self.alpha = alpha
        self.threshold = threshold
        self.topk = topk

    def forward(
        self,
        outputs,
        batched_targets,
        repeats=1,
        repeat_batch=1,
        out_is_valid=None,
        target_is_valid_padded=None,
    ):
        if repeats > 1 or repeat_batch > 1:
            raise AssertionError("BinaryOneToManyMatcher expects repeats <= 1.")

        pred_logits = _to_numpy(outputs["pred_logits"])
        pred_boxes = _to_numpy(outputs["pred_boxes"])
        batch_size, num_queries = pred_logits.shape[:2]
        out_prob = _sigmoid_np(pred_logits).squeeze(-1)
        out_bbox = pred_boxes
        num_boxes = _num_boxes_np(batched_targets)
        tgt_bbox = _to_numpy(batched_targets["boxes_padded"])
        if len(tgt_bbox) != batch_size:
            raise AssertionError("boxes_padded batch dimension mismatch.")

        num_targets = tgt_bbox.shape[1]
        if num_targets == 0:
            return _empty_int(), _empty_int(), _empty_int()

        iou, _ = _box_iou_np(
            _box_cxcywh_to_xyxy_np(out_bbox),
            _box_cxcywh_to_xyxy_np(tgt_bbox),
        )
        if iou.shape != (batch_size, num_queries, num_targets):
            raise AssertionError("unexpected IoU shape.")

        score = self.alpha * out_prob[:, :, None] + (1 - self.alpha) * iou
        if out_is_valid is not None:
            out_is_valid = _to_numpy(out_is_valid, dtype=bool)
            score = np.where(out_is_valid[:, :, None], score, -1e9)
        if target_is_valid_padded is not None:
            target_is_valid_padded = _to_numpy(target_is_valid_padded, dtype=bool)
            score = np.where(target_is_valid_padded[:, None, :], score, -1e9)

        quantile = 1 - self.topk / num_queries
        threshold_by_target = np.quantile(score, quantile, axis=1, keepdims=True)
        matches = (score > threshold_by_target) & (score > self.threshold)
        if out_is_valid is not None:
            matches &= out_is_valid[:, :, None]
        if target_is_valid_padded is not None:
            matches &= target_is_valid_padded[:, None, :]
        target_range = np.arange(num_targets, dtype=np.int64)[None, :]
        matches &= (target_range < num_boxes[:, None])[:, None, :]

        batch_idx, src_idx, tgt_idx = np.nonzero(matches)
        offsets = np.concatenate([[0], np.cumsum(num_boxes)[:-1]]).astype(np.int64)
        tgt_idx = tgt_idx.astype(np.int64) + offsets[batch_idx]
        return _to_mx_int(batch_idx), _to_mx_int(src_idx), _to_mx_int(tgt_idx)


__all__ = [
    "BinaryFocalHungarianMatcher",
    "BinaryHungarianMatcher",
    "BinaryHungarianMatcherV2",
    "BinaryOneToManyMatcher",
    "HungarianMatcher",
    "MLX_MATCHER_BASE_COMMIT",
    "TRAINING_MATCHER_CPU_BOUNDARY",
]
