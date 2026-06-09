from __future__ import annotations

from collections import defaultdict

import numpy as np

from sam3_mlx._unsupported import raise_unsupported, unsupported
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.perflib.masks_ops import mask_iou


def _to_numpy(value) -> np.ndarray:
    return to_numpy(value, copy=False)


def _linear_sum_assignment(cost_matrix: np.ndarray):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise unsupported(
            "sam3_mlx.perflib.associate_det_trk._linear_sum_assignment",
            reason="optional-dependency",
            detail=(
                "associate_det_trk requires SciPy's Hungarian solver for exact "
                "one-to-one track matching; SciPy is not a hard dependency of "
                "the MLX port."
            ),
        ) from exc
    return linear_sum_assignment(cost_matrix)


def associate_det_trk(
    det_masks,
    track_masks,
    iou_threshold=0.5,
    iou_threshold_trk=0.5,
    det_scores=None,
    new_det_thresh=0.0,
):
    det_np = _to_numpy(det_masks)
    track_np = _to_numpy(track_masks)
    if det_np.ndim != 3 or track_np.ndim != 3:
        raise ValueError("det_masks and track_masks must have shape (N, H, W).")

    num_det = det_np.shape[0]
    num_track = track_np.shape[0]
    if num_det == 0 or num_track == 0:
        return list(range(num_det)), [], {}, {}

    if det_np.shape[-2:] != track_np.shape[-2:]:
        raise_unsupported(
            "sam3_mlx.perflib.associate_det_trk.associate_det_trk(mask_resize=True)",
            reason="port-gap",
            detail=(
                "associate_det_trk mask resizing is not implemented in the MLX port."
            ),
            alternative="matching H/W masks",
        )

    det_binary = det_np > 0
    track_binary = track_np > 0
    iou = _to_numpy(mask_iou(det_binary, track_binary)).astype(np.float32, copy=False)
    iou_ge_det = iou >= iou_threshold
    iou_ge_det_any = iou_ge_det.any(axis=1)
    iou_ge_trk = iou >= iou_threshold_trk

    scores = None
    if det_scores is not None:
        scores = _to_numpy(det_scores).astype(np.float32, copy=False).reshape(-1)
        if scores.shape != (num_det,):
            raise ValueError("det_scores must have one score per detection mask.")

    cost_matrix = 1.0 - iou
    row_ind, col_ind = _linear_sum_assignment(cost_matrix)

    matched_trk = set()
    matched_det_scores = {}
    for det_idx, trk_idx in zip(row_ind, col_ind):
        det_idx = int(det_idx)
        trk_idx = int(trk_idx)
        if scores is not None:
            score = float(scores[det_idx])
            matched_det_scores[trk_idx] = [score, score * float(iou[det_idx, trk_idx])]
        if iou_ge_trk[det_idx, trk_idx]:
            matched_trk.add(trk_idx)

    unmatched_trk_indices = [
        trk_idx for trk_idx in range(num_track) if trk_idx not in matched_trk
    ]

    new_det_indices = []
    if scores is not None:
        for det_idx in range(num_det):
            if not iou_ge_det_any[det_idx] and scores[det_idx] >= new_det_thresh:
                new_det_indices.append(det_idx)

    det_to_matched_trk = defaultdict(list)
    for det_idx in range(num_det):
        for trk_idx in range(num_track):
            if iou_ge_det[det_idx, trk_idx]:
                det_to_matched_trk[det_idx].append(trk_idx)

    return (
        new_det_indices,
        unmatched_trk_indices,
        det_to_matched_trk,
        matched_det_scores,
    )


def associate_detections_to_trackers(*args, **kwargs):
    return associate_det_trk(*args, **kwargs)
