"""Deterministic, independently replayable image-parity evidence helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from sam3_mlx.release_contract import (
    COMPARISON_ALGORITHM,
    EVIDENCE_SCHEMA_VERSION,
    RELEASE_THRESHOLDS,
)


def _normalize_masks(value: Any, *, label: str) -> np.ndarray:
    masks = np.asarray(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"{label} masks must have shape (N, H, W), got {masks.shape}.")
    return masks.astype(np.bool_, copy=False)


def _normalize_boxes(value: Any, *, count: int, label: str) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float32)
    if boxes.shape != (count, 4):
        raise ValueError(
            f"{label} boxes must have shape ({count}, 4), got {boxes.shape}."
        )
    if not np.isfinite(boxes).all():
        raise ValueError(f"{label} boxes contain non-finite values.")
    return boxes


def _normalize_scores(value: Any, *, count: int, label: str) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float32).reshape(-1)
    if scores.shape != (count,):
        raise ValueError(
            f"{label} scores must have shape ({count},), got {scores.shape}."
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"{label} scores contain non-finite values.")
    return scores


def normalize_outputs(value: Mapping[str, Any], *, label: str) -> dict[str, np.ndarray]:
    """Normalize one oracle/runtime output set before comparison or storage."""

    masks = _normalize_masks(value["masks"], label=label)
    count = int(masks.shape[0])
    return {
        "masks": masks,
        "boxes": _normalize_boxes(value["boxes"], count=count, label=label),
        "scores": _normalize_scores(value["scores"], count=count, label=label),
    }


def mask_iou_matrix(left_masks: Any, right_masks: Any) -> np.ndarray:
    """Return the complete pairwise mask-IoU matrix in float64."""

    left = _normalize_masks(left_masks, label="official")
    right = _normalize_masks(right_masks, label="mlx")
    if left.shape[1:] != right.shape[1:]:
        raise ValueError(
            "Official and MLX masks must share spatial dimensions, got "
            f"{left.shape[1:]} and {right.shape[1:]} ."
        )
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=np.float64)

    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    intersection = np.logical_and(
        left_flat[:, None, :], right_flat[None, :, :]
    ).sum(axis=2, dtype=np.int64)
    union = np.logical_or(left_flat[:, None, :], right_flat[None, :, :]).sum(
        axis=2, dtype=np.int64
    )
    return np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union != 0,
    )


def optimal_assignment(scores: Any) -> list[tuple[int, int]]:
    """Solve a square maximum-weight assignment with deterministic Hungarian O(n^3)."""

    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Assignment score matrix must be square, got " f"shape {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Assignment score matrix contains non-finite values.")
    n = int(matrix.shape[0])
    if n == 0:
        return []

    # Standard one-indexed Hungarian minimization, applied to negative scores.
    # Iterating columns in ascending order and updating only on strict improvement
    # gives deterministic tie handling.
    cost = -matrix
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)

    for row in range(1, n + 1):
        p[0] = row
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=np.bool_)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, n + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minv[column]:
                    minv[column] = current
                    way[column] = column0
                if minv[column] < delta:
                    delta = minv[column]
                    column1 = column
            if not math.isfinite(float(delta)):
                raise RuntimeError("Hungarian assignment could not find an augmenting path.")
            for column in range(n + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minv[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = [(-1, -1)] * n
    for column in range(1, n + 1):
        row = int(p[column]) - 1
        assignment[row] = (row, column - 1)
    if any(row < 0 or column < 0 for row, column in assignment):
        raise RuntimeError("Hungarian assignment returned an incomplete matching.")
    return assignment


def match_masks(
    official_masks: Any,
    mlx_masks: Any,
) -> list[tuple[int, int, float]]:
    ious = mask_iou_matrix(official_masks, mlx_masks)
    if ious.shape[0] != ious.shape[1]:
        raise ValueError("Mask matching requires equal detection counts.")
    return [
        (official_index, mlx_index, float(ious[official_index, mlx_index]))
        for official_index, mlx_index in optimal_assignment(ious)
    ]


def compare_case(
    spec: Mapping[str, Any],
    official: Mapping[str, Any],
    mlx: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] = RELEASE_THRESHOLDS,
) -> dict[str, Any]:
    """Compare one case and return the canonical machine-readable projection."""

    official_outputs = normalize_outputs(official, label="official")
    mlx_outputs = normalize_outputs(mlx, label="mlx")
    if official_outputs["masks"].shape[1:] != mlx_outputs["masks"].shape[1:]:
        raise ValueError("Official and MLX output masks have different spatial shapes.")

    official_count = len(official_outputs["scores"])
    mlx_count = len(mlx_outputs["scores"])
    count_match = official_count == mlx_count
    matches = (
        match_masks(official_outputs["masks"], mlx_outputs["masks"])
        if count_match and official_count
        else []
    )
    mask_ious = [match[2] for match in matches]
    box_errors = [
        float(
            np.max(
                np.abs(
                    official_outputs["boxes"][official_index]
                    - mlx_outputs["boxes"][mlx_index]
                )
            )
        )
        for official_index, mlx_index, _ in matches
    ]
    score_errors = [
        float(
            abs(
                official_outputs["scores"][official_index]
                - mlx_outputs["scores"][mlx_index]
            )
        )
        for official_index, mlx_index, _ in matches
    ]

    mask_iou_min = min(mask_ious) if mask_ious else None
    mask_iou_mean = statistics.mean(mask_ious) if mask_ious else None
    box_l_inf_max = max(box_errors) if box_errors else None
    score_abs_max = max(score_errors) if score_errors else None
    passed = count_match and (
        official_count == 0
        or (
            mask_iou_min is not None
            and mask_iou_min >= float(thresholds["mask_iou_min"])
            and mask_iou_mean is not None
            and mask_iou_mean >= float(thresholds["mask_iou_mean_min"])
            and box_l_inf_max is not None
            and box_l_inf_max <= float(thresholds["box_l_inf_max"])
            and score_abs_max is not None
            and score_abs_max <= float(thresholds["score_abs_max"])
        )
    )

    return {
        "name": spec["name"],
        "resolution": int(spec["resolution"]),
        "prompt": spec.get("prompt"),
        "geometric_prompts": list(spec.get("geometric_prompts") or []),
        "status": "passed" if passed else "failed",
        "official_detection_count": official_count,
        "mlx_detection_count": mlx_count,
        "detection_count_match": count_match,
        "mask_iou_min": mask_iou_min,
        "mask_iou_mean": mask_iou_mean,
        "box_l_inf_max": box_l_inf_max,
        "score_abs_max": score_abs_max,
        "matches": [
            {
                "official_index": official_index,
                "mlx_index": mlx_index,
                "mask_iou": iou,
            }
            for official_index, mlx_index, iou in matches
        ],
    }


def write_evidence_bundle(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
    official_outputs: Sequence[Mapping[str, Any]],
    mlx_outputs: Sequence[Mapping[str, Any]],
) -> Path:
    """Write raw official and MLX arrays needed to replay every reported metric."""

    output_path = Path(path)
    if output_path.suffix != ".npz":
        raise ValueError("Evidence bundle path must end in .npz.")
    if len(official_outputs) != len(mlx_outputs):
        raise ValueError("Official and MLX evidence must have equal case counts.")

    payload: dict[str, np.ndarray] = {}
    canonical_metadata = {
        **dict(metadata),
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "comparison_algorithm": COMPARISON_ALGORITHM,
        "case_count": len(official_outputs),
    }
    payload["metadata_json"] = np.array(
        json.dumps(canonical_metadata, sort_keys=True, allow_nan=False)
    )
    for index, (official, mlx) in enumerate(
        zip(official_outputs, mlx_outputs, strict=True)
    ):
        for side, value in (("official", official), ("mlx", mlx)):
            normalized = normalize_outputs(value, label=side)
            for field in ("masks", "boxes", "scores"):
                payload[f"case_{index}_{side}_{field}"] = normalized[field]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_evidence_bundle(
    path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    """Load a raw evidence bundle without pickle/object-array deserialization."""

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported evidence schema_version="
                f"{metadata.get('schema_version')!r}."
            )
        if metadata.get("comparison_algorithm") != COMPARISON_ALGORITHM:
            raise ValueError("Evidence comparison algorithm does not match the release contract.")
        case_count = metadata.get("case_count")
        if isinstance(case_count, bool) or not isinstance(case_count, int) or case_count < 0:
            raise ValueError("Evidence case_count must be a non-negative integer.")

        official_outputs: list[dict[str, np.ndarray]] = []
        mlx_outputs: list[dict[str, np.ndarray]] = []
        for index in range(case_count):
            side_outputs = {}
            for side in ("official", "mlx"):
                value = {
                    field: np.array(archive[f"case_{index}_{side}_{field}"], copy=True)
                    for field in ("masks", "boxes", "scores")
                }
                side_outputs[side] = normalize_outputs(value, label=side)
            official_outputs.append(side_outputs["official"])
            mlx_outputs.append(side_outputs["mlx"])
    return metadata, official_outputs, mlx_outputs
