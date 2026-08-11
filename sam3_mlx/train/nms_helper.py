# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from importlib import import_module
from numbers import Real
from typing import Protocol, TypeGuard, TypedDict, cast
import warnings

import numpy as np
import numpy.typing as npt


type FloatArray = npt.NDArray[np.floating]
type BoolArray = npt.NDArray[np.bool_]
type Track = MutableMapping[str, object]
type TrackGroups = list[list[Track]]
type TrackIouFunction = Callable[[FloatArray, BoolArray, FloatArray], FloatArray]
type FrameIouFunction = Callable[[FloatArray, FloatArray], FloatArray]
type FrameNmsFunction = Callable[[FloatArray, FloatArray, float], list[int]]


class _TrackDetection(TypedDict):
    track_idx: int
    bboxes: FloatArray
    score: float


class _FrameDetection(TypedDict):
    track_idx: int
    bbox: FloatArray
    score: float


class _JitDecorator(Protocol):
    def __call__[**P, R](self, function: Callable[P, R], /) -> Callable[P, R]: ...


class _NumbaModule(Protocol):
    def jit(self, *, nopython: bool, parallel: bool = False) -> _JitDecorator: ...

    def prange(self, stop: int, /) -> Iterable[int]: ...


def _load_numba() -> _NumbaModule | None:
    try:
        return cast(_NumbaModule, import_module("numba"))
    except ImportError:
        warnings.warn(
            "Numba not found. Using slower pure Python implementations.",
            UserWarning,
            stacklevel=2,
        )
        return None


_numba = _load_numba()
HAS_NUMBA = _numba is not None
_parallel_range: Callable[[int], Iterable[int]] = (
    _numba.prange if _numba is not None else range
)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _bbox_values(bbox: object) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    if not _is_sequence(bbox) or len(bbox) != 4:
        raise ValueError("bbox must contain exactly four numeric values.")
    values = tuple(bbox)
    if not all(isinstance(value, Real) for value in values):
        raise TypeError("bbox values must be real numbers.")
    x, y, width, height = cast(tuple[Real, Real, Real, Real], values)
    return float(x), float(y), float(width), float(height)


def _track_groups(video_groups: object) -> TrackGroups:
    if not isinstance(video_groups, Mapping):
        raise TypeError("video_groups must be a mapping of videos to track lists.")
    groups: TrackGroups = []
    for tracks_value in cast(Mapping[object, object], video_groups).values():
        if not isinstance(tracks_value, list):
            raise TypeError("each video group must contain a list of tracks.")
        tracks: list[Track] = []
        for track_value in cast(list[object], tracks_value):
            if not isinstance(track_value, MutableMapping):
                raise TypeError("each track must be a mutable mapping.")
            tracks.append(cast(Track, track_value))
        groups.append(tracks)
    return groups


def _track_bboxes(track: Track) -> list[object]:
    bboxes = track.get("bboxes")
    if not isinstance(bboxes, list):
        raise TypeError("track 'bboxes' must be a list.")
    return cast(list[object], bboxes)


def _track_score(track: Track) -> float:
    score = track.get("score")
    if not isinstance(score, Real):
        raise TypeError("track 'score' must be a real number.")
    return float(score)


def is_zero_box(bbox: object) -> bool:
    """Return whether a box is absent or has only non-positive coordinates."""

    values = _bbox_values(bbox)
    return values is None or all(value <= 0 for value in values)


def convert_bbox_format(bbox: object) -> list[float]:
    """Convert a bbox from ``(x, y, w, h)`` to ``(x1, y1, x2, y2)``."""

    values = _bbox_values(bbox)
    if values is None:
        raise ValueError("bbox cannot be None.")
    x, y, width, height = values
    return [x, y, x + width, y + height]


def process_track_level_nms[T](video_groups: T, nms_threshold: float) -> T:
    """Apply track-level NMS to all videos, mutating and returning the input."""

    for tracks in _track_groups(video_groups):
        track_detections: list[_TrackDetection] = []
        for track_idx, track in enumerate(tracks):
            track_bboxes = _track_bboxes(track)
            if not track_bboxes:
                continue

            converted_bboxes: list[list[float]] = []
            valid_frames: list[bool] = []
            for bbox in track_bboxes:
                if bbox is not None and not is_zero_box(bbox):
                    converted_bboxes.append(convert_bbox_format(bbox))
                    valid_frames.append(True)
                else:
                    converted_bboxes.append([np.nan] * 4)
                    valid_frames.append(False)

            if any(valid_frames):
                track_detections.append(
                    {
                        "track_idx": track_idx,
                        "bboxes": np.asarray(converted_bboxes, dtype=np.float32),
                        "score": _track_score(track),
                    }
                )

        if track_detections:
            scores = np.asarray(
                [detection["score"] for detection in track_detections],
                dtype=np.float32,
            )
            keep = set(apply_track_nms(track_detections, scores, nms_threshold))
            for detection_idx, detection in enumerate(track_detections):
                if detection_idx not in keep:
                    track = tracks[detection["track_idx"]]
                    track["bboxes"] = [None] * len(_track_bboxes(track))

    return video_groups


def process_frame_level_nms[T](video_groups: T, nms_threshold: float) -> T:
    """Apply frame-level NMS to all videos, mutating and returning the input."""

    for tracks in _track_groups(video_groups):
        if not tracks:
            continue

        num_frames = len(_track_bboxes(tracks[0]))
        for frame_idx in range(num_frames):
            frame_detections: list[_FrameDetection] = []
            for track_idx, track in enumerate(tracks):
                track_bboxes = _track_bboxes(track)
                if len(track_bboxes) != num_frames:
                    raise ValueError("tracks in a video must have equal frame counts.")
                bbox = track_bboxes[frame_idx]
                if bbox is not None and not is_zero_box(bbox):
                    frame_detections.append(
                        {
                            "track_idx": track_idx,
                            "bbox": np.asarray(
                                convert_bbox_format(bbox), dtype=np.float32
                            ),
                            "score": _track_score(track),
                        }
                    )

            if frame_detections:
                bboxes = np.stack([detection["bbox"] for detection in frame_detections])
                scores = np.asarray(
                    [detection["score"] for detection in frame_detections],
                    dtype=np.float32,
                )
                keep = set(apply_frame_nms(bboxes, scores, nms_threshold))
                for detection_idx, detection in enumerate(frame_detections):
                    if detection_idx not in keep:
                        track_bboxes = _track_bboxes(tracks[detection["track_idx"]])
                        track_bboxes[frame_idx] = None

    return video_groups


def _compute_track_iou_matrix(
    bboxes_stacked: FloatArray, valid_masks: BoolArray, areas: FloatArray
) -> FloatArray:
    num_tracks = bboxes_stacked.shape[0]
    iou_matrix = np.zeros((num_tracks, num_tracks), dtype=np.float32)
    for i in _parallel_range(num_tracks):
        for j in range(i + 1, num_tracks):
            valid_ij = valid_masks[i] & valid_masks[j]
            if not valid_ij.any():
                continue
            bboxes_i = bboxes_stacked[i, valid_ij]
            bboxes_j = bboxes_stacked[j, valid_ij]
            area_i = areas[i, valid_ij]
            area_j = areas[j, valid_ij]
            inter_total = 0.0
            union_total = 0.0
            for k in range(bboxes_i.shape[0]):
                x1 = max(bboxes_i[k, 0], bboxes_j[k, 0])
                y1 = max(bboxes_i[k, 1], bboxes_j[k, 1])
                x2 = min(bboxes_i[k, 2], bboxes_j[k, 2])
                y2 = min(bboxes_i[k, 3], bboxes_j[k, 3])
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                union = area_i[k] + area_j[k] - intersection
                inter_total += float(intersection)
                union_total += float(union)
            if union_total > 0:
                iou_matrix[i, j] = inter_total / union_total
                iou_matrix[j, i] = iou_matrix[i, j]
    return iou_matrix


_compute_track_iou_matrix_numba: TrackIouFunction | None = None
if _numba is not None:
    _compute_track_iou_matrix_numba = _numba.jit(nopython=True, parallel=True)(
        _compute_track_iou_matrix
    )


def compute_track_iou_matrix(
    bboxes_stacked: FloatArray, valid_masks: BoolArray, areas: FloatArray
) -> FloatArray:
    """Compute pairwise track IoU using Numba when available."""

    if _compute_track_iou_matrix_numba is not None:
        return _compute_track_iou_matrix_numba(bboxes_stacked, valid_masks, areas)
    return _compute_track_iou_matrix(bboxes_stacked, valid_masks, areas)


def apply_track_nms(
    track_detections: Sequence[_TrackDetection],
    scores: FloatArray,
    nms_threshold: float,
) -> list[int]:
    """Apply NMS using IoU aggregated over each track's valid frames."""

    if not track_detections:
        return []
    bboxes_stacked = np.stack(
        [detection["bboxes"] for detection in track_detections], axis=0
    )
    valid_masks = cast(BoolArray, ~np.isnan(bboxes_stacked).any(axis=2))
    areas = (bboxes_stacked[:, :, 2] - bboxes_stacked[:, :, 0]) * (
        bboxes_stacked[:, :, 3] - bboxes_stacked[:, :, 1]
    )
    areas[~valid_masks] = 0
    iou_matrix = compute_track_iou_matrix(bboxes_stacked, valid_masks, areas)
    keep: list[int] = []
    order = np.argsort(-scores)
    suppress = np.zeros(len(track_detections), dtype=bool)
    for i in range(len(order)):
        current = int(order[i])
        if not suppress[current]:
            keep.append(current)
            remaining = order[i:]
            suppress[remaining] |= iou_matrix[current, remaining] >= nms_threshold
    return keep


def _compute_frame_ious(bbox: FloatArray, bboxes: FloatArray) -> FloatArray:
    ious = np.zeros(len(bboxes), dtype=np.float32)
    for i in _parallel_range(len(bboxes)):
        x1 = max(bbox[0], bboxes[i, 0])
        y1 = max(bbox[1], bboxes[i, 1])
        x2 = min(bbox[2], bboxes[i, 2])
        y2 = min(bbox[3], bboxes[i, 3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area2 = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
        union = area1 + area2 - intersection
        ious[i] = intersection / union if union > 0 else 0.0
    return ious


_compute_frame_ious_numba: FrameIouFunction | None = None
if _numba is not None:
    _compute_frame_ious_numba = _numba.jit(nopython=True, parallel=True)(
        _compute_frame_ious
    )


def compute_frame_ious(bbox: FloatArray, bboxes: FloatArray) -> FloatArray:
    """Compute frame box IoUs using Numba when available."""

    if _compute_frame_ious_numba is not None:
        return _compute_frame_ious_numba(bbox, bboxes)
    return _compute_frame_ious(bbox, bboxes)


def _apply_frame_nms(
    bboxes: FloatArray, scores: FloatArray, nms_threshold: float
) -> list[int]:
    order = np.argsort(-scores)
    keep: list[int] = []
    suppress = np.zeros(len(bboxes), dtype=bool)
    for i in range(len(order)):
        current = int(order[i])
        if not suppress[current]:
            keep.append(current)
            remaining = order[i + 1 :]
            if len(remaining) > 0:
                ious = _compute_frame_ious(bboxes[current], bboxes[remaining])
                suppress[remaining] |= ious >= nms_threshold
    return keep


def _compile_frame_nms(
    numba: _NumbaModule, compute_ious: FrameIouFunction
) -> FrameNmsFunction:
    def accelerated_frame_nms(
        bboxes: FloatArray, scores: FloatArray, nms_threshold: float
    ) -> list[int]:
        order = np.argsort(-scores)
        keep: list[int] = []
        suppress = np.zeros(len(bboxes), dtype=np.bool_)
        for i in range(len(order)):
            current = int(order[i])
            if not suppress[current]:
                keep.append(current)
                remaining = order[i + 1 :]
                if len(remaining) > 0:
                    ious = compute_ious(bboxes[current], bboxes[remaining])
                    suppress[remaining] |= ious >= nms_threshold
        return keep

    return numba.jit(nopython=True)(accelerated_frame_nms)


_apply_frame_nms_numba: FrameNmsFunction | None = None
if _numba is not None and _compute_frame_ious_numba is not None:
    _apply_frame_nms_numba = _compile_frame_nms(_numba, _compute_frame_ious_numba)


def apply_frame_nms(
    bboxes: FloatArray, scores: FloatArray, nms_threshold: float
) -> list[int]:
    """Apply frame-level NMS using Numba when available."""

    if _apply_frame_nms_numba is not None:
        return _apply_frame_nms_numba(bboxes, scores, nms_threshold)
    return _apply_frame_nms(bboxes, scores, nms_threshold)
