from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import pytest

from sam3_mlx.train import nms_helper


class _Track(TypedDict):
    bboxes: list[list[float] | None]
    score: float


def _overlapping_tracks() -> dict[str, list[_Track]]:
    return {
        "video": [
            {"bboxes": [[0.0, 0.0, 10.0, 10.0]], "score": 0.9},
            {"bboxes": [[0.0, 0.0, 10.0, 10.0]], "score": 0.8},
        ]
    }


def test_frame_iou_and_nms_fallback_contract(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nms_helper, "_compute_frame_ious_numba", None)
    monkeypatch.setattr(nms_helper, "_apply_frame_nms_numba", None)
    bbox = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32)
    bboxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        nms_helper.compute_frame_ious(bbox, bboxes),
        np.array([1.0, 0.0], dtype=np.float32),
    )
    assert nms_helper.apply_frame_nms(
        bboxes, np.array([0.9, 0.8], dtype=np.float32), 0.5
    ) == [0, 1]


def test_track_iou_fallback_aggregates_only_valid_frames(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(nms_helper, "_compute_track_iou_matrix_numba", None)
    bboxes = np.array(
        [
            [[0.0, 0.0, 10.0, 10.0], [np.nan] * 4],
            [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 5.0, 5.0]],
        ],
        dtype=np.float32,
    )
    valid = cast(npt.NDArray[np.bool_], ~np.isnan(bboxes).any(axis=2))
    areas = (bboxes[:, :, 2] - bboxes[:, :, 0]) * (bboxes[:, :, 3] - bboxes[:, :, 1])
    areas[~valid] = 0

    actual = nms_helper.compute_track_iou_matrix(bboxes, valid, areas)

    np.testing.assert_array_equal(
        actual, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    )


@pytest.mark.parametrize(
    "processor",
    [nms_helper.process_frame_level_nms, nms_helper.process_track_level_nms],
)
def test_group_processors_mutate_and_return_the_original_mapping(
    processor: Callable[[dict[str, list[_Track]], float], dict[str, list[_Track]]],
):
    groups = _overlapping_tracks()

    result = processor(groups, 0.5)

    assert result is groups
    assert groups["video"][0]["bboxes"] == [[0.0, 0.0, 10.0, 10.0]]
    assert groups["video"][1]["bboxes"] == [None]


def test_bbox_boundary_rejects_malformed_values():
    with pytest.raises(ValueError, match="exactly four"):
        nms_helper.convert_bbox_format([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="real numbers"):
        nms_helper.convert_bbox_format([1.0, 2.0, 3.0, "4"])
