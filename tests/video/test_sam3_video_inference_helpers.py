from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.sam3_video_inference import (
    _coerce_boxes,
    _coerce_points,
    _filter_outputs_by_removed_obj_ids,
    _state_to_video_outputs,
)


def test_coerce_boxes_converts_absolute_xywh_to_normalized_cxcywh_and_labels():
    boxes, labels = _coerce_boxes(
        boxes_xywh=[[10.0, 20.0, 30.0, 40.0]],
        box_labels=[False],
        rel_coordinates=False,
        orig_height=100,
        orig_width=200,
    )

    np.testing.assert_allclose(
        boxes,
        np.array([[0.125, 0.4, 0.15, 0.4]], dtype=np.float32),
    )
    np.testing.assert_array_equal(labels, np.array([False]))


def test_coerce_boxes_rejects_mismatched_labels_and_out_of_bounds_boxes():
    with pytest.raises(ValueError, match="one label per box"):
        _coerce_boxes(
            boxes_xywh=[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2]],
            box_labels=[True],
            rel_coordinates=True,
            orig_height=100,
            orig_width=200,
        )

    with pytest.raises(ValueError, match="within the image bounds"):
        _coerce_boxes(
            boxes_xywh=[[190.0, 20.0, 30.0, 40.0]],
            box_labels=None,
            rel_coordinates=False,
            orig_height=100,
            orig_width=200,
        )


def test_coerce_points_converts_absolute_xy_to_normalized_points_and_labels():
    points, labels = _coerce_points(
        points=[[50.0, 25.0], [100.0, 75.0]],
        point_labels=[True, False],
        rel_coordinates=False,
        orig_height=100,
        orig_width=200,
    )

    np.testing.assert_allclose(
        points,
        np.array([[0.25, 0.25], [0.5, 0.75]], dtype=np.float32),
    )
    np.testing.assert_array_equal(labels, np.array([True, False]))


def test_coerce_points_rejects_missing_points_and_out_of_bounds_points():
    with pytest.raises(ValueError, match="point_labels require points"):
        _coerce_points(
            points=None,
            point_labels=[True],
            rel_coordinates=True,
            orig_height=100,
            orig_width=200,
        )

    with pytest.raises(ValueError, match="within the image bounds"):
        _coerce_points(
            points=[[250.0, 25.0]],
            point_labels=[True],
            rel_coordinates=False,
            orig_height=100,
            orig_width=200,
        )


def test_state_to_video_outputs_converts_frame_state_to_official_video_schema():
    frame_state = {
        "masks": mx.array(
            [
                [[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]],
                [[[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]],
            ],
            dtype=mx.float32,
        ),
        "boxes": mx.array(
            [[0.0, 0.0, 3.0, 2.0], [1.0, 0.0, 3.0, 1.0]],
            dtype=mx.float32,
        ),
        "scores": mx.array([0.25, 0.75], dtype=mx.float32),
    }

    outputs = _state_to_video_outputs(
        frame_state,
        obj_id=None,
        orig_height=2,
        orig_width=3,
    )

    np.testing.assert_array_equal(outputs["out_obj_ids"], np.array([0, 1]))
    np.testing.assert_allclose(
        outputs["out_probs"], np.array([0.25, 0.75], dtype=np.float32)
    )
    np.testing.assert_allclose(
        outputs["out_boxes_xywh"],
        np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [1.0 / 3.0, 0.0, 2.0 / 3.0, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    assert outputs["out_binary_masks"].dtype == bool
    assert outputs["out_binary_masks"].shape == (2, 2, 3)


def test_state_to_video_outputs_rejects_obj_id_for_multiple_framewise_masks():
    frame_state = {
        "masks": np.ones((2, 2, 3), dtype=bool),
        "scores": np.array([0.25, 0.75], dtype=np.float32),
    }

    with pytest.raises(Sam3MlxUnsupportedError, match="exactly one mask") as exc_info:
        _state_to_video_outputs(
            frame_state,
            obj_id=7,
            orig_height=2,
            orig_width=3,
        )

    assert exc_info.value.reason == "video-multiplex"


def test_state_to_video_outputs_rejects_box_mask_count_mismatch():
    frame_state = {
        "masks": np.ones((2, 2, 3), dtype=bool),
        "boxes": np.array([[0.0, 0.0, 3.0, 2.0]], dtype=np.float32),
        "scores": np.array([0.25, 0.75], dtype=np.float32),
    }

    with pytest.raises(ValueError, match="same number of objects"):
        _state_to_video_outputs(
            frame_state,
            obj_id=None,
            orig_height=2,
            orig_width=3,
        )


def test_filter_outputs_by_removed_obj_ids_filters_every_video_output_field():
    outputs = {
        "out_obj_ids": np.array([2, 4, 6], dtype=np.int64),
        "out_probs": np.array([0.2, 0.4, 0.6], dtype=np.float32),
        "out_boxes_xywh": np.array(
            [[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
            dtype=np.float32,
        ),
        "out_binary_masks": np.array(
            [
                [[True, False]],
                [[False, True]],
                [[True, True]],
            ],
            dtype=bool,
        ),
    }

    filtered = _filter_outputs_by_removed_obj_ids(outputs, removed_obj_ids={4})

    np.testing.assert_array_equal(filtered["out_obj_ids"], np.array([2, 6]))
    np.testing.assert_array_equal(
        filtered["out_probs"], np.array([0.2, 0.6], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        filtered["out_boxes_xywh"],
        np.array([[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        filtered["out_binary_masks"],
        np.array([[[True, False]], [[True, True]]], dtype=bool),
    )
