import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.model import box_ops
from sam3_mlx.mlx_runtime import to_numpy


def test_model_masks_to_boxes_mlx_matches_exclusive_xyxy_contract(monkeypatch):
    masks = mx.array(
        np.array(
            [
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 1, 0],
                ],
                [
                    [0, 0, 0, 1],
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                ],
                np.zeros((3, 4), dtype=np.uint8),
            ],
            dtype=np.uint8,
        )
    )

    def fail_broadcast_to(*_args, **_kwargs):
        raise AssertionError("masks_to_boxes should not build dense HxW grids")

    monkeypatch.setattr(box_ops.mx, "broadcast_to", fail_broadcast_to)

    actual = box_ops.masks_to_boxes(masks)

    np.testing.assert_array_equal(
        to_numpy(actual),
        np.array(
            [
                [1.0, 1.0, 3.0, 3.0],
                [0.0, 0.0, 4.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_model_masks_to_boxes_mlx_empty_and_shape_errors():
    empty = box_ops.masks_to_boxes(mx.zeros((0, 3, 4), dtype=mx.bool_))

    assert empty.shape == (0, 4)
    np.testing.assert_array_equal(to_numpy(empty), np.zeros((0, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        box_ops.masks_to_boxes(mx.zeros((2, 3), dtype=mx.bool_))


def test_box_conversion_helpers_keep_batched_shape_and_values():
    boxes = mx.array(
        [
            [[0.5, 0.5, 0.25, 0.5], [0.25, 0.75, 0.5, 0.25]],
            [[1.0, 1.0, 0.2, 0.4], [0.4, 0.3, 0.2, 0.2]],
        ],
        dtype=mx.float32,
    )

    xyxy = box_ops.box_cxcywh_to_xyxy(boxes)
    roundtrip = box_ops.box_xyxy_to_cxcywh(xyxy)

    assert xyxy.shape == boxes.shape
    np.testing.assert_allclose(to_numpy(roundtrip), to_numpy(boxes), rtol=0, atol=1e-6)


def test_box_xywh_inter_union_avoids_array_truth_assertions():
    boxes1 = mx.array([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]])
    boxes2 = mx.array([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 1.0, 1.0]])

    inter, union = box_ops.box_xywh_inter_union(boxes1, boxes2)

    np.testing.assert_allclose(
        to_numpy(inter),
        np.array([1.0, 0.0], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(union),
        np.array([7.0, 5.0], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_box_xywh_inter_union_rejects_bad_last_dim():
    with pytest.raises(ValueError, match="last dimension 4"):
        box_ops.box_xywh_inter_union(
            mx.zeros((1, 3), dtype=mx.float32),
            mx.zeros((1, 4), dtype=mx.float32),
        )
