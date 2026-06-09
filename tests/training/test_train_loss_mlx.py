import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.train.loss import loss_fns
from sam3_mlx.train.loss.mask_sampling import point_sample
from sam3_mlx.train.loss.sigmoid_focal_loss import (
    sigmoid_focal_loss as elementwise_sigmoid_focal_loss,
)


def test_elementwise_sigmoid_focal_loss_matches_fixed_oracle_values():
    inputs = mx.array(
        [[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]],
        dtype=mx.float32,
    )
    targets = mx.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=mx.float32,
    )

    actual = elementwise_sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0)

    expected = np.array(
        [
            [0.00135267, 0.04332170, 0.00045089],
            [0.00566451, 0.01699354, 0.28305873],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=0, atol=1e-7)


def test_loss_fns_sigmoid_focal_loss_wrapper_pins_reduce_normalization():
    inputs = mx.array(
        [[-2.0, 0.0, 2.0], [1.0, -1.0, 0.5]],
        dtype=mx.float32,
    )
    targets = mx.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
        dtype=mx.float32,
    )

    reduced = loss_fns.sigmoid_focal_loss(
        inputs,
        targets,
        num_boxes=2,
        alpha=0.25,
        gamma=2.0,
    )
    unreduced = loss_fns.sigmoid_focal_loss(
        inputs,
        targets,
        num_boxes=2,
        alpha=0.25,
        gamma=2.0,
        reduce=False,
    )

    expected_unreduced = np.array(
        [
            [0.00135267, 0.04332170, 0.00045089],
            [0.00566451, 0.01699354, 0.28305873],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        to_numpy(reduced),
        np.array(0.058473676, dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        to_numpy(unreduced),
        expected_unreduced,
        rtol=0,
        atol=1e-7,
    )


def test_loss_fns_sigmoid_focal_loss_rejects_invalid_alpha():
    with pytest.raises(RuntimeError, match=r"Alpha should be in \[0,1\]"):
        loss_fns.sigmoid_focal_loss(
            mx.zeros((1, 2), dtype=mx.float32),
            mx.zeros((1, 2), dtype=mx.float32),
            num_boxes=1,
            alpha=1.5,
        )


def test_point_sample_matches_fixed_center_and_edge_contract():
    values = mx.array(
        [
            [
                [
                    [10.0, 2.0],
                    [4.0, 6.0],
                ]
            ]
        ],
        dtype=mx.float32,
    )
    points = mx.array(
        [
            [
                [0.5, 0.5],
                [1.0, 1.0],
            ]
        ],
        dtype=mx.float32,
    )

    actual = point_sample(values, points, align_corners=False)

    expected = np.array([[[5.5, 1.5]]], dtype=np.float32)
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=0, atol=1e-6)


def test_point_sample_rejects_unsupported_grid_sample_modes():
    values = mx.zeros((1, 1, 2, 2), dtype=mx.float32)
    points = mx.zeros((1, 1, 2), dtype=mx.float32)

    with pytest.raises(NotImplementedError, match="align_corners=False"):
        point_sample(values, points, align_corners=True)
    with pytest.raises(NotImplementedError, match="bilinear"):
        point_sample(values, points, mode="nearest")
    with pytest.raises(NotImplementedError, match="zero padding"):
        point_sample(values, points, padding_mode="border")
