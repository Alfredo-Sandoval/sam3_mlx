import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.model.edt import edt_triton
from sam3_mlx.model.sam3_tracker_utils import (
    _get_connected_components_with_padding,
    fill_holes_in_mask_scores,
    sample_one_point_from_error_center,
)
from sam3_mlx.perflib.connected_components import connected_components


def _to_numpy(value):
    if isinstance(value, mx.array):
        mx.eval(value)
    return np.asarray(value)


def test_edt_triton_mlx_input_matches_fixed_center_distance_field():
    mask = np.ones((1, 3, 3), dtype=bool)
    mask[0, 1, 1] = False
    expected = np.array(
        [
            [
                [np.sqrt(2.0), 1.0, np.sqrt(2.0)],
                [1.0, 0.0, 1.0],
                [np.sqrt(2.0), 1.0, np.sqrt(2.0)],
            ]
        ],
        dtype=np.float32,
    )

    actual = edt_triton(mx.array(mask))

    assert isinstance(actual, mx.array)
    np.testing.assert_allclose(_to_numpy(actual), expected, rtol=0, atol=1e-6)


def test_edt_triton_rejects_non_batched_2d_shape():
    with pytest.raises(AssertionError, match=r"shape \(B, H, W\)"):
        edt_triton(np.zeros((3, 3), dtype=bool))


def test_connected_components_mlx_input_uses_8_connected_contract():
    mask = np.array(
        [
            [
                [
                    [1, 0, 0, 1],
                    [0, 1, 0, 1],
                    [0, 0, 0, 0],
                    [1, 1, 0, 1],
                ]
            ]
        ],
        dtype=bool,
    )
    expected_labels = np.array(
        [
            [
                [
                    [1, 0, 0, 2],
                    [0, 1, 0, 2],
                    [0, 0, 0, 0],
                    [3, 3, 0, 4],
                ]
            ]
        ],
        dtype=np.int64,
    )
    expected_counts = np.array(
        [
            [
                [
                    [2, 0, 0, 2],
                    [0, 2, 0, 2],
                    [0, 0, 0, 0],
                    [2, 2, 0, 1],
                ]
            ]
        ],
        dtype=np.int64,
    )

    labels, counts = connected_components(mx.array(mask))

    assert isinstance(labels, mx.array)
    assert isinstance(counts, mx.array)
    np.testing.assert_array_equal(_to_numpy(labels), expected_labels)
    np.testing.assert_array_equal(_to_numpy(counts), expected_counts)


def test_tracker_connected_components_with_padding_keeps_4_connected_contract():
    mask = np.array(
        [
            [
                [
                    [1, 0, 0],
                    [0, 1, 1],
                    [0, 0, 1],
                ]
            ]
        ],
        dtype=bool,
    )
    expected_labels = np.array(
        [
            [
                [
                    [1, 0, 0],
                    [0, 2, 2],
                    [0, 0, 2],
                ]
            ]
        ],
        dtype=np.int32,
    )
    expected_counts = np.array(
        [
            [
                [
                    [1, 0, 0],
                    [0, 3, 3],
                    [0, 0, 3],
                ]
            ]
        ],
        dtype=np.int32,
    )

    labels, counts = _get_connected_components_with_padding(mx.array(mask))

    assert isinstance(labels, mx.array)
    assert isinstance(counts, mx.array)
    np.testing.assert_array_equal(_to_numpy(labels), expected_labels)
    np.testing.assert_array_equal(_to_numpy(counts), expected_counts)


def test_fill_holes_in_mask_scores_mlx_input_matches_fixed_cleanup_values():
    scores = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0, -1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0, -1.0],
                    [-1.0, -1.0, -1.0, 1.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    expected = scores.copy()
    expected[0, 0, 1, 1] = 0.1
    expected[0, 0, 3, 3] = -0.1

    actual = fill_holes_in_mask_scores(mx.array(scores), max_area=1)

    assert isinstance(actual, mx.array)
    np.testing.assert_allclose(_to_numpy(actual), expected, rtol=0, atol=1e-6)


def test_sample_one_point_from_error_center_mlx_uses_edt_center_contract():
    gt = np.zeros((1, 1, 5, 5), dtype=bool)
    gt[:, :, 1:4, 1:4] = True
    pred = np.zeros_like(gt)

    points, labels = sample_one_point_from_error_center(mx.array(gt), mx.array(pred))

    assert isinstance(points, mx.array)
    assert isinstance(labels, mx.array)
    np.testing.assert_array_equal(_to_numpy(points), np.array([[[2, 2]]]))
    np.testing.assert_array_equal(_to_numpy(labels), np.array([[1]]))
