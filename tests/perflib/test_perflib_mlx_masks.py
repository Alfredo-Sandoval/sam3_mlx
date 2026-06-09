import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from sam3_mlx.perflib import nms
from sam3_mlx.perflib import masks_ops
from sam3_mlx.perflib.iou import pairwise_iom, pairwise_iou
from sam3_mlx.perflib.masks_ops import mask_iom, mask_iou, masks_to_boxes
from sam3_mlx.perflib.nms import generic_nms, nms_masks
from tests._paths import PERFLIB_FIXTURE_ROOT

UPSTREAM_PERFLIB_TESTS_SOURCE_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"


def _to_numpy(value):
    mx.eval(value)
    return np.asarray(value)


def _load_official_masks_fixture(dtype: np.dtype) -> np.ndarray:
    image = Image.open(PERFLIB_FIXTURE_ROOT / "masks.tiff")
    frames = []
    for index in range(image.n_frames):
        image.seek(index)
        frames.append(np.asarray(image, dtype=dtype))
    return np.stack(frames, axis=0)


def test_mask_iou_and_iom_mlx_bool_masks_match_fixed_expected_values():
    pred = mx.array(
        [
            [[True, False, False], [True, True, False]],
            [[False, True, True], [False, False, False]],
        ]
    )
    gt = mx.array(
        [
            [[True, False, False], [False, True, False]],
            [[False, True, False], [False, False, False]],
            [[False, False, False], [False, False, False]],
        ]
    )

    iou = mask_iou(pred, gt)
    iom = mask_iom(pred, gt)

    assert isinstance(iou, mx.array)
    assert isinstance(iom, mx.array)
    np.testing.assert_allclose(
        _to_numpy(iou),
        np.array([[2.0 / 3.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _to_numpy(iom),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_mask_iou_rejects_uint_masks_in_bool_only_contract():
    pred = mx.array(np.ones((1, 2, 2), dtype=np.uint8))
    gt = mx.array(np.ones((1, 2, 2), dtype=bool))

    with pytest.raises(TypeError, match="boolean masks"):
        mask_iou(pred, gt)


def test_pairwise_iou_and_iom_mlx_uint_masks_match_fixed_expected_values():
    pred = mx.array(
        np.array(
            [
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]],
            ],
            dtype=np.uint8,
        )
    )
    gt = mx.array(
        np.array(
            [
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ],
            dtype=np.uint8,
        )
    )

    iou = pairwise_iou(pred, gt)
    iom = pairwise_iom(pred, gt)

    assert isinstance(iou, mx.array)
    assert isinstance(iom, mx.array)
    np.testing.assert_allclose(
        _to_numpy(iou),
        np.full((2, 2), 1.0 / (3.0 + 1e-6), dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        _to_numpy(iom),
        np.full((2, 2), 1.0 / (2.0 + 1e-8), dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )


def test_pairwise_iou_empty_mlx_masks_preserve_empty_pairwise_shape():
    pred = mx.array(np.zeros((0, 2, 2), dtype=np.uint8))
    gt = mx.array(np.array([[[1, 0], [0, 1]]], dtype=np.uint8))

    actual = pairwise_iou(pred, gt, eps=None)

    assert isinstance(actual, mx.array)
    np.testing.assert_array_equal(_to_numpy(actual), np.zeros((0, 1), dtype=np.float32))


def test_pairwise_iou_rejects_mismatched_mlx_spatial_shapes():
    pred = mx.array(np.zeros((1, 2, 2), dtype=np.uint8))
    gt = mx.array(np.zeros((1, 2, 1), dtype=np.uint8))

    with pytest.raises(ValueError, match="matching spatial shapes"):
        pairwise_iou(pred, gt)


def test_generic_nms_mlx_inputs_return_mlx_kept_indices():
    ious = mx.array(
        [
            [1.0, 0.2, 0.1, 0.8],
            [0.2, 1.0, 0.6, 0.1],
            [0.1, 0.6, 1.0, 0.3],
            [0.8, 0.1, 0.3, 1.0],
        ],
        dtype=mx.float32,
    )
    scores = mx.array([0.5, 0.9, 0.6, 0.7], dtype=mx.float32)

    actual = generic_nms(ious, scores, iou_threshold=0.5)

    assert isinstance(actual, mx.array)
    np.testing.assert_array_equal(_to_numpy(actual), np.array([1, 3], dtype=np.int64))


def test_nms_masks_mlx_uint_masks_keep_expected_detections():
    pred_probs = mx.array([0.8, 0.9, 0.7, 0.4], dtype=mx.float32)
    pred_masks = mx.array(
        np.array(
            [
                [[1, 1], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
                [[0, 1], [0, 1]],
            ],
            dtype=np.uint8,
        )
    )

    actual = nms_masks(
        pred_probs,
        pred_masks,
        prob_threshold=0.5,
        iou_threshold=0.5,
    )

    assert isinstance(actual, mx.array)
    np.testing.assert_array_equal(
        _to_numpy(actual),
        np.array([False, True, True, False]),
    )


def test_nms_masks_mlx_path_does_not_export_full_masks(monkeypatch):
    pred_probs = mx.array([0.8, 0.9, 0.7, 0.4], dtype=mx.float32)
    pred_masks = mx.array(
        np.array(
            [
                [[1, 1], [0, 0]],
                [[1, 1], [0, 0]],
                [[0, 0], [1, 1]],
                [[0, 1], [0, 1]],
            ],
            dtype=np.uint8,
        )
    )
    original_host_array = nms._host_array

    def guarded_host_array(value, *, dtype=None):
        if isinstance(value, mx.array) and len(value.shape) >= 3:
            raise AssertionError(f"full mask tensor exported to host: {value.shape}")
        return original_host_array(value, dtype=dtype)

    monkeypatch.setattr(nms, "_host_array", guarded_host_array)

    actual = nms_masks(
        pred_probs,
        pred_masks,
        prob_threshold=0.5,
        iou_threshold=0.5,
    )

    np.testing.assert_array_equal(
        _to_numpy(actual),
        np.array([False, True, True, False]),
    )


def test_nms_masks_mlx_no_valid_and_empty_masks_return_false_keep_masks():
    low_scores = mx.array([0.1, 0.2], dtype=mx.float32)
    bool_masks = mx.array(
        np.array(
            [
                [[True, False], [False, False]],
                [[False, True], [False, False]],
            ]
        )
    )
    no_valid = nms_masks(
        low_scores,
        bool_masks,
        prob_threshold=0.5,
        iou_threshold=0.5,
    )

    empty = nms_masks(
        mx.array(np.zeros((0,), dtype=np.float32)),
        mx.array(np.zeros((0, 2, 2), dtype=np.uint8)),
        prob_threshold=0.5,
        iou_threshold=0.5,
    )

    assert isinstance(no_valid, mx.array)
    assert isinstance(empty, mx.array)
    np.testing.assert_array_equal(_to_numpy(no_valid), np.array([False, False]))
    np.testing.assert_array_equal(_to_numpy(empty), np.zeros((0,), dtype=bool))


def test_masks_to_boxes_mlx_stays_on_device_and_matches_numpy_contract(monkeypatch):
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

    def fail_to_numpy(value):
        if isinstance(value, mx.array):
            raise AssertionError("MLX masks_to_boxes path exported masks to host")
        return np.asarray(value)

    monkeypatch.setattr(masks_ops, "_to_numpy", fail_to_numpy)

    actual = masks_to_boxes(masks, obj_ids=[10, 20, 30])

    assert isinstance(actual, mx.array)
    np.testing.assert_array_equal(
        _to_numpy(actual),
        np.array(
            [
                [1.0, 1.0, 2.0, 2.0],
                [0.0, 0.0, 3.0, 2.0],
                [4.0, 3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_masks_to_boxes_mlx_empty_and_shape_errors():
    empty = masks_to_boxes(
        mx.array(np.zeros((0, 2, 3), dtype=np.uint8)),
        obj_ids=[],
    )

    assert isinstance(empty, mx.array)
    np.testing.assert_array_equal(_to_numpy(empty), np.zeros((0, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="same length"):
        masks_to_boxes(mx.array(np.zeros((1, 2, 3), dtype=np.uint8)), obj_ids=[])

    with pytest.raises(ValueError, match="shape"):
        masks_to_boxes(mx.array(np.zeros((2, 3), dtype=np.uint8)), obj_ids=[1, 2])


def test_masks_to_boxes_matches_official_perflib_fixture_for_numpy_and_mlx():
    # Fixture and expected boxes come from official perflib/tests/tests.py at
    # UPSTREAM_PERFLIB_TESTS_SOURCE_COMMIT.
    expected = np.array(
        [
            [127, 2, 165, 40],
            [2, 50, 44, 92],
            [56, 63, 98, 100],
            [139, 68, 175, 104],
            [160, 112, 198, 145],
            [49, 138, 99, 182],
            [108, 148, 152, 213],
        ],
        dtype=np.float32,
    )

    for dtype in (np.float16, np.float32, np.float64):
        masks = _load_official_masks_fixture(dtype)
        obj_ids = [1 for _ in range(masks.shape[0])]

        numpy_boxes = masks_to_boxes(masks, obj_ids)
        mlx_boxes = masks_to_boxes(mx.array(masks), obj_ids)

        assert isinstance(numpy_boxes, np.ndarray)
        assert isinstance(mlx_boxes, mx.array)
        np.testing.assert_allclose(numpy_boxes, expected, rtol=0.0, atol=1e-4)
        np.testing.assert_allclose(_to_numpy(mlx_boxes), expected, rtol=0.0, atol=1e-4)


def test_numpy_inputs_still_return_numpy_outputs_for_overlap_helpers():
    pred = np.array([[[True, False], [False, True]]])
    gt = np.array([[[True, True], [False, False]]])

    bool_iou = mask_iou(pred, gt)
    bool_iom = mask_iom(pred, gt)
    numeric_iou = pairwise_iou(pred.astype(np.uint8), gt.astype(np.uint8))
    numeric_iom = pairwise_iom(pred.astype(np.uint8), gt.astype(np.uint8))

    assert isinstance(bool_iou, np.ndarray)
    assert isinstance(bool_iom, np.ndarray)
    assert isinstance(numeric_iou, np.ndarray)
    assert isinstance(numeric_iom, np.ndarray)
    np.testing.assert_allclose(
        bool_iou,
        np.array([[1.0 / 3.0]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        bool_iom,
        np.array([[0.5]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        numeric_iou,
        np.array([[1.0 / (3.0 + 1e-6)]], dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        numeric_iom,
        np.array([[1.0 / (2.0 + 1e-8)]], dtype=np.float32),
        rtol=0,
        atol=1e-7,
    )


def test_masks_to_boxes_numpy_inputs_still_return_numpy_outputs():
    masks = np.array(
        [
            [[False, True, False], [False, True, True]],
            [[False, False, False], [False, False, False]],
        ]
    )

    actual = masks_to_boxes(masks, obj_ids=[1, 2])

    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(
        actual,
        np.array(
            [
                [1.0, 0.0, 2.0, 1.0],
                [3.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
