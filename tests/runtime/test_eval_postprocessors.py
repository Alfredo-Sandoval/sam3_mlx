import mlx.core as mx
import numpy as np

from sam3_mlx.eval.postprocessors import PostProcessImage

UPSTREAM_EVAL_POSTPROCESSORS_SOURCE_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"


def _to_numpy(value):
    if isinstance(value, mx.array):
        mx.eval(value)
    return np.asarray(value)


def _sigmoid(value):
    value = np.asarray(value, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-value))


def test_postprocess_image_scales_boxes_and_applies_forced_labels():
    processor = PostProcessImage(
        max_dets_per_img=100,
        iou_type="bbox",
        use_presence=False,
    )
    outputs = {
        "pred_logits": mx.array(
            np.array(
                [
                    [[0.0, 2.0, -2.0], [3.0, -1.0, 0.0]],
                    [[-2.0, -1.0, 4.0], [1.0, 0.0, -3.0]],
                ],
                dtype=np.float32,
            )
        ),
        "pred_boxes": mx.array(
            np.array(
                [
                    [[0.5, 0.5, 0.2, 0.4], [0.25, 0.25, 0.5, 0.5]],
                    [[0.5, 0.5, 1.0, 1.0], [0.75, 0.25, 0.25, 0.5]],
                ],
                dtype=np.float32,
            )
        ),
        "presence_logit_dec": mx.array(np.zeros((2, 2), dtype=np.float32)),
    }

    results = processor(
        outputs,
        target_sizes_boxes=mx.array([[100, 200], [50, 80]], dtype=mx.int64),
        target_sizes_masks=mx.array([[100, 200], [50, 80]], dtype=mx.int64),
        forced_labels=mx.array([5, 9], dtype=mx.int64),
    )

    assert len(results) == 2
    np.testing.assert_allclose(
        results[0]["boxes"],
        np.array([[80.0, 30.0, 120.0, 70.0], [0.0, 0.0, 100.0, 50.0]]),
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        results[1]["boxes"],
        np.array([[0.0, 0.0, 80.0, 50.0], [50.0, 0.0, 70.0, 25.0]]),
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        results[0]["scores"],
        np.array([_sigmoid(2.0), _sigmoid(3.0)]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        results[1]["scores"],
        np.array([_sigmoid(4.0), _sigmoid(1.0)]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(results[0]["labels"], np.array([5, 5]))
    np.testing.assert_array_equal(results[1]["labels"], np.array([9, 9]))


def test_postprocess_image_detection_threshold_is_strictly_greater_than_threshold():
    processor = PostProcessImage(
        max_dets_per_img=100,
        iou_type="bbox",
        use_presence=False,
        detection_threshold=0.5,
    )
    outputs = {
        "pred_logits": mx.array(np.array([[[0.0], [2.0]]], dtype=np.float32)),
        "pred_boxes": mx.array(
            np.array([[[0.25, 0.5, 0.5, 1.0], [0.5, 0.5, 0.2, 0.2]]])
        ),
        "presence_logit_dec": mx.array(np.zeros((1, 2), dtype=np.float32)),
    }

    result = processor(
        outputs,
        target_sizes_boxes=mx.array([[10, 20]], dtype=mx.int64),
        target_sizes_masks=mx.array([[10, 20]], dtype=mx.int64),
    )[0]

    np.testing.assert_allclose(
        result["boxes"],
        np.array([[8.0, 4.0, 12.0, 6.0]]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result["scores"],
        np.array([_sigmoid(2.0)]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(result["labels"], np.array([1]))


def test_postprocess_image_resizes_masks_with_mlx_interpolation_contract():
    processor = PostProcessImage(
        max_dets_per_img=100,
        iou_type="segm",
        use_presence=False,
    )
    outputs = {
        "pred_logits": mx.array(np.array([[[2.0]]], dtype=np.float32)),
        "pred_masks": mx.array(
            np.array([[[[-4.0, 4.0], [4.0, -4.0]]]], dtype=np.float32)
        ),
        "presence_logit_dec": mx.array(np.zeros((1, 1), dtype=np.float32)),
    }

    result = processor(
        outputs,
        target_sizes_boxes=mx.array([[4, 4]], dtype=mx.int64),
        target_sizes_masks=mx.array([[4, 4]], dtype=mx.int64),
        consistent=False,
    )[0]

    assert result["boxes"] is None
    assert result["scores"] is None
    assert result["labels"] is None
    np.testing.assert_array_equal(
        result["masks"],
        np.array(
            [
                [
                    [False, False, True, True],
                    [False, False, True, True],
                    [True, True, False, False],
                    [True, True, False, False],
                ]
            ],
            dtype=bool,
        ),
    )


def test_postprocess_image_to_cpu_false_returns_mlx_arrays_for_tensor_outputs():
    processor = PostProcessImage(
        max_dets_per_img=100,
        iou_type="segm",
        to_cpu=False,
        use_presence=False,
    )
    outputs = {
        "pred_logits": mx.array(np.array([[[2.0]]], dtype=np.float32)),
        "pred_boxes": mx.array(np.array([[[0.5, 0.5, 1.0, 1.0]]], dtype=np.float32)),
        "pred_masks": mx.array(
            np.array([[[[-10.0, 10.0], [10.0, -10.0]]]], dtype=np.float32)
        ),
        "presence_logit_dec": mx.array(np.zeros((1, 1), dtype=np.float32)),
    }

    result = processor(
        outputs,
        target_sizes_boxes=mx.array([[8, 10]], dtype=mx.int64),
        target_sizes_masks=mx.array([[2, 2]], dtype=mx.int64),
        consistent=False,
    )[0]

    assert isinstance(result["boxes"], mx.array)
    assert isinstance(result["scores"], mx.array)
    assert isinstance(result["labels"], mx.array)
    assert isinstance(result["masks"], mx.array)
    np.testing.assert_allclose(
        _to_numpy(result["boxes"]),
        np.array([[0.0, 0.0, 10.0, 8.0]]),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        _to_numpy(result["masks"]),
        np.array([[[False, True], [True, False]]], dtype=bool),
    )


def test_postprocess_image_rle_output_is_compressed_and_filtered_by_detection_keep():
    processor = PostProcessImage(
        max_dets_per_img=100,
        iou_type="segm",
        use_presence=False,
        convert_mask_to_rle=True,
        detection_threshold=0.5,
    )
    outputs = {
        "pred_logits": mx.array(np.array([[[0.0], [2.0]]], dtype=np.float32)),
        "pred_boxes": mx.array(
            np.array(
                [[[0.25, 0.5, 0.5, 1.0], [0.5, 0.5, 0.2, 0.2]]],
                dtype=np.float32,
            )
        ),
        "pred_masks": mx.array(
            np.array(
                [
                    [
                        [[10.0, -10.0, -10.0], [-10.0, -10.0, -10.0]],
                        [[-10.0, 10.0, -10.0], [10.0, 10.0, -10.0]],
                    ]
                ],
                dtype=np.float32,
            )
        ),
        "presence_logit_dec": mx.array(np.zeros((1, 2), dtype=np.float32)),
    }

    result = processor(
        outputs,
        target_sizes_boxes=mx.array([[2, 3]], dtype=mx.int64),
        target_sizes_masks=mx.array([[2, 3]], dtype=mx.int64),
        consistent=False,
    )[0]

    assert result["masks_rle"] == [{"size": [2, 3], "counts": "132"}]
    assert "masks" not in result
    np.testing.assert_allclose(
        result["boxes"],
        np.array([[1.2, 0.8, 1.8, 1.2]]),
        rtol=0,
        atol=1e-6,
    )
