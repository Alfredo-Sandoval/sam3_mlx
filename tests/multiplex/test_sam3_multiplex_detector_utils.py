from types import MethodType

import numpy as np
import pytest
import mlx.core as mx

from sam3_mlx.model.sam3_multiplex_detector import Sam3MultiplexDetector
from sam3_mlx.model import sam3_multiplex_detector_utils as utils


def _is_mlx_array(value):
    return value.__class__.__module__.startswith("mlx.")


def _as_numpy(value):
    if _is_mlx_array(value):
        mx.eval(value)
        return np.array(value)
    return np.asarray(value)


def _maybe_mlx(value, as_mlx):
    return mx.array(value) if as_mlx else value


def test_forward_video_grounding_preserves_prev_encoder_backbone_payload():
    detector = Sam3MultiplexDetector.__new__(Sam3MultiplexDetector)
    detector.tracking_score_thresh = 0.0
    prev_encoder_out = {"backbone_out": {"sam2_backbone_out": {"sentinel": True}}}

    def fake_forward_grounding(self, **kwargs):
        del self, kwargs
        return {
            "pred_logits": mx.ones((1, 1, 1), dtype=mx.float32),
            "pred_boxes": mx.ones((1, 1, 4), dtype=mx.float32),
            "pred_boxes_xyxy": mx.ones((1, 1, 4), dtype=mx.float32),
            "pred_masks": mx.ones((1, 1, 2, 2), dtype=mx.float32),
            "prev_encoder_out": prev_encoder_out,
        }

    detector.forward_grounding = MethodType(fake_forward_grounding, detector)

    out, returned_backbone = detector.forward_video_grounding(backbone_out="backbone")

    assert returned_backbone == "backbone"
    assert out["prev_encoder_out"] is prev_encoder_out
    np.testing.assert_array_equal(_as_numpy(out["pred_object_ids"]), np.array([[0]]))


def test_forward_video_grounding_multigpu_exports_nested_backbone_cache():
    detector = Sam3MultiplexDetector.__new__(Sam3MultiplexDetector)
    detector.rank = 0
    detector.world_size = 1
    detector.is_multiplex = True

    sam2_fpn = [
        mx.ones((1, 2, 2, 2), dtype=mx.float32) * 1,
        mx.ones((1, 2, 1, 1), dtype=mx.float32) * 2,
        mx.ones((1, 2, 1, 1), dtype=mx.float32) * 3,
    ]
    interactive_fpn = [
        mx.ones((1, 2, 2, 2), dtype=mx.float32) * 4,
        mx.ones((1, 2, 1, 1), dtype=mx.float32) * 5,
        mx.ones((1, 2, 1, 1), dtype=mx.float32) * 6,
    ]

    def fake_forward_video_grounding(self, **kwargs):
        del self, kwargs
        return (
            {
                "pred_logits": mx.ones((1, 1, 1), dtype=mx.float32),
                "pred_boxes": mx.ones((1, 1, 4), dtype=mx.float32),
                "pred_boxes_xyxy": mx.ones((1, 1, 4), dtype=mx.float32),
                "pred_masks": mx.ones((1, 1, 2, 2), dtype=mx.float32),
                "prev_encoder_out": {
                    "backbone_out": {
                        "sam2_backbone_out": {
                            "backbone_fpn": sam2_fpn,
                            "vision_pos_enc": [sam2_fpn[0], sam2_fpn[1]],
                        },
                        "interactive": {
                            "backbone_fpn": interactive_fpn,
                            "vision_pos_enc": [
                                interactive_fpn[0],
                                interactive_fpn[1],
                            ],
                        },
                    },
                },
            },
            None,
        )

    detector.forward_video_grounding = MethodType(
        fake_forward_video_grounding,
        detector,
    )

    out, _ = detector.forward_video_grounding_multigpu(
        backbone_out={},
        find_inputs=["frame-0"],
        geometric_prompt=None,
        frame_idx=0,
        num_frames=1,
        multigpu_buffer={},
        return_sam2_backbone_feats=True,
    )

    assert out["sam2_backbone_fpn_0"] is sam2_fpn[0]
    assert out["sam2_backbone_fpn_1"] is sam2_fpn[1]
    assert out["sam2_backbone_fpn_2"] is sam2_fpn[2]
    assert out["sam2_backbone_pos_enc"] == [sam2_fpn[0], sam2_fpn[1]]
    assert out["interactive_backbone_fpn_0"] is interactive_fpn[0]
    assert out["interactive_backbone_fpn_1"] is interactive_fpn[1]
    assert out["interactive_backbone_fpn_2"] is interactive_fpn[2]
    assert out["interactive_backbone_pos_enc"] == [
        interactive_fpn[0],
        interactive_fpn[1],
    ]


def _single_nms_fixture():
    masks = np.array(
        [
            [[1, 1], [0, 0]],
            [[1, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[0, 1], [0, 0]],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.90, 0.80, 0.70, 0.65, 0.60], dtype=np.float32)
    return scores, masks


@pytest.mark.parametrize("as_mlx", [False, True])
def test_nms_masks_iou_mode_suppresses_identical_masks_at_near_one_threshold(
    as_mlx,
):
    scores = np.array([0.90, 0.80, 0.70], dtype=np.float32)
    masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]],
        ],
        dtype=np.float32,
    )
    expected_single = np.array([True, False, True])
    expected_batched = np.stack([expected_single, expected_single])

    single_keep = utils.nms_masks(
        _maybe_mlx(scores, as_mlx),
        _maybe_mlx(masks, as_mlx),
        prob_threshold=0.0,
        iou_threshold=0.9999995,
    )
    batched_keep = utils.nms_masks(
        _maybe_mlx(np.stack([scores, scores]), as_mlx),
        _maybe_mlx(np.stack([masks, masks]), as_mlx),
        prob_threshold=0.0,
        iou_threshold=0.9999995,
    )

    assert _is_mlx_array(single_keep) is as_mlx
    assert _is_mlx_array(batched_keep) is as_mlx
    np.testing.assert_array_equal(_as_numpy(single_keep), expected_single)
    np.testing.assert_array_equal(_as_numpy(batched_keep), expected_batched)


@pytest.mark.parametrize(
    ("nms_use_iom", "expected"),
    [
        (
            False,
            np.array([True, True, True, False, False]),
        ),
        (
            True,
            np.array([True, True, True, False, False]),
        ),
    ],
)
def test_nms_masks_single_mlx_matches_numpy_at_threshold_boundary(
    nms_use_iom, expected
):
    scores, masks = _single_nms_fixture()

    numpy_keep = utils.nms_masks(
        scores,
        masks,
        prob_threshold=0.60,
        iou_threshold=0.50,
        nms_use_iom=nms_use_iom,
    )
    mlx_keep = utils.nms_masks(
        mx.array(scores),
        mx.array(masks),
        prob_threshold=0.60,
        iou_threshold=0.50,
        nms_use_iom=nms_use_iom,
    )

    assert isinstance(numpy_keep, np.ndarray)
    assert _is_mlx_array(mlx_keep)
    np.testing.assert_array_equal(numpy_keep, expected)
    np.testing.assert_array_equal(_as_numpy(mlx_keep), numpy_keep)


@pytest.mark.parametrize("as_mlx", [False, True])
def test_nms_masks_single_iom_matches_official_source_area_denominator(as_mlx):
    scores = np.array([0.90, 0.80, 0.70], dtype=np.float32)
    masks = np.array(
        [
            [[1, 1, 0, 0]],
            [[1, 0, 0, 0]],
            [[0, 0, 1, 1]],
        ],
        dtype=np.float32,
    )

    keep = utils.nms_masks(
        _maybe_mlx(scores, as_mlx),
        _maybe_mlx(masks, as_mlx),
        prob_threshold=0.0,
        iou_threshold=0.50,
        nms_use_iom=True,
    )

    assert _is_mlx_array(keep) is as_mlx
    np.testing.assert_array_equal(
        _as_numpy(keep),
        np.array([True, True, True]),
    )


@pytest.mark.parametrize(
    ("nms_use_iom", "expected_first_row"),
    [
        (False, np.array([True, True, True, False, False])),
        (True, np.array([True, False, True, False, False])),
    ],
)
def test_nms_masks_batched_mlx_matches_numpy_and_keeps_no_valid_batch_empty(
    nms_use_iom, expected_first_row
):
    scores, masks = _single_nms_fixture()
    batch_scores = np.stack(
        [
            scores,
            np.array([0.10, 0.20, 0.30, 0.40, 0.60], dtype=np.float32),
        ]
    )
    batch_masks = np.stack([masks, masks])
    expected = np.stack(
        [
            expected_first_row,
            np.zeros(scores.shape[0], dtype=bool),
        ]
    )

    numpy_keep = utils.nms_masks(
        batch_scores,
        batch_masks,
        prob_threshold=0.60,
        iou_threshold=0.50,
        nms_use_iom=nms_use_iom,
    )
    mlx_keep = utils.nms_masks(
        mx.array(batch_scores),
        mx.array(batch_masks),
        prob_threshold=0.60,
        iou_threshold=0.50,
        nms_use_iom=nms_use_iom,
    )

    assert isinstance(numpy_keep, np.ndarray)
    assert _is_mlx_array(mlx_keep)
    np.testing.assert_array_equal(numpy_keep, expected)
    np.testing.assert_array_equal(_as_numpy(mlx_keep), numpy_keep)


def test_nms_masks_batched_numpy_skips_overlap_for_invalid_batches(monkeypatch):
    scores = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.90, 0.80, 0.10],
        ],
        dtype=np.float32,
    )
    masks = np.array(
        [
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
        ],
        dtype=np.float32,
    )
    original_pairwise = utils._pairwise_mask_iou_np
    overlap_candidate_counts = []

    def counted_pairwise(pred_masks, gt_masks):
        overlap_candidate_counts.append(pred_masks.shape[0])
        return original_pairwise(pred_masks, gt_masks)

    monkeypatch.setattr(utils, "_pairwise_mask_iou_np", counted_pairwise)

    keep = utils.nms_masks(
        scores,
        masks,
        prob_threshold=0.50,
        iou_threshold=0.50,
    )

    np.testing.assert_array_equal(
        keep,
        np.array(
            [
                [False, False, False],
                [True, True, False],
            ]
        ),
    )
    assert overlap_candidate_counts == [2]


def test_nms_masks_no_valid_single_mlx_matches_numpy():
    masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.10, 0.50, 0.60], dtype=np.float32)
    expected = np.zeros(scores.shape[0], dtype=bool)

    numpy_keep = utils.nms_masks(
        scores,
        masks,
        prob_threshold=0.60,
        iou_threshold=0.50,
    )
    mlx_keep = utils.nms_masks(
        mx.array(scores),
        mx.array(masks),
        prob_threshold=0.60,
        iou_threshold=0.50,
    )

    np.testing.assert_array_equal(numpy_keep, expected)
    np.testing.assert_array_equal(_as_numpy(mlx_keep), numpy_keep)


@pytest.mark.parametrize("as_mlx", [False, True])
def test_nms_masks_empty_inputs_preserve_single_and_batched_keep_shapes(as_mlx):
    single_scores = np.zeros((0,), dtype=np.float32)
    single_masks = np.zeros((0, 2, 2), dtype=np.float32)
    batched_scores = np.zeros((2, 0), dtype=np.float32)
    batched_masks = np.zeros((2, 0, 2, 2), dtype=np.float32)

    single_keep = utils.nms_masks(
        _maybe_mlx(single_scores, as_mlx),
        _maybe_mlx(single_masks, as_mlx),
        prob_threshold=0.60,
        iou_threshold=0.50,
    )
    batched_keep = utils.nms_masks(
        _maybe_mlx(batched_scores, as_mlx),
        _maybe_mlx(batched_masks, as_mlx),
        prob_threshold=0.60,
        iou_threshold=0.50,
    )

    assert _is_mlx_array(single_keep) is as_mlx
    assert _is_mlx_array(batched_keep) is as_mlx
    np.testing.assert_array_equal(_as_numpy(single_keep), np.zeros((0,), dtype=bool))
    np.testing.assert_array_equal(_as_numpy(batched_keep), np.zeros((2, 0), dtype=bool))


def test_pairwise_mask_iou_and_iom_mlx_match_numpy_outputs():
    pred_masks = np.array(
        [
            [[1, 1], [0, 0]],
            [[0, 0], [1, 0]],
        ],
        dtype=np.float32,
    )
    gt_masks = np.array(
        [
            [[1, 0], [0, 0]],
            [[1, 1], [0, 0]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.float32,
    )
    expected_iou = np.array(
        [
            [0.50, 1.00, 0.00],
            [0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    expected_iom = np.array(
        [
            [1.00, 1.00, 0.00],
            [0.00, 0.00, 0.00],
        ],
        dtype=np.float32,
    )

    numpy_iou = utils.perf_mask_iou(pred_masks, gt_masks)
    mlx_iou = utils.perf_mask_iou(mx.array(pred_masks), mx.array(gt_masks))
    numpy_iom = utils.perf_mask_iom(pred_masks, gt_masks)
    mlx_iom = utils.perf_mask_iom(mx.array(pred_masks), mx.array(gt_masks))

    np.testing.assert_allclose(numpy_iou, expected_iou, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_as_numpy(mlx_iou), numpy_iou, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(numpy_iom, expected_iom, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_as_numpy(mlx_iom), numpy_iom, rtol=1e-6, atol=1e-6)


def test_batched_mask_iou_and_iom_mlx_match_numpy_outputs():
    masks = np.array(
        [
            [
                [[1, 1], [0, 0]],
                [[1, 0], [0, 0]],
                [[0, 0], [1, 1]],
            ],
            [
                [[0, 0], [0, 0]],
                [[1, 0], [0, 0]],
                [[1, 1], [1, 1]],
            ],
        ],
        dtype=np.float32,
    )
    expected_iou = np.array(
        [
            [
                [1.00, 0.50, 0.00],
                [0.50, 1.00, 0.00],
                [0.00, 0.00, 1.00],
            ],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.00, 0.25],
                [0.00, 0.25, 1.00],
            ],
        ],
        dtype=np.float32,
    )
    expected_iom = np.array(
        [
            [
                [1.00, 1.00, 0.00],
                [1.00, 1.00, 0.00],
                [0.00, 0.00, 1.00],
            ],
            [
                [0.00, 0.00, 0.00],
                [0.00, 1.00, 1.00],
                [0.00, 1.00, 1.00],
            ],
        ],
        dtype=np.float32,
    )

    numpy_iou = utils._batched_mask_iou(masks)
    mlx_iou = utils._batched_mask_iou(mx.array(masks))
    numpy_iom = utils._batched_mask_iom(masks)
    mlx_iom = utils._batched_mask_iom(mx.array(masks))

    np.testing.assert_allclose(numpy_iou, expected_iou, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_as_numpy(mlx_iou), numpy_iou, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(numpy_iom, expected_iom, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_as_numpy(mlx_iom), numpy_iom, rtol=1e-6, atol=1e-6)


def test_pairwise_mask_overlap_empty_inputs_preserve_pairwise_shapes():
    empty_pred = np.zeros((0, 2, 2), dtype=np.float32)
    gt_masks = np.array([[[1, 0], [0, 1]]], dtype=np.float32)

    numpy_iou = utils.perf_mask_iou(empty_pred, gt_masks)
    mlx_iou = utils.perf_mask_iou(mx.array(empty_pred), mx.array(gt_masks))
    numpy_iom = utils.perf_mask_iom(empty_pred, gt_masks)
    mlx_iom = utils.perf_mask_iom(mx.array(empty_pred), mx.array(gt_masks))

    expected = np.zeros((0, 1), dtype=np.float32)
    np.testing.assert_array_equal(numpy_iou, expected)
    np.testing.assert_array_equal(_as_numpy(mlx_iou), expected)
    np.testing.assert_array_equal(numpy_iom, expected)
    np.testing.assert_array_equal(_as_numpy(mlx_iom), expected)


def test_batched_mask_overlap_zero_detections_preserve_empty_square_shape():
    masks = np.zeros((2, 0, 2, 2), dtype=np.float32)
    expected = np.zeros((2, 0, 0), dtype=np.float32)

    numpy_iou = utils._batched_mask_iou(masks)
    mlx_iou = utils._batched_mask_iou(mx.array(masks))
    numpy_iom = utils._batched_mask_iom(masks)
    mlx_iom = utils._batched_mask_iom(mx.array(masks))

    np.testing.assert_array_equal(numpy_iou, expected)
    np.testing.assert_array_equal(_as_numpy(mlx_iou), expected)
    np.testing.assert_array_equal(numpy_iom, expected)
    np.testing.assert_array_equal(_as_numpy(mlx_iom), expected)


@pytest.mark.parametrize("as_mlx", [False, True])
def test_nms_masks_rejects_invalid_shapes(as_mlx):
    with pytest.raises(ValueError, match="pred_probs must have shape"):
        utils.nms_masks(
            _maybe_mlx(np.zeros((1, 1, 1), dtype=np.float32), as_mlx),
            _maybe_mlx(np.zeros((1, 1, 2, 2), dtype=np.float32), as_mlx),
            prob_threshold=0.60,
            iou_threshold=0.50,
        )

    with pytest.raises(ValueError, match="same number of detections"):
        utils.nms_masks(
            _maybe_mlx(np.zeros((2,), dtype=np.float32), as_mlx),
            _maybe_mlx(np.zeros((3, 2, 2), dtype=np.float32), as_mlx),
            prob_threshold=0.60,
            iou_threshold=0.50,
        )

    with pytest.raises(ValueError, match="batched pred_masks"):
        utils.nms_masks(
            _maybe_mlx(np.zeros((2, 3), dtype=np.float32), as_mlx),
            _maybe_mlx(np.zeros((3, 2, 2), dtype=np.float32), as_mlx),
            prob_threshold=0.60,
            iou_threshold=0.50,
        )

    with pytest.raises(ValueError, match="leading dimensions"):
        utils.nms_masks(
            _maybe_mlx(np.zeros((2, 3), dtype=np.float32), as_mlx),
            _maybe_mlx(np.zeros((2, 2, 2, 2), dtype=np.float32), as_mlx),
            prob_threshold=0.60,
            iou_threshold=0.50,
        )


@pytest.mark.parametrize("as_mlx", [False, True])
@pytest.mark.parametrize("overlap_fn", [utils.perf_mask_iou, utils.perf_mask_iom])
def test_pairwise_mask_overlap_rejects_invalid_shapes(overlap_fn, as_mlx):
    valid_masks = np.zeros((1, 2, 2), dtype=np.float32)
    same_size_wrong_shape = np.zeros((1, 1, 4), dtype=np.float32)
    wrong_rank = np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="spatial dimensions must match"):
        overlap_fn(
            _maybe_mlx(valid_masks, as_mlx),
            _maybe_mlx(same_size_wrong_shape, as_mlx),
        )

    with pytest.raises(ValueError, match="mask overlap inputs"):
        overlap_fn(
            _maybe_mlx(wrong_rank, as_mlx),
            _maybe_mlx(valid_masks, as_mlx),
        )


@pytest.mark.parametrize("as_mlx", [False, True])
@pytest.mark.parametrize(
    "overlap_fn", [utils._batched_mask_iou, utils._batched_mask_iom]
)
def test_batched_mask_overlap_rejects_invalid_rank(overlap_fn, as_mlx):
    with pytest.raises(ValueError, match="batched masks"):
        overlap_fn(_maybe_mlx(np.zeros((2, 2, 2), dtype=np.float32), as_mlx))


def test_mlx_overlap_and_nms_paths_do_not_export_full_masks(monkeypatch):
    scores, masks = _single_nms_fixture()
    pred_masks = masks[:2]
    gt_masks = masks[2:5]
    original_to_numpy = utils._to_numpy

    def guarded_to_numpy(value):
        if _is_mlx_array(value) and len(value.shape) >= 3:
            raise AssertionError(f"full mask tensor exported to host: {value.shape}")
        return original_to_numpy(value)

    monkeypatch.setattr(utils, "_to_numpy", guarded_to_numpy)

    keep = utils.nms_masks(
        mx.array(scores),
        mx.array(masks),
        prob_threshold=0.60,
        iou_threshold=0.50,
    )
    iou = utils.perf_mask_iou(mx.array(pred_masks), mx.array(gt_masks))

    np.testing.assert_array_equal(
        _as_numpy(keep),
        np.array([True, True, True, False, False]),
    )
    np.testing.assert_allclose(
        _as_numpy(iou),
        utils.perf_mask_iou(pred_masks, gt_masks),
        rtol=1e-6,
        atol=1e-6,
    )


def test_batched_mlx_nms_exports_only_valid_overlap_submatrices(monkeypatch):
    scores, masks = _single_nms_fixture()
    batch_scores = np.stack(
        [
            scores,
            np.array([0.10, 0.20, 0.30, 0.40, 0.90], dtype=np.float32),
        ]
    )
    batch_masks = np.stack([masks, masks])
    expected = np.array(
        [
            [True, True, True, False, False],
            [False, False, False, False, True],
        ]
    )
    exported_mlx_shapes = []
    original_to_host = utils._to_host_postprocess_numpy

    def guarded_to_host(value):
        if _is_mlx_array(value):
            exported_mlx_shapes.append(value.shape)
            if len(value.shape) >= 3:
                raise AssertionError(f"unfiltered MLX tensor exported: {value.shape}")
        return original_to_host(value)

    monkeypatch.setattr(utils, "_to_host_postprocess_numpy", guarded_to_host)

    keep = utils.nms_masks(
        mx.array(batch_scores),
        mx.array(batch_masks),
        prob_threshold=0.60,
        iou_threshold=0.50,
    )

    np.testing.assert_array_equal(_as_numpy(keep), expected)
    assert (4, 4) in exported_mlx_shapes
    assert (1, 1) in exported_mlx_shapes


@pytest.mark.parametrize("as_mlx", [False, True])
def test_nms_masks_do_compile_still_fails_fast(as_mlx):
    scores, masks = _single_nms_fixture()

    with pytest.raises(NotImplementedError, match="do_compile=True"):
        utils.nms_masks(
            _maybe_mlx(scores, as_mlx),
            _maybe_mlx(masks, as_mlx),
            0.60,
            0.50,
            do_compile=True,
        )
