import numpy as np
import mlx.core as mx

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.train import matcher
from sam3_mlx.train.loss import loss_fns


def test_segment_miou_keeps_valid_count_branch_on_mlx(monkeypatch):
    def fail_host_export(*_args, **_kwargs):
        raise AssertionError("segment_miou should not export MLX arrays to NumPy")

    monkeypatch.setattr(loss_fns, "_to_numpy", fail_host_export)
    source = mx.array(
        [
            [[True, False], [True, False]],
            [[True, False], [False, False]],
            [[False, True], [False, True]],
        ],
        dtype=mx.bool_,
    )
    target = mx.array(
        [
            [[True, True], [False, False]],
            [[False, False], [False, False]],
            [[False, True], [False, True]],
        ],
        dtype=mx.bool_,
    )

    miou = loss_fns.segment_miou(source, target)

    np.testing.assert_allclose(
        to_numpy(miou),
        np.array((1.0 / 3.0 + 1.0) / 2.0, dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_segment_miou_all_empty_targets_returns_one_without_host_export(monkeypatch):
    def fail_host_export(*_args, **_kwargs):
        raise AssertionError("segment_miou should keep empty-target handling on MLX")

    monkeypatch.setattr(loss_fns, "_to_numpy", fail_host_export)
    source = mx.zeros((2, 2, 2), dtype=mx.bool_)
    target = mx.zeros((2, 2, 2), dtype=mx.bool_)

    miou = loss_fns.segment_miou(source, target)

    np.testing.assert_array_equal(to_numpy(miou), np.array(1.0, dtype=np.float32))


def test_iabce_target_arrays_stay_mlx_native_for_matched_soft_targets(monkeypatch):
    def fail_host_export(*_args, **_kwargs):
        raise AssertionError("IABCEMdetr target arrays should not round-trip via NumPy")

    monkeypatch.setattr(loss_fns, "_to_numpy", fail_host_export)
    criterion = loss_fns.IABCEMdetr(pos_weight=2.0, weak_loss=False)
    src_logits = mx.array(
        [
            [0.0, 0.5, -1.0],
            [1.0, -0.5, 2.0],
        ],
        dtype=mx.float32,
    )
    outputs = {
        "pred_boxes_xyxy": mx.array(
            [
                [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0]],
                [[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 1.0, 1.0]],
            ],
            dtype=mx.float32,
        )
    }
    targets = {
        "boxes_xyxy": mx.array(
            [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]],
            dtype=mx.float32,
        )
    }
    indices = (
        mx.array([0, 1], dtype=mx.int64),
        mx.array([1, 2], dtype=mx.int64),
        mx.array([0, 1], dtype=mx.int64),
    )

    target_classes, positive_targets = criterion._target_arrays(
        src_logits, outputs, targets, indices
    )

    expected_targets = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    expected_positive = expected_targets.copy()
    matched_probs = 1.0 / (1.0 + np.exp(-np.array([0.5, 2.0], dtype=np.float32)))
    expected_positive[0, 1] = matched_probs[0] ** criterion.alpha
    expected_positive[1, 2] = matched_probs[1] ** criterion.alpha

    np.testing.assert_array_equal(to_numpy(target_classes), expected_targets)
    np.testing.assert_allclose(
        to_numpy(positive_targets), expected_positive, rtol=1e-6, atol=1e-6
    )


def test_iabce_target_arrays_empty_match_stays_mlx_native(monkeypatch):
    def fail_host_export(*_args, **_kwargs):
        raise AssertionError("empty IABCEMdetr targets should remain MLX-native")

    monkeypatch.setattr(loss_fns, "_to_numpy", fail_host_export)
    criterion = loss_fns.IABCEMdetr(pos_weight=1.0, weak_loss=False)
    src_logits = mx.zeros((2, 3), dtype=mx.float32)
    outputs = {"pred_boxes_xyxy": mx.zeros((2, 3, 4), dtype=mx.float32)}
    targets = {"boxes_xyxy": mx.zeros((0, 4), dtype=mx.float32)}
    empty = mx.array([], dtype=mx.int64)

    target_classes, positive_targets = criterion._target_arrays(
        src_logits, outputs, targets, (empty, empty, empty)
    )

    expected = np.zeros((2, 3), dtype=np.float32)
    np.testing.assert_array_equal(to_numpy(target_classes), expected)
    np.testing.assert_array_equal(to_numpy(positive_targets), expected)


def test_training_matcher_cpu_boundary_is_named_and_returns_mlx_indices():
    assert "host" in matcher.TRAINING_MATCHER_CPU_BOUNDARY
    matcher_module = matcher.BinaryHungarianMatcher(
        cost_class=1.0, cost_bbox=1.0, cost_giou=1.0
    )
    outputs = {
        "pred_logits": mx.array([[[0.0], [3.0]]], dtype=mx.float32),
        "pred_boxes": mx.array(
            [[[0.8, 0.8, 0.2, 0.2], [0.5, 0.5, 0.4, 0.4]]],
            dtype=mx.float32,
        ),
    }
    targets = {
        "boxes": mx.array([[0.5, 0.5, 0.4, 0.4]], dtype=mx.float32),
        "num_boxes": mx.array([1], dtype=mx.int64),
    }

    batch_idx, src_idx, tgt_idx = matcher_module(outputs, targets)

    np.testing.assert_array_equal(to_numpy(batch_idx), np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(to_numpy(src_idx), np.array([1], dtype=np.int64))
    assert tgt_idx is None


def test_remaining_training_loss_cpu_boundary_is_documented_by_name():
    assert "instance_masks_to_semantic_masks" in loss_fns.TRAINING_LOSS_CPU_BOUNDARIES
    assert (
        "host"
        in loss_fns.TRAINING_LOSS_CPU_BOUNDARIES["instance_masks_to_semantic_masks"]
    )
