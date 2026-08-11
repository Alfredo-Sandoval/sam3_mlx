from typing import cast

import numpy as np
import mlx.core as mx
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.train import matcher
from sam3_mlx.train.loss import loss_fns
from sam3_mlx.train.loss.sam3_loss import DummyLoss, Sam3LossWrapper


class _TargetArraysProbe(loss_fns.IABCEMdetr):
    def call_target_arrays(
        self,
        src_logits: mx.array,
        outputs: loss_fns.IABCETargetArraysOutputMap,
        targets: loss_fns.IABCETargetArraysTargetMap,
        indices: loss_fns.MatchIndices,
    ) -> tuple[mx.array, mx.array]:
        return self._target_arrays(src_logits, outputs, targets, indices)


class _OverrideTargetArraysProbe(loss_fns.IABCEMdetr):
    override_called: bool = False

    def _target_arrays(
        self,
        src_logits: mx.array,
        outputs: loss_fns.IABCETargetArraysOutputMap,
        targets: loss_fns.IABCETargetArraysTargetMap,
        indices: loss_fns.MatchIndices,
    ) -> tuple[mx.array, mx.array]:
        del outputs, targets, indices
        self.override_called = True
        zeros = mx.zeros(src_logits.shape, dtype=mx.float32)
        return zeros, zeros


class _ConstantLoss:
    def __call__(
        self,
        *,
        outputs: object,
        targets: object,
        indices: object,
        num_boxes: mx.array,
        is_aux: bool,
    ) -> loss_fns.LossDict:
        del outputs, targets, indices, num_boxes
        scale = 2.0 if is_aux else 1.0
        return {
            loss_fns.CORE_LOSS_KEY: mx.array(scale, dtype=mx.float32),
            "loss_probe": mx.array(scale * 3.0, dtype=mx.float32),
        }


def test_segment_miou_keeps_valid_count_branch_on_mlx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_host_export(*_args: object, **_kwargs: object) -> None:
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


def test_loss_with_weights_base_get_loss_raises_not_implemented() -> None:
    criterion = loss_fns.LossWithWeights(weight_dict=None, compute_aux=False)

    with pytest.raises(NotImplementedError):
        criterion.get_loss()


def test_iabce_get_loss_honors_subclass_target_arrays_override() -> None:
    criterion = _OverrideTargetArraysProbe(pos_weight=1.0, weak_loss=False)
    outputs: loss_fns.IABCEOutputMap = {
        "pred_logits": mx.zeros((1, 2, 1), dtype=mx.float32),
        "pred_boxes_xyxy": mx.zeros((1, 2, 4), dtype=mx.float32),
    }
    targets: loss_fns.IABCETargetMap = {
        "boxes_xyxy": mx.zeros((0, 4), dtype=mx.float32)
    }
    indices: loss_fns.MatchIndices = (
        mx.array([], dtype=mx.int64),
        mx.array([], dtype=mx.int64),
    )

    losses = criterion.get_loss(outputs, targets, indices, num_boxes=1)

    assert criterion.override_called
    assert loss_fns.CORE_LOSS_KEY not in losses


def test_iabce_presence_missing_keys_keep_native_precedence() -> None:
    outputs: loss_fns.IABCEOutputMap = {
        "pred_logits": mx.zeros((1, 1, 1), dtype=mx.float32),
        "pred_boxes_xyxy": mx.zeros((1, 1, 4), dtype=mx.float32),
    }
    empty_targets: loss_fns.IABCETargetMap = {
        "boxes_xyxy": mx.zeros((0, 4), dtype=mx.float32)
    }
    indices: loss_fns.MatchIndices = (
        mx.array([], dtype=mx.int64),
        mx.array([], dtype=mx.int64),
    )

    with pytest.raises(KeyError, match="object_ids_padded"):
        loss_fns.IABCEMdetr(
            pos_weight=1.0, weak_loss=False, use_presence=True
        ).get_loss(outputs, empty_targets, indices, num_boxes=1)

    missing_boxes_targets: loss_fns.IABCETargetMap = {
        "boxes_xyxy": mx.zeros((0, 4), dtype=mx.float32),
        "object_ids_padded": mx.zeros((1, 1), dtype=mx.int64),
    }
    with pytest.raises(KeyError, match="boxes_padded"):
        loss_fns.IABCEMdetr(
            pos_weight=1.0, weak_loss=False, use_presence=True
        ).get_loss(outputs, missing_boxes_targets, indices, num_boxes=1)


def test_video_query_filter_preserves_none_target_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[loss_fns.MatchIndices] = []

    def capture_target_select(
        values: mx.array, indices: loss_fns.MatchIndices
    ) -> mx.array:
        captured.append(indices)
        return values

    monkeypatch.setattr(loss_fns, "_target_select", capture_target_select)
    criterion = loss_fns.Boxes(
        apply_loss_to_det_queries_in_video_grounding=False,
    )
    outputs: loss_fns.BoxesOutputMap = {
        "pred_boxes": mx.array([[[0.5, 0.5, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]]]),
        "pred_boxes_xyxy": mx.array([[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]]),
        "is_video_grounding_batch": True,
        "Q_det": 1,
    }
    targets: loss_fns.BoxesTargetMap = {
        "boxes": mx.array([[0.5, 0.5, 1.0, 1.0]]),
        "boxes_xyxy": mx.array([[0.0, 0.0, 1.0, 1.0]]),
    }
    indices: loss_fns.MatchIndices = (
        mx.array([0], dtype=mx.int64),
        mx.array([1], dtype=mx.int64),
        None,
    )

    criterion.get_loss(outputs, targets, indices, num_boxes=1)

    assert len(captured) == 2
    for filtered in captured:
        assert len(filtered) == 3
        assert filtered[2] is None
        np.testing.assert_array_equal(to_numpy(filtered[0]), np.array([0]))
        np.testing.assert_array_equal(to_numpy(filtered[1]), np.array([0]))


def test_segment_miou_all_empty_targets_returns_one_without_host_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_host_export(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("segment_miou should keep empty-target handling on MLX")

    monkeypatch.setattr(loss_fns, "_to_numpy", fail_host_export)
    source = mx.zeros((2, 2, 2), dtype=mx.bool_)
    target = mx.zeros((2, 2, 2), dtype=mx.bool_)

    miou = loss_fns.segment_miou(source, target)

    np.testing.assert_array_equal(to_numpy(miou), np.array(1.0, dtype=np.float32))


def test_iabce_target_arrays_stay_mlx_native_for_matched_soft_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_host_export(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("IABCEMdetr target arrays should not round-trip via NumPy")

    monkeypatch.setattr(loss_fns, "_to_numpy", fail_host_export)
    criterion = _TargetArraysProbe(pos_weight=2.0, weak_loss=False)
    src_logits = mx.array(
        [
            [0.0, 0.5, -1.0],
            [1.0, -0.5, 2.0],
        ],
        dtype=mx.float32,
    )
    outputs: loss_fns.IABCETargetArraysOutputMap = {
        "pred_boxes_xyxy": mx.array(
            [
                [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0]],
                [[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 3.0, 3.0], [0.0, 0.0, 1.0, 1.0]],
            ],
            dtype=mx.float32,
        )
    }
    targets: loss_fns.IABCETargetArraysTargetMap = {
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

    target_classes, positive_targets = criterion.call_target_arrays(
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


def test_iabce_target_arrays_empty_match_stays_mlx_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_host_export(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("empty IABCEMdetr targets should remain MLX-native")

    monkeypatch.setattr(loss_fns, "_to_numpy", fail_host_export)
    criterion = _TargetArraysProbe(pos_weight=1.0, weak_loss=False)
    src_logits = mx.zeros((2, 3), dtype=mx.float32)
    outputs: loss_fns.IABCETargetArraysOutputMap = {
        "pred_boxes_xyxy": mx.zeros((2, 3, 4), dtype=mx.float32),
    }
    targets: loss_fns.IABCETargetArraysTargetMap = {
        "boxes_xyxy": mx.zeros((0, 4), dtype=mx.float32)
    }
    empty = mx.array([], dtype=mx.int64)

    target_classes, positive_targets = criterion.call_target_arrays(
        src_logits, outputs, targets, (empty, empty, empty)
    )

    expected = np.zeros((2, 3), dtype=np.float32)
    np.testing.assert_array_equal(to_numpy(target_classes), expected)
    np.testing.assert_array_equal(to_numpy(positive_targets), expected)


def test_training_matcher_cpu_boundary_is_named_and_returns_mlx_indices() -> None:
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

    batch_idx, src_idx, tgt_idx = cast(
        tuple[mx.array, mx.array, mx.array | None], matcher_module(outputs, targets)
    )

    np.testing.assert_array_equal(to_numpy(batch_idx), np.array([0], dtype=np.int64))
    np.testing.assert_array_equal(to_numpy(src_idx), np.array([1], dtype=np.int64))
    assert tgt_idx is None


def test_sam3_loss_wrapper_accumulates_primary_and_auxiliary_outputs() -> None:
    wrapper = Sam3LossWrapper([_ConstantLoss()])
    outputs: dict[str, object] = {
        "indices": (
            mx.array([0], dtype=mx.int64),
            mx.array([0], dtype=mx.int64),
        ),
        "aux_outputs": [
            {
                "indices": (
                    mx.array([0], dtype=mx.int64),
                    mx.array([0], dtype=mx.int64),
                )
            }
        ],
    }

    losses = wrapper.compute_loss(outputs, {"num_boxes": mx.array([1], dtype=mx.int64)})

    np.testing.assert_array_equal(to_numpy(losses[loss_fns.CORE_LOSS_KEY]), 3.0)
    np.testing.assert_array_equal(to_numpy(losses["loss_probe"]), 3.0)
    np.testing.assert_array_equal(to_numpy(losses["loss_probe_aux_0"]), 6.0)


def test_sam3_loss_wrapper_rejects_malformed_auxiliary_outputs() -> None:
    wrapper = Sam3LossWrapper([_ConstantLoss()])
    outputs: dict[str, object] = {
        "indices": (mx.array([], dtype=mx.int64), mx.array([], dtype=mx.int64)),
        "aux_outputs": {"indices": ()},
    }

    with pytest.raises(TypeError, match="aux_outputs must be a list"):
        wrapper.compute_loss(outputs, {"num_boxes": mx.array([0], dtype=mx.int64)})


def test_dummy_loss_accumulate_preserves_existing_core_loss() -> None:
    dummy = DummyLoss()
    existing = mx.array(4.0, dtype=mx.float32)
    losses = {loss_fns.CORE_LOSS_KEY: existing}

    assert dummy.accumulate(losses)[loss_fns.CORE_LOSS_KEY] is existing


def test_remaining_training_loss_cpu_boundary_is_documented_by_name() -> None:
    assert "instance_masks_to_semantic_masks" in loss_fns.TRAINING_LOSS_CPU_BOUNDARIES
    assert (
        "host"
        in loss_fns.TRAINING_LOSS_CPU_BOUNDARIES["instance_masks_to_semantic_masks"]
    )
