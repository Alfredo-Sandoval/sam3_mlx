"""Frame-0 SAM forward and memory-conditioning contracts for Sam3TrackerBase.

Numerical parity against the official module is verified by
`scripts/tracker_parity.py --forward`, `--encode-memory`, and `--memory-attn`.
These torch-free tests lock in shapes, multimask selection, memory assembly, and
overlap constraints so the tracker port cannot silently regress to fail-fast
stubs or wrong contracts.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.data_misc import reshape_array, transpose_array
from sam3_mlx.model.sam3_tracker_base import (
    PointInputs,
    Sam3TrackerBase,
    TrackerFrameOutput,
    TrackerOutputState,
    TrackerStoredFrameOutput,
)
from sam3_mlx.model_builder import (
    create_tracker_maskmem_backbone,
    create_tracker_transformer,
)
from tests._json_contracts import require_mapping, require_real
from tests._paths import PORT_TRACKER_FIXTURE_ROOT

S = 72  # image_size / backbone_stride
IMAGE = 1008
type _SamHeadOutput = tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
]


class _TrackerHarness(Sam3TrackerBase):
    def forward_sam_heads(
        self,
        *,
        backbone_features: mx.array,
        point_inputs: PointInputs | None,
        high_res_features: list[mx.array] | None,
        multimask_output: bool,
    ) -> _SamHeadOutput:
        return self._forward_sam_heads(
            backbone_features=backbone_features,
            point_inputs=point_inputs,
            high_res_features=high_res_features,
            multimask_output=multimask_output,
        )

    def mask_as_output(
        self,
        backbone_features: mx.array,
        high_res_features: list[mx.array],
        mask_inputs: mx.array,
    ) -> _SamHeadOutput:
        return self._use_mask_as_output(
            backbone_features,
            high_res_features,
            mask_inputs,
        )

    def encode_new_memory(
        self,
        *,
        current_vision_feats: list[mx.array],
        feat_sizes: list[tuple[int, int]],
        pred_masks_high_res: mx.array,
        object_score_logits: mx.array,
        is_mask_from_pts: bool,
    ) -> tuple[mx.array, list[mx.array]]:
        return self._encode_new_memory(
            image=None,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=pred_masks_high_res,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )

    def prepare_memory_conditioned_features(
        self,
        *,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: list[mx.array],
        current_vision_pos_embeds: list[mx.array],
        feat_sizes: list[tuple[int, int]],
        output_dict: TrackerOutputState,
        num_frames: int,
    ) -> mx.array:
        return self._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            output_dict=output_dict,
            num_frames=num_frames,
        )

    def apply_non_overlapping_constraints(self, pred_masks: mx.array) -> mx.array:
        return self._apply_non_overlapping_constraints(pred_masks)

    def use_multimask(
        self,
        *,
        is_init_cond_frame: bool,
        point_inputs: PointInputs | None,
    ) -> bool:
        return self._use_multimask(is_init_cond_frame, point_inputs)


@pytest.fixture(scope="module")
def base() -> _TrackerHarness:
    model = _TrackerHarness(
        backbone=None,
        transformer=create_tracker_transformer(),
        maskmem_backbone=create_tracker_maskmem_backbone(),
        image_size=IMAGE,
        num_maskmem=7,
        backbone_stride=14,
        multimask_output_in_sam=True,
        forward_backbone_per_frame_for_eval=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        non_overlap_masks_for_mem_enc=False,
        max_cond_frames_in_attn=4,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )
    model.eval()
    return model


def _inputs() -> tuple[mx.array, list[mx.array], PointInputs]:
    mx.random.seed(0)
    backbone = mx.random.normal((1, 256, S, S))
    high_res = [
        mx.random.normal((1, 32, 4 * S, 4 * S)),
        mx.random.normal((1, 64, 2 * S, 2 * S)),
    ]
    point_inputs: PointInputs = {
        "point_coords": mx.array([[[504.0, 504.0]]]),
        "point_labels": mx.array([[1]], dtype=mx.int32),
    }
    return backbone, high_res, point_inputs


def test_forward_sam_heads_multimask_shapes(base: _TrackerHarness) -> None:
    backbone, high_res, point_inputs = _inputs()
    out = base.forward_sam_heads(
        backbone_features=backbone,
        point_inputs=point_inputs,
        high_res_features=high_res,
        multimask_output=True,
    )
    (low_multi, high_multi, ious, low_best, high_best, obj_ptr, obj_score) = out
    assert tuple(low_multi.shape) == (1, 3, 4 * S, 4 * S)
    assert tuple(high_multi.shape) == (1, 3, IMAGE, IMAGE)
    assert tuple(ious.shape) == (1, 3)
    assert tuple(low_best.shape) == (1, 1, 4 * S, 4 * S)
    assert tuple(high_best.shape) == (1, 1, IMAGE, IMAGE)
    assert tuple(obj_ptr.shape) == (1, 256)
    assert tuple(obj_score.shape) == (1, 1)


def test_forward_sam_heads_singlemask_shapes(base: _TrackerHarness) -> None:
    backbone, high_res, point_inputs = _inputs()
    out = base.forward_sam_heads(
        backbone_features=backbone,
        point_inputs=point_inputs,
        high_res_features=high_res,
        multimask_output=False,
    )
    low_multi, high_multi, ious, low_best, high_best, _, _ = out
    assert tuple(low_multi.shape) == (1, 1, 4 * S, 4 * S)
    assert tuple(high_multi.shape) == (1, 1, IMAGE, IMAGE)
    assert tuple(ious.shape) == (1, 1)
    # single-mask path returns the same tensors for best == multi
    assert tuple(low_best.shape) == (1, 1, 4 * S, 4 * S)
    assert tuple(high_best.shape) == (1, 1, IMAGE, IMAGE)


def test_forward_sam_heads_accepts_no_point_prompt(base: _TrackerHarness) -> None:
    backbone, high_res, _ = _inputs()
    # point_inputs=None must pad an empty (label -1) point, not crash
    out = base.forward_sam_heads(
        backbone_features=backbone,
        point_inputs=None,
        high_res_features=high_res,
        multimask_output=True,
    )
    assert tuple(out[0].shape) == (1, 3, 4 * S, 4 * S)
    # finite outputs everywhere an object is present (NO_OBJ_SCORE is finite too)
    assert bool(mx.isfinite(out[5]).all())  # obj_ptr


def test_use_mask_as_output_direct_prompt_logits_and_empty_object_pointer():
    model = _TrackerHarness(
        backbone=None,
        transformer=create_tracker_transformer(),
        maskmem_backbone=create_tracker_maskmem_backbone(),
        image_size=28,
        num_maskmem=2,
        backbone_stride=14,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        non_overlap_masks_for_mem_enc=False,
        max_cond_frames_in_attn=1,
    )
    model.eval()
    rng = np.random.default_rng(7)
    backbone = mx.array(rng.standard_normal((2, 256, 2, 2)).astype(np.float32))
    high_res = [
        mx.array(rng.standard_normal((2, 32, 8, 8)).astype(np.float32)),
        mx.array(rng.standard_normal((2, 64, 4, 4)).astype(np.float32)),
    ]
    mask_np = np.zeros((2, 1, 32, 32), dtype=np.float32)
    mask_np[0, 0] = 1.0

    out = model.mask_as_output(backbone, high_res, mx.array(mask_np))
    low_multi, high_multi, ious, low_best, high_best, obj_ptr, obj_score = out

    assert tuple(low_multi.shape) == (2, 1, 8, 8)
    assert tuple(high_multi.shape) == (2, 1, 32, 32)
    assert tuple(ious.shape) == (2, 1)
    assert tuple(low_best.shape) == (2, 1, 8, 8)
    assert tuple(high_best.shape) == (2, 1, 32, 32)
    assert tuple(obj_ptr.shape) == (2, 256)
    assert tuple(obj_score.shape) == (2, 1)
    np.testing.assert_allclose(to_numpy(obj_score), [[10.0], [-10.0]], atol=0.0)
    np.testing.assert_allclose(to_numpy(ious), np.ones((2, 1)), atol=0.0)
    np.testing.assert_allclose(to_numpy(high_best[0]), 10.0, atol=0.0)
    np.testing.assert_allclose(to_numpy(high_best[1]), -10.0, atol=0.0)
    assert bool(mx.isfinite(obj_ptr[0]).all())
    np.testing.assert_allclose(
        to_numpy(obj_ptr[1]),
        to_numpy(model.no_obj_ptr[0]),
        rtol=0,
        atol=1e-6,
    )


def test_encode_new_memory_shapes(base: _TrackerHarness) -> None:
    feat = mx.random.normal((S * S, 1, 256))  # (HW, B, C)
    masks = mx.random.normal((1, 1, IMAGE, IMAGE))
    object_score_logits = mx.array([[1.5]])
    maskmem_features, maskmem_pos_enc = base.encode_new_memory(
        current_vision_feats=[feat],
        feat_sizes=[(S, S)],
        pred_masks_high_res=masks,
        object_score_logits=object_score_logits,
        is_mask_from_pts=True,
    )
    # mem_dim=64 spatial memory at the backbone feature resolution
    assert tuple(maskmem_features.shape) == (1, 64, S, S)
    assert isinstance(maskmem_pos_enc, list)
    assert tuple(maskmem_pos_enc[0].shape) == (1, 64, S, S)


def _small_memory_output(side: int = 4) -> TrackerStoredFrameOutput:
    return TrackerStoredFrameOutput(
        {
            "maskmem_features": mx.ones((1, 64, side, side)),
            "maskmem_pos_enc": [mx.zeros((1, 64, side, side))],
            "obj_ptr": mx.ones((1, 256)),
            "object_score_logits": mx.ones((1, 1)),
            "pred_masks": mx.ones((1, 1, side, side)),
        }
    )


def _scored_memory_output(score: float) -> TrackerStoredFrameOutput:
    output = _small_memory_output()
    output["eff_iou_score"] = mx.array(score)
    return output


def test_prepare_memory_conditioned_features_initial_frame_adds_no_mem_embed(
    base: _TrackerHarness,
) -> None:
    side = 4
    feat = mx.zeros((side * side, 1, 256))
    out = base.prepare_memory_conditioned_features(
        frame_idx=0,
        is_init_cond_frame=True,
        current_vision_feats=[feat],
        current_vision_pos_embeds=[mx.zeros_like(feat)],
        feat_sizes=[(side, side)],
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        num_frames=1,
    )
    expected = reshape_array(
        transpose_array(feat + base.no_mem_embed, 1, 2, 0),
        1,
        256,
        side,
        side,
    )
    assert tuple(out.shape) == (1, 256, side, side)
    assert float(mx.max(mx.abs(out - expected))) < 1e-6


def test_prepare_memory_conditioned_features_uses_spatial_memory_and_obj_ptrs(
    base: _TrackerHarness,
) -> None:
    side = 4
    mx.random.seed(4)
    feat = mx.random.normal((side * side, 1, 256))
    pos = mx.random.normal((side * side, 1, 256))
    output_dict: TrackerOutputState = {
        "cond_frame_outputs": {0: _small_memory_output(side)},
        "non_cond_frame_outputs": {1: _small_memory_output(side)},
    }
    out = base.prepare_memory_conditioned_features(
        frame_idx=2,
        is_init_cond_frame=False,
        current_vision_feats=[feat],
        current_vision_pos_embeds=[pos],
        feat_sizes=[(side, side)],
        output_dict=output_dict,
        num_frames=3,
    )
    assert tuple(out.shape) == (1, 256, side, side)
    assert bool(mx.isfinite(out).all())


def _flatten_feature(value: mx.array) -> mx.array:
    batch_size, channels = value.shape[:2]
    return transpose_array(
        reshape_array(value, batch_size, channels, -1),
        2,
        0,
        1,
    )


def test_track_step_point_prompt_contract_without_memory_encoder(
    base: _TrackerHarness,
) -> None:
    backbone, high_res, point_inputs = _inputs()
    current_vision_feats = [
        _flatten_feature(high_res[0]),
        _flatten_feature(high_res[1]),
        _flatten_feature(backbone),
    ]
    current_vision_pos = [mx.zeros_like(x) for x in current_vision_feats]
    out = base.track_step(
        frame_idx=0,
        is_init_cond_frame=True,
        current_vision_feats=current_vision_feats,
        current_vision_pos_embeds=current_vision_pos,
        feat_sizes=[(4 * S, 4 * S), (2 * S, 2 * S), (S, S)],
        image=None,
        point_inputs=point_inputs,
        mask_inputs=None,
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        num_frames=1,
        run_mem_encoder=False,
    )
    assert tuple(out["pred_masks"].shape) == (1, 1, 4 * S, 4 * S)
    assert tuple(out["pred_masks_high_res"].shape) == (1, 1, IMAGE, IMAGE)
    assert tuple(out["obj_ptr"].shape) == (1, 256)
    assert tuple(out["object_score_logits"].shape) == (1, 1)
    assert out["maskmem_features"] is None
    assert out["maskmem_pos_enc"] is None


def test_track_step_preserves_transient_output_while_trimming_archived_frame(
    base: _TrackerHarness,
) -> None:
    backbone, high_res, point_inputs = _inputs()
    current_vision_feats = [
        _flatten_feature(high_res[0]),
        _flatten_feature(high_res[1]),
        _flatten_feature(backbone),
    ]
    old_output: TrackerFrameOutput = {
        "pred_masks": mx.ones((1, 1, 4 * S, 4 * S)),
        "pred_masks_high_res": mx.ones((1, 1, IMAGE, IMAGE)),
        "obj_ptr": mx.ones((1, 256)),
        "object_score_logits": mx.ones((1, 1)),
        "maskmem_features": mx.ones((1, 64, S, S)),
        "maskmem_pos_enc": [mx.zeros((1, 64, S, S))],
    }
    output_dict: TrackerOutputState = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {0: old_output},
    }
    old_trim_setting = base.trim_past_non_cond_mem_for_eval
    base.trim_past_non_cond_mem_for_eval = True
    try:
        current = base.track_step(
            frame_idx=base.num_maskmem,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=[
                mx.zeros_like(value) for value in current_vision_feats
            ],
            feat_sizes=[(4 * S, 4 * S), (2 * S, 2 * S), (S, S)],
            image=None,
            point_inputs=point_inputs,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=base.num_maskmem + 1,
            run_mem_encoder=False,
        )
    finally:
        base.trim_past_non_cond_mem_for_eval = old_trim_setting

    assert "pred_masks_high_res" in current
    assert current["maskmem_features"] is None
    assert set(output_dict["non_cond_frame_outputs"][0]) == {
        "pred_masks",
        "obj_ptr",
        "object_score_logits",
    }


def test_frame_filter_keeps_thresholded_history_and_must_include_neighbor(
    base: _TrackerHarness,
) -> None:
    old_use_memory_selection = base.use_memory_selection
    old_mf_threshold = base.mf_threshold
    base.use_memory_selection = True
    base.mf_threshold = 0.5
    output_dict: TrackerOutputState = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {
            1: _scored_memory_output(0.4),
            2: _scored_memory_output(0.9),
            3: _scored_memory_output(0.2),
        },
    }
    try:
        assert base.frame_filter(
            output_dict, False, frame_idx=4, num_frames=5, r=1
        ) == [2, 3]
        assert (
            base.frame_filter(output_dict, False, frame_idx=0, num_frames=5, r=1) == []
        )
    finally:
        base.use_memory_selection = old_use_memory_selection
        base.mf_threshold = old_mf_threshold


def test_apply_non_overlapping_constraints_clamps_losing_objects(
    base: _TrackerHarness,
) -> None:
    pred_masks = mx.array(
        [
            [[[5.0, 1.0], [0.0, -20.0]]],
            [[[3.0, 2.0], [4.0, -30.0]]],
        ]
    )
    constrained = to_numpy(base.apply_non_overlapping_constraints(pred_masks))
    expected = np.array(
        [
            [[[5.0, -10.0], [-10.0, -20.0]]],
            [[[-10.0, 2.0], [4.0, -30.0]]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(constrained, expected, atol=0.0)


def test_use_multimask_logic(base: _TrackerHarness) -> None:
    pts1: PointInputs = {
        "point_coords": mx.zeros((1, 1, 2), dtype=mx.float32),
        "point_labels": mx.zeros((1, 1), dtype=mx.int32),
    }
    # multimask_output_for_tracking=True, range [0,1] -> 1 point qualifies
    assert base.use_multimask(is_init_cond_frame=True, point_inputs=pts1) is True
    assert base.use_multimask(is_init_cond_frame=False, point_inputs=pts1) is True
    # 2 points exceed multimask_max_pt_num=1 -> no multimask
    pts2: PointInputs = {
        "point_coords": mx.zeros((1, 2, 2), dtype=mx.float32),
        "point_labels": mx.zeros((1, 2), dtype=mx.int32),
    }
    assert base.use_multimask(is_init_cond_frame=True, point_inputs=pts2) is False
    # no points -> num_pts=0, still within [0,1]
    assert base.use_multimask(is_init_cond_frame=True, point_inputs=None) is True


def test_direct_mask_parity_fixture_is_current() -> None:
    fixture = PORT_TRACKER_FIXTURE_ROOT / "direct_mask_parity.json"
    payload: object = json.loads(fixture.read_text())
    data = require_mapping(payload, "direct mask parity fixture")
    atol = require_real(data.get("atol"), "direct mask parity atol")
    worst_max_abs = require_real(
        data.get("worst_max_abs"),
        "direct mask parity worst_max_abs",
    )
    results = require_mapping(data.get("results"), "direct mask parity results")
    object_score = require_mapping(
        results.get("object_score_logits"),
        "direct mask parity object_score_logits",
    )
    ious = require_mapping(results.get("ious"), "direct mask parity ious")

    assert atol == 2e-3
    assert worst_max_abs <= atol
    assert require_real(object_score.get("max_abs"), "object score max_abs") == 0.0
    assert require_real(ious.get("max_abs"), "ious max_abs") == 0.0


def test_cal_mem_score_matches_formula(base: _TrackerHarness) -> None:
    object_score_logits = mx.array([[2.0]])
    iou_score = mx.array([[0.8]])
    score = base.cal_mem_score(object_score_logits, iou_score)
    # object_score_norm = sigmoid(2)*2 - 1; score = norm * iou (mean over 1 elem)
    expected = (float(mx.sigmoid(mx.array(2.0))) * 2 - 1) * 0.8
    assert abs(float(score) - expected) < 1e-6
    # negative logit -> normalized score 0
    assert abs(float(base.cal_mem_score(mx.array([[-3.0]]), mx.array([[0.9]])))) < 1e-6
