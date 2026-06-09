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

from sam3_mlx.model.sam3_tracker_base import Sam3TrackerBase
from sam3_mlx.model_builder import (
    _create_tracker_maskmem_backbone,
    _create_tracker_transformer,
)
from tests._paths import PORT_TRACKER_FIXTURE_ROOT

S = 72  # image_size / backbone_stride
IMAGE = 1008


@pytest.fixture(scope="module")
def base():
    model = Sam3TrackerBase(
        backbone=None,
        transformer=_create_tracker_transformer(),
        maskmem_backbone=_create_tracker_maskmem_backbone(),
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


def _inputs():
    mx.random.seed(0)
    backbone = mx.random.normal((1, 256, S, S))
    high_res = [
        mx.random.normal((1, 32, 4 * S, 4 * S)),
        mx.random.normal((1, 64, 2 * S, 2 * S)),
    ]
    point_inputs = {
        "point_coords": mx.array([[[504.0, 504.0]]]),
        "point_labels": mx.array([[1]], dtype=mx.int32),
    }
    return backbone, high_res, point_inputs


def test_forward_sam_heads_multimask_shapes(base):
    backbone, high_res, point_inputs = _inputs()
    out = base._forward_sam_heads(
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


def test_forward_sam_heads_singlemask_shapes(base):
    backbone, high_res, point_inputs = _inputs()
    out = base._forward_sam_heads(
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


def test_forward_sam_heads_accepts_no_point_prompt(base):
    backbone, high_res, _ = _inputs()
    # point_inputs=None must pad an empty (label -1) point, not crash
    out = base._forward_sam_heads(
        backbone_features=backbone,
        point_inputs=None,
        high_res_features=high_res,
        multimask_output=True,
    )
    assert tuple(out[0].shape) == (1, 3, 4 * S, 4 * S)
    # finite outputs everywhere an object is present (NO_OBJ_SCORE is finite too)
    assert bool(mx.isfinite(out[5]).all())  # obj_ptr


def test_use_mask_as_output_direct_prompt_logits_and_empty_object_pointer():
    model = Sam3TrackerBase(
        backbone=None,
        transformer=_create_tracker_transformer(),
        maskmem_backbone=_create_tracker_maskmem_backbone(),
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

    out = model._use_mask_as_output(backbone, high_res, mx.array(mask_np))
    low_multi, high_multi, ious, low_best, high_best, obj_ptr, obj_score = out

    assert tuple(low_multi.shape) == (2, 1, 8, 8)
    assert tuple(high_multi.shape) == (2, 1, 32, 32)
    assert tuple(ious.shape) == (2, 1)
    assert tuple(low_best.shape) == (2, 1, 8, 8)
    assert tuple(high_best.shape) == (2, 1, 32, 32)
    assert tuple(obj_ptr.shape) == (2, 256)
    assert tuple(obj_score.shape) == (2, 1)
    np.testing.assert_allclose(np.array(obj_score), [[10.0], [-10.0]], atol=0.0)
    np.testing.assert_allclose(np.array(ious), np.ones((2, 1)), atol=0.0)
    np.testing.assert_allclose(np.array(high_best[0]), 10.0, atol=0.0)
    np.testing.assert_allclose(np.array(high_best[1]), -10.0, atol=0.0)
    assert bool(mx.isfinite(obj_ptr[0]).all())
    np.testing.assert_allclose(
        np.array(obj_ptr[1]),
        np.array(model.no_obj_ptr[0]),
        rtol=0,
        atol=1e-6,
    )


def test_encode_new_memory_shapes(base):
    feat = mx.random.normal((S * S, 1, 256))  # (HW, B, C)
    masks = mx.random.normal((1, 1, IMAGE, IMAGE))
    object_score_logits = mx.array([[1.5]])
    maskmem_features, maskmem_pos_enc = base._encode_new_memory(
        image=None,
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


def _small_memory_output(side=4):
    return {
        "maskmem_features": mx.ones((1, 64, side, side)),
        "maskmem_pos_enc": [mx.zeros((1, 64, side, side))],
        "obj_ptr": mx.ones((1, 256)),
    }


def test_prepare_memory_conditioned_features_initial_frame_adds_no_mem_embed(base):
    side = 4
    feat = mx.zeros((side * side, 1, 256))
    out = base._prepare_memory_conditioned_features(
        frame_idx=0,
        is_init_cond_frame=True,
        current_vision_feats=[feat],
        current_vision_pos_embeds=[mx.zeros_like(feat)],
        feat_sizes=[(side, side)],
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        num_frames=1,
    )
    expected = (feat + base.no_mem_embed).transpose(1, 2, 0).reshape(1, 256, side, side)
    assert tuple(out.shape) == (1, 256, side, side)
    assert float(mx.max(mx.abs(out - expected))) < 1e-6


def test_prepare_memory_conditioned_features_uses_spatial_memory_and_obj_ptrs(base):
    side = 4
    mx.random.seed(4)
    feat = mx.random.normal((side * side, 1, 256))
    pos = mx.random.normal((side * side, 1, 256))
    output_dict = {
        "cond_frame_outputs": {0: _small_memory_output(side)},
        "non_cond_frame_outputs": {1: _small_memory_output(side)},
    }
    out = base._prepare_memory_conditioned_features(
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


def test_track_step_point_prompt_contract_without_memory_encoder(base):
    backbone, high_res, point_inputs = _inputs()
    current_vision_feats = [
        high_res[0].reshape(1, 32, -1).transpose(2, 0, 1),
        high_res[1].reshape(1, 64, -1).transpose(2, 0, 1),
        backbone.reshape(1, 256, -1).transpose(2, 0, 1),
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


def test_frame_filter_keeps_thresholded_history_and_must_include_neighbor(base):
    old_use_memory_selection = base.use_memory_selection
    old_mf_threshold = base.mf_threshold
    base.use_memory_selection = True
    base.mf_threshold = 0.5
    output_dict = {
        "non_cond_frame_outputs": {
            1: {"eff_iou_score": mx.array(0.4)},
            2: {"eff_iou_score": mx.array(0.9)},
            3: {"eff_iou_score": mx.array(0.2)},
        }
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


def test_apply_non_overlapping_constraints_clamps_losing_objects(base):
    pred_masks = mx.array(
        [
            [[[5.0, 1.0], [0.0, -20.0]]],
            [[[3.0, 2.0], [4.0, -30.0]]],
        ]
    )
    constrained = np.array(base._apply_non_overlapping_constraints(pred_masks))
    expected = np.array(
        [
            [[[5.0, -10.0], [-10.0, -20.0]]],
            [[[-10.0, 2.0], [4.0, -30.0]]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(constrained, expected, atol=0.0)


def test_use_multimask_logic(base):
    pts1 = {"point_labels": mx.zeros((1, 1), dtype=mx.int32)}
    # multimask_output_for_tracking=True, range [0,1] -> 1 point qualifies
    assert base._use_multimask(is_init_cond_frame=True, point_inputs=pts1) is True
    assert base._use_multimask(is_init_cond_frame=False, point_inputs=pts1) is True
    # 2 points exceed multimask_max_pt_num=1 -> no multimask
    pts2 = {"point_labels": mx.zeros((1, 2), dtype=mx.int32)}
    assert base._use_multimask(is_init_cond_frame=True, point_inputs=pts2) is False
    # no points -> num_pts=0, still within [0,1]
    assert base._use_multimask(is_init_cond_frame=True, point_inputs=None) is True


def test_direct_mask_parity_fixture_is_current():
    fixture = PORT_TRACKER_FIXTURE_ROOT / "direct_mask_parity.json"
    data = json.loads(fixture.read_text())

    assert data["atol"] == 2e-3
    assert data["worst_max_abs"] <= data["atol"]
    assert data["results"]["object_score_logits"]["max_abs"] == 0.0
    assert data["results"]["ious"]["max_abs"] == 0.0


def test_cal_mem_score_matches_formula(base):
    object_score_logits = mx.array([[2.0]])
    iou_score = mx.array([[0.8]])
    score = base.cal_mem_score(object_score_logits, iou_score)
    # object_score_norm = sigmoid(2)*2 - 1; score = norm * iou (mean over 1 elem)
    expected = (float(mx.sigmoid(mx.array(2.0))) * 2 - 1) * 0.8
    assert abs(float(score) - expected) < 1e-6
    # negative logit -> normalized score 0
    assert abs(float(base.cal_mem_score(mx.array([[-3.0]]), mx.array([[0.9]])))) < 1e-6
