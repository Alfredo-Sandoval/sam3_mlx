from __future__ import annotations

from collections import OrderedDict
from types import MethodType, SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.model.multiplex_utils import (
    MultiplexState,
)
from sam3_mlx.model.video_tracking_multiplex_demo import (
    Sam3VideoTrackingMultiplexDemo,
    VideoTrackingMultiplexDemo,
)


def _demo_model(
    cls: type[VideoTrackingMultiplexDemo] = VideoTrackingMultiplexDemo,
) -> VideoTrackingMultiplexDemo:
    model = cls.__new__(cls)
    model.always_start_from_first_ann_frame = False
    model.clear_non_cond_mem_around_input = False
    model.clear_non_cond_mem_for_multi_obj = False
    model.fill_hole_area = 0
    model.non_overlap_masks_for_output = False
    model.use_memory_selection = False
    model.backbone = None
    model.image_size = 8
    model.input_mask_size = 8
    model.memory_temporal_stride_for_eval = 1
    model.num_maskmem = 1
    model.add_all_frames_to_correct_as_cond = False
    model.is_dynamic_model = True
    model.multimask_output_in_sam = False
    model.multimask_output_for_tracking = False
    model.multimask_min_pt_num = 1
    model.multimask_max_pt_num = 1
    model.num_multimask_outputs = 1
    return model


def _state(*, num_frames: int = 2) -> dict:
    return {
        "num_frames": num_frames,
        "video_height": 4,
        "video_width": 4,
        "images": None,
        "cached_features": {},
        "point_inputs_per_obj": {0: {}, 1: {}},
        "mask_inputs_per_obj": {0: {}, 1: {}},
        "obj_id_to_idx": OrderedDict([(10, 0), (20, 1)]),
        "obj_idx_to_id": OrderedDict([(0, 10), (1, 20)]),
        "obj_ids": [10, 20],
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        "output_dict_per_obj": {
            0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            1: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        },
        "temp_output_dict_per_obj": {
            0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            1: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        },
        "consolidated_frame_inds": {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        },
        "frames_already_tracked": {},
        "first_ann_frame_idx": None,
        "multiplex_state": SimpleNamespace(total_valid_entries=2),
        "tracking_has_started": False,
        "user_refined_frames_per_obj": {},
    }


def _empty_state(*, num_frames: int = 2) -> dict:
    state = _state(num_frames=num_frames)
    state["point_inputs_per_obj"] = {}
    state["mask_inputs_per_obj"] = {}
    state["obj_id_to_idx"] = OrderedDict()
    state["obj_idx_to_id"] = OrderedDict()
    state["obj_ids"] = []
    state["output_dict_per_obj"] = {}
    state["temp_output_dict_per_obj"] = {}
    state["multiplex_state"] = None
    return state


def _stage(values: list[float]) -> dict:
    masks = mx.array(
        np.stack(
            [np.full((1, 2, 2), value, dtype=np.float32) for value in values],
            axis=0,
        )
    )
    return {
        "pred_masks": masks,
        "object_score_logits": mx.array(
            np.arange(1, len(values) + 1, dtype=np.float32)[:, None]
        ),
    }


def _packed_stage(values: list[float]) -> dict:
    stage = _stage(values)
    stage.update(
        {
            "maskmem_features": mx.array([[[10.0], [20.0]]], dtype=mx.float32),
            "maskmem_pos_enc": [mx.array([[[1.0], [2.0]]], dtype=mx.float32)],
            "obj_ptr": mx.array([[[0.1], [0.2]]], dtype=mx.float32),
            "conditioning_objects": {0, 1},
            "local_obj_id_to_idx": OrderedDict([(10, 0), (20, 1)]),
        }
    )
    return stage


def _to_numpy(value):
    mx.eval(value)
    return np.asarray(value)


def test_demo_propagate_yields_existing_conditioning_frame_and_slices_objects():
    model = _demo_model()
    state = _state(num_frames=2)
    state["output_dict"]["cond_frame_outputs"][0] = _stage([3.0, 4.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            run_mem_encoder=False,
        )
    )

    assert len(outputs) == 1
    frame_idx, obj_ids, low_res_masks, video_res_masks = outputs[0]
    assert frame_idx == 0
    assert obj_ids == [10, 20]
    assert low_res_masks.shape == (2, 1, 2, 2)
    assert video_res_masks.shape == (2, 1, 4, 4)
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[3.0], [4.0]], dtype=np.float32),
    )
    assert state["frames_already_tracked"][0] == {"reverse": False}
    assert 0 in state["output_dict_per_obj"][0]["cond_frame_outputs"]
    assert 0 in state["output_dict_per_obj"][1]["cond_frame_outputs"]
    np.testing.assert_array_equal(
        _to_numpy(
            state["output_dict_per_obj"][1]["cond_frame_outputs"][0]["pred_masks"]
        ),
        np.full((1, 1, 2, 2), 4.0, dtype=np.float32),
    )


def test_sam3_demo_propagate_uses_seeded_state_path():
    model = _demo_model(Sam3VideoTrackingMultiplexDemo)
    state = _state(num_frames=1)
    state["output_dict"]["cond_frame_outputs"][0] = _stage([7.0, 8.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            run_mem_encoder=False,
        )
    )

    assert len(outputs) == 1
    frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores = outputs[0]
    assert frame_idx == 0
    assert obj_ids == [10, 20]
    assert low_res_masks.shape == (2, 1, 2, 2)
    assert video_res_masks.shape == (2, 1, 4, 4)
    assert obj_scores.shape == (2, 1)
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[7.0], [8.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(obj_scores),
        np.array([[1.0], [2.0]], dtype=np.float32),
    )


def test_sam3_demo_propagate_accepts_multiplex_preflight_keyword():
    model = _demo_model(Sam3VideoTrackingMultiplexDemo)
    state = _state(num_frames=1)
    state["output_dict"]["cond_frame_outputs"][0] = _stage([7.0, 8.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            run_mem_encoder=False,
            propagate_preflight=False,
        )
    )

    assert len(outputs) == 1
    frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores = outputs[0]
    assert frame_idx == 0
    assert obj_ids == [10, 20]
    assert low_res_masks.shape == (2, 1, 2, 2)
    assert video_res_masks.shape == (2, 1, 4, 4)
    np.testing.assert_array_equal(
        _to_numpy(obj_scores),
        np.array([[1.0], [2.0]], dtype=np.float32),
    )


def test_sam3_demo_propagate_filters_object_scores():
    model = _demo_model(Sam3VideoTrackingMultiplexDemo)
    state = _state(num_frames=1)
    state["output_dict"]["cond_frame_outputs"][0] = _stage([7.0, 8.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            obj_ids=[20],
            run_mem_encoder=False,
        )
    )

    assert len(outputs) == 1
    frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores = outputs[0]
    assert frame_idx == 0
    assert obj_ids == [20]
    assert low_res_masks.shape == (1, 1, 2, 2)
    assert video_res_masks.shape == (1, 1, 4, 4)
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[8.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(obj_scores),
        np.array([[2.0]], dtype=np.float32),
    )


def test_demo_add_new_masks_initializes_state_and_committed_outputs():
    model = _demo_model()
    state = _empty_state(num_frames=2)
    masks = mx.array(
        np.stack(
            [
                np.eye(4, dtype=np.float32),
                np.fliplr(np.eye(4, dtype=np.float32)),
            ],
            axis=0,
        )
    )
    calls = []

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        current_out = _stage([0.25, 0.75])
        current_out["conditioning_objects"] = {0, 1}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((1, 2, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame

    frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_masks(
        state,
        frame_idx=0,
        obj_ids=[10, 20],
        masks=masks,
    )

    assert frame_idx == 0
    assert obj_ids == [10, 20]
    assert low_res_masks is None
    assert video_res_masks.shape == (2, 1, 4, 4)
    assert state["multiplex_state"].object_ids == [10, 20]
    assert state["first_ann_frame_idx"] == 0
    assert state["consolidated_frame_inds"]["cond_frame_outputs"] == {0}
    assert 0 in state["output_dict"]["cond_frame_outputs"]
    assert calls[0]["is_init_cond_frame"] is True
    assert calls[0]["new_object_masks"] is None
    assert tuple(calls[0]["mask_inputs"].shape) == (2, 1, 8, 8)
    np.testing.assert_array_equal(
        _to_numpy(state["mask_inputs_per_obj"][0][0]),
        _to_numpy(masks[0:1, None] > 0.5),
    )
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[1024.0], [-1024.0]], dtype=np.float32),
    )
    assert 0 in state["output_dict_per_obj"][1]["cond_frame_outputs"]
    assert (
        "pred_masks_video_res"
        in state["output_dict_per_obj"][0]["cond_frame_outputs"][0]
    )


def test_demo_add_new_masks_state_can_propagate_seeded_frame():
    model = _demo_model()
    state = _empty_state(num_frames=1)

    def fake_run_single_frame(**kwargs):
        current_out = _stage([0.25, 0.75])
        current_out["conditioning_objects"] = {0, 1}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((1, 2, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame
    model.add_new_masks(
        state,
        frame_idx=0,
        obj_ids=[10, 20],
        masks=mx.ones((2, 4, 4), dtype=mx.float32),
    )

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            run_mem_encoder=False,
        )
    )

    assert len(outputs) == 1
    frame_idx, obj_ids, low_res_masks, video_res_masks = outputs[0]
    assert frame_idx == 0
    assert obj_ids == [10, 20]
    assert low_res_masks.shape == (2, 1, 2, 2)
    assert video_res_masks.shape == (2, 1, 4, 4)
    assert state["tracking_has_started"] is True


def test_demo_add_new_masks_marks_existing_state_for_dynamic_mask_insert():
    model = _demo_model()
    state = _state(num_frames=1)
    state["multiplex_state"] = MultiplexState(
        [[0, 1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
        object_ids=[10, 20],
    )
    state["output_dict"]["cond_frame_outputs"][0] = _stage([0.25, 0.75])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)
    calls = []

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        current_out = _stage([0.25, 0.75, 0.5])
        current_out["conditioning_objects"] = {0, 1, 2}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((2, 2, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame

    frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_masks(
        state,
        frame_idx=0,
        obj_ids=[30],
        masks=mx.ones((1, 4, 4), dtype=mx.float32),
    )

    assert frame_idx == 0
    assert obj_ids == [10, 20, 30]
    assert low_res_masks is None
    assert video_res_masks.shape == (3, 1, 4, 4)
    assert calls[0]["add_to_existing_state"] is True
    assert calls[0]["new_object_idxs"] == [2]
    assert calls[0]["new_object_ids"] == [30]
    assert calls[0]["allow_new_buckets"] is True
    assert tuple(calls[0]["mask_inputs"].shape) == (1, 1, 8, 8)
    assert tuple(calls[0]["new_object_masks"].shape) == (1, 1, 8, 8)


def test_demo_add_new_masks_reconditions_existing_state_object():
    model = _demo_model()
    state = _state(num_frames=1)
    original_multiplex_state = state["multiplex_state"]
    calls = []

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        current_out = _stage([0.25, 0.75])
        current_out["conditioning_objects"] = {1}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((1, 2, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame

    frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_masks(
        state,
        frame_idx=0,
        obj_ids=[20],
        masks=mx.ones((1, 4, 4), dtype=mx.float32),
        reconditioning=True,
    )

    assert frame_idx == 0
    assert obj_ids == [10, 20]
    assert low_res_masks is None
    assert video_res_masks.shape == (2, 1, 4, 4)
    assert state["multiplex_state"] is original_multiplex_state
    assert state["obj_ids"] == [10, 20]
    assert calls[0]["reconditioning"] is True
    assert calls[0]["new_object_idxs"] == [1]
    assert calls[0]["new_object_ids"] == [20]
    assert calls[0]["allow_new_buckets"] is False
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[0.25], [1024.0]], dtype=np.float32),
    )
    assert 0 in state["output_dict_per_obj"][1]["cond_frame_outputs"]


def test_demo_run_single_frame_adds_point_masks_to_existing_state():
    model = _demo_model()
    state = _state(num_frames=1)
    state["multiplex_state"] = MultiplexState(
        [[0, 1]],
        allowed_bucket_capacity=2,
        object_ids=[10, 20],
    )
    existing_out = _stage([0.25, 0.75])
    existing_out["conditioning_objects"] = {0, 1}
    existing_out["pred_masks_high_res"] = _stage([2.5, 7.5])["pred_masks"]
    existing_out["obj_ptr"] = mx.zeros((1, 2, 1), dtype=mx.float32)
    state["output_dict"]["cond_frame_outputs"][0] = existing_out
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)
    point_inputs = {
        "point_coords": mx.array([[[4.0, 2.0]]], dtype=mx.float32),
        "point_labels": mx.array([[1]], dtype=mx.int32),
    }
    add_calls = []
    head_calls = []

    def fake_get_image_feature(inference_state, frame_idx, batch_size):
        assert inference_state is state
        assert frame_idx == 0
        assert batch_size == 1
        features = {
            "interactive": {
                "vision_feats": [
                    mx.ones((4, 1, 2), dtype=mx.float32),
                    mx.ones((1, 1, 2), dtype=mx.float32),
                ],
                "feat_sizes": [(2, 2), (1, 1)],
            },
            "sam2_backbone_out": {
                "vision_feats": [mx.ones((1, 1, 2), dtype=mx.float32)],
                "feat_sizes": [(1, 1)],
            },
        }
        return mx.ones((1, 3, 4, 4), dtype=mx.float32), features

    def fake_get_interactive_pix_mem(vision_feats, feat_sizes):
        assert len(vision_feats) == 2
        assert feat_sizes == [(2, 2), (1, 1)]
        return mx.full((1, 2, 1, 1), 3.0, dtype=mx.float32)

    def fake_forward_sam_heads(**kwargs):
        head_calls.append(kwargs)
        return {"low_res_masks": mx.full((1, 1, 2, 2), 9.0, dtype=mx.float32)}

    def fake_add_new_masks_to_existing_state(**kwargs):
        add_calls.append(kwargs)
        kwargs["prev_output"]["pred_masks"] = mx.concatenate(
            [kwargs["prev_output"]["pred_masks"], kwargs["new_masks"]],
            axis=0,
        )
        kwargs["prev_output"]["object_score_logits"] = mx.concatenate(
            [
                kwargs["prev_output"]["object_score_logits"],
                mx.array([[3.0]], dtype=mx.float32),
            ],
            axis=0,
        )
        kwargs["prev_output"]["conditioning_objects"].update(kwargs["obj_idxs_in_mask"])

    model._get_image_feature = fake_get_image_feature
    model._get_interactive_pix_mem = fake_get_interactive_pix_mem
    model._forward_sam_heads = fake_forward_sam_heads
    model.add_new_masks_to_existing_state = fake_add_new_masks_to_existing_state

    current_out, pred_masks = model._run_single_frame_inference(
        inference_state=state,
        output_dict=state["output_dict"],
        frame_idx=0,
        batch_size=1,
        is_init_cond_frame=True,
        point_inputs=point_inputs,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=False,
        add_to_existing_state=True,
        new_object_idxs=[2],
        new_object_ids=[30],
        allow_new_buckets=True,
        prefer_new_buckets=True,
    )

    assert current_out is existing_out
    assert pred_masks.shape == (3, 1, 2, 2)
    assert head_calls[0]["point_inputs"] is point_inputs
    assert head_calls[0]["objects_to_interact"] == [2]
    assert head_calls[0]["multiplex_state"] is state["multiplex_state"]
    assert add_calls[0]["prev_output"] is existing_out
    assert add_calls[0]["obj_idxs_in_mask"] == [2]
    assert add_calls[0]["obj_ids_in_mask"] == [30]
    assert add_calls[0]["are_masks_from_pts"] is True
    assert add_calls[0]["allow_new_buckets"] is True
    assert add_calls[0]["prefer_new_buckets"] is True
    np.testing.assert_array_equal(
        _to_numpy(add_calls[0]["new_masks"]),
        np.full((1, 1, 2, 2), 9.0, dtype=np.float32),
    )


def test_demo_run_single_frame_reconditions_point_masks_in_existing_state():
    model = _demo_model()
    state = _state(num_frames=1)
    state["multiplex_state"] = MultiplexState(
        [[0, 1]],
        allowed_bucket_capacity=2,
        object_ids=[10, 20],
    )
    existing_out = _stage([0.25, 0.75])
    existing_out["conditioning_objects"] = {0}
    existing_out["pred_masks_high_res"] = _stage([2.5, 7.5])["pred_masks"]
    existing_out["obj_ptr"] = mx.zeros((1, 2, 1), dtype=mx.float32)
    state["output_dict"]["cond_frame_outputs"][0] = existing_out
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)
    point_inputs = {
        "point_coords": mx.array([[[6.0, 2.0]]], dtype=mx.float32),
        "point_labels": mx.array([[0]], dtype=mx.int32),
    }
    prev_logits = mx.array(
        np.stack(
            [
                np.full((1, 2, 2), -6.0, dtype=np.float32),
                np.full((1, 2, 2), 6.0, dtype=np.float32),
            ],
            axis=0,
        )
    )
    head_calls = []
    recondition_calls = []

    def fake_get_image_feature(inference_state, frame_idx, batch_size):
        assert inference_state is state
        assert frame_idx == 0
        assert batch_size == 1
        features = {
            "interactive": {
                "vision_feats": [
                    mx.ones((4, 1, 2), dtype=mx.float32),
                    mx.ones((1, 1, 2), dtype=mx.float32),
                ],
                "feat_sizes": [(2, 2), (1, 1)],
            },
            "sam2_backbone_out": {
                "vision_feats": [mx.ones((1, 1, 2), dtype=mx.float32)],
                "feat_sizes": [(1, 1)],
            },
        }
        return mx.ones((1, 3, 4, 4), dtype=mx.float32), features

    def fake_get_interactive_pix_mem(vision_feats, feat_sizes):
        assert len(vision_feats) == 2
        assert feat_sizes == [(2, 2), (1, 1)]
        return mx.full((1, 2, 1, 1), 4.0, dtype=mx.float32)

    def fake_forward_sam_heads(**kwargs):
        head_calls.append(kwargs)
        return {"low_res_masks": mx.full((1, 1, 2, 2), 11.0, dtype=mx.float32)}

    def fake_recondition_masks_in_existing_state(**kwargs):
        recondition_calls.append(kwargs)
        kwargs["prev_output"]["pred_masks"] = mx.concatenate(
            [kwargs["prev_output"]["pred_masks"][:1], kwargs["new_masks"]],
            axis=0,
        )
        kwargs["prev_output"]["object_score_logits"] = mx.array(
            [[1.0], [9.0]],
            dtype=mx.float32,
        )
        kwargs["prev_output"]["conditioning_objects"].update(kwargs["obj_idxs_in_mask"])

    model._get_image_feature = fake_get_image_feature
    model._get_interactive_pix_mem = fake_get_interactive_pix_mem
    model._forward_sam_heads = fake_forward_sam_heads
    model.recondition_masks_in_existing_state = fake_recondition_masks_in_existing_state

    current_out, pred_masks = model._run_single_frame_inference(
        inference_state=state,
        output_dict=state["output_dict"],
        frame_idx=0,
        batch_size=1,
        is_init_cond_frame=False,
        point_inputs=point_inputs,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=prev_logits,
        reconditioning=True,
        new_object_idxs=[1],
        new_object_ids=[20],
        objects_to_interact=[1],
    )

    assert current_out is existing_out
    assert pred_masks.shape == (2, 1, 2, 2)
    assert head_calls[0]["point_inputs"] is point_inputs
    assert head_calls[0]["objects_to_interact"] == [1]
    assert head_calls[0]["multiplex_state"] is state["multiplex_state"]
    np.testing.assert_array_equal(
        _to_numpy(head_calls[0]["mask_inputs"]),
        np.full((1, 1, 2, 2), 6.0, dtype=np.float32),
    )
    assert recondition_calls[0]["prev_output"] is existing_out
    assert recondition_calls[0]["obj_idxs_in_mask"] == [1]
    assert recondition_calls[0]["obj_ids_in_mask"] == [20]
    assert recondition_calls[0]["add_mask_to_memory"] is True
    assert recondition_calls[0]["are_masks_from_pts"] is True
    np.testing.assert_array_equal(
        _to_numpy(recondition_calls[0]["new_masks"]),
        np.full((1, 1, 2, 2), 11.0, dtype=np.float32),
    )


def test_demo_add_new_points_initializes_state_and_committed_outputs():
    model = _demo_model()
    state = _empty_state(num_frames=2)
    calls = []

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        current_out = _stage([0.5])
        current_out["conditioning_objects"] = {0}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((1, 1, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame

    frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_points(
        state,
        frame_idx=0,
        obj_id=10,
        points=mx.array([[0.5, 0.25]], dtype=mx.float32),
        labels=mx.array([1], dtype=mx.int32),
        clear_old_points=True,
        rel_coordinates=True,
    )

    assert frame_idx == 0
    assert obj_ids == [10]
    assert low_res_masks is None
    assert video_res_masks.shape == (1, 1, 4, 4)
    assert state["multiplex_state"].object_ids == [10]
    assert state["first_ann_frame_idx"] == 0
    assert state["consolidated_frame_inds"]["cond_frame_outputs"] == {0}
    assert calls[0]["is_init_cond_frame"] is True
    assert calls[0]["mask_inputs"] is None
    np.testing.assert_array_equal(
        _to_numpy(calls[0]["point_inputs"]["point_coords"]),
        np.array([[[4.0, 2.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(calls[0]["point_inputs"]["point_labels"]),
        np.array([[1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _to_numpy(state["point_inputs_per_obj"][0][0]["point_coords"]),
        np.array([[[4.0, 2.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[0.5]], dtype=np.float32),
    )
    assert 0 in state["output_dict_per_obj"][0]["cond_frame_outputs"]


def test_demo_add_new_points_accumulates_before_tracking_starts():
    model = _demo_model()
    state = _empty_state(num_frames=2)
    calls = []

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        current_out = _stage([0.5])
        current_out["conditioning_objects"] = {0}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((1, 1, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame
    model.add_new_points(
        state,
        frame_idx=0,
        obj_id=10,
        points=mx.array([[0.5, 0.25]], dtype=mx.float32),
        labels=mx.array([1], dtype=mx.int32),
        clear_old_points=True,
        rel_coordinates=True,
    )
    model.add_new_points(
        state,
        frame_idx=0,
        obj_id=10,
        points=mx.array([[0.25, 0.75]], dtype=mx.float32),
        labels=mx.array([0], dtype=mx.int32),
        clear_old_points=False,
        rel_coordinates=True,
    )

    assert len(calls) == 2
    expected_coords = np.array([[[4.0, 2.0], [2.0, 6.0]]], dtype=np.float32)
    expected_labels = np.array([[1, 0]], dtype=np.int32)
    np.testing.assert_array_equal(
        _to_numpy(calls[1]["point_inputs"]["point_coords"]),
        expected_coords,
    )
    np.testing.assert_array_equal(
        _to_numpy(calls[1]["point_inputs"]["point_labels"]),
        expected_labels,
    )
    np.testing.assert_array_equal(
        _to_numpy(state["point_inputs_per_obj"][0][0]["point_coords"]),
        expected_coords,
    )
    np.testing.assert_array_equal(
        _to_numpy(state["point_inputs_per_obj"][0][0]["point_labels"]),
        expected_labels,
    )


def test_demo_add_new_points_adds_dynamic_object_to_existing_state():
    model = _demo_model()
    state = _state(num_frames=1)
    state["multiplex_state"] = MultiplexState(
        [[0, 1]],
        allowed_bucket_capacity=2,
        object_ids=[10, 20],
    )
    state["output_dict"]["cond_frame_outputs"][0] = _stage([0.25, 0.75])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)
    calls = []

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        kwargs["inference_state"]["multiplex_state"].add_objects(
            [2],
            object_ids=[30],
            allow_new_buckets=True,
            prefer_new_buckets=True,
        )
        current_out = _stage([0.25, 0.75, 0.5])
        current_out["conditioning_objects"] = {0, 1, 2}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((2, 2, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame

    frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_points(
        state,
        frame_idx=0,
        obj_id=30,
        points=mx.array([[0.25, 0.75]], dtype=mx.float32),
        labels=mx.array([1], dtype=mx.int32),
        clear_old_points=True,
        rel_coordinates=True,
    )

    assert frame_idx == 0
    assert obj_ids == [10, 20, 30]
    assert low_res_masks is None
    assert video_res_masks.shape == (3, 1, 4, 4)
    assert state["multiplex_state"].assignments == [[0, 1], [2, -1]]
    assert state["multiplex_state"].object_ids == [10, 20, 30]
    assert state["obj_id_to_idx"] == OrderedDict([(10, 0), (20, 1), (30, 2)])
    assert calls[0]["batch_size"] == 1
    assert calls[0]["add_to_existing_state"] is True
    assert calls[0]["new_object_idxs"] == [2]
    assert calls[0]["new_object_ids"] == [30]
    assert calls[0]["allow_new_buckets"] is True
    assert calls[0]["prefer_new_buckets"] is True
    assert calls[0]["is_init_cond_frame"] is True
    assert calls[0]["reverse"] is False
    np.testing.assert_array_equal(
        _to_numpy(calls[0]["point_inputs"]["point_coords"]),
        np.array([[[2.0, 6.0]]], dtype=np.float32),
    )
    assert 0 in state["output_dict_per_obj"][2]["cond_frame_outputs"]
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[0.25], [0.75], [0.5]], dtype=np.float32),
    )


def test_demo_add_new_points_refines_existing_multi_object_state():
    model = _demo_model()
    calls = []

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        batch_size = int(kwargs["batch_size"])
        if kwargs["reconditioning"]:
            current_out = _stage([8.0, 1.5])
            current_out["conditioning_objects"] = {0}
        else:
            current_out = _stage([0.5 + idx for idx in range(batch_size)])
            current_out["conditioning_objects"] = set(range(batch_size))
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros(
            (1, current_out["pred_masks"].shape[0], 1),
            dtype=mx.float32,
        )
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame
    multi_state = _empty_state(num_frames=1)
    model.add_new_masks(
        multi_state,
        frame_idx=0,
        obj_ids=[10, 20],
        masks=mx.ones((2, 4, 4), dtype=mx.float32),
    )
    list(
        model.propagate_in_video(
            multi_state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            run_mem_encoder=False,
        )
    )

    frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_points(
        multi_state,
        frame_idx=0,
        obj_id=10,
        points=mx.array([[0.25, 0.75]], dtype=mx.float32),
        labels=mx.array([0], dtype=mx.int32),
        clear_old_points=False,
        rel_coordinates=True,
    )

    refine_call = calls[-1]
    assert frame_idx == 0
    assert obj_ids == [10, 20]
    assert low_res_masks is None
    assert video_res_masks.shape == (2, 1, 4, 4)
    assert refine_call["batch_size"] == 1
    assert refine_call["add_to_existing_state"] is False
    assert refine_call["reconditioning"] is True
    assert refine_call["new_object_idxs"] == [0]
    assert refine_call["new_object_ids"] == [10]
    assert refine_call["objects_to_interact"] == [0]
    assert refine_call["allow_new_buckets"] is False
    assert refine_call["prefer_new_buckets"] is False
    assert refine_call["is_init_cond_frame"] is True
    assert multi_state["user_refined_frames_per_obj"][10] == {0}
    np.testing.assert_array_equal(
        _to_numpy(refine_call["point_inputs"]["point_coords"]),
        np.array([[[2.0, 6.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(
            multi_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"][
                :, :, 0, 0
            ]
        ),
        np.array([[8.0], [1.5]], dtype=np.float32),
    )


def test_demo_add_new_points_gap_fills_existing_multi_object_state():
    model = _demo_model()
    calls = []

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        batch_size = int(kwargs["batch_size"])
        if kwargs.get("objects_to_interact") == [0]:
            current_out = _stage([8.0, 1.5])
            current_out["conditioning_objects"] = {0}
        else:
            current_out = _stage([0.5 + idx for idx in range(batch_size)])
            current_out["conditioning_objects"] = set(range(batch_size))
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros(
            (1, current_out["pred_masks"].shape[0], 1),
            dtype=mx.float32,
        )
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame
    state = _empty_state(num_frames=2)
    model.add_new_masks(
        state,
        frame_idx=0,
        obj_ids=[10, 20],
        masks=mx.ones((2, 4, 4), dtype=mx.float32),
    )

    frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_points(
        state,
        frame_idx=1,
        obj_id=10,
        points=mx.array([[0.25, 0.75]], dtype=mx.float32),
        labels=mx.array([0], dtype=mx.int32),
        clear_old_points=False,
        rel_coordinates=True,
    )

    gap_fill_call = calls[-1]
    assert frame_idx == 1
    assert obj_ids == [10, 20]
    assert low_res_masks is None
    assert video_res_masks.shape == (2, 1, 4, 4)
    assert gap_fill_call["batch_size"] == 2
    assert gap_fill_call["add_to_existing_state"] is False
    assert gap_fill_call["reconditioning"] is False
    assert gap_fill_call["objects_to_interact"] == [0]
    assert gap_fill_call["new_object_idxs"] == [0]
    assert gap_fill_call["new_object_ids"] == [10]
    assert gap_fill_call["is_init_cond_frame"] is False
    assert gap_fill_call["allow_new_buckets"] is False
    assert gap_fill_call["prefer_new_buckets"] is False
    assert 1 in state["output_dict"]["cond_frame_outputs"]
    assert 1 not in state["output_dict"]["non_cond_frame_outputs"]
    assert state["consolidated_frame_inds"]["cond_frame_outputs"] == {0, 1}
    np.testing.assert_array_equal(
        _to_numpy(gap_fill_call["point_inputs"]["point_coords"]),
        np.array([[[2.0, 6.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(
            state["output_dict"]["cond_frame_outputs"][1]["pred_masks"][:, :, 0, 0]
        ),
        np.array([[8.0], [1.5]], dtype=np.float32),
    )


def test_demo_add_new_points_state_can_propagate_seeded_frame():
    model = _demo_model()
    state = _empty_state(num_frames=1)

    def fake_run_single_frame(**kwargs):
        current_out = _stage([0.5])
        current_out["conditioning_objects"] = {0}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((1, 1, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame
    model.add_new_points(
        state,
        frame_idx=0,
        obj_id=10,
        points=mx.array([[0.5, 0.25]], dtype=mx.float32),
        labels=mx.array([1], dtype=mx.int32),
        clear_old_points=True,
        rel_coordinates=True,
    )

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            run_mem_encoder=False,
        )
    )

    assert len(outputs) == 1
    frame_idx, obj_ids, low_res_masks, video_res_masks = outputs[0]
    assert frame_idx == 0
    assert obj_ids == [10]
    assert low_res_masks.shape == (1, 1, 2, 2)
    assert video_res_masks.shape == (1, 1, 4, 4)
    assert state["tracking_has_started"] is True


def test_demo_add_new_points_refines_tracked_single_object_frame():
    model = _demo_model()
    state = _empty_state(num_frames=2)
    calls = []
    values = [0.5, 10.0, 40.0, 1.25]

    def fake_run_single_frame(**kwargs):
        calls.append(kwargs)
        current_out = _stage([values[len(calls) - 1]])
        current_out["conditioning_objects"] = {0}
        current_out["maskmem_features"] = None
        current_out["maskmem_pos_enc"] = None
        current_out["obj_ptr"] = mx.zeros((1, 1, 1), dtype=mx.float32)
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = fake_run_single_frame
    model.add_new_points(
        state,
        frame_idx=0,
        obj_id=10,
        points=mx.array([[0.5, 0.25]], dtype=mx.float32),
        labels=mx.array([1], dtype=mx.int32),
        clear_old_points=True,
        rel_coordinates=True,
    )
    list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
            run_mem_encoder=False,
        )
    )
    assert 1 in state["output_dict"]["non_cond_frame_outputs"]
    assert state["frames_already_tracked"][1] == {"reverse": False}

    frame_idx, obj_ids, low_res_masks, video_res_masks = model.add_new_points(
        state,
        frame_idx=1,
        obj_id=10,
        points=mx.array([[0.125, 0.5]], dtype=mx.float32),
        labels=mx.array([1], dtype=mx.int32),
        clear_old_points=True,
        rel_coordinates=True,
    )

    assert frame_idx == 1
    assert obj_ids == [10]
    assert low_res_masks is None
    assert video_res_masks.shape == (1, 1, 4, 4)
    assert calls[2]["is_init_cond_frame"] is True
    assert calls[2]["prev_sam_mask_logits"] is None
    assert 1 in state["output_dict"]["cond_frame_outputs"]
    assert 1 not in state["output_dict"]["non_cond_frame_outputs"]
    assert state["consolidated_frame_inds"]["cond_frame_outputs"] == {0, 1}
    assert 1 not in state["consolidated_frame_inds"]["non_cond_frame_outputs"]
    assert state["user_refined_frames_per_obj"][10] == {1}
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[40.0]], dtype=np.float32),
    )

    model.add_new_points(
        state,
        frame_idx=1,
        obj_id=10,
        points=mx.array([[0.75, 0.25]], dtype=mx.float32),
        labels=mx.array([0], dtype=mx.int32),
        clear_old_points=False,
        rel_coordinates=True,
    )

    assert calls[3]["is_init_cond_frame"] is False
    np.testing.assert_array_equal(
        _to_numpy(calls[3]["prev_sam_mask_logits"]),
        np.full((1, 1, 2, 2), 32.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(calls[3]["point_inputs"]["point_coords"]),
        np.array([[[1.0, 4.0], [6.0, 2.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(calls[3]["point_inputs"]["point_labels"]),
        np.array([[1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _to_numpy(
            state["output_dict"]["cond_frame_outputs"][1]["pred_masks"][:, :, 0, 0]
        ),
        np.array([[1.25]], dtype=np.float32),
    )


def test_demo_propagate_filters_requested_object_ids():
    model = _demo_model()
    state = _state(num_frames=1)
    state["output_dict"]["cond_frame_outputs"][0] = _stage([3.0, 4.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            obj_ids=[20],
            run_mem_encoder=False,
        )
    )

    assert len(outputs) == 1
    frame_idx, obj_ids, low_res_masks, video_res_masks = outputs[0]
    assert frame_idx == 0
    assert obj_ids == [20]
    assert low_res_masks.shape == (1, 1, 2, 2)
    assert video_res_masks.shape == (1, 1, 4, 4)
    np.testing.assert_array_equal(
        _to_numpy(video_res_masks[:, :, 0, 0]),
        np.array([[4.0]], dtype=np.float32),
    )


def test_demo_propagate_clears_nearby_non_conditioning_memory():
    model = _demo_model()
    model.clear_non_cond_mem_around_input = True
    model.clear_non_cond_mem_for_multi_obj = True
    state = _state(num_frames=3)
    state["output_dict"]["cond_frame_outputs"][1] = _stage([3.0, 4.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(1)
    for frame_idx, value in [(0, 5.0), (2, 6.0)]:
        state["output_dict"]["non_cond_frame_outputs"][frame_idx] = _stage(
            [value, value + 1.0]
        )
        state["consolidated_frame_inds"]["non_cond_frame_outputs"].add(frame_idx)
        state["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame_idx] = {
            "pred_masks": mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            "object_score_logits": mx.zeros((1, 1), dtype=mx.float32),
        }

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=1,
            max_frame_num_to_track=0,
            run_mem_encoder=False,
        )
    )

    assert [frame_idx for frame_idx, *_ in outputs] == [1]
    assert state["output_dict"]["non_cond_frame_outputs"] == {}
    assert state["consolidated_frame_inds"]["non_cond_frame_outputs"] == set()
    assert state["output_dict_per_obj"][0]["non_cond_frame_outputs"] == {}


def test_demo_propagate_runs_and_stores_non_conditioning_frame():
    model = _demo_model()
    state = _state(num_frames=2)
    state["output_dict"]["cond_frame_outputs"][0] = _stage([1.0, 2.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)
    calls = []

    def fake_run_single_frame(self, **kwargs):
        calls.append(kwargs)
        current_out = _stage([5.0, 6.0])
        return current_out, current_out["pred_masks"]

    model._run_single_frame_inference = MethodType(fake_run_single_frame, model)

    outputs = list(
        model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
            run_mem_encoder=False,
        )
    )

    assert [frame_idx for frame_idx, *_ in outputs] == [0, 1]
    assert calls[0]["frame_idx"] == 1
    assert calls[0]["batch_size"] == 2
    assert calls[0]["run_mem_encoder"] is False
    assert 1 in state["output_dict"]["non_cond_frame_outputs"]
    assert state["output_dict"]["non_cond_frame_outputs"][1][
        "local_obj_id_to_idx"
    ] == OrderedDict([(10, 0), (20, 1)])
    assert state["frames_already_tracked"][1] == {"reverse": False}
    np.testing.assert_array_equal(
        _to_numpy(outputs[1][3][:, :, 0, 0]),
        np.array([[5.0], [6.0]], dtype=np.float32),
    )


def test_demo_propagate_requires_conditioning_and_validates_obj_filtering():
    model = _demo_model()
    state = _state(num_frames=1)

    with pytest.raises(RuntimeError, match="No points are provided"):
        list(model.propagate_in_video(state))

    state["output_dict"]["cond_frame_outputs"][0] = _stage([1.0, 2.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)
    with pytest.raises(ValueError, match="Unknown obj_ids"):
        list(model.propagate_in_video(state, obj_ids=[30]))
    with pytest.raises(ValueError, match="duplicate object ids"):
        list(model.propagate_in_video(state, obj_ids=[10, 10]))


def test_demo_clear_all_points_in_video_resets_seeded_state():
    model = _demo_model()
    state = _state(num_frames=1)
    state["output_dict"]["cond_frame_outputs"][0] = _stage([1.0, 2.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)
    state["point_inputs_per_obj"][0][0] = {"point_coords": mx.zeros((1, 1, 2))}
    state["tracking_has_started"] = True
    state["frames_already_tracked"][0] = {"reverse": False}

    model.clear_all_points_in_video(state)

    assert state["obj_ids"] == []
    assert state["obj_id_to_idx"] == OrderedDict()
    assert state["output_dict"] == {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    assert state["point_inputs_per_obj"] == {}
    assert state["multiplex_state"] is None
    assert state["tracking_has_started"] is False


def test_demo_remove_object_slices_packed_state_and_per_object_outputs():
    model = _demo_model()
    state = _state(num_frames=1)
    state["multiplex_state"] = MultiplexState(
        [[0, 1]],
        allowed_bucket_capacity=2,
        object_ids=[10, 20],
    )
    state["output_dict"]["cond_frame_outputs"][0] = _packed_stage([3.0, 4.0])
    state["consolidated_frame_inds"]["cond_frame_outputs"].add(0)
    state["point_inputs_per_obj"][0][0] = {"point_coords": mx.zeros((1, 1, 2))}
    state["point_inputs_per_obj"][1][0] = {"point_coords": mx.ones((1, 1, 2))}
    model._add_output_per_object(
        state,
        0,
        state["output_dict"]["cond_frame_outputs"][0],
        "cond_frame_outputs",
    )

    remaining_obj_ids, updated_frames = model.remove_object(
        state,
        10,
        need_output=True,
    )

    assert remaining_obj_ids == [20]
    assert len(updated_frames) == 1
    assert state["obj_id_to_idx"] == OrderedDict([(20, 0)])
    assert state["obj_idx_to_id"] == OrderedDict([(0, 20)])
    assert state["multiplex_state"].object_ids == [20]
    out = state["output_dict"]["cond_frame_outputs"][0]
    assert out["local_obj_id_to_idx"] == OrderedDict([(20, 0)])
    assert out["conditioning_objects"] == {0}
    np.testing.assert_array_equal(
        _to_numpy(out["pred_masks"][:, :, 0, 0]),
        np.array([[4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(out["object_score_logits"]),
        np.array([[2.0]], dtype=np.float32),
    )
    assert set(state["output_dict_per_obj"]) == {0}
    np.testing.assert_array_equal(
        _to_numpy(
            state["output_dict_per_obj"][0]["cond_frame_outputs"][0]["pred_masks"]
        ),
        np.full((1, 1, 2, 2), 4.0, dtype=np.float32),
    )


def test_demo_remove_objects_ignores_missing_non_strict_and_errors_strict():
    model = _demo_model()
    state = _state(num_frames=1)

    remaining_obj_ids, updated_frames = model.remove_objects(
        state,
        [30],
        strict=False,
    )

    assert remaining_obj_ids == [10, 20]
    assert updated_frames == []
    with pytest.raises(ValueError, match="do not exist"):
        model.remove_objects(state, [30], strict=True)


def test_demo_get_image_feature_computes_and_caches_loader_backed_frame():
    model = _demo_model()
    model.backbone = object()
    image_stack = mx.ones((2, 3, 4, 4), dtype=mx.float32)
    state = _state(num_frames=2)
    state["images"] = image_stack
    prepared_features = {
        "interactive": {
            "vision_feats": [mx.zeros((4, 1, 2), dtype=mx.float32)],
            "feat_sizes": [(2, 2)],
        },
        "sam2_backbone_out": {
            "vision_feats": [mx.ones((4, 1, 2), dtype=mx.float32)],
            "feat_sizes": [(2, 2)],
        },
    }
    calls = []

    def fake_forward_image(image, **kwargs):
        calls.append((image, kwargs))
        return {"raw": "backbone"}

    def fake_prepare(backbone_out):
        assert backbone_out == {"raw": "backbone"}
        return prepared_features

    model.forward_image = fake_forward_image
    model._prepare_backbone_features = fake_prepare

    image, features = model._get_image_feature(state, frame_idx=1, batch_size=2)

    assert features is prepared_features
    assert tuple(image.shape) == (1, 3, 4, 4)
    assert calls[0][1] == {
        "need_sam3_out": True,
        "need_interactive_out": True,
        "need_propagation_out": True,
    }
    assert set(state["cached_features"]) == {1}
    np.testing.assert_array_equal(
        _to_numpy(image), np.ones((1, 3, 4, 4), dtype=np.float32)
    )
