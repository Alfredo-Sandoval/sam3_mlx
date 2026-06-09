"""Tracker predictor contracts.

These tests cover the tracker predictor state-machine slice. The numerical
tracker kernels remain covered in ``tests/port/tracker/test_tracker_forward.py`` and
``scripts/tracker_parity.py``; this file locks the higher-level predictor state
machine without invoking the unported image backbone.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx import Sam3MlxUnsupportedError
from sam3_mlx.model.sam3_tracking_predictor import Sam3TrackerPredictor
from tests._paths import PORT_TRACKER_FIXTURE_ROOT, REPO_ROOT


PREDICTOR_KWARGS = dict(
    image_size=28,
    num_maskmem=2,
    backbone_stride=14,
    max_cond_frames_in_attn=1,
    multimask_output_in_sam=True,
    multimask_output_for_tracking=True,
    multimask_min_pt_num=0,
    multimask_max_pt_num=1,
)


def _to_numpy(value):
    mx.eval(value)
    return np.asarray(value)


class _ScriptedPredictor(Sam3TrackerPredictor):
    """Predictor with scripted frame outputs for state-machine tests."""

    def __init__(self):
        super().__init__(**PREDICTOR_KWARGS)
        self.eval()
        self.calls = []
        self.mask_calls = []
        self.memory_calls = []

    def _frame_out(self, frame_idx, run_mem_encoder, batch_size=1):
        value = float(frame_idx + 1)
        mask = mx.full(
            (batch_size, 1, self.low_res_mask_size, self.low_res_mask_size),
            value,
            dtype=mx.float32,
        )
        out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": mask,
            "obj_ptr": mx.full((batch_size, self.hidden_dim), value, dtype=mx.float32),
            "object_score_logits": mx.full((batch_size, 1), value, dtype=mx.float32),
        }
        if run_mem_encoder:
            side = self.sam_image_embedding_size
            out["maskmem_features"] = mx.full(
                (batch_size, self.mem_dim, side, side),
                value + 10.0,
                dtype=mx.float32,
            )
            out["maskmem_pos_enc"] = [
                mx.full(
                    (batch_size, self.mem_dim, side, side),
                    value + 20.0,
                    dtype=mx.float32,
                )
            ]
        return out, mask

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
    ):
        del inference_state, output_dict, prev_sam_mask_logits
        if mask_inputs is not None:
            self.mask_calls.append(
                {
                    "frame_idx": frame_idx,
                    "mask_shape": tuple(mask_inputs.shape),
                    "has_points": point_inputs is not None,
                }
            )
        self.calls.append(
            {
                "frame_idx": frame_idx,
                "batch_size": batch_size,
                "is_init_cond_frame": is_init_cond_frame,
                "has_points": point_inputs is not None,
                "reverse": reverse,
                "run_mem_encoder": run_mem_encoder,
                "use_prev_mem_frame": use_prev_mem_frame,
            }
        )
        return self._frame_out(frame_idx, run_mem_encoder, batch_size=batch_size)

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
    ):
        del inference_state, object_score_logits
        self.memory_calls.append(
            {
                "frame_idx": frame_idx,
                "batch_size": batch_size,
                "high_res_shape": tuple(high_res_masks.shape),
                "is_mask_from_pts": is_mask_from_pts,
            }
        )
        side = self.sam_image_embedding_size
        return (
            mx.full((batch_size, self.mem_dim, side, side), 7.0, dtype=mx.float32),
            [mx.full((batch_size, self.mem_dim, side, side), 8.0, dtype=mx.float32)],
        )


class _FakeTrackerBackbone:
    """Backbone stub that emits the tracker/SAM2 feature contract."""

    def __init__(self):
        self.calls = []

    def forward_image(self, image):
        self.calls.append(tuple(image.shape))
        return {
            "sam2_backbone_out": {
                "backbone_fpn": [
                    mx.zeros((1, 256, 8, 8), dtype=mx.float32),
                    mx.zeros((1, 256, 4, 4), dtype=mx.float32),
                    mx.zeros((1, 256, 2, 2), dtype=mx.float32),
                ],
                "vision_pos_enc": [
                    mx.ones((1, 256, 8, 8), dtype=mx.float32),
                    mx.ones((1, 256, 4, 4), dtype=mx.float32),
                    mx.ones((1, 256, 2, 2), dtype=mx.float32),
                ],
            }
        }


def test_init_state_and_object_mapping_contract():
    predictor = _ScriptedPredictor()
    state = predictor.init_state(video_height=6, video_width=10, num_frames=3)

    assert state["device"] == "mlx"
    assert state["storage_device"] == "mlx"
    assert state["video_height"] == 6
    assert state["video_width"] == 10
    assert state["num_frames"] == 3
    assert state["cached_features"] == {}
    assert state["tracking_has_started"] is False

    assert predictor._obj_id_to_idx(state, "cell") == 0
    assert predictor._obj_idx_to_id(state, 0) == "cell"
    assert predictor._get_obj_num(state) == 1
    assert predictor._obj_id_to_idx(state, "second") == 1
    assert predictor._obj_idx_to_id(state, 1) == "second"
    assert predictor._get_obj_num(state) == 2
    assert state["obj_ids"] == ["cell", "second"]


def test_add_new_points_or_box_scales_prompts_and_returns_video_masks():
    predictor = _ScriptedPredictor()
    state = predictor.init_state(video_height=6, video_width=10, num_frames=2)

    frame_idx, obj_ids, low_res_masks, video_res_masks = (
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id="cell",
            points=[[0.25, 0.5]],
            labels=[1],
            box=[0.0, 0.0, 1.0, 1.0],
        )
    )

    assert frame_idx == 0
    assert obj_ids == ["cell"]
    assert low_res_masks is None
    assert tuple(video_res_masks.shape) == (1, 1, 6, 10)
    np.testing.assert_allclose(
        _to_numpy(video_res_masks), np.ones((1, 1, 6, 10)), rtol=0, atol=1e-6
    )

    point_inputs = state["point_inputs_per_obj"][0][0]
    np.testing.assert_allclose(
        _to_numpy(point_inputs["point_coords"]),
        np.array([[[0.0, 0.0], [28.0, 28.0], [7.0, 14.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(point_inputs["point_labels"]),
        np.array([[2, 3, 1]], dtype=np.int32),
    )
    assert predictor.calls == [
        {
            "frame_idx": 0,
            "batch_size": 1,
            "is_init_cond_frame": True,
            "has_points": True,
            "reverse": False,
            "run_mem_encoder": False,
            "use_prev_mem_frame": False,
        }
    ]


def test_add_new_mask_stores_video_mask_and_clears_same_frame_points():
    predictor = _ScriptedPredictor()
    state = predictor.init_state(video_height=6, video_width=10, num_frames=2)
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id="cell",
        points=[[0.25, 0.5]],
        labels=[1],
    )
    mask = np.zeros((6, 10), dtype=np.float32)
    mask[1:4, 2:5] = 1.0

    frame_idx, obj_ids, low_res_masks, video_res_masks = predictor.add_new_mask(
        state,
        frame_idx=np.int64(0),
        obj_id="cell",
        mask=mask,
    )

    assert frame_idx == 0
    assert obj_ids == ["cell"]
    assert low_res_masks is None
    assert tuple(video_res_masks.shape) == (1, 1, 6, 10)
    logits = _to_numpy(video_res_masks)
    assert logits[0, 0, 2, 3] == 1024.0
    assert logits[0, 0, 0, 0] == -1024.0
    assert predictor.mask_calls == [
        {
            "frame_idx": 0,
            "mask_shape": (1, 1, predictor.input_mask_size, predictor.input_mask_size),
            "has_points": False,
        }
    ]
    assert state["point_inputs_per_obj"][0] == {}
    stored_mask = state["mask_inputs_per_obj"][0][0]
    assert isinstance(next(iter(state["mask_inputs_per_obj"][0])), int)
    assert tuple(stored_mask.shape) == (1, 1, 6, 10)
    np.testing.assert_array_equal(_to_numpy(stored_mask)[0, 0], mask.astype(bool))
    assert (
        state["temp_output_dict_per_obj"][0]["cond_frame_outputs"][0]["pred_masks"]
        is None
    )


def test_get_orig_video_res_output_applies_fill_hole_cleanup():
    predictor = Sam3TrackerPredictor(fill_hole_area=1, **PREDICTOR_KWARGS)
    predictor.eval()
    state = predictor.init_state(video_height=4, video_width=4, num_frames=1)
    scores = mx.array(
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
        dtype=mx.float32,
    )

    low_res_masks, video_res_masks = predictor._get_orig_video_res_output(
        state,
        scores,
    )

    assert low_res_masks is scores
    expected = _to_numpy(scores).copy()
    expected[0, 0, 1, 1] = 0.1
    expected[0, 0, 3, 3] = -0.1
    np.testing.assert_allclose(_to_numpy(video_res_masks), expected, rtol=0, atol=1e-6)


def test_tracker_object_wise_non_overlap_and_shrinkage_helpers():
    predictor = Sam3TrackerPredictor(**PREDICTOR_KWARGS)
    predictor.eval()
    pred_masks = mx.array(
        [
            [[[5.0, 4.0], [-2.0, 3.0]]],
            [[[2.0, -1.0], [6.0, 7.0]]],
        ],
        dtype=mx.float32,
    )
    obj_scores = mx.array([[0.2], [0.9]], dtype=mx.float32)

    constrained = predictor._apply_object_wise_non_overlapping_constraints(
        pred_masks,
        obj_scores,
        background_value=-10.0,
    )

    np.testing.assert_allclose(
        _to_numpy(constrained),
        np.array(
            [
                [[[-10.0, 4.0], [-10.0, -10.0]]],
                [[[2.0, -10.0], [6.0, 7.0]]],
            ],
            dtype=np.float32,
        ),
    )

    shrunk = predictor._suppress_shrinked_masks(
        pred_masks,
        constrained,
        shrink_threshold=0.5,
    )

    np.testing.assert_allclose(
        _to_numpy(shrunk),
        np.array(
            [
                [[[-10.0, -10.0], [-10.0, -10.0]]],
                [[[2.0, -1.0], [6.0, 7.0]]],
            ],
            dtype=np.float32,
        ),
    )


def test_preflight_commits_prompt_frame_and_propagation_tracks_next_frame():
    predictor = _ScriptedPredictor()
    state = predictor.init_state(video_height=6, video_width=10, num_frames=3)
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id="cell",
        points=[[0.25, 0.5]],
        labels=[1],
    )

    predictor.propagate_in_video_preflight(state)

    assert state["tracking_has_started"] is True
    assert state["first_ann_frame_idx"] == 0
    assert state["temp_output_dict_per_obj"][0]["cond_frame_outputs"] == {}
    assert set(state["output_dict"]["cond_frame_outputs"]) == {0}
    assert state["consolidated_frame_inds"]["cond_frame_outputs"] == {0}
    assert predictor.memory_calls == [
        {
            "frame_idx": 0,
            "batch_size": 1,
            "high_res_shape": (1, 1, 28, 28),
            "is_mask_from_pts": True,
        }
    ]

    outputs = list(
        predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
            reverse=False,
        )
    )

    assert [frame_idx for frame_idx, *_ in outputs] == [0, 1]
    assert outputs[0][1] == ["cell"]
    assert tuple(outputs[0][2].shape) == (
        1,
        1,
        predictor.low_res_mask_size,
        predictor.low_res_mask_size,
    )
    assert tuple(outputs[0][3].shape) == (1, 1, 6, 10)
    np.testing.assert_allclose(_to_numpy(outputs[1][2]), np.full((1, 1, 8, 8), 2.0))
    np.testing.assert_allclose(
        _to_numpy(outputs[1][4]), np.array([[2.0]], dtype=np.float32)
    )
    assert set(state["output_dict"]["non_cond_frame_outputs"]) == {1}
    assert state["frames_already_tracked"] == {
        0: {"reverse": False},
        1: {"reverse": False},
    }
    assert predictor.calls[-1] == {
        "frame_idx": 1,
        "batch_size": 1,
        "is_init_cond_frame": False,
        "has_points": False,
        "reverse": False,
        "run_mem_encoder": True,
        "use_prev_mem_frame": True,
    }


def test_multi_object_prompt_propagation_and_removal_updates_state():
    predictor = _ScriptedPredictor()
    state = predictor.init_state(video_height=6, video_width=10, num_frames=3)
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id="cell",
        points=[[0.25, 0.5]],
        labels=[1],
    )

    frame_idx, obj_ids, _, video_res_masks = predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id="nucleus",
        points=[[0.75, 0.5]],
        labels=[1],
    )

    assert frame_idx == 0
    assert obj_ids == ["cell", "nucleus"]
    assert tuple(video_res_masks.shape) == (2, 1, 6, 10)
    assert state["obj_id_to_idx"] == {"cell": 0, "nucleus": 1}
    assert sorted(state["temp_output_dict_per_obj"]) == [0, 1]

    predictor.propagate_in_video_preflight(state)

    assert state["tracking_has_started"] is True
    assert tuple(state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].shape) == (
        2,
        1,
        predictor.low_res_mask_size,
        predictor.low_res_mask_size,
    )
    assert predictor.memory_calls[-1] == {
        "frame_idx": 0,
        "batch_size": 2,
        "high_res_shape": (2, 1, 28, 28),
        "is_mask_from_pts": True,
    }
    assert tuple(
        state["output_dict_per_obj"][1]["cond_frame_outputs"][0]["pred_masks"].shape
    ) == (1, 1, predictor.low_res_mask_size, predictor.low_res_mask_size)

    outputs = list(
        predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
        )
    )

    assert [frame_idx for frame_idx, *_ in outputs] == [0, 1]
    assert outputs[1][1] == ["cell", "nucleus"]
    assert tuple(outputs[1][2].shape) == (
        2,
        1,
        predictor.low_res_mask_size,
        predictor.low_res_mask_size,
    )
    assert predictor.calls[-1]["batch_size"] == 2

    remaining_obj_ids, updated_frames = predictor.remove_object(state, "cell")

    assert remaining_obj_ids == ["nucleus"]
    assert len(updated_frames) == 1
    assert updated_frames[0][0] == 0
    assert tuple(updated_frames[0][1].shape) == (1, 1, 6, 10)
    assert state["obj_id_to_idx"] == {"nucleus": 0}
    assert state["obj_idx_to_id"] == {0: "nucleus"}
    assert state["obj_ids"] == ["nucleus"]
    assert sorted(state["point_inputs_per_obj"]) == [0]
    assert tuple(state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].shape) == (
        1,
        1,
        predictor.low_res_mask_size,
        predictor.low_res_mask_size,
    )
    assert tuple(
        state["output_dict"]["non_cond_frame_outputs"][1]["pred_masks"].shape
    ) == (
        1,
        1,
        predictor.low_res_mask_size,
        predictor.low_res_mask_size,
    )


def test_propagate_in_video_filters_requested_obj_ids_in_requested_order():
    predictor = _ScriptedPredictor()
    state = predictor.init_state(video_height=6, video_width=10, num_frames=3)
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id="cell",
        points=[[0.25, 0.5]],
        labels=[1],
    )
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id="nucleus",
        points=[[0.75, 0.5]],
        labels=[1],
    )
    predictor.propagate_in_video_preflight(state)

    side = predictor.low_res_mask_size
    cond_out = state["output_dict"]["cond_frame_outputs"][0]
    cond_out["pred_masks"] = mx.concat(
        [
            mx.full((1, 1, side, side), 10.0, dtype=mx.float32),
            mx.full((1, 1, side, side), 20.0, dtype=mx.float32),
        ],
        axis=0,
    )
    cond_out["object_score_logits"] = mx.array([[0.1], [0.9]], dtype=mx.float32)

    ordered = list(
        predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=0,
            obj_ids=["nucleus", "cell"],
        )
    )

    assert ordered[0][1] == ["nucleus", "cell"]
    np.testing.assert_allclose(
        _to_numpy(ordered[0][2][:, :, 0, 0]),
        np.array([[20.0], [10.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _to_numpy(ordered[0][4]),
        np.array([[0.9], [0.1]], dtype=np.float32),
    )

    subset = list(
        predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
            obj_ids=["nucleus"],
        )
    )

    assert [frame_idx for frame_idx, *_ in subset] == [0, 1]
    assert subset[0][1] == ["nucleus"]
    assert subset[1][1] == ["nucleus"]
    assert tuple(subset[1][2].shape) == (
        1,
        1,
        predictor.low_res_mask_size,
        predictor.low_res_mask_size,
    )
    assert predictor.calls[-1]["batch_size"] == 2
    assert tuple(
        state["output_dict"]["non_cond_frame_outputs"][1]["pred_masks"].shape
    ) == (
        2,
        1,
        predictor.low_res_mask_size,
        predictor.low_res_mask_size,
    )


def test_propagate_in_video_rejects_invalid_requested_obj_ids():
    predictor = _ScriptedPredictor()
    state = predictor.init_state(video_height=6, video_width=10, num_frames=2)
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id="cell",
        points=[[0.25, 0.5]],
        labels=[1],
    )
    predictor.propagate_in_video_preflight(state)

    cases = [
        ([], "at least one"),
        (["cell", "cell"], "duplicate"),
        (["missing"], "Unknown obj_ids"),
    ]
    for obj_ids, message in cases:
        with pytest.raises(ValueError, match=message):
            list(
                predictor.propagate_in_video(
                    state,
                    start_frame_idx=0,
                    max_frame_num_to_track=0,
                    obj_ids=obj_ids,
                )
            )


def test_remove_single_object_clears_tracker_state():
    predictor = _ScriptedPredictor()
    state = predictor.init_state(video_height=6, video_width=10, num_frames=2)
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id="cell",
        points=[[0.25, 0.5]],
        labels=[1],
    )
    predictor.propagate_in_video_preflight(state)
    list(
        predictor.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=1,
        )
    )

    remaining_obj_ids, updated_frames = predictor.remove_object(state, "cell")

    assert remaining_obj_ids == []
    assert updated_frames == []
    assert state["obj_id_to_idx"] == {}
    assert state["obj_idx_to_id"] == {}
    assert state["obj_ids"] == []
    assert state["point_inputs_per_obj"] == {}
    assert state["mask_inputs_per_obj"] == {}
    assert state["output_dict"] == {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }
    assert state["output_dict_per_obj"] == {}
    assert state["temp_output_dict_per_obj"] == {}
    assert state["tracking_has_started"] is False
    assert state["frames_already_tracked"] == {}
    assert state["first_ann_frame_idx"] is None

    assert predictor.remove_object(state, "missing") == ([], [])
    with pytest.raises(RuntimeError, match="Cannot remove object id missing"):
        predictor.remove_object(state, "missing", strict=True)


def test_cached_feature_lookup_prepares_backbone_features_and_rejects_misses():
    predictor = Sam3TrackerPredictor(**PREDICTOR_KWARGS)
    predictor.eval()
    image = mx.zeros((1, 3, 28, 28), dtype=mx.float32)
    backbone_out = {
        "backbone_fpn": [
            mx.zeros((1, 32, 8, 8), dtype=mx.float32),
            mx.zeros((1, 64, 4, 4), dtype=mx.float32),
            mx.zeros((1, 256, 2, 2), dtype=mx.float32),
        ],
        "vision_pos_enc": [
            mx.ones((1, 32, 8, 8), dtype=mx.float32),
            mx.ones((1, 64, 4, 4), dtype=mx.float32),
            mx.ones((1, 256, 2, 2), dtype=mx.float32),
        ],
    }
    state = predictor.init_state(
        video_height=6,
        video_width=10,
        num_frames=1,
        cached_features={0: (image, backbone_out)},
    )

    out_image, prepared, vision_feats, vision_pos, feat_sizes = (
        predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    )

    assert out_image is image
    assert prepared is not backbone_out
    assert feat_sizes == [(8, 8), (4, 4), (2, 2)]
    assert tuple(vision_feats[-1].shape) == (4, 1, 256)
    assert tuple(vision_pos[0].shape) == (64, 1, 32)

    batched_image, batched_prepared, batched_vision_feats, batched_vision_pos, _ = (
        predictor._get_image_feature(state, frame_idx=0, batch_size=2)
    )
    assert tuple(batched_image.shape) == (2, 3, 28, 28)
    assert tuple(batched_prepared["backbone_fpn"][0].shape) == (2, 32, 8, 8)
    assert tuple(batched_vision_feats[-1].shape) == (4, 2, 256)
    assert tuple(batched_vision_pos[0].shape) == (64, 2, 32)
    with pytest.raises(RuntimeError, match="not cached"):
        predictor._get_image_feature(state, frame_idx=99, batch_size=1)


def test_cache_miss_uses_tracker_backbone_and_stores_prepared_features():
    backbone = _FakeTrackerBackbone()
    predictor = Sam3TrackerPredictor(backbone=backbone, **PREDICTOR_KWARGS)
    predictor.eval()
    images = mx.zeros((2, 3, 28, 28), dtype=mx.float32)
    state = predictor.init_state(images=images)

    out_image, prepared, vision_feats, vision_pos, feat_sizes = (
        predictor._get_image_feature(state, frame_idx=1, batch_size=1)
    )

    assert backbone.calls == [(1, 3, 28, 28)]
    assert tuple(out_image.shape) == (1, 3, 28, 28)
    assert state["cached_features"][1][0] is out_image
    assert state["video_height"] == 28
    assert state["video_width"] == 28
    assert state["num_frames"] == 2
    assert tuple(prepared["backbone_fpn"][0].shape) == (1, 32, 8, 8)
    assert tuple(prepared["backbone_fpn"][1].shape) == (1, 64, 4, 4)
    assert feat_sizes == [(8, 8), (4, 4), (2, 2)]
    assert tuple(vision_feats[-1].shape) == (4, 1, 256)
    assert tuple(vision_pos[0].shape) == (64, 1, 256)

    # The second lookup should use the cached feature instead of the backbone.
    predictor._get_image_feature(state, frame_idx=1, batch_size=1)
    assert backbone.calls == [(1, 3, 28, 28)]


def test_init_state_video_path_loads_mlx_frames_for_backbone_cache_misses():
    backbone = _FakeTrackerBackbone()
    predictor = Sam3TrackerPredictor(backbone=backbone, **PREDICTOR_KWARGS)
    predictor.eval()

    state = predictor.init_state(video_path="<load-dummy-video-2>")

    assert state["num_frames"] == 2
    assert state["video_height"] == 480
    assert state["video_width"] == 640
    assert tuple(state["images"].shape) == (2, 3, 28, 28)
    assert state["cached_features"] == {}

    predictor._get_image_feature(state, frame_idx=0, batch_size=1)
    assert backbone.calls == [(1, 3, 28, 28)]
    assert 0 in state["cached_features"]


def test_deferred_predictor_paths_raise_canonical_errors():
    predictor = _ScriptedPredictor()

    with pytest.raises(Sam3MlxUnsupportedError, match="Async"):
        predictor.init_state(
            video_path="<load-dummy-video-1>",
            async_loading_frames=True,
        )
    with pytest.raises(Sam3MlxUnsupportedError, match="only has an effect"):
        Sam3TrackerPredictor(clear_non_cond_mem_for_multi_obj=True)


def test_predictor_loop_parity_fixture_is_current():
    fixture = PORT_TRACKER_FIXTURE_ROOT / "predictor_loop_parity.json"
    data = json.loads(fixture.read_text())

    assert data["atol"] == 2e-3
    assert data["worst_max_abs"] <= data["atol"]
    assert data["results"]["prop[1].video_res_masks"]["max_abs"] == 0.0
    proof_artifact = REPO_ROOT / data["proof_artifact"]
    assert proof_artifact.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
