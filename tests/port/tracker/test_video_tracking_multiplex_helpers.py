from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from sam3_mlx.model.data_misc import NestedTensor
from sam3_mlx.model.multiplex_utils import (
    MultiplexState,
    UnsupportedMultiplexRuntimeError,
)
from sam3_mlx.model.memory import SimpleMaskEncoder
from sam3_mlx.model.video_tracking_multiplex import (
    NO_OBJ_SCORE,
    VideoTrackingMultiplex,
    _append,
    _merge,
    concat_points,
)
from tests._paths import PORT_TRACKER_FIXTURE_ROOT

OFFICIAL_SAM3_VIDEO_TRACKING_MULTIPLEX_COMMIT = (
    "2814fa619404a722d03e9a012e083e4f293a4e53"
)
PACKED_ADD_PARITY_FIXTURE = (
    PORT_TRACKER_FIXTURE_ROOT / "multiplex_packed_add_parity.json"
)


def _to_numpy(value):
    mx.eval(value)
    return np.asarray(value)


def _constant_rows(values, tail_shape):
    return mx.array(
        np.stack(
            [np.full(tail_shape, value, dtype=np.float32) for value in values],
            axis=0,
        )
    )


def _packed_low_res_mask_patterns():
    base = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    return mx.array(np.stack([base, base + 100.0], axis=0), dtype=mx.float32)


class _ReconditionHarness:
    add_new_masks = VideoTrackingMultiplex.add_new_masks
    add_new_masks_to_existing_state = (
        VideoTrackingMultiplex.add_new_masks_to_existing_state
    )
    _current_frame_output = staticmethod(VideoTrackingMultiplex._current_frame_output)
    _prepared_features_from_state = VideoTrackingMultiplex._prepared_features_from_state
    propagate_in_video_preflight = VideoTrackingMultiplex.propagate_in_video_preflight
    recondition_masks_in_existing_state = (
        VideoTrackingMultiplex.recondition_masks_in_existing_state
    )

    def __init__(
        self,
        *,
        use_memory_selection=True,
        use_obj_ptrs_in_encoder=True,
        save_image_features=False,
        low_res_masks=None,
    ):
        self.use_mask_input_as_output_without_sam = True
        self.use_memory_selection = use_memory_selection
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.save_image_features = save_image_features
        self.low_res_masks = low_res_masks
        self.mask_calls = []
        self.memory_calls = []
        self.pix_mem_calls = []

    def _get_interactive_pix_mem(self, vision_feats, feat_sizes):
        self.pix_mem_calls.append(
            {
                "vision_feats": vision_feats,
                "feat_sizes": feat_sizes,
            }
        )
        return mx.array([[1.0]], dtype=mx.float32)

    def _use_mask_as_output(
        self,
        *,
        backbone_features,
        high_res_features,
        mask_inputs,
        multiplex_state,
        objects_in_mask,
    ):
        self.mask_calls.append(
            {
                "backbone_features": backbone_features,
                "high_res_features": high_res_features,
                "mask_inputs": mask_inputs,
                "multiplex_state": multiplex_state,
                "objects_in_mask": objects_in_mask,
            }
        )
        return {
            "low_res_masks": (
                self.low_res_masks
                if self.low_res_masks is not None
                else _constant_rows([7.0, 8.0], (1, 2, 2))
            ),
            "high_res_masks": _constant_rows([70.0, 80.0], (1, 4, 4)),
            "object_score_logits": mx.array([[101.0], [202.0]], dtype=mx.float32),
            "ious": mx.array([[0.91], [0.82]], dtype=mx.float32),
            "obj_ptr": mx.array(
                [[100.0, 101.0], [200.0, 201.0]],
                dtype=mx.float32,
            ),
        }

    def _encode_new_memory(
        self,
        *,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        conditioning_objects,
        is_mask_from_pts,
        multiplex_state,
    ):
        self.memory_calls.append(
            {
                "image": image,
                "current_vision_feats": current_vision_feats,
                "feat_sizes": feat_sizes,
                "pred_masks_high_res": pred_masks_high_res,
                "object_score_logits": object_score_logits,
                "conditioning_objects": conditioning_objects.copy(),
                "is_mask_from_pts": is_mask_from_pts,
                "multiplex_state": multiplex_state,
            }
        )
        return (
            mx.array([[9.0, 9.5]], dtype=mx.float32),
            [mx.array([[8.0, 8.5]], dtype=mx.float32)],
        )


class _RecordingSimpleMaskEncoder(SimpleMaskEncoder):
    def __init__(self):
        self.calls = []

    def __call__(self, pix_feat, masks, skip_mask_sigmoid=False):
        self.calls.append(
            {
                "pix_feat": pix_feat,
                "masks": masks,
                "skip_mask_sigmoid": skip_mask_sigmoid,
            }
        )
        return {
            "vision_features": mx.zeros_like(pix_feat),
            "vision_pos_enc": [mx.ones_like(pix_feat)],
        }


class _MemoryHarness:
    _apply_non_overlapping_constraints = (
        VideoTrackingMultiplex._apply_non_overlapping_constraints
    )
    _apply_object_wise_non_overlapping_constraints = (
        VideoTrackingMultiplex._apply_object_wise_non_overlapping_constraints
    )
    _encode_new_memory = VideoTrackingMultiplex._encode_new_memory
    _maybe_clone = VideoTrackingMultiplex._maybe_clone

    def __init__(self):
        self.hidden_dim = 2
        self.mem_dim = 2
        self.training = False
        self.non_overlap_masks_for_mem_enc = False
        self.apply_sigmoid_to_mask_logits_for_mem_enc = False
        self.binarize_mask_from_pts_for_mem_enc = False
        self.sigmoid_scale_for_mem_enc = 1.0
        self.sigmoid_bias_for_mem_enc = 0.0
        self.add_object_conditional_embeddings = False
        self.add_object_unconditional_embeddings = False
        self.condition_as_mask_input = False
        self.condition_as_mask_input_fg = 1.0
        self.condition_as_mask_input_bg = 0.0
        self.no_obj_embed_spatial = mx.array(
            [[1.0, 10.0], [2.0, 20.0]],
            dtype=mx.float32,
        )
        self.object_score_logit_threshold = 0.0
        self.maskmem_backbone = _RecordingSimpleMaskEncoder()


def test_video_tracking_multiplex_object_wise_non_overlap_handles_bool_masks():
    harness = _MemoryHarness()
    pred_masks = mx.array(
        [
            [[[True, True, False], [False, False, False]]],
            [[[False, True, True], [False, True, False]]],
        ],
        dtype=mx.bool_,
    )
    obj_scores = mx.array([[0.8], [0.2]], dtype=mx.float32)

    constrained = harness._apply_object_wise_non_overlapping_constraints(
        pred_masks,
        obj_scores,
        background_value=0,
    )

    np.testing.assert_array_equal(
        _to_numpy(constrained),
        np.array(
            [
                [[[True, True, False], [False, False, False]]],
                [[[False, False, True], [False, True, False]]],
            ],
            dtype=bool,
        ),
    )


class _MaskOutputHarness:
    _use_mask_as_output = VideoTrackingMultiplex._use_mask_as_output

    def __init__(self, *, use_obj_ptrs_in_encoder=False):
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.pred_obj_scores = False
        self.use_no_obj_ptr = False


class _IdentityProjection:
    def __call__(self, value):
        return value


class _PromptEncoderStub:
    mask_input_size = (2, 2)

    def __init__(self):
        self.calls = []

    def __call__(self, *, points, boxes, masks):
        del boxes
        point_coords, point_labels = points
        self.calls.append(
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
                "masks": masks,
            }
        )
        batch = point_coords.shape[0]
        return (
            mx.zeros((batch, 1, 2), dtype=mx.float32),
            mx.zeros((batch, 2, 2, 2), dtype=mx.float32),
        )

    def get_dense_pe(self):
        return mx.zeros((1, 2, 2, 2), dtype=mx.float32)


class _InteractiveMaskDecoderStub:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        repeat_image,
        high_res_features,
    ):
        self.calls.append(
            {
                "image_embeddings": image_embeddings,
                "image_pe": image_pe,
                "sparse_prompt_embeddings": sparse_prompt_embeddings,
                "dense_prompt_embeddings": dense_prompt_embeddings,
                "multimask_output": multimask_output,
                "repeat_image": repeat_image,
                "high_res_features": high_res_features,
            }
        )
        return (
            _constant_rows([5.0, 6.0], (1, 2, 2)),
            mx.ones((2, 1), dtype=mx.float32),
            mx.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=mx.float32),
            mx.array([[1.0], [-1.0]], dtype=mx.float32),
        )


class _PropagationMaskDecoderStub:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        image_embeddings,
        image_pe,
        high_res_features,
        multimask_output,
        extra_per_object_embeddings,
    ):
        self.calls.append(
            {
                "image_embeddings": image_embeddings,
                "image_pe": image_pe,
                "high_res_features": high_res_features,
                "multimask_output": multimask_output,
                "extra_per_object_embeddings": extra_per_object_embeddings,
            }
        )
        return {
            "masks": mx.array(
                np.array(
                    [
                        [
                            [[[5.0, 5.0], [5.0, 5.0]]],
                            [[[6.0, 6.0], [6.0, 6.0]]],
                        ],
                        [
                            [[[7.0, 7.0], [7.0, 7.0]]],
                            [[[8.0, 8.0], [8.0, 8.0]]],
                        ],
                    ],
                    dtype=np.float32,
                )
            ),
            "iou_pred": mx.array([[[0.9], [0.8]], [[0.7], [0.6]]], dtype=mx.float32),
            "sam_tokens_out": mx.array(
                [
                    [[[1.0, 2.0]], [[3.0, 4.0]]],
                    [[[5.0, 6.0]], [[7.0, 8.0]]],
                ],
                dtype=mx.float32,
            ),
            "object_score_logits": mx.array(
                [[1.0, -1.0], [-1.0, 1.0]], dtype=mx.float32
            ),
        }


class _SamHeadsHarness:
    _forward_sam_heads = VideoTrackingMultiplex._forward_sam_heads
    _maybe_clone = VideoTrackingMultiplex._maybe_clone

    def __init__(self):
        self.sam_prompt_embed_dim = 2
        self.sam_image_embedding_size = 2
        self.image_size = 8
        self.pred_obj_scores = True
        self.object_score_logit_threshold = 0.0
        self.stability_score_attentuation = False
        self.decode_mask_with_shared_tokens = False
        self.add_output_suppression_embeddings = True
        self.output_valid_embed = mx.array([[1.0, 10.0], [2.0, 20.0]], dtype=mx.float32)
        self.output_invalid_embed = mx.array(
            [[-1.0, -10.0], [-2.0, -20.0]],
            dtype=mx.float32,
        )
        self.use_obj_ptrs_in_encoder = True
        self.use_no_obj_ptr = True
        self.use_linear_no_obj_ptr = False
        self.fixed_no_obj_ptr = False
        self.no_obj_ptr = mx.array(
            [[100.0, 100.0], [200.0, 200.0]],
            dtype=mx.float32,
        )
        self.interactive_obj_ptr_proj = _IdentityProjection()
        self.obj_ptr_proj = _IdentityProjection()
        self.interactive_sam_prompt_encoder = _PromptEncoderStub()
        self.interactive_sam_mask_decoder = _InteractiveMaskDecoderStub()
        self.sam_mask_decoder = _PropagationMaskDecoderStub()

    def get_propagation_dense_pe(self):
        return mx.zeros((1, 2, 2, 2), dtype=mx.float32)


class _EncoderRecorder:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        src,
        src_key_padding_mask,
        src_pos,
        prompt,
        prompt_pos,
        prompt_key_padding_mask,
        feat_sizes,
        num_obj_ptr_tokens,
    ):
        self.calls.append(
            {
                "src": src,
                "src_key_padding_mask": src_key_padding_mask,
                "src_pos": src_pos,
                "prompt": prompt,
                "prompt_pos": prompt_pos,
                "prompt_key_padding_mask": prompt_key_padding_mask,
                "feat_sizes": feat_sizes,
                "num_obj_ptr_tokens": num_obj_ptr_tokens,
            }
        )
        return {"memory": src + 10.0}


class _TransformerRecorder:
    def __init__(self):
        self.encoder = _EncoderRecorder()


class _DecoupledEncoderRecorder:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        image,
        src,
        memory_image,
        memory,
        image_pos,
        src_pos,
        memory_image_pos,
        memory_pos,
        num_obj_ptr_tokens,
    ):
        self.calls.append(
            {
                "image": image,
                "src": src,
                "memory_image": memory_image,
                "memory": memory,
                "image_pos": image_pos,
                "src_pos": src_pos,
                "memory_image_pos": memory_image_pos,
                "memory_pos": memory_pos,
                "num_obj_ptr_tokens": num_obj_ptr_tokens,
            }
        )
        return {"memory": src + 20.0}


class _DecoupledTransformerRecorder:
    def __init__(self):
        self.encoder = _DecoupledEncoderRecorder()


class _MemoryConditioningHarness:
    _prepare_memory_conditioned_features = (
        VideoTrackingMultiplex._prepare_memory_conditioned_features
    )
    _broadcast_to_buckets = VideoTrackingMultiplex._broadcast_to_buckets
    _get_tpos_enc = VideoTrackingMultiplex._get_tpos_enc
    frame_filter = VideoTrackingMultiplex.frame_filter

    def __init__(self):
        self.hidden_dim = 2
        self.mem_dim = 2
        self.num_maskmem = 2
        self.save_image_features = False
        self.training = False
        self.memory_temporal_stride_for_eval = 1
        self.use_memory_selection = False
        self.max_cond_frames_in_attn = -1
        self.keep_first_cond_frame = False
        self.use_obj_ptrs_in_encoder = True
        self.max_obj_ptrs_in_encoder = 4
        self.only_obj_ptrs_in_the_past_for_eval = False
        self.use_signed_tpos_enc_to_obj_ptrs = False
        self.add_tpos_enc_to_obj_ptrs = False
        self.sincos_tpos_enc = True
        self.proj_tpos_enc_in_obj_ptrs = False
        self.use_maskmem_tpos_v2 = False
        self.maskmem_tpos_enc = mx.zeros((2, 1, 1, 2), dtype=mx.float32)
        self.obj_ptr_tpos_proj = _IdentityProjection()
        self.transformer = _TransformerRecorder()


class _TrackStepHarness:
    _track_step_aux = VideoTrackingMultiplex._track_step_aux
    track_step = VideoTrackingMultiplex.track_step
    _trim_output_and_memory = VideoTrackingMultiplex._trim_output_and_memory
    _get_interactive_pix_mem = VideoTrackingMultiplex._get_interactive_pix_mem
    _use_mask_as_output = VideoTrackingMultiplex._use_mask_as_output
    _use_multimask = VideoTrackingMultiplex._use_multimask
    cal_mem_score = VideoTrackingMultiplex.cal_mem_score

    def __init__(self):
        self.hidden_dim = 2
        self.directly_add_no_mem_embed = True
        self.interactivity_no_mem_embed = mx.zeros((1, 1, 2), dtype=mx.float32)
        self.use_mask_input_as_output_without_sam = True
        self.use_obj_ptrs_in_encoder = False
        self.use_memory_selection = True
        self.save_image_features = False
        self.num_maskmem = 2
        self.num_correction_pt_per_frame = 0
        self.offload_output_to_cpu_for_eval = False
        self.trim_past_non_cond_mem_for_eval = False
        self.memory_temporal_stride_for_eval = 1
        self.max_obj_ptrs_in_encoder = 4
        self.mf_threshold = 0.5
        self.training = False
        self.multimask_output_in_sam = False
        self.multimask_output_for_tracking = False
        self.multimask_min_pt_num = 1
        self.multimask_max_pt_num = 1
        self.num_multimask_outputs = 1
        self.iter_use_prev_mask_pred = False
        self.pt_sampling_for_eval = "uniform"
        self.add_all_frames_to_correct_as_cond = False
        self.memory_calls = []
        self.prepare_calls = []
        self.forward_calls = []
        self.add_mask_calls = []
        self.recondition_calls = []

    def add_new_masks_to_existing_state(self, **kwargs):
        self.add_mask_calls.append(kwargs)

    def recondition_masks_in_existing_state(self, **kwargs):
        self.recondition_calls.append(kwargs)

    def _encode_new_memory(
        self,
        *,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        conditioning_objects,
        is_mask_from_pts,
        multiplex_state,
    ):
        self.memory_calls.append(
            {
                "image": image,
                "current_vision_feats": current_vision_feats,
                "feat_sizes": feat_sizes,
                "pred_masks_high_res": pred_masks_high_res,
                "object_score_logits": object_score_logits,
                "conditioning_objects": conditioning_objects.copy(),
                "is_mask_from_pts": is_mask_from_pts,
                "multiplex_state": multiplex_state,
            }
        )
        return (
            mx.ones((multiplex_state.num_buckets, 2, 1, 1), dtype=mx.float32),
            [mx.zeros((multiplex_state.num_buckets, 2, 1, 1), dtype=mx.float32)],
        )

    def _prepare_memory_conditioned_features(self, **kwargs):
        self.prepare_calls.append(kwargs)
        state = kwargs["multiplex_state"]
        return mx.full((state.num_buckets, 2, 2, 2), 4.0, dtype=mx.float32)

    def _forward_sam_heads(
        self,
        backbone_features,
        *,
        point_inputs=None,
        mask_inputs=None,
        interactive_high_res_features=None,
        propagation_high_res_features=None,
        multimask_output=False,
        multiplex_state,
        objects_to_interact=None,
        **kwargs,
    ):
        del kwargs
        self.forward_calls.append(
            {
                "backbone_features": backbone_features,
                "point_inputs": point_inputs,
                "mask_inputs": mask_inputs,
                "interactive_high_res_features": interactive_high_res_features,
                "propagation_high_res_features": propagation_high_res_features,
                "multimask_output": multimask_output,
                "objects_to_interact": objects_to_interact,
            }
        )
        if point_inputs is None and mask_inputs is None:
            values = [1.0, 2.0, 3.0][: multiplex_state.total_valid_entries]
        else:
            values = [90.0] * len(objects_to_interact)
        rows = len(values)
        low_tail = getattr(self, "sam_output_low_tail", (1, 2, 2))
        high_tail = getattr(self, "sam_output_high_tail", (1, 4, 4))
        return {
            "low_res_multimasks": _constant_rows(values, low_tail),
            "high_res_multimasks": _constant_rows(values, high_tail),
            "low_res_masks": _constant_rows(values, low_tail),
            "high_res_masks": _constant_rows(values, high_tail),
            "ious": mx.ones((rows, 1), dtype=mx.float32),
            "object_score_logits": mx.ones((rows, 1), dtype=mx.float32),
        }


class _ForwardTrackingHarness:
    forward = VideoTrackingMultiplex.forward
    forward_image = VideoTrackingMultiplex.forward_image
    _target_segments_as_masks = VideoTrackingMultiplex._target_segments_as_masks
    _prepare_prompt_inputs_meta = VideoTrackingMultiplex._prepare_prompt_inputs_meta
    _prepare_dynamic_vos_eval_prompt_inputs = (
        VideoTrackingMultiplex._prepare_dynamic_vos_eval_prompt_inputs
    )
    _prepare_conditional_frames = VideoTrackingMultiplex._prepare_conditional_frames
    prepare_prompt_inputs = VideoTrackingMultiplex.prepare_prompt_inputs
    _prepare_backbone_features = VideoTrackingMultiplex._prepare_backbone_features
    _prepare_backbone_features_per_frame = (
        VideoTrackingMultiplex._prepare_backbone_features_per_frame
    )
    forward_tracking = VideoTrackingMultiplex.forward_tracking

    def __init__(self):
        self.num_feature_levels = 1
        self.training = False
        self.share_necks = False
        self.use_high_res_features_in_sam = False
        self.forward_backbone_per_frame_for_eval = False
        self.prob_to_use_pt_input_for_train = 0.0
        self.prob_to_use_box_input_for_train = 0.0
        self.prob_to_use_pt_input_for_eval = 0.0
        self.prob_to_use_box_input_for_eval = 0.0
        self.num_frames_to_correct_for_train = 1
        self.num_frames_to_correct_for_eval = 1
        self.rand_frames_to_correct_for_train = False
        self.rand_frames_to_correct_for_eval = False
        self.num_init_cond_frames_for_train = 1
        self.num_init_cond_frames_for_eval = 1
        self.rand_init_cond_frames_for_train = True
        self.rand_init_cond_frames_for_eval = False
        self.pt_sampling_for_eval = "uniform"
        self.rng = np.random.default_rng(seed=42)
        self.add_all_frames_to_correct_as_cond = False
        self.add_all_transition_frames_as_cond = False
        self.backbone = SimpleNamespace(forward_image=self._forward_image)
        self.multiplex_controller = SimpleNamespace(
            get_state=self._get_state,
        )
        self.backbone_calls = []
        self.track_calls = []

    def _forward_image(self, image, **kwargs):
        self.backbone_calls.append({"image": image, **kwargs})
        image_tensor = image.tensors if isinstance(image, NestedTensor) else image
        batch = image_tensor.shape[0]
        call_value = float(len(self.backbone_calls))

        def _neck(value):
            features = mx.full((batch, 2, 2, 2), value, dtype=mx.float32)
            pos = mx.zeros_like(features)
            return {"backbone_fpn": [features], "vision_pos_enc": [pos]}

        output = {}
        if kwargs.get("need_interactive_out"):
            output["interactive"] = _neck(10.0 + call_value)
        if kwargs.get("need_propagation_out"):
            output["sam2_backbone_out"] = _neck(20.0 + call_value)
        return output

    def _get_state(self, num_valid_entries, *, dtype, random):
        self.state_call = {
            "num_valid_entries": num_valid_entries,
            "dtype": dtype,
            "random": random,
        }
        return MultiplexState(
            [[0, 1]],
            dtype=dtype,
            allowed_bucket_capacity=2,
        )

    def track_step(self, **kwargs):
        self.track_calls.append(kwargs)
        frame_idx = kwargs["frame_idx"]
        return {
            "conditioning_objects": {frame_idx},
            "pred_masks": _constant_rows([frame_idx + 1.0], (1, 2, 2)),
            "object_score_logits": mx.array([[frame_idx + 0.5]], dtype=mx.float32),
            "multistep_point_inputs": [kwargs["point_inputs"]],
            "obj_ptr": mx.array([[frame_idx, frame_idx + 10.0]], dtype=mx.float32),
        }


def _interactive_features():
    return {
        "vision_feats": [mx.ones((4, 1, 2), dtype=mx.float32)],
        "feat_sizes": [(2, 2)],
    }


def _propagation_features():
    return {
        "vision_feats": [mx.ones((4, 1, 2), dtype=mx.float32)],
        "vision_masks": [None],
        "vision_pos_embeds": [mx.zeros((4, 1, 2), dtype=mx.float32)],
        "feat_sizes": [(2, 2)],
    }


def _assert_prepared_feature_value(features, value):
    assert set(features) == {
        "feat_sizes",
        "vision_feats",
        "vision_masks",
        "vision_pos_embeds",
    }
    assert features["feat_sizes"] == [(2, 2)]
    np.testing.assert_array_equal(
        _to_numpy(features["vision_feats"][0]),
        np.full((4, 1, 2), value, dtype=np.float32),
    )


def _forward_backbone_out():
    interactive_feature = mx.array(
        np.arange(2 * 2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2, 2)
    )
    propagation_feature = interactive_feature + 100.0
    interactive_pos = mx.zeros_like(interactive_feature)
    propagation_pos = mx.zeros_like(propagation_feature)
    feature_mask = mx.array(
        np.array(
            [
                [[False, True], [False, False]],
                [[True, False], [False, True]],
            ],
            dtype=bool,
        )
    )
    return {
        "interactive": {
            "backbone_fpn": [{"tensors": interactive_feature, "mask": feature_mask}],
            "vision_pos_enc": [interactive_pos],
        },
        "sam2_backbone_out": {
            "backbone_fpn": [propagation_feature],
            "vision_pos_enc": [propagation_pos],
        },
        "num_frames": 2,
        "init_cond_frames": [0],
        "frames_not_in_init_cond": [1],
        "frames_to_add_correction_pt": [],
        "gt_masks_per_frame": {
            0: mx.ones((2, 1, 4, 4), dtype=mx.float32),
            1: mx.ones((2, 1, 4, 4), dtype=mx.float32) * 2.0,
        },
        "mask_inputs_per_frame": {
            0: mx.ones((2, 1, 4, 4), dtype=mx.float32),
        },
        "point_inputs_per_frame": {},
    }


def _forward_input():
    return SimpleNamespace(
        img_batch=mx.array(
            np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
        ),
        find_inputs=[
            SimpleNamespace(img_ids=mx.array([0, 0], dtype=mx.int64)),
            SimpleNamespace(img_ids=mx.array([1, 1], dtype=mx.int64)),
        ],
    )


def _forward_input_with_targets():
    input_data = _forward_input()
    input_data.find_targets = [
        SimpleNamespace(
            segments=mx.array(
                np.array(
                    [
                        [[1.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [0.0, 1.0]],
                    ],
                    dtype=np.float32,
                )
            )
        ),
        SimpleNamespace(
            segments=mx.array(
                np.array(
                    [
                        [[0.0, 1.0], [0.0, 0.0]],
                        [[0.0, 0.0], [1.0, 0.0]],
                    ],
                    dtype=np.float32,
                )
            )
        ),
    ]
    return input_data


def _dynamic_eval_forward_input():
    input_data = _forward_input()
    input_data.img_batch = mx.array(
        np.arange(3 * 3 * 2 * 2, dtype=np.float32).reshape(3, 3, 2, 2)
    )
    input_data.find_inputs = [
        SimpleNamespace(img_ids=mx.array([0, 0], dtype=mx.int64)),
        SimpleNamespace(img_ids=mx.array([1, 1], dtype=mx.int64)),
        SimpleNamespace(img_ids=mx.array([2, 2], dtype=mx.int64)),
    ]
    input_data.find_targets = [
        SimpleNamespace(
            segments=_constant_rows([100.0, 101.0], (2, 2)),
            num_boxes=mx.array([1, 1], dtype=mx.int64),
        ),
        SimpleNamespace(
            segments=_constant_rows([200.0, 201.0], (2, 2)),
            num_boxes=mx.array([1, 1], dtype=mx.int64),
        ),
        SimpleNamespace(
            segments=_constant_rows([300.0, 301.0], (2, 2)),
            num_boxes=mx.array([1, 1], dtype=mx.int64),
        ),
    ]
    input_data.find_metadatas = [
        SimpleNamespace(coco_image_id=[100, 101, 102]),
    ]
    input_data.visible_objects_per_frame = {
        0: {1},
        1: {1},
        2: {0, 1},
    }
    return input_data


def test_concat_points_appends_mlx_points_and_labels_along_prompt_axis():
    old_inputs = {
        "point_coords": mx.array([[[1.0, 2.0]]], dtype=mx.float32),
        "point_labels": mx.array([[1]], dtype=mx.int32),
    }
    new_points = mx.array([[[3.0, 4.0], [5.0, 6.0]]], dtype=mx.float32)
    new_labels = mx.array([[0, 1]], dtype=mx.int32)

    merged = concat_points(old_inputs, new_points, new_labels)

    np.testing.assert_array_equal(
        _to_numpy(merged["point_coords"]),
        np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(merged["point_labels"]),
        np.array([[1, 0, 1]], dtype=np.int32),
    )


def test_concat_points_without_old_inputs_returns_new_inputs_unchanged():
    new_points = np.array([[[3.0, 4.0]]], dtype=np.float32)
    new_labels = np.array([[1]], dtype=np.int32)

    merged = concat_points(None, new_points, new_labels)

    assert merged == {"point_coords": new_points, "point_labels": new_labels}


def test_append_extends_existing_stage_output_and_honors_non_strict_missing_key():
    stage = {"pred_masks": mx.array([[1.0], [2.0]], dtype=mx.float32)}
    sam_output = {"low_res_masks": mx.array([[3.0], [4.0]], dtype=mx.float32)}

    _append(stage, sam_output, "pred_masks", "low_res_masks", dim=0)

    np.testing.assert_array_equal(
        _to_numpy(stage["pred_masks"]),
        np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
    )

    _append(stage, sam_output, "missing", "low_res_masks", strict=False)
    assert "missing" not in stage

    with pytest.raises(AssertionError, match="missing not found"):
        _append(stage, sam_output, "missing", "low_res_masks", strict=True)


def test_merge_replaces_mlx_rows_by_index_and_preserves_destination_dtype():
    stage = {
        "obj_ptr": mx.array(
            [[10.0, 10.5], [20.0, 20.5], [30.0, 30.5]],
            dtype=mx.float32,
        )
    }
    sam_output = {
        "obj_ptr": mx.array(
            [[1.0, 1.5], [2.0, 2.5]],
            dtype=mx.float16,
        )
    }

    _merge(stage, sam_output, "obj_ptr", "obj_ptr", [2, 0])

    assert stage["obj_ptr"].dtype == mx.float32
    np.testing.assert_array_equal(
        _to_numpy(stage["obj_ptr"]),
        np.array([[2.0, 2.5], [20.0, 20.5], [1.0, 1.5]], dtype=np.float32),
    )


def test_merge_replaces_numpy_rows_by_index_and_honors_non_strict_missing_key():
    stage = {
        "object_score_logits": np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    }
    sam_output = {"object_score_logits": np.array([[1.25], [2.75]], dtype=np.float64)}

    _merge(
        stage,
        sam_output,
        "object_score_logits",
        "object_score_logits",
        [1, 2],
    )

    assert stage["object_score_logits"].dtype == np.float32
    np.testing.assert_array_equal(
        stage["object_score_logits"],
        np.array([[10.0], [1.25], [2.75]], dtype=np.float32),
    )

    _merge(stage, sam_output, "missing", "object_score_logits", [0], strict=False)
    assert "missing" not in stage

    with pytest.raises(AssertionError, match="missing not found"):
        _merge(stage, sam_output, "missing", "object_score_logits", [0], strict=True)


def test_encode_new_memory_muxes_masks_and_adds_no_object_spatial_embedding():
    model = _MemoryHarness()
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    current_vision_feat = mx.arange(4 * 2 * 2).reshape(4, 2, 2).astype(mx.float32)
    pred_masks_high_res = _constant_rows([1.0, 2.0, 3.0], (1, 4, 4))
    object_score_logits = mx.array([[1.0], [-1.0], [-1.0]], dtype=mx.float32)

    maskmem_features, maskmem_pos_enc = model._encode_new_memory(
        image=None,
        current_vision_feats=[current_vision_feat],
        feat_sizes=[(2, 2)],
        pred_masks_high_res=pred_masks_high_res,
        object_score_logits=object_score_logits,
        is_mask_from_pts=False,
        conditioning_objects={0},
        multiplex_state=multiplex_state,
    )

    call = model.maskmem_backbone.calls[0]
    assert call["skip_mask_sigmoid"] is True
    np.testing.assert_array_equal(
        _to_numpy(call["pix_feat"]),
        _to_numpy(current_vision_feat.transpose(1, 2, 0).reshape(2, 2, 2, 2)),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["masks"]),
        _to_numpy(multiplex_state.mux(pred_masks_high_res).reshape(2, 2, 4, 4)),
    )

    expected_features = np.zeros((2, 2, 2, 2), dtype=np.float32)
    expected_features[0, 0] = 2.0
    expected_features[0, 1] = 20.0
    expected_features[1, 0] = 3.0
    expected_features[1, 1] = 30.0
    np.testing.assert_array_equal(_to_numpy(maskmem_features), expected_features)
    np.testing.assert_array_equal(
        _to_numpy(maskmem_pos_enc[0]),
        np.ones((2, 2, 2, 2), dtype=np.float32),
    )


def test_encode_new_memory_appends_condition_channels_for_mask_input_mode():
    model = _MemoryHarness()
    model.no_obj_embed_spatial = None
    model.condition_as_mask_input = True
    model.condition_as_mask_input_fg = 5.0
    model.condition_as_mask_input_bg = -5.0
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    current_vision_feat = mx.zeros((4, 2, 2), dtype=mx.float32)
    pred_masks_high_res = _constant_rows([1.0, 2.0, 3.0], (1, 4, 4))

    model._encode_new_memory(
        image=None,
        current_vision_feats=[current_vision_feat],
        feat_sizes=[(2, 2)],
        pred_masks_high_res=pred_masks_high_res,
        object_score_logits=None,
        is_mask_from_pts=False,
        conditioning_objects={1},
        multiplex_state=multiplex_state,
    )

    mask_channels = _to_numpy(model.maskmem_backbone.calls[0]["masks"])
    assert mask_channels.shape == (2, 4, 4, 4)
    np.testing.assert_array_equal(
        mask_channels[:, :2],
        _to_numpy(multiplex_state.mux(pred_masks_high_res).reshape(2, 2, 4, 4)),
    )
    expected_conditions = np.zeros((2, 2, 4, 4), dtype=np.float32)
    expected_conditions[0, 0] = -5.0
    expected_conditions[0, 1] = 5.0
    expected_conditions[1, 0] = -5.0
    np.testing.assert_array_equal(mask_channels[:, 2:], expected_conditions)


def test_use_mask_as_output_converts_binary_masks_without_object_pointers():
    model = _MaskOutputHarness()
    multiplex_state = MultiplexState(
        [[0, 1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    mask_inputs = mx.array(
        np.stack(
            [
                np.ones((1, 8, 8), dtype=np.float32),
                np.zeros((1, 8, 8), dtype=np.float32),
            ],
            axis=0,
        )
    )

    output = model._use_mask_as_output(
        backbone_features=mx.zeros((1, 2, 2, 2), dtype=mx.float32),
        high_res_features=[],
        mask_inputs=mask_inputs,
        multiplex_state=multiplex_state,
        objects_in_mask=[0, 1],
    )

    assert "obj_ptr" not in output
    np.testing.assert_array_equal(
        _to_numpy(output["high_res_masks"]),
        np.stack(
            [
                np.full((1, 8, 8), 10.0, dtype=np.float32),
                np.full((1, 8, 8), -10.0, dtype=np.float32),
            ],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        _to_numpy(output["low_res_masks"]),
        np.stack(
            [
                np.full((1, 2, 2), 10.0, dtype=np.float32),
                np.full((1, 2, 2), -10.0, dtype=np.float32),
            ],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        _to_numpy(output["ious"]),
        np.ones((2, 1), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(output["object_score_logits"]),
        np.array([[10.0], [-10.0]], dtype=np.float32),
    )


def test_use_mask_as_output_requires_sam_heads_for_object_pointers():
    model = _MaskOutputHarness(use_obj_ptrs_in_encoder=True)
    multiplex_state = MultiplexState([[0]], dtype=mx.float32)

    with pytest.raises(
        UnsupportedMultiplexRuntimeError,
        match="VideoTrackingMultiplex._use_mask_as_output\\(obj_ptrs\\)",
    ):
        model._use_mask_as_output(
            backbone_features=mx.zeros((1, 2, 2, 2), dtype=mx.float32),
            high_res_features=[],
            mask_inputs=mx.zeros((1, 1, 4, 4), dtype=mx.float32),
            multiplex_state=multiplex_state,
            objects_in_mask=[0],
        )


def test_use_mask_as_output_reuses_singleton_features_for_object_pointers():
    model = _SamHeadsHarness()
    model.interactive_mask_downsample = lambda value: value
    multiplex_state = MultiplexState(
        [[0, 1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    mask_inputs = mx.array(
        np.stack(
            [
                np.ones((1, 4, 4), dtype=np.float32),
                np.zeros((1, 4, 4), dtype=np.float32),
            ],
            axis=0,
        )
    )

    output = VideoTrackingMultiplex._use_mask_as_output(
        model,
        backbone_features=mx.zeros((1, 2, 2, 2), dtype=mx.float32),
        high_res_features=[mx.zeros((1, 2, 4, 4), dtype=mx.float32)],
        mask_inputs=mask_inputs,
        multiplex_state=multiplex_state,
        objects_in_mask=[0, 1],
    )

    prompt_call = model.interactive_sam_prompt_encoder.calls[0]
    assert prompt_call["masks"].shape == (2, 1, 2, 2)
    decoder_call = model.interactive_sam_mask_decoder.calls[0]
    assert decoder_call["repeat_image"] is True
    assert decoder_call["image_embeddings"].shape == (1, 2, 2, 2)
    assert decoder_call["dense_prompt_embeddings"].shape == (2, 2, 2, 2)
    assert decoder_call["high_res_features"][0].shape == (1, 2, 4, 4)
    assert output["high_res_masks"].shape == (2, 1, 4, 4)
    assert output["obj_ptr"].shape == (2, 2)


def test_forward_sam_heads_interactive_mask_path_returns_object_pointers():
    model = _SamHeadsHarness()
    multiplex_state = MultiplexState(
        [[0, 1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )

    output = model._forward_sam_heads(
        backbone_features=mx.zeros((2, 2, 2, 2), dtype=mx.float32),
        mask_inputs=mx.ones((2, 1, 4, 4), dtype=mx.float32),
        interactive_high_res_features=[mx.zeros((2, 2, 4, 4), dtype=mx.float32)],
        multiplex_state=multiplex_state,
        objects_to_interact=[0, 1],
    )

    prompt_call = model.interactive_sam_prompt_encoder.calls[0]
    np.testing.assert_array_equal(
        _to_numpy(prompt_call["point_labels"]),
        -np.ones((2, 1), dtype=np.int32),
    )
    assert prompt_call["masks"].shape == (2, 1, 2, 2)
    decoder_call = model.interactive_sam_mask_decoder.calls[0]
    assert decoder_call["repeat_image"] is False
    assert decoder_call["high_res_features"][0].shape == (2, 2, 4, 4)

    expected_low_res = _to_numpy(_constant_rows([5.0, NO_OBJ_SCORE], (1, 2, 2)))
    np.testing.assert_array_equal(
        _to_numpy(output["low_res_multimasks"]), expected_low_res
    )
    np.testing.assert_array_equal(
        _to_numpy(output["object_score_logits"]),
        np.array([[1.0], [-1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(output["obj_ptr"]),
        np.array([[1.0, 2.0], [203.0, 204.0]], dtype=np.float32),
    )


def test_forward_sam_heads_propagation_path_demuxes_masks_and_object_pointers():
    model = _SamHeadsHarness()
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )

    output = model._forward_sam_heads(
        backbone_features=mx.zeros((2, 2, 2, 2), dtype=mx.float32),
        propagation_high_res_features=[mx.zeros((2, 2, 4, 4), dtype=mx.float32)],
        multiplex_state=multiplex_state,
    )

    decoder_call = model.sam_mask_decoder.calls[0]
    assert decoder_call["multimask_output"] is False
    np.testing.assert_array_equal(
        _to_numpy(decoder_call["extra_per_object_embeddings"]),
        np.array(
            [[[1.0, 10.0], [2.0, 20.0]], [[1.0, 10.0], [-2.0, -20.0]]],
            dtype=np.float32,
        ),
    )

    np.testing.assert_array_equal(
        _to_numpy(output["low_res_multimasks"]),
        np.array(
            [
                [[[5.0, 5.0], [5.0, 5.0]]],
                [[[NO_OBJ_SCORE, NO_OBJ_SCORE], [NO_OBJ_SCORE, NO_OBJ_SCORE]]],
                [[[NO_OBJ_SCORE, NO_OBJ_SCORE], [NO_OBJ_SCORE, NO_OBJ_SCORE]]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        _to_numpy(output["ious"]),
        np.array([[0.9], [0.8], [0.7]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(output["object_score_logits"]),
        np.array([1.0, -1.0, -1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(output["obj_ptr"]),
        np.array([[1.0, 2.0], [203.0, 204.0], [105.0, 106.0]], dtype=np.float32),
    )


def test_forward_sam_heads_propagation_requires_high_res_features():
    model = _SamHeadsHarness()

    with pytest.raises(ValueError, match="propagation_high_res_features"):
        model._forward_sam_heads(
            backbone_features=mx.zeros((1, 2, 2, 2), dtype=mx.float32),
            multiplex_state=MultiplexState([[0]], dtype=mx.float32),
        )


def test_prepare_memory_conditioned_features_broadcasts_frame_and_adds_muxed_prompts():
    model = _MemoryConditioningHarness()
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    current_feat = mx.arange(8).reshape(4, 1, 2).astype(mx.float32)
    maskmem_features = mx.array(
        np.array(
            [
                [[[1.0]], [[10.0]]],
                [[[2.0]], [[20.0]]],
            ],
            dtype=np.float32,
        )
    )
    maskmem_pos = mx.zeros_like(maskmem_features)
    obj_ptr = mx.array(
        np.array(
            [
                [[100.0, 101.0], [200.0, 201.0]],
                [[300.0, 301.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        )
    )
    output_dict = {
        "cond_frame_outputs": {
            0: {
                "conditioning_objects": {0, 1, 2},
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": [maskmem_pos],
                "obj_ptr": obj_ptr,
            }
        },
        "non_cond_frame_outputs": {},
    }

    out = model._prepare_memory_conditioned_features(
        frame_idx=1,
        is_init_cond_frame=False,
        current_vision_feats=[current_feat],
        current_vision_masks=[None],
        current_vision_pos_embeds=[mx.zeros_like(current_feat)],
        feat_sizes=[(2, 2)],
        output_dict=output_dict,
        num_frames=3,
        multiplex_state=multiplex_state,
    )

    call = model.transformer.encoder.calls[0]
    assert call["src"].shape == (4, 2, 2)
    assert call["prompt"].shape == (3, 2, 2)
    assert call["num_obj_ptr_tokens"] == 2
    np.testing.assert_array_equal(
        _to_numpy(call["src"][:, 0]),
        _to_numpy(current_feat[:, 0]),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["src"][:, 1]),
        _to_numpy(current_feat[:, 0]),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["prompt"][0]),
        np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["prompt"][1:]),
        np.array(
            [
                [[100.0, 101.0], [300.0, 301.0]],
                [[200.0, 201.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert out.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(
        _to_numpy(out[0]),
        _to_numpy((current_feat[:, 0] + 10.0).transpose(1, 0).reshape(2, 2, 2)),
    )


def test_prepare_memory_conditioned_features_threads_saved_image_features():
    model = _MemoryConditioningHarness()
    model.save_image_features = True
    model.use_obj_ptrs_in_encoder = False
    model.transformer = _DecoupledTransformerRecorder()
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    current_feat = mx.arange(8).reshape(4, 1, 2).astype(mx.float32)
    current_pos = (mx.arange(8).reshape(4, 1, 2) + 50).astype(mx.float32)
    maskmem_features = mx.arange(16).reshape(2, 2, 2, 2).astype(mx.float32)
    maskmem_pos = (mx.arange(16).reshape(2, 2, 2, 2) + 100).astype(mx.float32)
    image_features = (mx.arange(8).reshape(4, 1, 2) + 200).astype(mx.float32)
    image_pos = (mx.arange(8).reshape(4, 1, 2) + 300).astype(mx.float32)
    output_dict = {
        "cond_frame_outputs": {
            0: {
                "conditioning_objects": {0, 1, 2},
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": [maskmem_pos],
                "image_features": image_features,
                "image_pos_enc": image_pos,
            }
        },
        "non_cond_frame_outputs": {},
    }

    out = model._prepare_memory_conditioned_features(
        frame_idx=1,
        is_init_cond_frame=False,
        current_vision_feats=[current_feat],
        current_vision_masks=[None],
        current_vision_pos_embeds=[current_pos],
        feat_sizes=[(2, 2)],
        output_dict=output_dict,
        num_frames=3,
        multiplex_state=multiplex_state,
    )

    call = model.transformer.encoder.calls[0]
    assert call["num_obj_ptr_tokens"] == 0
    np.testing.assert_array_equal(_to_numpy(call["image"]), _to_numpy(current_feat))
    np.testing.assert_array_equal(_to_numpy(call["image_pos"]), _to_numpy(current_pos))
    np.testing.assert_array_equal(
        _to_numpy(call["src"][:, 0]),
        _to_numpy(current_feat[:, 0]),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["src"][:, 1]),
        _to_numpy(current_feat[:, 0]),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["memory"]),
        _to_numpy(maskmem_features.reshape(2, 2, 4).transpose(2, 0, 1)),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["memory_pos"]),
        _to_numpy(maskmem_pos.reshape(2, 2, 4).transpose(2, 0, 1)),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["memory_image"]),
        _to_numpy(image_features),
    )
    np.testing.assert_array_equal(
        _to_numpy(call["memory_image_pos"]),
        _to_numpy(image_pos),
    )
    assert out.shape == (2, 2, 2, 2)
    np.testing.assert_array_equal(
        _to_numpy(out[0]),
        _to_numpy((current_feat[:, 0] + 20.0).transpose(1, 0).reshape(2, 2, 2)),
    )


def test_prepare_memory_conditioned_features_save_image_falls_back_without_image_memory():
    model = _MemoryConditioningHarness()
    model.save_image_features = True
    model.transformer = _DecoupledTransformerRecorder()
    multiplex_state = MultiplexState(
        [[0, 1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    current_feat = mx.arange(8).reshape(4, 1, 2).astype(mx.float32)
    output_dict = {
        "cond_frame_outputs": {
            0: {
                "conditioning_objects": {0, 1},
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "obj_ptr": mx.array(
                    [[[10.0, 11.0], [20.0, 21.0]]],
                    dtype=mx.float32,
                ),
            }
        },
        "non_cond_frame_outputs": {},
    }

    out = model._prepare_memory_conditioned_features(
        frame_idx=1,
        is_init_cond_frame=False,
        current_vision_feats=[current_feat],
        current_vision_masks=[None],
        current_vision_pos_embeds=[mx.zeros_like(current_feat)],
        feat_sizes=[(2, 2)],
        output_dict=output_dict,
        num_frames=3,
        multiplex_state=multiplex_state,
    )

    assert model.transformer.encoder.calls == []
    np.testing.assert_array_equal(
        _to_numpy(out),
        _to_numpy(current_feat.transpose(1, 2, 0).reshape(1, 2, 2, 2)),
    )


def test_track_step_mask_input_records_conditioning_output_and_memory():
    model = _TrackStepHarness()
    multiplex_state = MultiplexState(
        [[0, 1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    mask_inputs = mx.array(
        np.stack(
            [
                np.ones((1, 8, 8), dtype=np.float32),
                np.zeros((1, 8, 8), dtype=np.float32),
            ],
            axis=0,
        )
    )

    out = model.track_step(
        frame_idx=0,
        is_init_cond_frame=True,
        backbone_features_interactive=_interactive_features(),
        backbone_features_propagation=_propagation_features(),
        image="frame",
        point_inputs=None,
        mask_inputs=mask_inputs,
        gt_masks=None,
        frames_to_add_correction_pt=[],
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        num_frames=1,
        multiplex_state=multiplex_state,
    )

    assert out["conditioning_objects"] == {0, 1}
    np.testing.assert_array_equal(
        _to_numpy(out["object_score_logits"]),
        np.array([[10.0], [-10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(out["pred_masks"]),
        np.stack(
            [
                np.full((1, 2, 2), 10.0, dtype=np.float32),
                np.full((1, 2, 2), -10.0, dtype=np.float32),
            ],
            axis=0,
        ),
    )
    assert len(model.memory_calls) == 1
    assert model.memory_calls[0]["image"] == "frame"
    assert model.memory_calls[0]["conditioning_objects"] == {0, 1}
    assert model.memory_calls[0]["is_mask_from_pts"] is False
    assert out["maskmem_features"].shape == (1, 2, 1, 1)
    assert len(out["multistep_pred_ious"]) == 1


def test_track_step_routes_new_masks_to_reconditioning_helper_when_flagged():
    model = _TrackStepHarness()
    multiplex_state = MultiplexState(
        [[0, 1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    new_masks = _constant_rows([7.0], (1, 4, 4))

    out = model.track_step(
        frame_idx=1,
        is_init_cond_frame=False,
        backbone_features_interactive=_interactive_features(),
        backbone_features_propagation=_propagation_features(),
        image="frame",
        point_inputs=None,
        mask_inputs=None,
        gt_masks=None,
        frames_to_add_correction_pt=[],
        output_dict={
            "cond_frame_outputs": {0: {"conditioning_objects": {0, 1}}},
            "non_cond_frame_outputs": {},
        },
        num_frames=2,
        run_mem_encoder=True,
        multiplex_state=multiplex_state,
        new_object_masks=new_masks,
        new_object_idxs=[1],
        new_object_ids=[20],
        reconditioning=True,
    )

    assert model.add_mask_calls == []
    assert len(model.recondition_calls) == 1
    call = model.recondition_calls[0]
    assert call["obj_idxs_in_mask"] == [1]
    assert call["obj_ids_in_mask"] == [20]
    assert call["multiplex_state"] is multiplex_state
    assert call["prev_output"] is out
    assert call["add_mask_to_memory"] is True
    np.testing.assert_array_equal(
        _to_numpy(call["new_masks"]),
        _to_numpy(new_masks),
    )


def test_track_step_propagation_and_interaction_replaces_selected_rows():
    model = _TrackStepHarness()
    model.use_memory_selection = False
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    point_inputs = {
        "point_coords": mx.array([[[1.0, 2.0]]], dtype=mx.float32),
        "point_labels": mx.array([[1]], dtype=mx.int32),
    }

    out = model.track_step(
        frame_idx=1,
        is_init_cond_frame=False,
        backbone_features_interactive=_interactive_features(),
        backbone_features_propagation=_propagation_features(),
        image=None,
        point_inputs=point_inputs,
        mask_inputs=None,
        gt_masks=None,
        frames_to_add_correction_pt=[],
        output_dict={
            "cond_frame_outputs": {0: {"conditioning_objects": {0}}},
            "non_cond_frame_outputs": {},
        },
        num_frames=2,
        run_mem_encoder=False,
        multiplex_state=multiplex_state,
        objects_to_interact=[1],
    )

    assert len(model.prepare_calls) == 1
    assert model.forward_calls[0]["point_inputs"] is None
    np.testing.assert_array_equal(
        _to_numpy(model.forward_calls[1]["mask_inputs"]),
        _to_numpy(_constant_rows([2.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(out["pred_masks"]),
        _to_numpy(_constant_rows([1.0, 90.0, 3.0], (1, 2, 2))),
    )
    assert out["conditioning_objects"] == {1}
    assert out["maskmem_features"] is None
    assert out["maskmem_pos_enc"] is None


def test_track_step_correction_points_update_selected_rows_and_multistep_outputs():
    model = _TrackStepHarness()
    model.num_correction_pt_per_frame = 1
    model.use_memory_selection = False
    model.sam_output_high_tail = (1, 8, 8)

    out = model.track_step(
        frame_idx=0,
        is_init_cond_frame=True,
        backbone_features_interactive=_interactive_features(),
        backbone_features_propagation=_propagation_features(),
        image=None,
        point_inputs=None,
        mask_inputs=mx.ones((1, 1, 8, 8), dtype=mx.float32),
        gt_masks=mx.zeros((1, 1, 8, 8), dtype=mx.float32),
        frames_to_add_correction_pt=[0],
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        num_frames=1,
        run_mem_encoder=False,
        multiplex_state=MultiplexState([[0]], dtype=mx.float32),
        objects_to_interact=[0],
    )

    assert len(model.forward_calls) == 1
    correction_call = model.forward_calls[0]
    assert correction_call["objects_to_interact"] == [0]
    assert correction_call["point_inputs"]["point_coords"].shape == (1, 1, 2)
    assert correction_call["point_inputs"]["point_labels"].shape == (1, 1)
    np.testing.assert_array_equal(
        _to_numpy(correction_call["point_inputs"]["point_labels"]),
        np.array([[0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _to_numpy(out["pred_masks"]),
        _to_numpy(_constant_rows([90.0], (1, 2, 2))),
    )
    assert out["multistep_pred_masks"].shape == (1, 2, 2, 2)
    assert out["multistep_pred_masks_high_res"].shape == (1, 2, 8, 8)
    assert len(out["multistep_pred_ious"]) == 2
    assert out["multistep_point_inputs"][0] is None
    assert out["multistep_point_inputs"][1]["point_coords"].shape == (1, 1, 2)
    assert out["conditioning_objects"] == {0}
    assert out["maskmem_features"] is None
    assert out["maskmem_pos_enc"] is None


def test_track_step_training_correction_points_remain_explicit_boundary():
    model = _TrackStepHarness()
    model.training = True
    model.num_correction_pt_per_frame = 1

    with pytest.raises(
        UnsupportedMultiplexRuntimeError,
        match="VideoTrackingMultiplex._track_step_aux\\(training correction-points\\)",
    ):
        model.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            backbone_features_interactive=_interactive_features(),
            backbone_features_propagation=_propagation_features(),
            image=None,
            point_inputs=None,
            mask_inputs=mx.ones((1, 1, 8, 8), dtype=mx.float32),
            gt_masks=mx.zeros((1, 1, 8, 8), dtype=mx.float32),
            frames_to_add_correction_pt=[0],
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=1,
            run_mem_encoder=False,
            multiplex_state=MultiplexState([[0]], dtype=mx.float32),
            objects_to_interact=[0],
        )


def test_trim_output_and_memory_prunes_old_non_conditioning_heavy_tensors():
    model = _TrackStepHarness()
    model.trim_past_non_cond_mem_for_eval = True
    old_out = {
        "conditioning_objects": {0},
        "pred_masks": mx.array([[1.0]], dtype=mx.float32),
        "pred_masks_high_res": mx.array([[2.0]], dtype=mx.float32),
        "object_score_logits": mx.array([[1.0]], dtype=mx.float32),
        "multistep_point_inputs": [None],
        "maskmem_features": mx.array([[3.0]], dtype=mx.float32),
        "maskmem_pos_enc": [mx.array([[4.0]], dtype=mx.float32)],
        "eff_iou_score": mx.array(0.2, dtype=mx.float32),
    }
    output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {2: old_out},
    }

    returned = model._trim_output_and_memory(
        frame_idx=4,
        output_dict=output_dict,
        current_out={"conditioning_objects": set()},
        memory_encoder_was_used=True,
    )

    assert returned == {"conditioning_objects": set()}
    trimmed = output_dict["non_cond_frame_outputs"][2]
    assert set(trimmed) == {
        "conditioning_objects",
        "pred_masks",
        "object_score_logits",
        "multistep_point_inputs",
    }
    assert trimmed["pred_masks"] is old_out["pred_masks"]


def test_trim_output_and_memory_compacts_current_output_for_eval_offload():
    model = _TrackStepHarness()
    model.offload_output_to_cpu_for_eval = True
    model.use_obj_ptrs_in_encoder = True
    model.save_image_features = True
    current_out = {
        "conditioning_objects": {0},
        "pred_masks": mx.array([[1.0]], dtype=mx.float32),
        "pred_masks_high_res": mx.array([[2.0]], dtype=mx.float32),
        "object_score_logits": mx.array([[3.0]], dtype=mx.float32),
        "obj_ptr": mx.array([[4.0, 5.0]], dtype=mx.float32),
        "maskmem_features": mx.array([[6.0]], dtype=mx.float32),
        "maskmem_pos_enc": [mx.array([[7.0]], dtype=mx.float32)],
        "image_features": mx.array([[8.0]], dtype=mx.float32),
        "image_pos_enc": mx.array([[9.0]], dtype=mx.float32),
        "multistep_point_inputs": [None],
        "multistep_pred_masks": mx.array([[10.0]], dtype=mx.float32),
        "multistep_pred_masks_high_res": mx.array([[11.0]], dtype=mx.float32),
    }

    trimmed = model._trim_output_and_memory(
        frame_idx=0,
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        current_out=current_out,
        memory_encoder_was_used=True,
    )

    assert set(trimmed) == {
        "conditioning_objects",
        "pred_masks",
        "pred_masks_high_res",
        "object_score_logits",
        "multistep_point_inputs",
        "obj_ptr",
        "maskmem_features",
        "maskmem_pos_enc",
        "image_features",
        "image_pos_enc",
    }
    assert trimmed["maskmem_features"] is current_out["maskmem_features"]
    assert trimmed["image_features"] is current_out["image_features"]


def test_prepare_backbone_features_flattens_precomputed_neck_outputs_and_masks():
    model = _ForwardTrackingHarness()
    prepared = model._prepare_backbone_features(_forward_backbone_out())

    assert set(prepared) == {"interactive", "sam2_backbone_out"}
    interactive = prepared["interactive"]
    assert interactive["feat_sizes"] == [(2, 2)]
    assert interactive["vision_feats"][0].shape == (4, 2, 2)
    np.testing.assert_array_equal(
        _to_numpy(interactive["vision_feats"][0][:, 0]),
        np.array(
            [
                [0.0, 4.0],
                [1.0, 5.0],
                [2.0, 6.0],
                [3.0, 7.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        _to_numpy(interactive["vision_masks"][0]),
        np.array(
            [
                [False, True, False, False],
                [True, False, False, True],
            ],
            dtype=bool,
        ),
    )


def test_prepare_prompt_inputs_builds_mask_prompts_and_frame_metadata():
    model = _ForwardTrackingHarness()
    backbone_out = model.prepare_prompt_inputs({}, _forward_input_with_targets())

    assert backbone_out["num_frames"] == 2
    assert backbone_out["use_pt_input"] is False
    assert backbone_out["init_cond_frames"] == [0]
    assert backbone_out["frames_not_in_init_cond"] == [1]
    assert backbone_out["frames_to_add_correction_pt"] == []
    assert set(backbone_out["gt_masks_per_frame"]) == {0, 1}
    assert backbone_out["gt_masks_per_frame"][0].shape == (2, 1, 2, 2)
    assert set(backbone_out["mask_inputs_per_frame"]) == {0}
    assert backbone_out["point_inputs_per_frame"] == {}
    np.testing.assert_array_equal(
        _to_numpy(backbone_out["mask_inputs_per_frame"][0]),
        _to_numpy(backbone_out["gt_masks_per_frame"][0]),
    )


def test_prepare_prompt_inputs_dynamic_eval_builds_transition_metadata_and_masks():
    model = _ForwardTrackingHarness()
    model.is_dynamic_vos_evaluation = True
    input_data = _dynamic_eval_forward_input()

    backbone_out = model.prepare_prompt_inputs({}, input_data)

    assert backbone_out["init_cond_frames"] == [0]
    assert backbone_out["frames_not_in_init_cond"] == [1, 2]
    assert backbone_out["transition_points"] == {2}
    assert backbone_out["object_appearance_order"] == [1, 0]
    assert backbone_out["valid_idx_per_frame"] == {
        0: [0],
        1: [0],
        2: [0, 1],
    }
    assert backbone_out["valid_objects_prior_to_each_transition"] == {2: [0]}
    assert backbone_out["new_idx_per_transition"] == {2: [1]}
    np.testing.assert_array_equal(
        _to_numpy(backbone_out["mask_inputs_per_frame"][0]),
        _to_numpy(_constant_rows([101.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(backbone_out["gt_masks_per_frame"][2]),
        _to_numpy(_constant_rows([301.0, 300.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(input_data.find_targets[2].segments),
        _to_numpy(_constant_rows([301.0], (2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(input_data.find_targets[2].num_boxes),
        np.array([1], dtype=np.int64),
    )


def test_forward_dynamic_eval_prepares_transitions_and_restores_output_order():
    model = _ForwardTrackingHarness()
    model.is_dynamic_vos_evaluation = True
    input_data = _dynamic_eval_forward_input()

    outputs, queries = model.forward(input_data)

    assert queries is None
    assert [call["frame_idx"] for call in model.track_calls] == [0, 1, 2]
    transition_call = model.track_calls[2]
    assert transition_call["new_object_idxs"] == [1]
    np.testing.assert_array_equal(
        _to_numpy(transition_call["new_object_masks"]),
        _to_numpy(_constant_rows([300.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(transition_call["gt_masks"]),
        _to_numpy(_constant_rows([301.0], (1, 2, 2))),
    )
    assert len(outputs) == 3
    np.testing.assert_array_equal(
        _to_numpy(outputs[0]["pred_masks"]),
        _to_numpy(_constant_rows([0.0, 1.0, 0.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(outputs[1]["pred_masks"]),
        _to_numpy(_constant_rows([0.0, 2.0, 0.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(outputs[2]["pred_masks"]),
        _to_numpy(_constant_rows([0.0, 3.0, 0.0], (1, 2, 2))),
    )


def test_prepare_prompt_inputs_dynamic_eval_skips_empty_prefix_frames():
    model = _ForwardTrackingHarness()
    model.is_dynamic_vos_evaluation = True
    input_data = _dynamic_eval_forward_input()
    input_data.visible_objects_per_frame = {
        0: set(),
        1: {0},
        2: {0, 1},
    }

    backbone_out = model.prepare_prompt_inputs({}, input_data)

    assert backbone_out["init_cond_frames"] == [1]
    assert backbone_out["frames_not_in_init_cond"] == [2]
    assert set(backbone_out["mask_inputs_per_frame"]) == {1}
    assert backbone_out["transition_points"] == {2}
    assert backbone_out["object_appearance_order"] == [0, 1]
    assert backbone_out["valid_idx_per_frame"] == {
        0: [],
        1: [0],
        2: [0, 1],
    }
    assert input_data.find_targets[0].segments.shape == (0, 2, 2)
    np.testing.assert_array_equal(
        _to_numpy(backbone_out["mask_inputs_per_frame"][1]),
        _to_numpy(_constant_rows([200.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(input_data.find_targets[2].segments),
        _to_numpy(_constant_rows([300.0], (2, 2))),
    )


def test_forward_precomputes_features_and_returns_stage_outputs_tuple():
    model = _ForwardTrackingHarness()

    outputs, queries = model.forward(_forward_input_with_targets())

    assert queries is None
    assert len(outputs) == 2
    assert len(model.backbone_calls) == 1
    assert model.backbone_calls[0]["need_interactive_out"] is True
    assert model.backbone_calls[0]["need_propagation_out"] is True
    assert model.state_call["num_valid_entries"] == 2
    np.testing.assert_array_equal(
        _to_numpy(model.track_calls[0]["mask_inputs"]),
        np.array(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 1.0]]],
            ],
            dtype=np.float32,
        ),
    )
    assert "obj_ptr" not in outputs[0]


def test_forward_defers_backbone_features_per_frame_for_eval():
    model = _ForwardTrackingHarness()
    model.forward_backbone_per_frame_for_eval = True

    outputs, queries = model.forward(_forward_input_with_targets())

    assert queries is None
    assert len(outputs) == 2
    assert len(model.backbone_calls) == 2
    assert model.backbone_calls[0]["need_interactive_out"] is True
    assert model.backbone_calls[1]["need_interactive_out"] is False
    _assert_prepared_feature_value(
        model.track_calls[0]["backbone_features_interactive"],
        11.0,
    )
    assert model.track_calls[1]["backbone_features_interactive"] is None


def test_forward_tracking_routes_precomputed_frames_and_returns_state_dict():
    model = _ForwardTrackingHarness()
    output = model.forward_tracking(
        _forward_backbone_out(),
        _forward_input(),
        return_dict=True,
        objects_to_interact=[0],
    )

    assert set(output["cond_frame_outputs"]) == {0}
    assert set(output["non_cond_frame_outputs"]) == {1}
    assert output["multiplex_state"].assignments == [[0, 1]]
    assert model.state_call == {
        "num_valid_entries": 2,
        "dtype": mx.float32,
        "random": False,
    }
    assert len(model.track_calls) == 2

    first_call, second_call = model.track_calls
    assert first_call["frame_idx"] == 0
    assert first_call["is_init_cond_frame"] is True
    assert first_call["objects_to_interact"] == [0]
    np.testing.assert_array_equal(
        _to_numpy(first_call["image"]),
        np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2),
    )
    np.testing.assert_array_equal(
        _to_numpy(first_call["backbone_features_interactive"]["vision_feats"][0][:, 0]),
        np.array(
            [
                [0.0, 4.0],
                [1.0, 5.0],
                [2.0, 6.0],
                [3.0, 7.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        _to_numpy(first_call["mask_inputs"]),
        np.ones((2, 1, 4, 4), dtype=np.float32),
    )
    assert second_call["frame_idx"] == 1
    assert second_call["is_init_cond_frame"] is False
    assert second_call["mask_inputs"] is None
    assert (
        second_call["output_dict"]["cond_frame_outputs"] is output["cond_frame_outputs"]
    )


def test_forward_tracking_list_output_is_frame_ordered_and_strips_obj_ptr():
    model = _ForwardTrackingHarness()
    outputs = model.forward_tracking(
        _forward_backbone_out(),
        _forward_input(),
        return_dict=False,
    )

    assert len(outputs) == 2
    assert [out["conditioning_objects"] for out in outputs] == [{0}, {1}]
    assert all("obj_ptr" not in out for out in outputs)
    np.testing.assert_array_equal(
        _to_numpy(outputs[1]["pred_masks"]),
        _to_numpy(_constant_rows([2.0], (1, 2, 2))),
    )


def test_forward_tracking_splits_transition_masks_and_marks_transition_conditional():
    model = _ForwardTrackingHarness()
    model.add_all_transition_frames_as_cond = True
    backbone_out = _forward_backbone_out()
    backbone_out["transition_points"] = [1]
    backbone_out["new_idx_per_transition"] = {1: [2]}
    backbone_out["valid_objects_prior_to_each_transition"] = {1: [0, 1]}
    backbone_out["gt_masks_per_frame"][1] = _constant_rows(
        [10.0, 20.0, 30.0],
        (1, 4, 4),
    )

    output = model.forward_tracking(
        backbone_out,
        _forward_input(),
        return_dict=True,
    )

    assert set(output["cond_frame_outputs"]) == {0, 1}
    assert output["non_cond_frame_outputs"] == {}
    transition_call = model.track_calls[1]
    assert transition_call["new_object_idxs"] == [2]
    np.testing.assert_array_equal(
        _to_numpy(transition_call["new_object_masks"]),
        _to_numpy(_constant_rows([30.0], (1, 4, 4))),
    )
    np.testing.assert_array_equal(
        _to_numpy(transition_call["gt_masks"]),
        _to_numpy(_constant_rows([10.0, 20.0], (1, 4, 4))),
    )


def test_forward_tracking_dynamic_eval_pads_missing_frames_and_restores_object_order():
    model = _ForwardTrackingHarness()
    model.is_dynamic_vos_evaluation = True
    backbone_out = {
        "num_frames": 3,
        "init_cond_frames": [0],
        "frames_not_in_init_cond": [2],
        "frames_to_add_correction_pt": [],
        "gt_masks_per_frame": {
            0: mx.ones((2, 1, 4, 4), dtype=mx.float32),
            2: mx.ones((2, 1, 4, 4), dtype=mx.float32),
        },
        "mask_inputs_per_frame": {
            0: mx.ones((2, 1, 4, 4), dtype=mx.float32),
        },
        "point_inputs_per_frame": {},
        "transition_points": [],
        "object_appearance_order": [1, 0],
    }
    input_data = _forward_input()
    input_data.img_batch = mx.array(
        np.arange(3 * 3 * 2 * 2, dtype=np.float32).reshape(3, 3, 2, 2)
    )
    input_data.find_inputs = [
        SimpleNamespace(img_ids=mx.array([0, 0], dtype=mx.int64)),
        SimpleNamespace(img_ids=mx.array([1, 1], dtype=mx.int64)),
        SimpleNamespace(img_ids=mx.array([2, 2], dtype=mx.int64)),
    ]
    input_data.find_metadatas = [
        SimpleNamespace(coco_image_id=[100, 101, 102]),
    ]

    outputs = model.forward_tracking(backbone_out, input_data, return_dict=False)

    assert len(outputs) == 3
    assert all("obj_ptr" not in output for output in outputs)
    np.testing.assert_array_equal(
        _to_numpy(outputs[0]["pred_masks"]),
        _to_numpy(_constant_rows([0.0, 1.0, 0.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(outputs[1]["pred_masks"]),
        np.zeros((3, 1, 2, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(outputs[2]["pred_masks"]),
        _to_numpy(_constant_rows([0.0, 3.0, 0.0], (1, 2, 2))),
    )


def test_forward_tracking_computes_per_frame_backbone_when_features_are_missing():
    model = _ForwardTrackingHarness()
    backbone_out = {
        "num_frames": 2,
        "init_cond_frames": [0],
        "frames_not_in_init_cond": [1],
        "frames_to_add_correction_pt": [0],
        "gt_masks_per_frame": {
            0: mx.ones((2, 1, 2, 2), dtype=mx.float32),
            1: mx.ones((2, 1, 2, 2), dtype=mx.float32),
        },
        "mask_inputs_per_frame": {0: mx.ones((2, 1, 2, 2), dtype=mx.float32)},
        "point_inputs_per_frame": {},
    }

    output = model.forward_tracking(backbone_out, _forward_input(), return_dict=True)

    assert set(output["cond_frame_outputs"]) == {0}
    assert set(output["non_cond_frame_outputs"]) == {1}
    assert len(model.backbone_calls) == 2
    assert model.backbone_calls[0]["need_interactive_out"] is True
    assert model.backbone_calls[0]["need_propagation_out"] is True
    assert model.backbone_calls[1]["need_interactive_out"] is False
    assert model.backbone_calls[1]["need_propagation_out"] is True
    np.testing.assert_array_equal(
        _to_numpy(model.backbone_calls[0]["image"]),
        _to_numpy(_forward_input().img_batch[:1]),
    )
    np.testing.assert_array_equal(
        _to_numpy(model.backbone_calls[1]["image"]),
        _to_numpy(_forward_input().img_batch[1:2]),
    )
    first_call = model.track_calls[0]
    second_call = model.track_calls[1]
    _assert_prepared_feature_value(first_call["backbone_features_interactive"], 11.0)
    _assert_prepared_feature_value(first_call["backbone_features_propagation"], 21.0)
    assert second_call["backbone_features_interactive"] is None
    _assert_prepared_feature_value(second_call["backbone_features_propagation"], 22.0)


def test_prepare_backbone_features_per_frame_preserves_nested_tensor_mask():
    model = _ForwardTrackingHarness()
    tensors = mx.array(np.arange(3 * 3 * 2 * 2, dtype=np.float32).reshape(3, 3, 2, 2))
    mask = mx.array(
        np.array(
            [
                [[False, False], [False, True]],
                [[True, False], [False, False]],
                [[False, True], [True, False]],
            ],
            dtype=bool,
        )
    )

    image, features = model._prepare_backbone_features_per_frame(
        NestedTensor(tensors, mask),
        mx.array([1, 1], dtype=mx.int64),
        need_interactive_out=True,
        need_propagation_out=True,
    )

    assert isinstance(image, NestedTensor)
    assert isinstance(model.backbone_calls[0]["image"], NestedTensor)
    np.testing.assert_array_equal(
        _to_numpy(image.tensors),
        _to_numpy(tensors[1:2]),
    )
    np.testing.assert_array_equal(
        _to_numpy(image.mask),
        _to_numpy(mask[1:2]),
    )
    assert set(features) == {"interactive", "sam2_backbone_out"}
    assert features["interactive"]["vision_feats"][0].shape == (4, 1, 2)


def test_recondition_masks_in_existing_state_updates_rows_pointers_and_memory():
    model = _ReconditionHarness()
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
    )
    existing_pointers = mx.array(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
        dtype=mx.float32,
    )
    prev_output = {
        "conditioning_objects": {1},
        "pred_masks": _constant_rows([10.0, 20.0, 30.0], (1, 2, 2)),
        "pred_masks_high_res": _constant_rows([11.0, 22.0, 33.0], (1, 4, 4)),
        "object_score_logits": mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32),
        "iou_score": mx.array([0.1, 0.2, 0.3], dtype=mx.float32),
        "input_masks": _constant_rows([111.0, 222.0, 333.0], (1, 4, 4)),
        "obj_ptr": multiplex_state.mux(existing_pointers),
    }
    interactive_pix_feat = mx.array([[1.0]], dtype=mx.float32)
    interactive_high_res_features = [mx.array([[2.0]], dtype=mx.float32)]
    propagation_vision_feats = [mx.array([[3.0]], dtype=mx.float32)]
    propagation_feat_sizes = [(2, 2)]
    new_masks = _constant_rows([40.0, 50.0], (1, 4, 4))

    model.recondition_masks_in_existing_state(
        interactive_pix_feat=interactive_pix_feat,
        interactive_high_res_features=interactive_high_res_features,
        propagation_vision_feats=propagation_vision_feats,
        propagation_feat_sizes=propagation_feat_sizes,
        new_masks=new_masks,
        obj_idxs_in_mask=[2, 0],
        obj_ids_in_mask=[300, 100],
        prev_output=prev_output,
        multiplex_state=multiplex_state,
        are_masks_from_pts=True,
    )

    assert model.mask_calls[0]["objects_in_mask"] == [2, 0]
    assert model.mask_calls[0]["multiplex_state"] is multiplex_state
    np.testing.assert_array_equal(
        _to_numpy(prev_output["pred_masks"]),
        _to_numpy(_constant_rows([8.0, 20.0, 7.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["pred_masks_high_res"]),
        _to_numpy(_constant_rows([80.0, 22.0, 70.0], (1, 4, 4))),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["object_score_logits"]),
        np.array([[202.0], [2.0], [101.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _to_numpy(prev_output["iou_score"]),
        np.array([0.82, 0.2, 0.91], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["input_masks"]),
        _to_numpy(_constant_rows([50.0, 222.0, 40.0], (1, 4, 4))),
    )
    np.testing.assert_array_equal(
        _to_numpy(multiplex_state.demux(prev_output["obj_ptr"])),
        np.array(
            [[200.0, 201.0], [20.0, 21.0], [100.0, 101.0]],
            dtype=np.float32,
        ),
    )

    assert prev_output["conditioning_objects"] == {0, 1, 2}
    assert len(model.memory_calls) == 1
    assert model.memory_calls[0]["image"] is None
    assert model.memory_calls[0]["current_vision_feats"] is propagation_vision_feats
    assert model.memory_calls[0]["feat_sizes"] == propagation_feat_sizes
    assert model.memory_calls[0]["conditioning_objects"] == {0, 1, 2}
    assert model.memory_calls[0]["is_mask_from_pts"] is True
    np.testing.assert_array_equal(
        _to_numpy(model.memory_calls[0]["pred_masks_high_res"]),
        _to_numpy(prev_output["pred_masks_high_res"]),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["maskmem_features"]),
        np.array([[9.0, 9.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["maskmem_pos_enc"][0]),
        np.array([[8.0, 8.5]], dtype=np.float32),
    )


def test_add_new_masks_to_existing_state_appends_rows_pointers_and_memory():
    model = _ReconditionHarness()
    multiplex_state = MultiplexState(
        [[0, -1], [1, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
        object_ids=[100, 200],
    )
    existing_pointers = mx.array(
        [[10.0, 11.0], [20.0, 21.0]],
        dtype=mx.float32,
    )
    prev_output = {
        "conditioning_objects": {1},
        "pred_masks": _constant_rows([10.0, 20.0], (1, 2, 2)),
        "pred_masks_high_res": _constant_rows([11.0, 22.0], (1, 2, 2)),
        "object_score_logits": mx.array([[1.0], [2.0]], dtype=mx.float32),
        "iou_score": mx.array([0.1, 0.2], dtype=mx.float32),
        "input_masks": _constant_rows([111.0, 222.0], (1, 4, 4)),
        "obj_ptr": multiplex_state.mux(existing_pointers),
    }
    propagation_vision_feats = [mx.array([[3.0]], dtype=mx.float32)]
    propagation_feat_sizes = [(2, 2)]
    new_masks = _constant_rows([40.0, 50.0], (1, 4, 4))

    model.add_new_masks_to_existing_state(
        interactive_pix_feat=mx.array([[1.0]], dtype=mx.float32),
        interactive_high_res_features=[mx.array([[2.0]], dtype=mx.float32)],
        propagation_vision_feats=propagation_vision_feats,
        propagation_feat_sizes=propagation_feat_sizes,
        new_masks=new_masks,
        obj_idxs_in_mask=[2, 3],
        obj_ids_in_mask=[300, 400],
        prev_output=prev_output,
        multiplex_state=multiplex_state,
        are_masks_from_pts=True,
    )

    assert multiplex_state.assignments == [[0, 2], [1, 3]]
    assert multiplex_state.object_ids == [100, 200, 300, 400]
    assert model.mask_calls[0]["objects_in_mask"] == [2, 3]
    assert model.mask_calls[0]["multiplex_state"] is multiplex_state

    np.testing.assert_array_equal(
        _to_numpy(prev_output["pred_masks"]),
        _to_numpy(_constant_rows([10.0, 20.0, 7.0, 8.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["pred_masks_high_res"]),
        _to_numpy(_constant_rows([11.0, 22.0, 70.0, 80.0], (1, 4, 4))),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["object_score_logits"]),
        np.array([[1.0], [2.0], [101.0], [202.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _to_numpy(prev_output["iou_score"]),
        np.array([0.1, 0.2, 0.91, 0.82], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["input_masks"]),
        _to_numpy(_constant_rows([111.0, 222.0, 40.0, 50.0], (1, 4, 4))),
    )
    np.testing.assert_array_equal(
        _to_numpy(multiplex_state.demux(prev_output["obj_ptr"])),
        np.array(
            [[10.0, 11.0], [20.0, 21.0], [100.0, 101.0], [200.0, 201.0]],
            dtype=np.float32,
        ),
    )

    assert prev_output["conditioning_objects"] == {1, 2, 3}
    assert len(model.memory_calls) == 1
    assert model.memory_calls[0]["conditioning_objects"] == {1, 2, 3}
    assert model.memory_calls[0]["is_mask_from_pts"] is True
    np.testing.assert_array_equal(
        _to_numpy(model.memory_calls[0]["pred_masks_high_res"]),
        _to_numpy(prev_output["pred_masks_high_res"]),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["maskmem_features"]),
        np.array([[9.0, 9.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["maskmem_pos_enc"][0]),
        np.array([[8.0, 8.5]], dtype=np.float32),
    )


def test_add_new_masks_to_existing_state_matches_official_packed_add_fixture():
    fixture = json.loads(PACKED_ADD_PARITY_FIXTURE.read_text())
    assert fixture["official_commit"] == OFFICIAL_SAM3_VIDEO_TRACKING_MULTIPLEX_COMMIT
    assert fixture["case"] == "packed_append_with_memory_reencode"
    assert fixture["component"] == (
        "VideoTrackingMultiplex.add_new_masks_to_existing_state"
    )
    for metric in fixture["metrics"].values():
        assert metric["max_abs"] <= fixture["atol"]

    model = _ReconditionHarness(low_res_masks=_packed_low_res_mask_patterns())
    multiplex_state = MultiplexState(
        [[0, -1], [1, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
        object_ids=[100, 200],
    )
    existing_pointers = mx.array(
        [[10.0, 11.0], [20.0, 21.0]],
        dtype=mx.float32,
    )
    prev_output = {
        "conditioning_objects": {1},
        "pred_masks": _constant_rows([10.0, 20.0], (1, 2, 2)),
        "pred_masks_high_res": _constant_rows([11.0, 22.0], (1, 2, 2)),
        "object_score_logits": mx.array([[1.0], [2.0]], dtype=mx.float32),
        "iou_score": mx.array([0.1, 0.2], dtype=mx.float32),
        "input_masks": _constant_rows([111.0, 222.0], (1, 4, 4)),
        "obj_ptr": multiplex_state.mux(existing_pointers),
    }

    model.add_new_masks_to_existing_state(
        interactive_pix_feat=mx.array([[1.0]], dtype=mx.float32),
        interactive_high_res_features=[mx.array([[2.0]], dtype=mx.float32)],
        propagation_vision_feats=[mx.array([[3.0]], dtype=mx.float32)],
        propagation_feat_sizes=[(2, 2)],
        new_masks=_constant_rows([40.0, 50.0], (1, 4, 4)),
        obj_idxs_in_mask=[2, 3],
        obj_ids_in_mask=[300, 400],
        prev_output=prev_output,
        multiplex_state=multiplex_state,
        are_masks_from_pts=True,
    )

    expected = fixture["official"]
    assert multiplex_state.assignments == expected["assignments"]
    assert multiplex_state.object_ids == expected["object_ids"]
    assert (
        sorted(prev_output["conditioning_objects"]) == expected["conditioning_objects"]
    )
    assert (
        model.mask_calls[0]["objects_in_mask"] == expected["mask_call_objects_in_mask"]
    )
    assert (
        sorted(model.memory_calls[0]["conditioning_objects"])
        == expected["memory_call"]["conditioning_objects"]
    )
    assert (
        model.memory_calls[0]["is_mask_from_pts"]
        == expected["memory_call"]["is_mask_from_pts"]
    )

    def assert_matches_fixture(name, observed):
        payload = expected[name]
        observed_np = _to_numpy(observed)
        assert list(observed_np.shape) == payload["shape"]
        assert str(observed_np.dtype) == payload["dtype"]
        np.testing.assert_allclose(
            observed_np,
            np.asarray(payload["values"], dtype=np.float32),
            rtol=0.0,
            atol=fixture["atol"],
        )

    assert_matches_fixture("pred_masks", prev_output["pred_masks"])
    assert_matches_fixture("pred_masks_high_res", prev_output["pred_masks_high_res"])
    assert_matches_fixture("object_score_logits", prev_output["object_score_logits"])
    assert_matches_fixture("iou_score", prev_output["iou_score"])
    assert_matches_fixture("input_masks", prev_output["input_masks"])
    assert_matches_fixture(
        "demux_obj_ptr",
        multiplex_state.demux(prev_output["obj_ptr"]),
    )
    assert_matches_fixture("maskmem_features", prev_output["maskmem_features"])
    assert_matches_fixture("maskmem_pos_enc_0", prev_output["maskmem_pos_enc"][0])

    memory_call = expected["memory_call"]
    np.testing.assert_allclose(
        _to_numpy(model.memory_calls[0]["pred_masks_high_res"]),
        np.asarray(memory_call["pred_masks_high_res"]["values"], dtype=np.float32),
        rtol=0.0,
        atol=fixture["atol"],
    )
    np.testing.assert_allclose(
        _to_numpy(model.memory_calls[0]["object_score_logits"]),
        np.asarray(memory_call["object_score_logits"]["values"], dtype=np.float32),
        rtol=0.0,
        atol=fixture["atol"],
    )


def test_add_new_masks_adapter_mutates_current_packed_state():
    model = _ReconditionHarness()
    multiplex_state = MultiplexState(
        [[0, -1, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=3,
        object_ids=[10],
    )
    existing_pointers = mx.array([[10.0, 11.0]], dtype=mx.float32)
    prev_output = {
        "conditioning_objects": {0},
        "pred_masks": _constant_rows([10.0], (1, 2, 2)),
        "pred_masks_high_res": _constant_rows([11.0], (1, 2, 2)),
        "object_score_logits": mx.array([[1.0]], dtype=mx.float32),
        "iou_score": mx.array([0.1], dtype=mx.float32),
        "input_masks": _constant_rows([111.0], (1, 4, 4)),
        "obj_ptr": multiplex_state.mux(existing_pointers),
    }
    interactive_vision_feats = [
        mx.arange(4, dtype=mx.float32).reshape(4, 1, 1),
        mx.ones((1, 1, 1), dtype=mx.float32),
    ]
    propagation_vision_feats = [mx.array([[3.0]], dtype=mx.float32)]
    cached_backbone = {
        "interactive": {
            "vision_feats": interactive_vision_feats,
            "feat_sizes": [(2, 2), (1, 1)],
        },
        "sam2_backbone_out": {
            "vision_feats": propagation_vision_feats,
            "feat_sizes": [(2, 2)],
        },
    }
    state = {
        "obj_ids": [10],
        "multiplex_state": multiplex_state,
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {2: prev_output},
        },
        "cached_features": {
            2: (mx.zeros((1, 3, 4, 4), dtype=mx.float32), cached_backbone)
        },
    }

    frame_idx, obj_ids, _, _ = model.add_new_masks(
        inference_state=state,
        frame_idx=2,
        obj_ids=mx.array([20, 30], dtype=mx.int64),
        masks=_constant_rows([40.0, 50.0], (4, 4)),
        add_mask_to_memory=True,
    )
    model.propagate_in_video_preflight(state, run_mem_encoder=True)

    assert frame_idx == 2
    assert obj_ids == [10, 20, 30]
    assert state["obj_ids"] == [10, 20, 30]
    assert state["backbone_out"] is cached_backbone
    assert state["tracking_has_started"] is True
    assert multiplex_state.assignments == [[0, 1, 2]]
    assert multiplex_state.object_ids == [10, 20, 30]
    assert model.pix_mem_calls[0]["vision_feats"] is interactive_vision_feats
    assert model.pix_mem_calls[0]["feat_sizes"] == [(2, 2), (1, 1)]
    assert model.mask_calls[0]["objects_in_mask"] == [1, 2]
    assert model.mask_calls[0]["high_res_features"][0].shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(
        _to_numpy(model.mask_calls[0]["mask_inputs"]),
        _to_numpy(_constant_rows([40.0, 50.0], (1, 4, 4))),
    )

    np.testing.assert_array_equal(
        _to_numpy(prev_output["pred_masks"]),
        _to_numpy(_constant_rows([10.0, 7.0, 8.0], (1, 2, 2))),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["pred_masks_high_res"]),
        _to_numpy(_constant_rows([11.0, 70.0, 80.0], (1, 4, 4))),
    )
    np.testing.assert_array_equal(
        _to_numpy(prev_output["object_score_logits"]),
        np.array([[1.0], [101.0], [202.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _to_numpy(prev_output["iou_score"]),
        np.array([0.1, 0.91, 0.82], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        _to_numpy(multiplex_state.demux(prev_output["obj_ptr"])),
        np.array(
            [[10.0, 11.0], [100.0, 101.0], [200.0, 201.0]],
            dtype=np.float32,
        ),
    )
    assert prev_output["conditioning_objects"] == {0, 1, 2}
    assert model.memory_calls[0]["current_vision_feats"] is propagation_vision_feats
    assert model.memory_calls[0]["feat_sizes"] == [(2, 2)]
    assert model.memory_calls[0]["conditioning_objects"] == {0, 1, 2}
    np.testing.assert_array_equal(
        _to_numpy(prev_output["maskmem_features"]),
        np.array([[9.0, 9.5]], dtype=np.float32),
    )


def test_recondition_masks_in_existing_state_requires_mask_output_mode():
    model = _ReconditionHarness()
    model.use_mask_input_as_output_without_sam = False

    with pytest.raises(
        UnsupportedMultiplexRuntimeError,
        match="VideoTrackingMultiplex.recondition_masks_in_existing_state",
    ):
        model.recondition_masks_in_existing_state(
            interactive_pix_feat=None,
            interactive_high_res_features=[],
            propagation_vision_feats=None,
            propagation_feat_sizes=None,
            new_masks=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            obj_idxs_in_mask=[0],
            obj_ids_in_mask=None,
            prev_output={"conditioning_objects": set()},
            multiplex_state=MultiplexState([[0]], dtype=mx.float32),
            add_mask_to_memory=False,
        )

    with pytest.raises(
        UnsupportedMultiplexRuntimeError,
        match="VideoTrackingMultiplex.add_new_masks_to_existing_state",
    ):
        model.add_new_masks_to_existing_state(
            interactive_pix_feat=None,
            interactive_high_res_features=[],
            propagation_vision_feats=None,
            propagation_feat_sizes=None,
            new_masks=mx.zeros((1, 1, 2, 2), dtype=mx.float32),
            obj_idxs_in_mask=[0],
            obj_ids_in_mask=None,
            prev_output={"conditioning_objects": set()},
            multiplex_state=MultiplexState([[0]], dtype=mx.float32),
            add_mask_to_memory=False,
        )
