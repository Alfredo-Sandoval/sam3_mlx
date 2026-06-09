from __future__ import annotations

from copy import deepcopy

try:
    from typing import NotRequired, Required, TypedDict
except ImportError:
    from typing_extensions import NotRequired, Required, TypedDict
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sam3_mlx.model.data_misc import NestedTensor, interpolate
from sam3_mlx.model.memory import SimpleMaskEncoder
from sam3_mlx.model.multiplex_mask_decoder import MLP, MultiplexMaskDecoder
from sam3_mlx.model.multiplex_utils import (
    MultiplexController,
    MultiplexState,
    raise_unsupported_multiplex_runtime,
)
from sam3_mlx.model.sam3_tracker_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)
from sam3_mlx.sam.common import Conv2dNCHW
from sam3_mlx.sam.mask_decoder import MaskDecoder
from sam3_mlx.sam.prompt_encoder import PositionEmbeddingRandom, PromptEncoder
from sam3_mlx.sam.transformer import TwoWayTransformer


NO_OBJ_SCORE = -1024.0
neck_outs = ["interactive", "sam2_backbone_out"]


def _trunc_normal(shape, std: float = 0.02) -> mx.array:
    return mx.random.truncated_normal(lower=-2.0, upper=2.0, shape=shape) * std


class SAMOutput(TypedDict, total=True):
    low_res_multimasks: Any
    high_res_multimasks: Any
    ious: Any
    low_res_masks: Any
    high_res_masks: Any
    object_score_logits: Any
    obj_ptr: NotRequired[Any]


class StageOutput(TypedDict, total=False):
    conditioning_objects: Required[set[int]]
    pred_masks: Any
    pred_masks_high_res: Any
    point_inputs: dict[str, Any]
    mask_inputs: Any
    object_score_logits: Any
    obj_ptr: Any
    maskmem_features: Any
    maskmem_pos_enc: list[Any]
    image_features: Any
    image_pos_enc: Any
    iou_score: Any
    eff_iou_score: Any
    multistep_pred_masks: Any
    multistep_pred_masks_high_res: Any
    multistep_pred_multimasks: list[Any]
    multistep_pred_multimasks_high_res: list[Any]
    multistep_pred_ious: list[Any]
    multistep_point_inputs: list[dict[str, Any]]
    multistep_object_score_logits: list[Any]


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


def _concat(values: list[Any], axis: int) -> Any:
    if any(_is_mlx_array(value) for value in values):
        return mx.concat(values, axis=axis)
    return np.concatenate(values, axis=axis)


def concat_points(
    old_point_inputs: Any, new_points: Any, new_labels: Any
) -> dict[str, Any]:
    """Add new points and labels to previous point inputs."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = _concat([old_point_inputs["point_coords"], new_points], axis=1)
        labels = _concat([old_point_inputs["point_labels"], new_labels], axis=1)
    return {"point_coords": points, "point_labels": labels}


def _append(
    d1: StageOutput,
    d2: SAMOutput,
    k1: str,
    k2: str,
    dim: int = 0,
    strict: bool = True,
) -> None:
    if strict:
        assert k1 in d1, f"{k1} not found"
    elif k1 not in d1:
        return
    d1[k1] = _concat([d1[k1], d2[k2]], axis=dim)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if _is_mlx_array(value):
        mx.eval(value)
        return np.array(value)
    return np.asarray(value)


def _merge(
    d1: StageOutput,
    d2: SAMOutput,
    k1: str,
    k2: str,
    d2_idx: list[int],
    strict: bool = True,
) -> None:
    if strict:
        assert k1 in d1, f"{k1} not found"
    elif k1 not in d1:
        return

    if _is_mlx_array(d1[k1]) or _is_mlx_array(d2[k2]):
        merged = _to_numpy(d1[k1]).copy()
        source = _to_numpy(d2[k2]).astype(merged.dtype, copy=False)
        merged[d2_idx] = source
        d1[k1] = mx.array(merged)
        return

    d1[k1][d2_idx] = np.asarray(d2[k2], dtype=d1[k1].dtype)


def _replace_rows(value: Any, indices: list[int], replacement: Any) -> Any:
    if _is_mlx_array(value) or _is_mlx_array(replacement):
        updated = _to_numpy(value).copy()
        source = _to_numpy(replacement).astype(updated.dtype, copy=False)
        updated[indices] = source
        return mx.array(updated)

    updated = np.asarray(value).copy()
    source = np.asarray(replacement, dtype=updated.dtype)
    updated[indices] = source
    return updated


def _take_rows(value: Any, indices: list[int]) -> Any:
    if _is_mlx_array(value):
        return mx.take(value, mx.array(indices, dtype=mx.int32), axis=0)
    return np.asarray(value)[indices]


def _to_scalar(value: Any) -> float:
    if _is_mlx_array(value):
        return float(_to_numpy(value))
    return float(value)


def _feature_tensor(feature_map: Any) -> Any:
    if hasattr(feature_map, "tensors"):
        return feature_map.tensors
    if isinstance(feature_map, dict) and "tensors" in feature_map:
        return feature_map["tensors"]
    return feature_map


def _feature_mask(feature_map: Any) -> Any | None:
    if hasattr(feature_map, "mask"):
        return feature_map.mask
    if isinstance(feature_map, dict):
        return feature_map.get("mask")
    return None


def _take_along_axis(value: Any, indices: Any, axis: int) -> Any:
    if _is_mlx_array(value):
        return mx.take(value, mx.array(indices, dtype=mx.int32), axis=axis)
    return np.take(np.asarray(value), np.asarray(indices, dtype=np.int64), axis=axis)


def _take_image_batch(value: Any, indices: Any) -> Any:
    if isinstance(value, NestedTensor):
        tensors = _take_along_axis(value.tensors, indices, axis=0)
        mask = (
            _take_along_axis(value.mask, indices, axis=0)
            if value.mask is not None
            else None
        )
        return NestedTensor(tensors, mask)
    return _take_along_axis(value, indices, axis=0)


def _replace_feature_tensor(feature_map: Any, tensor: Any) -> Any:
    if isinstance(feature_map, NestedTensor):
        return NestedTensor(tensor, feature_map.mask)
    if isinstance(feature_map, dict):
        updated = dict(feature_map)
        updated["tensors"] = tensor
        return updated
    return tensor


class VideoTrackingMultiplex(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        maskmem_backbone: nn.Module,
        multiplex_controller: MultiplexController,
        num_maskmem: int = 7,
        image_size: int = 512,
        backbone_stride: int = 16,
        **kwargs: Any,
    ):
        super().__init__()
        sam_mask_decoder_extra_args = deepcopy(
            kwargs.get("sam_mask_decoder_extra_args", None)
        )
        interactive_sam_mask_decoder_extra_args = deepcopy(sam_mask_decoder_extra_args)
        if sam_mask_decoder_extra_args is not None:
            sam_mask_decoder_extra_args["dynamic_multimask_via_stability"] = False

        self.backbone = backbone
        self.use_high_res_features_in_sam = kwargs.get(
            "use_high_res_features_in_sam", False
        )
        self.num_feature_levels = 3 if self.use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = kwargs.get("use_obj_ptrs_in_encoder", False)
        self.max_obj_ptrs_in_encoder = kwargs.get("max_obj_ptrs_in_encoder", 16)
        if self.use_obj_ptrs_in_encoder:
            self.interactive_mask_downsample = Conv2dNCHW(1, 1, kernel_size=4, stride=4)

        self.add_tpos_enc_to_obj_ptrs = kwargs.get("add_tpos_enc_to_obj_ptrs", True)
        self.proj_tpos_enc_in_obj_ptrs = kwargs.get("proj_tpos_enc_in_obj_ptrs", False)
        if self.proj_tpos_enc_in_obj_ptrs:
            assert self.add_tpos_enc_to_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = kwargs.get(
            "use_signed_tpos_enc_to_obj_ptrs", False
        )
        self.only_obj_ptrs_in_the_past_for_eval = kwargs.get(
            "only_obj_ptrs_in_the_past_for_eval", False
        )
        self.multiplex_controller = multiplex_controller
        self.save_image_features = kwargs.get("save_image_features", False)
        self.multiplex_count = self.multiplex_controller.multiplex_count

        assert transformer.decoder is None, "transformer should be encoder-only"
        self.transformer = transformer
        self.hidden_dim = transformer.d_model

        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj,
            "weight",
        ):
            mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
            assert mem_dim == self.hidden_dim, (
                "there should be no compression of memory embeddings"
            )
        self.num_maskmem = num_maskmem
        self.sincos_tpos_enc = kwargs.get("sincos_tpos_enc", True)
        self.use_maskmem_tpos_v2 = kwargs.get("use_maskmem_tpos_v2", False)
        self.maskmem_tpos_enc = _trunc_normal((num_maskmem, 1, 1, self.mem_dim))
        self.interactivity_no_mem_embed = _trunc_normal((1, 1, self.hidden_dim))
        self.directly_add_no_mem_embed = kwargs.get("directly_add_no_mem_embed", False)

        self.apply_sigmoid_to_mask_logits_for_mem_enc = kwargs.get(
            "apply_sigmoid_to_mask_logits_for_mem_enc", False
        )
        self.sigmoid_scale_for_mem_enc = kwargs.get("sigmoid_scale_for_mem_enc", 1.0)
        self.sigmoid_bias_for_mem_enc = kwargs.get("sigmoid_bias_for_mem_enc", 0.0)
        self.binarize_mask_from_pts_for_mem_enc = kwargs.get(
            "binarize_mask_from_pts_for_mem_enc", False
        )
        self.non_overlap_masks_for_mem_enc = kwargs.get(
            "non_overlap_masks_for_mem_enc", False
        )
        self.memory_temporal_stride_for_eval = kwargs.get(
            "memory_temporal_stride_for_eval", 1
        )
        self.use_mask_input_as_output_without_sam = kwargs.get(
            "use_mask_input_as_output_without_sam", False
        )
        self.multimask_output_in_sam = kwargs.get("multimask_output_in_sam", False)
        self.multimask_min_pt_num = kwargs.get("multimask_min_pt_num", 1)
        self.multimask_max_pt_num = kwargs.get("multimask_max_pt_num", 1)
        self.multimask_output_for_tracking = kwargs.get(
            "multimask_output_for_tracking", False
        )
        self.use_multimask_token_for_obj_ptr = kwargs.get(
            "use_multimask_token_for_obj_ptr", False
        )
        self.use_best_iou_mask_for_mem_enc = kwargs.get(
            "use_best_iou_mask_for_mem_enc", False
        )
        self.iou_prediction_use_sigmoid = kwargs.get(
            "iou_prediction_use_sigmoid", False
        )
        self.object_score_logit_threshold = kwargs.get(
            "object_score_logit_threshold", 0.0
        )
        self.stability_score_attentuation = kwargs.get(
            "stability_score_attentuation", False
        )
        self.iter_use_prev_mask_pred = kwargs.get("iter_use_prev_mask_pred", False)

        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        self.input_mask_size = self.low_res_mask_size * 4
        self.forward_backbone_per_frame_for_eval = kwargs.get(
            "forward_backbone_per_frame_for_eval", False
        )
        self.offload_output_to_cpu_for_eval = kwargs.get(
            "offload_output_to_cpu_for_eval", False
        )
        self.trim_past_non_cond_mem_for_eval = kwargs.get(
            "trim_past_non_cond_mem_for_eval", False
        )
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.interactive_sam_mask_decoder_extra_args = (
            interactive_sam_mask_decoder_extra_args
        )
        self.pred_obj_scores = kwargs.get("pred_obj_scores", False)
        self.pred_obj_scores_mlp = kwargs.get("pred_obj_scores_mlp", False)
        self.fixed_no_obj_ptr = kwargs.get("fixed_no_obj_ptr", False)
        self.use_no_obj_ptr = kwargs.get("use_no_obj_ptr", True)
        self.use_linear_no_obj_ptr = kwargs.get("use_linear_no_obj_ptr", False)
        if (
            self.pred_obj_scores
            and self.use_obj_ptrs_in_encoder
            and self.use_no_obj_ptr
        ):
            if self.use_linear_no_obj_ptr:
                self.no_obj_ptr_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            else:
                self.no_obj_ptr = _trunc_normal((self.multiplex_count, self.hidden_dim))

        self.use_mlp_for_obj_ptr_proj = kwargs.get("use_mlp_for_obj_ptr_proj", False)
        self.no_obj_embed_spatial = None
        if kwargs.get("no_obj_embed_spatial", False):
            self.no_obj_embed_spatial = _trunc_normal(
                (self.multiplex_count, self.hidden_dim)
            )
        self.num_multimask_outputs = kwargs.get("num_multimask_outputs", 3)
        self.decode_mask_with_shared_tokens = kwargs.get(
            "decode_mask_with_shared_tokens", False
        )
        self.decode_mask_attribute_with_shared_tokens = kwargs.get(
            "decode_mask_attribute_with_shared_tokens", False
        )
        self.share_necks = kwargs.get("share_necks", False)

        self.add_output_suppression_embeddings = kwargs.get(
            "add_output_suppression_embeddings", False
        )
        if self.add_output_suppression_embeddings:
            self.output_valid_embed = _trunc_normal(
                (self.multiplex_count, self.hidden_dim)
            )
            self.output_invalid_embed = _trunc_normal(
                (self.multiplex_count, self.hidden_dim)
            )
        self.add_object_conditional_embeddings = kwargs.get(
            "add_object_conditional_embeddings", False
        )
        add_object_unconditional_embeddings = kwargs.get(
            "add_object_unconditional_embeddings",
            self.add_object_conditional_embeddings,
        )
        self.add_object_unconditional_embeddings = add_object_unconditional_embeddings
        if self.add_object_conditional_embeddings:
            self.obj_cond_embed = _trunc_normal((self.multiplex_count, self.hidden_dim))
            if self.add_object_unconditional_embeddings:
                self.obj_non_cond_embed = _trunc_normal(
                    (self.multiplex_count, self.hidden_dim)
                )
        self.condition_as_mask_input = kwargs.get("condition_as_mask_input", False)
        self.condition_as_mask_input_fg = kwargs.get("condition_as_mask_input_fg", 1.0)
        self.condition_as_mask_input_bg = kwargs.get("condition_as_mask_input_bg", 0.0)
        self.is_dynamic_model = kwargs.get("is_dynamic_model", False)

        self._build_sam_heads()

        self.prob_to_use_pt_input_for_train = kwargs.get(
            "prob_to_use_pt_input_for_train", 0.0
        )
        self.prob_to_use_box_input_for_train = kwargs.get(
            "prob_to_use_box_input_for_train", 0.0
        )
        self.prob_to_use_pt_input_for_eval = kwargs.get(
            "prob_to_use_pt_input_for_eval", 0.0
        )
        self.prob_to_use_box_input_for_eval = kwargs.get(
            "prob_to_use_box_input_for_eval", 0.0
        )
        self.num_frames_to_correct_for_train = kwargs.get(
            "num_frames_to_correct_for_train", 1
        )
        self.num_frames_to_correct_for_eval = kwargs.get(
            "num_frames_to_correct_for_eval", 1
        )
        self.rand_frames_to_correct_for_train = kwargs.get(
            "rand_frames_to_correct_for_train", False
        )
        self.rand_frames_to_correct_for_eval = kwargs.get(
            "rand_frames_to_correct_for_eval", False
        )
        self.prob_correct_all_objects_for_train = kwargs.get(
            "prob_correct_all_objects_for_train", 0.0
        )
        self.ratio_of_objects_to_correct_for_train = kwargs.get(
            "ratio_of_objects_to_correct_for_train", 1.0
        )
        self.rand_objects_to_correct_for_train = kwargs.get(
            "rand_objects_to_correct_for_train", True
        )
        self.force_correct_all_for_conditional_inputs = kwargs.get(
            "force_correct_all_for_conditional_inputs", False
        )
        self.num_init_cond_frames_for_train = kwargs.get(
            "num_init_cond_frames_for_train", 1
        )
        self.num_init_cond_frames_for_eval = kwargs.get(
            "num_init_cond_frames_for_eval", 1
        )
        self.rand_init_cond_frames_for_train = kwargs.get(
            "rand_init_cond_frames_for_train", True
        )
        self.rand_init_cond_frames_for_eval = kwargs.get(
            "rand_init_cond_frames_for_eval", False
        )
        self.max_cond_frames_in_attn = kwargs.get("max_cond_frames_in_attn", -1)
        self.keep_first_cond_frame = kwargs.get("keep_first_cond_frame", False)
        self.add_all_frames_to_correct_as_cond = kwargs.get(
            "add_all_frames_to_correct_as_cond", False
        )
        self.num_correction_pt_per_frame = kwargs.get("num_correction_pt_per_frame", 7)
        self.pt_sampling_for_eval = kwargs.get("pt_sampling_for_eval", "center")
        self.prob_to_sample_from_gt_for_train = kwargs.get(
            "prob_to_sample_from_gt_for_train", 0.0
        )
        self.rng = np.random.default_rng(seed=42)
        self.rng2 = (
            np.random.default_rng(seed=42)
            if kwargs.get("randomness_fix", False)
            else self.rng
        )
        self.use_memory_selection = kwargs.get("use_memory_selection", False)
        self.mf_threshold = kwargs.get("mf_threshold", 0.01)
        self.compile_all_components = kwargs.get("compile_all_components", False)
        if self.compile_all_components:
            raise_unsupported_multiplex_runtime(
                "VideoTrackingMultiplex.compile_all_components"
            )

    def _build_sam_heads(self) -> None:
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.image_pe_layer = PositionEmbeddingRandom(self.hidden_dim // 2)
        self.interactive_sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.interactive_sam_mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.interactive_sam_mask_decoder_extra_args or {}),
        )
        self.sam_mask_decoder = MultiplexMaskDecoder(
            multiplex_count=self.multiplex_count,
            num_multimask_outputs=self.num_multimask_outputs,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.hidden_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            decode_mask_with_shared_tokens=self.decode_mask_with_shared_tokens,
            decode_mask_attribute_with_shared_tokens=(
                self.decode_mask_attribute_with_shared_tokens
            ),
            multimask_outputs_only=(
                self.num_multimask_outputs > 0 and self.multimask_output_in_sam
            ),
            **(self.sam_mask_decoder_extra_args or {}),
        )
        if self.use_obj_ptrs_in_encoder:
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    3,
                )
                self.interactive_obj_ptr_proj = MLP(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    3,
                )
            else:
                self.obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.interactive_obj_ptr_proj = nn.Linear(
                    self.hidden_dim,
                    self.hidden_dim,
                )
        else:
            self.obj_ptr_proj = nn.Identity()
            self.interactive_obj_ptr_proj = nn.Identity()
        self.obj_ptr_tpos_proj = (
            nn.Linear(self.hidden_dim, self.mem_dim)
            if self.proj_tpos_enc_in_obj_ptrs
            else nn.Identity()
        )

    def _maybe_clone(self, x: Any) -> Any:
        return mx.array(x) if _is_mlx_array(x) else np.array(x, copy=True)

    def get_propagation_dense_pe(self) -> mx.array:
        return self.image_pe_layer(
            (self.sam_image_embedding_size, self.sam_image_embedding_size)
        )[None, ...]

    def forward_image(
        self,
        img_batch: Any,
        *,
        need_sam3_out: bool = False,
        need_interactive_out: bool = False,
        need_propagation_out: bool = False,
    ) -> dict[str, Any]:
        """Run the image backbone and prepare SAM high-res features."""
        if self.backbone is None or not hasattr(self.backbone, "forward_image"):
            raise_unsupported_multiplex_runtime(
                "VideoTrackingMultiplex.forward_image(backbone)"
            )

        if self.share_necks:
            need_propagation_out = need_interactive_out or need_propagation_out
            need_interactive_out = False
            try:
                backbone_out = self.backbone.forward_image(
                    img_batch,
                    need_sam3_out=need_sam3_out,
                    need_sam2_out=need_propagation_out,
                )
            except TypeError:
                raise_unsupported_multiplex_runtime(
                    "VideoTrackingMultiplex.forward_image(share_necks)"
                )
            backbone_out["interactive"] = backbone_out["sam2_backbone_out"]
        else:
            backbone_out = self.backbone.forward_image(
                img_batch,
                need_sam3_out=need_sam3_out,
                need_interactive_out=need_interactive_out,
                need_propagation_out=need_propagation_out,
            )

        if self.use_high_res_features_in_sam:
            if need_interactive_out and "interactive" in backbone_out:
                interactive_fpn = list(backbone_out["interactive"]["backbone_fpn"])
                interactive_fpn[0] = _replace_feature_tensor(
                    interactive_fpn[0],
                    self.interactive_sam_mask_decoder.conv_s0(
                        _feature_tensor(interactive_fpn[0])
                    ),
                )
                interactive_fpn[1] = _replace_feature_tensor(
                    interactive_fpn[1],
                    self.interactive_sam_mask_decoder.conv_s1(
                        _feature_tensor(interactive_fpn[1])
                    ),
                )
                backbone_out["interactive"] = dict(backbone_out["interactive"])
                backbone_out["interactive"]["backbone_fpn"] = interactive_fpn
            if need_propagation_out and "sam2_backbone_out" in backbone_out:
                propagation_fpn = list(
                    backbone_out["sam2_backbone_out"]["backbone_fpn"]
                )
                propagation_fpn[0] = _replace_feature_tensor(
                    propagation_fpn[0],
                    self.sam_mask_decoder.conv_s0(_feature_tensor(propagation_fpn[0])),
                )
                propagation_fpn[1] = _replace_feature_tensor(
                    propagation_fpn[1],
                    self.sam_mask_decoder.conv_s1(_feature_tensor(propagation_fpn[1])),
                )
                backbone_out["sam2_backbone_out"] = dict(
                    backbone_out["sam2_backbone_out"]
                )
                backbone_out["sam2_backbone_out"]["backbone_fpn"] = propagation_fpn

        return backbone_out

    def _target_segments_as_masks(self, target: Any) -> Any:
        segments = target["segments"] if isinstance(target, dict) else target.segments
        if len(segments.shape) == 3:
            return mx.expand_dims(segments, axis=1)
        if len(segments.shape) == 4 and segments.shape[1] == 1:
            return segments
        raise ValueError("find_targets segments must have shape [B,H,W] or [B,1,H,W]")

    def _prepare_prompt_inputs_meta(
        self,
        backbone_out: dict[str, Any],
        input: Any,
        start_frame_idx: int = 0,
    ) -> dict[str, Any]:
        find_targets = getattr(input, "find_targets", None)
        if find_targets is None:
            raise ValueError("forward requires input.find_targets for mask prompts")

        gt_masks_per_frame = {
            stage_id: self._target_segments_as_masks(targets)
            for stage_id, targets in enumerate(find_targets)
        }
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = len(find_targets)
        backbone_out["num_frames"] = num_frames

        if getattr(self, "training", False):
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval

        if num_frames == 1:
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        if num_init_cond_frames < 1:
            raise ValueError("num_init_cond_frames must be at least 1")
        if start_frame_idx < 0 or start_frame_idx >= num_frames:
            raise ValueError("start_frame_idx must refer to an existing frame")

        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            num_init_cond_frames = int(self.rng.integers(1, num_init_cond_frames + 1))
        if num_init_cond_frames > num_frames - start_frame_idx:
            raise ValueError("num_init_cond_frames exceeds available frames")
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            num_frames_to_correct = int(
                self.rng.integers(num_init_cond_frames, num_frames_to_correct + 1)
            )
        backbone_out["use_pt_input"] = use_pt_input

        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]
        else:
            remaining = np.arange(start_frame_idx + 1, num_frames)
            if num_init_cond_frames - 1 > remaining.size:
                raise ValueError("not enough frames for init conditioning")
            sampled = self.rng.choice(
                remaining,
                num_init_cond_frames - 1,
                replace=False,
            )
            init_cond_frames = [start_frame_idx] + [int(x) for x in sampled]
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]

        if not use_pt_input:
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            if num_frames_to_correct <= num_init_cond_frames:
                raise ValueError(
                    "num_frames_to_correct must exceed num_init_cond_frames"
                )
            extra_num = num_frames_to_correct - num_init_cond_frames
            candidates = backbone_out["frames_not_in_init_cond"]
            if extra_num > len(candidates):
                raise ValueError("not enough non-conditioning frames to correct")
            extra = self.rng.choice(candidates, extra_num, replace=False)
            frames_to_add_correction_pt = init_cond_frames + [int(x) for x in extra]
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt
        if getattr(self, "is_dynamic_vos_evaluation", False) and not getattr(
            self,
            "training",
            False,
        ):
            self._prepare_dynamic_vos_eval_prompt_inputs(
                backbone_out,
                input,
                start_frame_idx=start_frame_idx,
            )
        return backbone_out

    def _prepare_dynamic_vos_eval_prompt_inputs(
        self,
        backbone_out: dict[str, Any],
        input: Any,
        *,
        start_frame_idx: int,
    ) -> None:
        visible_objects_per_frame = getattr(input, "visible_objects_per_frame", None)
        if visible_objects_per_frame is None:
            raise ValueError(
                "dynamic VOS evaluation requires input.visible_objects_per_frame."
            )

        num_frames = backbone_out["num_frames"]
        gt_masks_per_frame = backbone_out["gt_masks_per_frame"]
        init_cond_frames = sorted(backbone_out["init_cond_frames"])
        frames_not_in_init_cond = list(backbone_out["frames_not_in_init_cond"])

        if len(visible_objects_per_frame.get(start_frame_idx, set())) == 0:
            if len(init_cond_frames) != 1:
                raise ValueError(
                    "empty dynamic VOS start frame requires one initial cond frame."
                )
            for stage_id in range(start_frame_idx, num_frames):
                if len(visible_objects_per_frame.get(stage_id, set())) > 0:
                    init_cond_frames = [stage_id]
                    break
            for stage_id in range(init_cond_frames[0] + 1):
                if stage_id in frames_not_in_init_cond:
                    frames_not_in_init_cond.remove(stage_id)
            backbone_out["init_cond_frames"] = init_cond_frames
            backbone_out["frames_not_in_init_cond"] = frames_not_in_init_cond

        valid_idx_per_frame: dict[int, list[int]] = {}
        valid_idx_prior_to_each_transition: dict[int, list[int]] = {}
        new_idx_per_transition: dict[int, list[int]] = {}

        object_appearance_order: list[int] = []
        object_appear_at_stage: dict[int, int] = {}
        transition_points: list[int] = []
        stage_to_new_objects: dict[int, list[int]] = {}
        for stage_id in range(start_frame_idx, num_frames):
            visible_objects = sorted(
                int(obj_id) for obj_id in visible_objects_per_frame.get(stage_id, set())
            )
            for obj_id in visible_objects:
                if obj_id in object_appear_at_stage:
                    continue
                object_appear_at_stage[obj_id] = stage_id
                object_appearance_order.append(obj_id)
                stage_to_new_objects.setdefault(stage_id, []).append(obj_id)
                if stage_id not in init_cond_frames:
                    transition_points.append(stage_id)

        objects_seen_so_far: list[int] = []
        for stage_id in range(start_frame_idx, num_frames):
            if stage_id in transition_points:
                new_objects = stage_to_new_objects.get(stage_id, [])
                num_objects_before = len(objects_seen_so_far)
                valid_idx_prior_to_each_transition[stage_id] = list(
                    range(num_objects_before)
                )
                new_idx_per_transition[stage_id] = list(
                    range(num_objects_before, num_objects_before + len(new_objects))
                )
                objects_seen_so_far.extend(new_objects)

            if stage_id in init_cond_frames:
                valid_idx_per_frame[stage_id] = list(
                    range(len(stage_to_new_objects.get(stage_id, [])))
                )
                objects_seen_so_far.extend(stage_to_new_objects.get(stage_id, []))
            else:
                valid_idx_per_frame[stage_id] = list(range(len(objects_seen_so_far)))

        for stage_id in range(start_frame_idx, num_frames):
            remapped = _take_rows(gt_masks_per_frame[stage_id], object_appearance_order)
            gt_masks_per_frame[stage_id] = _take_rows(
                remapped,
                valid_idx_per_frame[stage_id],
            )

        for stage_id, targets in enumerate(input.find_targets):
            if stage_id < start_frame_idx or stage_id >= num_frames:
                continue
            if stage_id in transition_points:
                prev_objects = valid_idx_prior_to_each_transition[stage_id]
                target_masks = _take_rows(gt_masks_per_frame[stage_id], prev_objects)
            else:
                target_masks = gt_masks_per_frame[stage_id]
            targets.segments = target_masks.squeeze(1)
            if hasattr(targets, "num_boxes"):
                targets.num_boxes = targets.num_boxes[: targets.segments.shape[0]]

        backbone_out["valid_idx_per_frame"] = valid_idx_per_frame
        backbone_out["new_idx_per_transition"] = new_idx_per_transition
        backbone_out["valid_objects_prior_to_each_transition"] = (
            valid_idx_prior_to_each_transition
        )
        backbone_out["transition_points"] = set(transition_points)
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        backbone_out["object_appearance_order"] = object_appearance_order

    def _prepare_conditional_frames(
        self,
        backbone_out: dict[str, Any],
    ) -> dict[str, Any]:
        init_cond_frames = backbone_out["init_cond_frames"]
        gt_masks_per_frame = backbone_out["gt_masks_per_frame"]
        use_pt_input = backbone_out["use_pt_input"]

        prob_to_use_box_input = (
            self.prob_to_use_box_input_for_train
            if getattr(self, "training", False)
            else self.prob_to_use_box_input_for_eval
        )

        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
                continue

            if self.rng.random() < prob_to_use_box_input:
                points, labels = sample_box_points(gt_masks_per_frame[t])
            else:
                points, labels = get_next_point(
                    gt_masks=gt_masks_per_frame[t],
                    pred_masks=None,
                    method=(
                        "uniform"
                        if getattr(self, "training", False)
                        else self.pt_sampling_for_eval
                    ),
                )
            backbone_out["point_inputs_per_frame"][t] = {
                "point_coords": points,
                "point_labels": labels,
            }
        return backbone_out

    def prepare_prompt_inputs(
        self,
        backbone_out: dict[str, Any],
        input: Any,
        start_frame_idx: int = 0,
    ) -> dict[str, Any]:
        backbone_out = self._prepare_prompt_inputs_meta(
            backbone_out,
            input,
            start_frame_idx,
        )
        return self._prepare_conditional_frames(backbone_out)

    def _prepare_backbone_features(
        self,
        backbone_out: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Flatten precomputed neck features into ``(HW, B, C)`` form."""
        backbone_features = {}
        for neck_k in neck_outs:
            if neck_k not in backbone_out:
                continue
            neck_out = backbone_out[neck_k]
            if len(neck_out["backbone_fpn"]) != len(neck_out["vision_pos_enc"]):
                raise ValueError("backbone_fpn and vision_pos_enc length mismatch")
            if len(neck_out["backbone_fpn"]) < self.num_feature_levels:
                raise ValueError("not enough feature levels for multiplex tracking")

            feature_maps = neck_out["backbone_fpn"][-self.num_feature_levels :]
            pos_maps = neck_out["vision_pos_enc"][-self.num_feature_levels :]
            feature_tensors = [_feature_tensor(x) for x in feature_maps]
            feat_sizes = [(x.shape[-2], x.shape[-1]) for x in pos_maps]
            vision_feats = [
                x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
                for x in feature_tensors
            ]
            vision_pos_embeds = [
                x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
                for x in pos_maps
            ]
            vision_masks = [_feature_mask(x) for x in feature_maps]
            vision_masks = [
                mask.reshape(mask.shape[0], -1) if mask is not None else None
                for mask in vision_masks
            ]

            backbone_features[neck_k] = {
                "vision_feats": vision_feats,
                "vision_masks": vision_masks,
                "vision_pos_embeds": vision_pos_embeds,
                "feat_sizes": feat_sizes,
            }
        return backbone_features

    def _prepare_backbone_features_per_frame(
        self,
        img_batch: Any,
        img_ids: Any,
        *,
        need_interactive_out: bool = False,
        need_propagation_out: bool = False,
    ) -> tuple[Any, dict[str, dict[str, Any]]]:
        """Compute and flatten backbone features for a batch of identical image ids (one frame)."""
        img_ids_np = _to_numpy(img_ids).reshape(-1).astype(np.int64)
        if img_ids_np.size == 0:
            raise ValueError("img_ids must contain at least one image id")
        if not np.all(img_ids_np == img_ids_np[0]):
            raise ValueError("all image ids for a multiplex stage must match")
        unique_img_ids = np.array([int(img_ids_np[0])], dtype=np.int64)

        image = _take_image_batch(img_batch, unique_img_ids)
        backbone_out = self.forward_image(
            image,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )
        return image, self._prepare_backbone_features(backbone_out)

    def _get_tpos_enc(
        self,
        rel_pos_list: list[int],
        device: Any = None,
        max_abs_pos: int | None = None,
        dummy: bool = False,
    ) -> mx.array:
        del device
        if dummy:
            return mx.zeros((len(rel_pos_list), self.mem_dim), dtype=mx.float32)
        if not self.sincos_tpos_enc:
            raise_unsupported_multiplex_runtime(
                "VideoTrackingMultiplex._get_tpos_enc(sincos_tpos_enc=False)"
            )

        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        pos_enc = mx.array(rel_pos_list, dtype=mx.float32) / t_diff_max
        tpos_dim = self.hidden_dim if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
        pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
        return self.obj_ptr_tpos_proj(pos_enc)

    def _get_interactive_pix_mem(
        self,
        features: list[Any],
        feat_sizes: list[tuple[int, int]],
    ) -> Any:
        if not self.directly_add_no_mem_embed:
            raise ValueError("directly_add_no_mem_embed is required for interactivity")
        pix_feat_with_mem = features[-1] + self.interactivity_no_mem_embed
        B = features[-1].shape[1]
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        return pix_feat_with_mem.transpose(1, 2, 0).reshape(B, C, H, W)

    def cal_mem_score(self, object_score_logits: Any, iou_score: Any) -> Any:
        object_score_norm = mx.where(
            object_score_logits > 0,
            mx.sigmoid(object_score_logits) * 2 - 1,
            mx.zeros_like(object_score_logits),
        )
        return (object_score_norm * iou_score).mean()

    def frame_filter(
        self,
        output_dict: dict[str, dict[int, StageOutput]],
        track_in_reverse: bool,
        frame_idx: int,
        num_frames: int,
        r: int,
    ) -> list[int]:
        if (frame_idx == 0 and not track_in_reverse) or (
            frame_idx == num_frames - 1 and track_in_reverse
        ):
            return []

        max_num = min(num_frames, self.max_obj_ptrs_in_encoder)
        if not track_in_reverse:
            start = frame_idx - 1
            end = 0
            step = -r
            must_include = frame_idx - 1
        else:
            start = frame_idx + 1
            end = num_frames
            step = r
            must_include = frame_idx + 1

        valid_indices: list[int] = []
        for i in range(start, end, step):
            out = output_dict["non_cond_frame_outputs"].get(i)
            if out is None or "eff_iou_score" not in out:
                continue
            if _to_scalar(out["eff_iou_score"]) > self.mf_threshold:
                valid_indices.insert(0, i)
            if len(valid_indices) >= max_num - 1:
                break

        if must_include not in valid_indices:
            valid_indices.append(must_include)
        return valid_indices

    def _broadcast_to_buckets(self, value: Any, num_buckets: int) -> Any:
        if value.shape[1] == num_buckets:
            return value
        if value.shape[1] != 1:
            raise ValueError(
                "multiplex frame features must have batch 1 or num_buckets: "
                f"{value.shape[1]} != 1 or {num_buckets}"
            )
        return mx.broadcast_to(value, (value.shape[0], num_buckets, value.shape[2]))

    def _prepare_memory_conditioned_features(
        self,
        *,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: list[Any],
        current_vision_masks: list[Any | None],
        current_vision_pos_embeds: list[Any],
        feat_sizes: list[tuple[int, int]],
        output_dict: dict[str, dict[int, StageOutput]],
        num_frames: int | None,
        track_in_reverse: bool = False,
        use_prev_mem_frame: bool = True,
        multiplex_state: MultiplexState,
    ) -> Any:
        """Fuse current-frame visual features with previous multiplex memories."""
        B = multiplex_state.num_buckets
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        vision_feat = self._broadcast_to_buckets(current_vision_feats[-1], B)
        vision_pos_embed = self._broadcast_to_buckets(current_vision_pos_embeds[-1], B)

        vision_mask = current_vision_masks[-1]
        if vision_mask is not None:
            if vision_mask.shape[0] == 1:
                vision_mask = mx.broadcast_to(vision_mask, (B, vision_mask.shape[1]))
            elif vision_mask.shape[0] != B:
                raise ValueError(
                    "multiplex vision mask must have batch 1 or num_buckets"
                )

        if self.num_maskmem == 0:
            return vision_feat.transpose(1, 2, 0).reshape(B, C, H, W)

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if is_init_cond_frame or not use_prev_mem_frame:
            raise_unsupported_multiplex_runtime(
                "VideoTrackingMultiplex._prepare_memory_conditioned_features(init-or-no-prev-memory)"
            )

        to_cat_prompt: list[Any] = []
        to_cat_prompt_pos_embed: list[Any] = []
        to_cat_image_feat: list[Any] = []
        to_cat_image_pos_embed: list[Any] = []
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise ValueError("at least one conditioning frame is required")

        cond_outputs = output_dict["cond_frame_outputs"]
        selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
            frame_idx,
            cond_outputs,
            self.max_cond_frames_in_attn,
            keep_first_cond_frame=self.keep_first_cond_frame,
        )
        t_pos_and_prevs = [
            ((frame_idx - t) * tpos_sign_mul, out, True)
            for t, out in selected_cond_outputs.items()
        ]

        r = (
            1
            if getattr(self, "training", False)
            else self.memory_temporal_stride_for_eval
        )
        valid_indices: list[int] = []
        if self.use_memory_selection:
            if num_frames is None:
                raise ValueError(
                    "num_frames is required when use_memory_selection=True"
                )
            valid_indices = self.frame_filter(
                output_dict,
                track_in_reverse,
                frame_idx,
                num_frames,
                r,
            )

        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if self.use_memory_selection:
                if t_rel > len(valid_indices):
                    continue
                prev_frame_idx = valid_indices[-t_rel]
            elif t_rel == 1:
                prev_frame_idx = (
                    frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
                )
            elif not track_in_reverse:
                prev_frame_idx = ((frame_idx - 2) // r) * r
                prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
            else:
                prev_frame_idx = -(-(frame_idx + 2) // r) * r
                prev_frame_idx = prev_frame_idx + (t_rel - 2) * r

            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx)
            if out is None:
                out = unselected_cond_outputs.get(prev_frame_idx)
            t_pos_and_prevs.append((t_pos, out, False))

        for t_pos, prev, is_selected_cond_frame in t_pos_and_prevs:
            if prev is None:
                continue
            feats = prev.get("maskmem_features")
            maskmem_pos_list = prev.get("maskmem_pos_enc")
            if feats is None or not maskmem_pos_list:
                continue
            maskmem_enc = maskmem_pos_list[-1]
            if maskmem_enc is None or feats.shape[0] == 0:
                continue

            if len(feats.shape) == 5:
                feats = multiplex_state.demux(feats)
                prev["maskmem_features"] = feats
            if len(maskmem_enc.shape) == 5:
                maskmem_enc = multiplex_state.demux(maskmem_enc)
                prev["maskmem_pos_enc"][-1] = maskmem_enc

            seq_len = feats.shape[-2] * feats.shape[-1]
            to_cat_prompt.append(
                feats.reshape(feats.shape[0], self.mem_dim, seq_len).transpose(2, 0, 1)
            )
            maskmem_enc = maskmem_enc.reshape(
                maskmem_enc.shape[0], self.mem_dim, seq_len
            ).transpose(2, 0, 1)

            if self.use_maskmem_tpos_v2:
                if t_pos <= 0 or t_pos >= self.num_maskmem:
                    tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - 1]
                else:
                    tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
            else:
                t = t_pos if not is_selected_cond_frame else 0
                tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t - 1]
            to_cat_prompt_pos_embed.append(maskmem_enc + tpos_enc)

            if self.save_image_features:
                image_feat = prev.get("image_features")
                image_pos_embed = prev.get("image_pos_enc")
                if image_feat is None or image_pos_embed is None:
                    to_cat_prompt.pop()
                    to_cat_prompt_pos_embed.pop()
                    continue
                to_cat_image_feat.append(image_feat)
                to_cat_image_pos_embed.append(image_pos_embed + tpos_enc)

        if self.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = (
                self.max_obj_ptrs_in_encoder
                if num_frames is None
                else min(num_frames, self.max_obj_ptrs_in_encoder)
            )
            if (
                not getattr(self, "training", False)
                and self.only_obj_ptrs_in_the_past_for_eval
            ):
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs

            pos_and_outs_for_ptr = [
                (
                    (
                        (frame_idx - t) * tpos_sign_mul
                        if self.use_signed_tpos_enc_to_obj_ptrs
                        else abs(frame_idx - t)
                    ),
                    out,
                    True,
                )
                for t, out in ptr_cond_outputs.items()
            ]

            for t_diff in range(1, max_obj_ptrs_in_encoder):
                if not self.use_memory_selection:
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                else:
                    if -t_diff <= -len(valid_indices):
                        break
                    t = valid_indices[-t_diff]
                out = output_dict["non_cond_frame_outputs"].get(
                    t,
                    unselected_cond_outputs.get(t),
                )
                if out is not None:
                    pos_and_outs_for_ptr.append((t_diff, out, False))

            filtered_ptrs = [
                (pos, out, is_cond)
                for pos, out, is_cond in pos_and_outs_for_ptr
                if "obj_ptr" in out
            ]
            if filtered_ptrs:
                pos_list, out_list, _ = zip(*filtered_ptrs)
                obj_ptrs = mx.concat([out["obj_ptr"] for out in out_list], axis=1)
                obj_ptrs = obj_ptrs.transpose(1, 0, 2)
                if self.add_tpos_enc_to_obj_ptrs:
                    obj_pos = self._get_tpos_enc(
                        list(pos_list),
                        max_abs_pos=max_obj_ptrs_in_encoder,
                    )
                else:
                    obj_pos = self._get_tpos_enc(list(pos_list), dummy=True)
                obj_pos = mx.broadcast_to(
                    obj_pos[:, None, :], (obj_pos.shape[0], B, obj_pos.shape[1])
                )
                if self.mem_dim != C:
                    raise ValueError("multiplex object pointers require mem_dim == C")
                obj_pos = mx.repeat(obj_pos, multiplex_state.multiplex_count, axis=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]

        if len(to_cat_prompt) == 0:
            return vision_feat.transpose(1, 2, 0).reshape(B, C, H, W)

        prompt = mx.concat(to_cat_prompt, axis=0)
        prompt_pos_embed = mx.concat(to_cat_prompt_pos_embed, axis=0)
        if self.save_image_features:
            if vision_mask is not None:
                raise ValueError(
                    "save_image_features memory fusion does not support vision masks"
                )
            if len(to_cat_image_feat) == 0 or len(to_cat_image_pos_embed) == 0:
                return vision_feat.transpose(1, 2, 0).reshape(B, C, H, W)
            image_feat = mx.concat(to_cat_image_feat, axis=0)
            image_pos_embed = mx.concat(to_cat_image_pos_embed, axis=0)
            encoder_out = self.transformer.encoder(
                image=current_vision_feats[-1],
                src=vision_feat,
                memory_image=image_feat,
                memory=prompt,
                image_pos=current_vision_pos_embeds[-1],
                src_pos=vision_pos_embed,
                memory_image_pos=image_pos_embed,
                memory_pos=prompt_pos_embed,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
        else:
            encoder_out = self.transformer.encoder(
                src=vision_feat,
                src_key_padding_mask=vision_mask,
                src_pos=vision_pos_embed,
                prompt=prompt,
                prompt_pos=prompt_pos_embed,
                prompt_key_padding_mask=None,
                feat_sizes=feat_sizes,
                num_obj_ptr_tokens=num_obj_ptr_tokens,
            )
        return encoder_out["memory"].transpose(1, 2, 0).reshape(B, C, H, W)

    def _apply_non_overlapping_constraints(self, pred_masks: mx.array) -> mx.array:
        batch_size = pred_masks.shape[0]
        if batch_size == 1:
            return pred_masks

        max_obj_inds = mx.argmax(pred_masks, axis=0).reshape(
            (1,) + pred_masks.shape[1:]
        )
        batch_obj_inds = mx.arange(batch_size).reshape(batch_size, 1, 1, 1)
        keep = max_obj_inds == batch_obj_inds
        return mx.where(
            keep,
            pred_masks,
            mx.minimum(pred_masks, mx.array(-10.0, dtype=pred_masks.dtype)),
        )

    def _apply_object_wise_non_overlapping_constraints(
        self,
        pred_masks: mx.array,
        obj_scores: mx.array,
        background_value: float = -10.0,
    ) -> mx.array:
        pred_masks_single_score = mx.where(
            pred_masks > 0,
            obj_scores[..., None, None],
            mx.array(background_value, dtype=obj_scores.dtype),
        )
        pixel_level_non_overlapping_masks = self._apply_non_overlapping_constraints(
            pred_masks_single_score
        )
        return mx.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            mx.minimum(pred_masks, mx.array(background_value, dtype=pred_masks.dtype)),
        )

    def _encode_new_memory(
        self,
        image: Any,
        current_vision_feats: list[Any],
        feat_sizes: list[tuple[int, int]],
        pred_masks_high_res: Any,
        object_score_logits: Any,
        is_mask_from_pts: bool,
        *,
        conditioning_objects: set[int] | None = None,
        multiplex_state: MultiplexState,
    ) -> tuple[Any, list[Any]]:
        """Encode the current image and its multiplexed masks into memory."""
        if current_vision_feats is None or feat_sizes is None:
            raise ValueError(
                "current_vision_feats and feat_sizes are required for memory encoding"
            )

        B = current_vision_feats[-1].shape[1]
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].transpose(1, 2, 0).reshape(B, C, H, W)

        if self.non_overlap_masks_for_mem_enc and not getattr(self, "training", False):
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )

        if self.apply_sigmoid_to_mask_logits_for_mem_enc:
            assert not self.binarize_mask_from_pts_for_mem_enc, (
                "haven't been trained this way; beware of hardcoded config override"
            )
            binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
            if binarize and not getattr(self, "training", False):
                mask_for_mem = (pred_masks_high_res > 0).astype(mx.float32)
            else:
                mask_for_mem = mx.sigmoid(pred_masks_high_res)
            if self.sigmoid_scale_for_mem_enc != 1.0:
                mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
            if self.sigmoid_bias_for_mem_enc != 0.0:
                mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        else:
            mask_for_mem = pred_masks_high_res

        if self.add_object_conditional_embeddings or self.condition_as_mask_input:
            if conditioning_objects is None:
                conditioning_objects = set()
                unconditioning_objects = sorted(
                    multiplex_state.get_all_valid_object_idx()
                )
            else:
                conditioning_objects = set(conditioning_objects)
                all_objects_idx = multiplex_state.get_all_valid_object_idx()
                unconditioning_objects = sorted(
                    [idx for idx in all_objects_idx if idx not in conditioning_objects]
                )

        mux_mask_for_mem = multiplex_state.mux(mask_for_mem)
        if mux_mask_for_mem.shape[2] != 1:
            raise ValueError("mask_for_mem must have a singleton channel dimension")
        mux_mask_for_mem = mux_mask_for_mem.reshape(
            mux_mask_for_mem.shape[:2] + mux_mask_for_mem.shape[3:]
        )

        if self.condition_as_mask_input:
            cond_values_np = np.full(
                (mask_for_mem.shape[0],),
                self.condition_as_mask_input_bg,
                dtype=np.float32,
            )
            if conditioning_objects:
                cond_values_np[list(conditioning_objects)] = (
                    self.condition_as_mask_input_fg
                )
            cond_values = mx.array(cond_values_np, dtype=mask_for_mem.dtype)
            embedded_conditions = cond_values.reshape(-1, 1, 1, 1) * mx.ones_like(
                mask_for_mem
            )
            embedded_conditions = multiplex_state.mux(embedded_conditions)
            embedded_conditions = embedded_conditions.reshape(
                embedded_conditions.shape[:2] + embedded_conditions.shape[3:]
            )
            mux_mask_for_mem = mx.concat(
                [mux_mask_for_mem, embedded_conditions],
                axis=1,
            )

        if isinstance(self.maskmem_backbone, SimpleMaskEncoder):
            maskmem_out = self.maskmem_backbone(
                pix_feat,
                mux_mask_for_mem,
                skip_mask_sigmoid=True,
            )
        else:
            maskmem_out = self.maskmem_backbone(image, pix_feat, mux_mask_for_mem)

        maskmem_features = self._maybe_clone(maskmem_out["vision_features"])
        maskmem_pos_enc = [self._maybe_clone(m) for m in maskmem_out["vision_pos_enc"]]

        if self.no_obj_embed_spatial is not None:
            if object_score_logits is None:
                raise ValueError(
                    "object_score_logits are required when no_obj_embed_spatial is set"
                )
            obj_expected = multiplex_state.total_valid_entries
            obj_current = object_score_logits.shape[0]
            if obj_current != obj_expected:
                if obj_current < obj_expected:
                    pad_shape = (obj_expected - obj_current,) + tuple(
                        object_score_logits.shape[1:]
                    )
                    obj_pad = mx.zeros(pad_shape, dtype=object_score_logits.dtype)
                    object_score_logits = mx.concat(
                        [object_score_logits, obj_pad],
                        axis=0,
                    )
                else:
                    object_score_logits = object_score_logits[:obj_expected]

            object_score_logits = multiplex_state.mux(object_score_logits)
            is_obj_appearing = (
                object_score_logits > self.object_score_logit_threshold
            ).astype(mx.float32)
            no_obj_embed_spatial = mx.broadcast_to(
                self.no_obj_embed_spatial[None, :, :],
                (
                    multiplex_state.num_buckets,
                    self.no_obj_embed_spatial.shape[0],
                    self.no_obj_embed_spatial.shape[1],
                ),
            )
            no_obj_embed = ((1 - is_obj_appearing) * no_obj_embed_spatial).sum(axis=1)
            maskmem_features = maskmem_features + no_obj_embed[..., None, None]

        if self.add_object_conditional_embeddings:
            obj_cond_embed = mx.broadcast_to(
                self.obj_cond_embed[None, :, :],
                (
                    multiplex_state.num_buckets,
                    self.obj_cond_embed.shape[0],
                    self.obj_cond_embed.shape[1],
                ),
            )
            obj_cond_embed = multiplex_state.demux(obj_cond_embed)
            obj_merged_embed = obj_cond_embed

            if self.add_object_unconditional_embeddings:
                obj_non_cond_embed = mx.broadcast_to(
                    self.obj_non_cond_embed[None, :, :],
                    (
                        multiplex_state.num_buckets,
                        self.obj_non_cond_embed.shape[0],
                        self.obj_non_cond_embed.shape[1],
                    ),
                )
                obj_non_cond_embed = multiplex_state.demux(obj_non_cond_embed)
                if unconditioning_objects:
                    obj_merged_embed = _replace_rows(
                        obj_merged_embed,
                        unconditioning_objects,
                        obj_non_cond_embed[unconditioning_objects],
                    )

            obj_merged_embed = multiplex_state.mux(obj_merged_embed).sum(axis=1)
            maskmem_features = maskmem_features + obj_merged_embed[..., None, None]

        if len(maskmem_features.shape) == 5:
            maskmem_features = multiplex_state.demux(maskmem_features)

        demuxed_pos_enc = []
        for pos_enc in maskmem_pos_enc:
            if pos_enc is not None and len(pos_enc.shape) == 5:
                pos_enc = multiplex_state.demux(pos_enc)
            demuxed_pos_enc.append(pos_enc)

        return maskmem_features, demuxed_pos_enc

    def _use_mask_as_output(
        self,
        backbone_features: Any,
        high_res_features: list[Any],
        mask_inputs: Any,
        multiplex_state: MultiplexState,
        objects_in_mask: list[int] | None = None,
    ) -> SAMOutput:
        """
        Directly convert binary mask inputs into SAM-style mask logits.

        The no-object-pointer path is MLX-native. Object-pointer extraction
        still requires the multiplex SAM-head forward and remains explicit.
        """
        if len(mask_inputs.shape) != 4 or mask_inputs.shape[1] != 1:
            raise ValueError("mask_inputs must have shape [N, 1, H, W].")
        if objects_in_mask is None:
            objects_in_mask = list(range(multiplex_state.total_valid_entries))
        if mask_inputs.shape[0] != len(objects_in_mask):
            raise ValueError(
                "mask_inputs batch must match objects_in_mask length: "
                f"{mask_inputs.shape[0]} != {len(objects_in_mask)}"
            )

        out_scale, out_bias = 20.0, -10.0
        dtype = getattr(backbone_features, "dtype", mx.float32)
        mask_inputs_float = mask_inputs.astype(dtype)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = interpolate(
            high_res_masks,
            size=(
                max(1, high_res_masks.shape[-2] // 4),
                max(1, high_res_masks.shape[-1] // 4),
            ),
            align_corners=False,
            mode="bilinear",
        )
        ious = mx.ones((mask_inputs.shape[0], 1), dtype=dtype)

        is_obj_appearing = mx.any(
            mask_inputs.reshape(mask_inputs.shape[0], -1).astype(mx.float32) > 0.0,
            axis=1,
        )[:, None]
        lambda_is_obj_appearing = is_obj_appearing.astype(dtype)
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias

        outputs: SAMOutput = {
            "low_res_multimasks": low_res_masks,
            "high_res_multimasks": high_res_masks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
        }

        if self.use_obj_ptrs_in_encoder:
            forward_sam_heads = getattr(self, "_forward_sam_heads", None)
            if forward_sam_heads is None:
                raise_unsupported_multiplex_runtime(
                    "VideoTrackingMultiplex._use_mask_as_output(obj_ptrs)"
                )
            sam_outputs = forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.interactive_mask_downsample(mask_inputs_float),
                interactive_high_res_features=high_res_features,
                gt_masks=mask_inputs,
                objects_to_interact=objects_in_mask,
                multiplex_state=multiplex_state,
            )
            obj_ptr = sam_outputs["obj_ptr"]

            if self.pred_obj_scores and self.use_no_obj_ptr:
                if self.use_linear_no_obj_ptr:
                    obj_ptr = lambda_is_obj_appearing * obj_ptr + (
                        1 - lambda_is_obj_appearing
                    ) * self.no_obj_ptr_linear(obj_ptr)
                else:
                    if self.fixed_no_obj_ptr:
                        obj_ptr = lambda_is_obj_appearing * obj_ptr
                    selected_no_obj_ptr = mx.broadcast_to(
                        self.no_obj_ptr[None, :, :],
                        (
                            multiplex_state.num_buckets,
                            self.no_obj_ptr.shape[0],
                            self.no_obj_ptr.shape[1],
                        ),
                    )
                    selected_no_obj_ptr = multiplex_state.demux(selected_no_obj_ptr)
                    selected_no_obj_ptr = selected_no_obj_ptr[objects_in_mask]
                    obj_ptr = (
                        obj_ptr + (1 - lambda_is_obj_appearing) * selected_no_obj_ptr
                    )

            outputs["obj_ptr"] = obj_ptr

        return outputs

    def _forward_sam_heads(
        self,
        backbone_features: Any,
        *,
        point_inputs: dict[str, Any] | None = None,
        mask_inputs: Any | None = None,
        interactive_high_res_features: list[Any] | None = None,
        propagation_high_res_features: list[Any] | None = None,
        multimask_output: bool = False,
        gt_masks: Any = None,
        multiplex_state: MultiplexState,
        objects_to_interact: list[int] | None = None,
    ) -> SAMOutput:
        """Forward the interactive SAM prompt encoder and mask decoder path."""
        del gt_masks
        image_batch = backbone_features.shape[0]
        if backbone_features.shape[1] != self.sam_prompt_embed_dim:
            raise ValueError("backbone feature channels must match prompt embed dim")
        if backbone_features.shape[2] != self.sam_image_embedding_size:
            raise ValueError("backbone feature height must match SAM image embedding")
        if backbone_features.shape[3] != self.sam_image_embedding_size:
            raise ValueError("backbone feature width must match SAM image embedding")

        is_interactive = point_inputs is not None or mask_inputs is not None
        if is_interactive:
            if interactive_high_res_features is None:
                raise ValueError("interactive_high_res_features are required")
            if objects_to_interact is None:
                raise ValueError("objects_to_interact are required")

            if point_inputs is not None:
                sam_point_coords = point_inputs["point_coords"]
                sam_point_labels = point_inputs["point_labels"]
                prompt_batch = sam_point_coords.shape[0]
                if sam_point_labels.shape[0] != prompt_batch:
                    raise ValueError(
                        "point prompt labels must match point coords batch"
                    )
            else:
                if mask_inputs is None:
                    raise ValueError("interactive inference requires points or masks")
                prompt_batch = mask_inputs.shape[0]
                sam_point_coords = mx.zeros((prompt_batch, 1, 2), dtype=mx.float32)
                sam_point_labels = -mx.ones((prompt_batch, 1), dtype=mx.int32)

            if image_batch not in (1, prompt_batch):
                raise ValueError(
                    "interactive image batch must be 1 or match prompt batch"
                )
            repeat_image = image_batch == 1 and prompt_batch != 1

            if mask_inputs is not None:
                if len(mask_inputs.shape) != 4 or tuple(mask_inputs.shape[:2]) != (
                    prompt_batch,
                    1,
                ):
                    raise ValueError(
                        "mask_inputs must have shape [prompt_batch, 1, H, W]"
                    )
                if tuple(mask_inputs.shape[-2:]) != tuple(
                    self.interactive_sam_prompt_encoder.mask_input_size
                ):
                    sam_mask_prompt = interpolate(
                        mask_inputs.astype(mx.float32),
                        size=self.interactive_sam_prompt_encoder.mask_input_size,
                        align_corners=False,
                        mode="bilinear",
                    )
                else:
                    sam_mask_prompt = mask_inputs
            else:
                sam_mask_prompt = None

            sparse_embeddings, dense_embeddings = self.interactive_sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels),
                boxes=None,
                masks=sam_mask_prompt,
            )
            sparse_embeddings = self._maybe_clone(sparse_embeddings)
            dense_embeddings = self._maybe_clone(dense_embeddings)
            image_pe = self._maybe_clone(
                self.interactive_sam_prompt_encoder.get_dense_pe()
            )
            (
                low_res_multimasks,
                ious,
                sam_output_tokens,
                object_score_logits,
            ) = self.interactive_sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=repeat_image,
                high_res_features=interactive_high_res_features,
            )
        else:
            if propagation_high_res_features is None:
                raise ValueError("propagation_high_res_features are required")
            if self.add_output_suppression_embeddings:
                output_valid_embed = self.output_valid_embed[None, :, :]
                output_invalid_embed = self.output_invalid_embed[None, :, :]
                valid_object_mask = multiplex_state.get_valid_object_mask()[
                    ..., None
                ].astype(mx.float32)
                output_merged_embed = (
                    valid_object_mask * output_valid_embed
                    + (1 - valid_object_mask) * output_invalid_embed
                )
            else:
                output_merged_embed = None

            image_pe = self._maybe_clone(self.get_propagation_dense_pe())
            out = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=image_pe,
                high_res_features=propagation_high_res_features,
                multimask_output=multimask_output,
                extra_per_object_embeddings=output_merged_embed,
            )
            low_res_multimasks = multiplex_state.demux(out["masks"])
            ious = multiplex_state.demux(out["iou_pred"])
            sam_output_tokens = multiplex_state.demux(out["sam_tokens_out"])
            object_score_logits = multiplex_state.demux(out["object_score_logits"])

        low_res_multimasks = self._maybe_clone(low_res_multimasks)
        ious = self._maybe_clone(ious)
        object_score_logits = self._maybe_clone(object_score_logits)
        sam_output_tokens = self._maybe_clone(sam_output_tokens)

        is_obj_appearing = None
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.object_score_logit_threshold
            if len(is_obj_appearing.shape) > 1 and is_obj_appearing.shape[-1] == 1:
                is_obj_appearing = is_obj_appearing.reshape(is_obj_appearing.shape[:-1])
            appearing_mask_shape = (is_obj_appearing.shape[0],) + (1,) * (
                len(low_res_multimasks.shape) - 1
            )
            low_res_multimasks = mx.where(
                is_obj_appearing.reshape(appearing_mask_shape),
                low_res_multimasks,
                mx.array(NO_OBJ_SCORE, dtype=low_res_multimasks.dtype),
            )

        low_res_multimasks = low_res_multimasks.astype(mx.float32)
        high_res_multimasks = interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            if self.stability_score_attentuation:
                stability_scores = self.sam_mask_decoder._get_stability_scores(
                    low_res_multimasks
                )
                ious = ious * stability_scores
            best_iou_inds = mx.argmax(ious, axis=-1)
            batch_inds = mx.arange(ious.shape[0])
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]
            if sam_output_tokens.shape[1] > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            if multimask_output and not is_interactive:
                assert self.decode_mask_with_shared_tokens
                low_res_masks = low_res_multimasks[:, 0:1]
                high_res_masks = high_res_multimasks[:, 0:1]
            else:
                low_res_masks = low_res_multimasks
                high_res_masks = high_res_multimasks

        outputs: SAMOutput = {
            "low_res_multimasks": low_res_multimasks,
            "high_res_multimasks": high_res_multimasks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
        }

        if self.use_obj_ptrs_in_encoder:
            obj_ptr = (
                self.interactive_obj_ptr_proj(sam_output_token)
                if is_interactive
                else self.obj_ptr_proj(sam_output_token)
            )
            if self.pred_obj_scores and self.use_no_obj_ptr:
                assert is_obj_appearing is not None
                lambda_is_obj_appearing = is_obj_appearing.reshape(-1, 1).astype(
                    mx.float32
                )
                if self.use_linear_no_obj_ptr:
                    obj_ptr = lambda_is_obj_appearing * obj_ptr + (
                        1 - lambda_is_obj_appearing
                    ) * self.no_obj_ptr_linear(obj_ptr)
                else:
                    if self.fixed_no_obj_ptr:
                        obj_ptr = lambda_is_obj_appearing * obj_ptr
                    selected_no_obj_ptr = mx.broadcast_to(
                        self.no_obj_ptr[None, :, :],
                        (
                            multiplex_state.num_buckets,
                            self.no_obj_ptr.shape[0],
                            self.no_obj_ptr.shape[1],
                        ),
                    )
                    selected_no_obj_ptr = multiplex_state.demux(selected_no_obj_ptr)
                    if is_interactive:
                        selected_no_obj_ptr = selected_no_obj_ptr[objects_to_interact]
                    obj_ptr = (
                        obj_ptr + (1 - lambda_is_obj_appearing) * selected_no_obj_ptr
                    )
            outputs["obj_ptr"] = obj_ptr

        return outputs

    def _use_multimask(self, is_init_cond_frame: bool, point_inputs: Any) -> bool:
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].shape[1]
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
            and self.num_multimask_outputs > 0
        )

    def _track_step_aux(
        self,
        *,
        frame_idx: int,
        is_init_cond_frame: bool,
        backbone_features_interactive: dict[str, Any] | None,
        backbone_features_propagation: dict[str, Any] | None,
        image: Any,
        point_inputs: dict[str, Any] | None,
        mask_inputs: Any,
        gt_masks: Any,
        frames_to_add_correction_pt: list[int],
        output_dict: dict[str, dict[int, StageOutput]],
        num_frames: int,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: Any = None,
        multiplex_state: MultiplexState,
        objects_to_interact: list[int] | None = None,
        need_aux_output: bool = False,
    ) -> tuple[StageOutput, dict[str, Any]]:
        current_out: StageOutput = {
            "conditioning_objects": set(),
            "point_inputs": point_inputs,
            "mask_inputs": mask_inputs,
        }

        mode = None
        if mask_inputs is not None:
            mode = "mask_as_output"
        elif point_inputs is None:
            mode = "propagation_only"
        elif prev_sam_mask_logits is not None:
            if objects_to_interact is None:
                raise ValueError(
                    "objects_to_interact must be specified when refining "
                    "prev_sam_mask_logits"
                )
            mode = "interaction_only"
        elif is_init_cond_frame:
            mode = "interaction_only"
        elif objects_to_interact is not None:
            if getattr(self, "training", False):
                raise_unsupported_multiplex_runtime(
                    "VideoTrackingMultiplex._track_step_aux(training propagation_and_interaction)"
                )
            mode = "propagation_and_interaction"

        if mode is None:
            raise ValueError(
                "Unable to determine tracking case. "
                f"mask_inputs={mask_inputs is not None}, "
                f"point_inputs={point_inputs is not None}, "
                f"prev_sam_mask_logits={prev_sam_mask_logits is not None}, "
                f"objects_to_interact={objects_to_interact}, "
                f"is_init_cond_frame={is_init_cond_frame}"
            )

        interactive_vision_feats = interactive_feat_sizes = None
        interactive_high_res_features = None
        if backbone_features_interactive is not None:
            interactive_vision_feats = backbone_features_interactive["vision_feats"]
            interactive_feat_sizes = backbone_features_interactive["feat_sizes"]
            if len(interactive_vision_feats) > 1:
                interactive_high_res_features = [
                    x.transpose(1, 2, 0).reshape(x.shape[1], x.shape[2], *s)
                    for x, s in zip(
                        interactive_vision_feats[:-1],
                        interactive_feat_sizes[:-1],
                    )
                ]
        elif mode in (
            "interaction_only",
            "propagation_and_interaction",
            "mask_as_output",
        ):
            raise ValueError("interactive features are required for prompt interaction")

        propagation_vision_feats = propagation_vision_masks = None
        propagation_vision_pos_embeds = propagation_feat_sizes = None
        propagation_high_res_features = None
        if backbone_features_propagation is not None:
            propagation_vision_feats = backbone_features_propagation["vision_feats"]
            propagation_vision_masks = backbone_features_propagation["vision_masks"]
            propagation_vision_pos_embeds = backbone_features_propagation[
                "vision_pos_embeds"
            ]
            propagation_feat_sizes = backbone_features_propagation["feat_sizes"]
            if len(propagation_vision_feats) > 1:
                propagation_high_res_features = [
                    x.transpose(1, 2, 0).reshape(x.shape[1], x.shape[2], *s)
                    for x, s in zip(
                        propagation_vision_feats[:-1],
                        propagation_feat_sizes[:-1],
                    )
                ]
        elif (
            mode in ("propagation_only", "propagation_and_interaction")
            or run_mem_encoder
        ):
            raise ValueError("propagation features are required")

        interactive_pix_feat = None
        if mode == "mask_as_output":
            if not self.use_mask_input_as_output_without_sam:
                raise_unsupported_multiplex_runtime(
                    "VideoTrackingMultiplex._track_step_aux(mask_as_output)"
                )
            assert interactive_vision_feats is not None
            assert interactive_feat_sizes is not None
            interactive_pix_feat = self._get_interactive_pix_mem(
                interactive_vision_feats,
                interactive_feat_sizes,
            )
            sam_outputs = self._use_mask_as_output(
                backbone_features=interactive_pix_feat,
                high_res_features=interactive_high_res_features,
                mask_inputs=mask_inputs,
                multiplex_state=multiplex_state,
            )
            current_out["conditioning_objects"].update(range(mask_inputs.shape[0]))
        else:
            propagation_out = None
            if mode in ("propagation_only", "propagation_and_interaction"):
                assert propagation_vision_feats is not None
                assert propagation_vision_masks is not None
                assert propagation_vision_pos_embeds is not None
                assert propagation_feat_sizes is not None
                pix_feat_with_mem = self._prepare_memory_conditioned_features(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init_cond_frame,
                    current_vision_feats=propagation_vision_feats[-1:],
                    current_vision_masks=propagation_vision_masks[-1:],
                    current_vision_pos_embeds=propagation_vision_pos_embeds[-1:],
                    feat_sizes=propagation_feat_sizes[-1:],
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=track_in_reverse,
                    multiplex_state=multiplex_state,
                )
                propagation_out = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    propagation_high_res_features=propagation_high_res_features,
                    multimask_output=self._use_multimask(
                        is_init_cond_frame,
                        point_inputs=None,
                    ),
                    objects_to_interact=list(
                        range(multiplex_state.total_valid_entries)
                    ),
                    multiplex_state=multiplex_state,
                )

            interaction_out = None
            if mode in ("interaction_only", "propagation_and_interaction"):
                assert interactive_vision_feats is not None
                assert interactive_feat_sizes is not None
                interactive_pix_feat = self._get_interactive_pix_mem(
                    interactive_vision_feats,
                    interactive_feat_sizes,
                )
                if mask_inputs is not None or point_inputs is None:
                    raise ValueError("interaction mode requires point inputs only")
                if prev_sam_mask_logits is not None:
                    assert objects_to_interact is not None
                    if not self.iter_use_prev_mask_pred:
                        raise ValueError("iter_use_prev_mask_pred is required")
                    mask_inputs = _take_rows(prev_sam_mask_logits, objects_to_interact)
                elif mode == "propagation_and_interaction":
                    assert objects_to_interact is not None
                    assert propagation_out is not None
                    mask_inputs = _take_rows(
                        propagation_out["low_res_masks"],
                        objects_to_interact,
                    )

                if objects_to_interact is not None:
                    if point_inputs["point_coords"].shape[0] != len(
                        objects_to_interact
                    ):
                        raise ValueError(
                            "point prompt batch must match objects_to_interact"
                        )
                    if point_inputs["point_labels"].shape[0] != len(
                        objects_to_interact
                    ):
                        raise ValueError(
                            "point label batch must match objects_to_interact"
                        )

                interaction_objects = (
                    objects_to_interact
                    if objects_to_interact is not None
                    else sorted(multiplex_state.get_all_valid_object_idx())
                )
                interaction_out = self._forward_sam_heads(
                    backbone_features=interactive_pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    interactive_high_res_features=interactive_high_res_features,
                    multimask_output=self._use_multimask(
                        is_init_cond_frame,
                        point_inputs=point_inputs,
                    ),
                    objects_to_interact=interaction_objects,
                    multiplex_state=multiplex_state,
                )
                current_out["conditioning_objects"].update(interaction_objects)

            if propagation_out is None and interaction_out is not None:
                sam_outputs = interaction_out
            elif interaction_out is None and propagation_out is not None:
                sam_outputs = propagation_out
            else:
                assert propagation_out is not None and interaction_out is not None
                assert objects_to_interact is not None
                sam_outputs = propagation_out
                for key in (
                    "low_res_multimasks",
                    "high_res_multimasks",
                    "low_res_masks",
                    "high_res_masks",
                    "ious",
                    "object_score_logits",
                    "obj_ptr",
                ):
                    if key in sam_outputs and key in interaction_out:
                        sam_outputs[key] = _replace_rows(
                            sam_outputs[key],
                            objects_to_interact,
                            interaction_out[key],
                        )

        low_res_multimasks = sam_outputs["low_res_multimasks"]
        high_res_multimasks = sam_outputs["high_res_multimasks"]
        ious = sam_outputs["ious"]
        low_res_masks = sam_outputs["low_res_masks"]
        high_res_masks = sam_outputs["high_res_masks"]
        object_score_logits = sam_outputs["object_score_logits"]
        obj_ptr = sam_outputs["obj_ptr"] if self.use_obj_ptrs_in_encoder else None

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        if (
            frame_idx in frames_to_add_correction_pt
            and self.num_correction_pt_per_frame > 0
        ):
            if getattr(self, "training", False):
                raise_unsupported_multiplex_runtime(
                    "VideoTrackingMultiplex._track_step_aux(training correction-points)"
                )
            if gt_masks is None:
                raise ValueError("correction points require gt_masks")
            if objects_to_interact is None:
                raise ValueError("correction points require objects_to_interact")
            if interactive_vision_feats is None or interactive_feat_sizes is None:
                raise ValueError("correction points require interactive features")
            if point_inputs is not None:
                if point_inputs["point_coords"].shape[0] != len(objects_to_interact):
                    raise ValueError(
                        "point prompt batch must match objects_to_interact"
                    )
                if point_inputs["point_labels"].shape[0] != len(objects_to_interact):
                    raise ValueError("point label batch must match objects_to_interact")

            all_pred_masks = [low_res_masks]
            all_pred_high_res_masks = [high_res_masks]
            all_pred_multimasks = [low_res_multimasks]
            all_pred_high_res_multimasks = [high_res_multimasks]
            all_pred_ious = [ious]
            all_point_inputs = [point_inputs]
            all_object_score_logits = [object_score_logits]

            for _ in range(self.num_correction_pt_per_frame):
                pred_for_new_pt = high_res_masks > 0
                new_points, new_labels = get_next_point(
                    gt_masks=_take_rows(gt_masks, objects_to_interact),
                    pred_masks=_take_rows(pred_for_new_pt, objects_to_interact),
                    method=self.pt_sampling_for_eval,
                )
                point_inputs = concat_points(point_inputs, new_points, new_labels)
                if self.iter_use_prev_mask_pred:
                    mask_inputs = _take_rows(low_res_masks, objects_to_interact)

                pix_feat_with_mem = self._get_interactive_pix_mem(
                    interactive_vision_feats,
                    interactive_feat_sizes,
                )
                correction_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    interactive_high_res_features=interactive_high_res_features,
                    propagation_high_res_features=propagation_high_res_features,
                    multimask_output=self._use_multimask(
                        is_init_cond_frame,
                        point_inputs,
                    ),
                    gt_masks=gt_masks,
                    objects_to_interact=objects_to_interact,
                    multiplex_state=multiplex_state,
                )

                low_res_masks = _replace_rows(
                    low_res_masks,
                    objects_to_interact,
                    correction_outputs["low_res_masks"],
                )
                high_res_masks = _replace_rows(
                    high_res_masks,
                    objects_to_interact,
                    correction_outputs["high_res_masks"],
                )
                low_res_multimasks = _replace_rows(
                    low_res_multimasks,
                    objects_to_interact,
                    correction_outputs["low_res_multimasks"],
                )
                high_res_multimasks = _replace_rows(
                    high_res_multimasks,
                    objects_to_interact,
                    correction_outputs["high_res_multimasks"],
                )
                ious = _replace_rows(
                    ious, objects_to_interact, correction_outputs["ious"]
                )
                object_score_logits = _replace_rows(
                    object_score_logits,
                    objects_to_interact,
                    correction_outputs["object_score_logits"],
                )
                if self.use_obj_ptrs_in_encoder:
                    assert obj_ptr is not None
                    obj_ptr = _replace_rows(
                        obj_ptr,
                        objects_to_interact,
                        correction_outputs["obj_ptr"],
                    )

                all_pred_masks.append(low_res_masks)
                all_pred_high_res_masks.append(high_res_masks)
                all_pred_multimasks.append(low_res_multimasks)
                all_pred_high_res_multimasks.append(high_res_multimasks)
                all_pred_ious.append(ious)
                all_point_inputs.append(point_inputs)
                all_object_score_logits.append(object_score_logits)

            current_out["multistep_pred_masks"] = _concat(all_pred_masks, axis=1)
            current_out["multistep_pred_masks_high_res"] = _concat(
                all_pred_high_res_masks,
                axis=1,
            )
            current_out["multistep_pred_multimasks"] = all_pred_multimasks
            current_out["multistep_pred_multimasks_high_res"] = (
                all_pred_high_res_multimasks
            )
            current_out["multistep_pred_ious"] = all_pred_ious
            current_out["multistep_point_inputs"] = all_point_inputs
            current_out["multistep_object_score_logits"] = all_object_score_logits

            if self.add_all_frames_to_correct_as_cond:
                current_out["conditioning_objects"].update(objects_to_interact)

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        if self.use_obj_ptrs_in_encoder:
            assert obj_ptr is not None
            current_out["obj_ptr"] = multiplex_state.mux(obj_ptr)

        if self.use_memory_selection:
            iou_score = mx.max(ious, axis=-1)
            current_out["iou_score"] = iou_score
            current_out["eff_iou_score"] = self.cal_mem_score(
                object_score_logits,
                iou_score,
            )
        current_out["object_score_logits"] = object_score_logits

        if run_mem_encoder and self.num_maskmem > 0:
            assert propagation_vision_feats is not None
            assert propagation_feat_sizes is not None
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                conditioning_objects=current_out["conditioning_objects"],
                multiplex_state=multiplex_state,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        if self.save_image_features:
            if (
                propagation_vision_feats is None
                or propagation_vision_pos_embeds is None
            ):
                raise ValueError(
                    "propagation features are required to save image features"
                )
            current_out["image_features"] = propagation_vision_feats[-1]
            current_out["image_pos_enc"] = propagation_vision_pos_embeds[-1]

        aux_output: dict[str, Any] = {}
        if need_aux_output:
            if interactive_pix_feat is None:
                if interactive_vision_feats is None or interactive_feat_sizes is None:
                    raise ValueError("interactive features are required for aux output")
                interactive_pix_feat = self._get_interactive_pix_mem(
                    interactive_vision_feats,
                    interactive_feat_sizes,
                )
            aux_output["interactive_pix_feat"] = interactive_pix_feat
            aux_output["interactive_high_res_features"] = interactive_high_res_features
            aux_output["propagation_vision_feats"] = propagation_vision_feats
            aux_output["propagation_feat_sizes"] = propagation_feat_sizes

        return current_out, aux_output

    def _trim_output_and_memory(
        self,
        frame_idx: int,
        output_dict: dict[str, dict[int, StageOutput]],
        current_out: StageOutput,
        memory_encoder_was_used: bool,
    ) -> StageOutput:
        if self.offload_output_to_cpu_for_eval and not getattr(self, "training", False):
            trimmed_out: StageOutput = {
                "conditioning_objects": current_out["conditioning_objects"],
                "pred_masks": current_out["pred_masks"],
                "pred_masks_high_res": current_out["pred_masks_high_res"],
                "object_score_logits": current_out["object_score_logits"],
                "multistep_point_inputs": current_out["multistep_point_inputs"],
            }
            if self.use_obj_ptrs_in_encoder:
                trimmed_out["obj_ptr"] = current_out["obj_ptr"]
            if memory_encoder_was_used and self.num_maskmem > 0:
                trimmed_out["maskmem_features"] = current_out["maskmem_features"]
                trimmed_out["maskmem_pos_enc"] = current_out["maskmem_pos_enc"]
            if self.save_image_features:
                trimmed_out["image_features"] = current_out["image_features"]
                trimmed_out["image_pos_enc"] = current_out["image_pos_enc"]
            current_out = trimmed_out

        def _trim_past_out(past_out: StageOutput | None) -> StageOutput | None:
            if past_out is None:
                return None
            trimmed_past_out: StageOutput = {
                "conditioning_objects": past_out["conditioning_objects"],
                "pred_masks": past_out["pred_masks"],
                "object_score_logits": past_out["object_score_logits"],
                "multistep_point_inputs": past_out["multistep_point_inputs"],
            }
            if self.use_obj_ptrs_in_encoder:
                trimmed_past_out["obj_ptr"] = past_out["obj_ptr"]
            return trimmed_past_out

        if self.trim_past_non_cond_mem_for_eval and not getattr(
            self, "training", False
        ):
            r = self.memory_temporal_stride_for_eval
            past_frame_idx = frame_idx - r * self.num_maskmem
            past_out = output_dict["non_cond_frame_outputs"].get(past_frame_idx)
            if past_out is not None:
                if (
                    self.use_memory_selection
                    and _to_scalar(past_out.get("eff_iou_score", 0)) < self.mf_threshold
                ) or not self.use_memory_selection:
                    output_dict["non_cond_frame_outputs"][past_frame_idx] = (
                        _trim_past_out(past_out)
                    )

            if self.use_memory_selection:
                far_old_frame_idx = frame_idx - 20 * self.max_obj_ptrs_in_encoder
                past_out = output_dict["non_cond_frame_outputs"].get(far_old_frame_idx)
                if past_out is not None:
                    output_dict["non_cond_frame_outputs"][far_old_frame_idx] = (
                        _trim_past_out(past_out)
                    )

        del memory_encoder_was_used
        return current_out

    def track_step(
        self,
        *,
        frame_idx: int,
        is_init_cond_frame: bool,
        backbone_features_interactive: dict[str, Any] | None,
        backbone_features_propagation: dict[str, Any] | None,
        image: Any,
        point_inputs: dict[str, Any] | None,
        mask_inputs: Any,
        gt_masks: Any,
        frames_to_add_correction_pt: list[int],
        output_dict: dict[str, dict[int, StageOutput]],
        num_frames: int,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: Any = None,
        multiplex_state: MultiplexState,
        objects_to_interact: list[int] | None = None,
        new_object_masks: Any = None,
        new_object_idxs: list[int] | None = None,
        new_object_ids: list[int] | None = None,
        are_new_masks_from_pts: bool = False,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
        reconditioning: bool = False,
    ) -> StageOutput:
        current_out, aux_out = self._track_step_aux(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=backbone_features_interactive,
            backbone_features_propagation=backbone_features_propagation,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            gt_masks=gt_masks,
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            run_mem_encoder=(run_mem_encoder and new_object_masks is None),
            prev_sam_mask_logits=prev_sam_mask_logits,
            multiplex_state=multiplex_state,
            objects_to_interact=objects_to_interact,
            need_aux_output=(new_object_masks is not None),
        )

        if new_object_masks is not None:
            if new_object_idxs is None:
                raise ValueError("new_object_idxs are required with new_object_masks")
            mask_state_kwargs = dict(
                interactive_pix_feat=aux_out["interactive_pix_feat"],
                interactive_high_res_features=aux_out["interactive_high_res_features"],
                propagation_vision_feats=aux_out["propagation_vision_feats"],
                propagation_feat_sizes=aux_out["propagation_feat_sizes"],
                new_masks=new_object_masks,
                obj_idxs_in_mask=new_object_idxs,
                obj_ids_in_mask=new_object_ids,
                prev_output=current_out,
                multiplex_state=multiplex_state,
                add_mask_to_memory=run_mem_encoder,
            )
            if reconditioning:
                self.recondition_masks_in_existing_state(**mask_state_kwargs)
            else:
                self.add_new_masks_to_existing_state(
                    **mask_state_kwargs,
                    are_masks_from_pts=are_new_masks_from_pts,
                    allow_new_buckets=allow_new_buckets,
                    prefer_new_buckets=prefer_new_buckets,
                )

        return self._trim_output_and_memory(
            frame_idx=frame_idx,
            output_dict=output_dict,
            current_out=current_out,
            memory_encoder_was_used=run_mem_encoder,
        )

    def _prepared_features_from_state(
        self,
        inference_state: dict[str, Any],
        *,
        frame_idx: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        backbone_out = inference_state.get("backbone_out")
        if not isinstance(backbone_out, dict):
            cached_features = inference_state.get("cached_features")
            cached = (
                cached_features.get(int(frame_idx))
                if frame_idx is not None and isinstance(cached_features, dict)
                else None
            )
            if cached is None:
                raise ValueError(
                    "SAM2 state must contain backbone_out or cached_features[frame_idx] "
                    "for mask addition."
                )
            if not (isinstance(cached, tuple) and len(cached) == 2):
                raise TypeError("cached_features values must be (image, backbone_out).")
            _, backbone_out = cached
            if not isinstance(backbone_out, dict):
                raise ValueError(
                    "cached_features current frame backbone_out must be a dict."
                )
            inference_state["backbone_out"] = backbone_out
        if "interactive" not in backbone_out or "sam2_backbone_out" not in backbone_out:
            raise ValueError(
                "backbone_out must contain interactive and sam2_backbone_out features."
            )
        interactive = backbone_out["interactive"]
        propagation = backbone_out["sam2_backbone_out"]
        if (
            isinstance(interactive, dict)
            and "vision_feats" in interactive
            and isinstance(propagation, dict)
            and "vision_feats" in propagation
        ):
            return backbone_out
        return self._prepare_backbone_features(backbone_out)

    @staticmethod
    def _current_frame_output(
        inference_state: dict[str, Any],
        frame_idx: int,
    ) -> StageOutput:
        output_dict = inference_state.get("output_dict")
        if not isinstance(output_dict, dict):
            raise ValueError("SAM2 state must contain output_dict for mask addition.")

        matches: list[StageOutput] = []
        for storage_key in ("non_cond_frame_outputs", "cond_frame_outputs"):
            storage = output_dict.get(storage_key, {})
            if isinstance(storage, dict) and frame_idx in storage:
                matches.append(storage[frame_idx])
        if len(matches) != 1:
            raise ValueError(
                "SAM2 state must contain exactly one current frame output for mask "
                f"addition at frame_idx={frame_idx}; found {len(matches)}."
            )
        return matches[0]

    def add_new_masks(
        self,
        *,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_ids: Any,
        masks: Any,
        add_mask_to_memory: bool = True,
        are_masks_from_pts: bool = False,
    ) -> tuple[int, list[int], None, None]:
        """Official-shaped adapter for adding detector masks to packed MLX state."""
        frame_idx = int(frame_idx)
        obj_ids_np = _to_numpy(obj_ids).astype(np.int64).reshape(-1)
        obj_ids_list = [int(obj_id) for obj_id in obj_ids_np.tolist()]
        if len(obj_ids_list) == 0:
            return frame_idx, inference_state.setdefault("obj_ids", []), None, None

        new_masks = masks if _is_mlx_array(masks) else mx.array(masks)
        new_masks = new_masks.astype(mx.float32)
        if new_masks.ndim == 3:
            new_masks = new_masks[:, None, :, :]
        if new_masks.ndim != 4 or new_masks.shape[1] != 1:
            raise ValueError(
                "masks must have shape (N, H, W) or (N, 1, H, W), "
                f"got {new_masks.shape}."
            )
        if new_masks.shape[0] != len(obj_ids_list):
            raise ValueError(
                "masks batch must match obj_ids; got "
                f"{new_masks.shape[0]} masks for {len(obj_ids_list)} ids."
            )

        multiplex_state = inference_state.get("multiplex_state")
        if multiplex_state is None:
            raise ValueError(
                "SAM2 state must contain multiplex_state for mask addition."
            )
        prev_output = self._current_frame_output(inference_state, frame_idx)
        backbone_features = self._prepared_features_from_state(
            inference_state,
            frame_idx=frame_idx,
        )

        interactive = backbone_features["interactive"]
        interactive_vision_feats = interactive["vision_feats"]
        interactive_feat_sizes = interactive["feat_sizes"]
        interactive_high_res_features = None
        if len(interactive_vision_feats) > 1:
            interactive_high_res_features = [
                x.transpose(1, 2, 0).reshape(x.shape[1], x.shape[2], *s)
                for x, s in zip(
                    interactive_vision_feats[:-1],
                    interactive_feat_sizes[:-1],
                )
            ]
        interactive_pix_feat = self._get_interactive_pix_mem(
            interactive_vision_feats,
            interactive_feat_sizes,
        )

        propagation = backbone_features["sam2_backbone_out"]
        propagation_vision_feats = propagation.get("vision_feats")
        propagation_feat_sizes = propagation.get("feat_sizes")
        if add_mask_to_memory and (
            propagation_vision_feats is None or propagation_feat_sizes is None
        ):
            raise ValueError(
                "sam2_backbone_out vision_feats and feat_sizes are required when "
                "add_mask_to_memory=True."
            )

        self.add_new_masks_to_existing_state(
            interactive_pix_feat=interactive_pix_feat,
            interactive_high_res_features=interactive_high_res_features,
            propagation_vision_feats=propagation_vision_feats,
            propagation_feat_sizes=propagation_feat_sizes,
            new_masks=new_masks,
            obj_idxs_in_mask=list(range(len(obj_ids_list))),
            obj_ids_in_mask=obj_ids_list,
            prev_output=prev_output,
            multiplex_state=multiplex_state,
            add_mask_to_memory=add_mask_to_memory,
            are_masks_from_pts=are_masks_from_pts,
        )

        state_obj_ids = inference_state.setdefault("obj_ids", [])
        for obj_id in obj_ids_list:
            if obj_id not in state_obj_ids:
                state_obj_ids.append(obj_id)
        if getattr(multiplex_state, "object_ids", None) is not None:
            inference_state["obj_ids"] = [
                int(obj_id) for obj_id in multiplex_state.object_ids
            ]
        return frame_idx, inference_state["obj_ids"], None, None

    def propagate_in_video_preflight(
        self,
        sam2_state: dict[str, Any],
        run_mem_encoder: bool = True,
    ) -> None:
        del run_mem_encoder
        sam2_state["tracking_has_started"] = True

    def forward_tracking(
        self,
        backbone_out: dict[str, Any],
        input: Any,
        return_dict: bool = False,
        objects_to_interact: list[int] | None = None,
    ) -> Any:
        """Forward tracking over precomputed or per-frame backbone features."""
        img_feats_already_computed = (
            "interactive" in backbone_out or "sam2_backbone_out" in backbone_out
        )
        backbone_features = (
            self._prepare_backbone_features(backbone_out)
            if img_feats_already_computed
            else None
        )

        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        transition_points = backbone_out.get("transition_points", [])
        new_idx_per_transition = backbone_out.get("new_idx_per_transition", {})
        valid_objects_prior_to_each_transition = backbone_out.get(
            "valid_objects_prior_to_each_transition",
            {},
        )

        cond_frame_outputs: dict[int, StageOutput] = {}
        non_cond_frame_outputs: dict[int, StageOutput] = {}
        output_dict = {
            "cond_frame_outputs": cond_frame_outputs,
            "non_cond_frame_outputs": non_cond_frame_outputs,
        }
        first_stage = processing_order[0]
        gt_masks_per_frame = backbone_out["gt_masks_per_frame"]
        multiplex_state = self.multiplex_controller.get_state(
            gt_masks_per_frame[first_stage].shape[0],
            dtype=mx.float32,
            random=getattr(self, "training", False),
        )

        point_inputs_per_frame = backbone_out.get("point_inputs_per_frame", {})
        mask_inputs_per_frame = backbone_out.get("mask_inputs_per_frame", {})

        for stage_id in processing_order:
            img_ids = _to_numpy(input.find_inputs[stage_id].img_ids).reshape(-1)
            if img_ids.size == 0:
                raise ValueError("img_ids must contain at least one image id")
            if not np.all(img_ids == img_ids[0]):
                raise ValueError("all image ids for a multiplex stage must match")
            img_ids = np.array([int(img_ids[0])], dtype=np.int64)

            if img_feats_already_computed:
                assert backbone_features is not None
                current_image = _take_image_batch(input.img_batch, img_ids)
                current_backbone_features = {}
                for neck_k, neck_out in backbone_features.items():
                    current_backbone_features[neck_k] = {
                        "vision_feats": [
                            _take_along_axis(x, img_ids, axis=1)
                            for x in neck_out["vision_feats"]
                        ],
                        "vision_masks": [
                            _take_along_axis(x, img_ids, axis=0)
                            if x is not None
                            else None
                            for x in neck_out["vision_masks"]
                        ],
                        "vision_pos_embeds": [
                            _take_along_axis(x, img_ids, axis=1)
                            for x in neck_out["vision_pos_embeds"]
                        ],
                        "feat_sizes": neck_out["feat_sizes"],
                    }
            else:
                need_interactive_out = (
                    stage_id in frames_to_add_correction_pt
                    or stage_id in init_cond_frames
                    or stage_id in transition_points
                )
                (
                    current_image,
                    current_backbone_features,
                ) = self._prepare_backbone_features_per_frame(
                    input.img_batch,
                    img_ids,
                    need_interactive_out=need_interactive_out,
                    need_propagation_out=True,
                )

            gt_masks = gt_masks_per_frame.get(stage_id)
            if stage_id in transition_points:
                if gt_masks is None:
                    raise ValueError("transition points require gt masks")
                new_object_idxs = new_idx_per_transition[stage_id]
                if new_object_idxs != sorted(new_object_idxs):
                    raise ValueError("new_object_idxs must be sorted")
                valid_prior = valid_objects_prior_to_each_transition.get(stage_id)
                if valid_prior is not None and new_object_idxs[0] != len(valid_prior):
                    raise ValueError("new_object_idxs must start after prior objects")
                if new_object_idxs[-1] != gt_masks.shape[0] - 1:
                    raise ValueError("new_object_idxs must end at the last gt mask")
                new_object_masks = _take_rows(gt_masks, new_object_idxs)
                gt_masks = gt_masks[: new_object_idxs[0]]
            else:
                new_object_masks = None
                new_object_idxs = None

            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                backbone_features_interactive=current_backbone_features.get(
                    "interactive"
                ),
                backbone_features_propagation=current_backbone_features.get(
                    "sam2_backbone_out"
                ),
                image=current_image,
                point_inputs=point_inputs_per_frame.get(stage_id),
                mask_inputs=mask_inputs_per_frame.get(stage_id),
                gt_masks=gt_masks,
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                multiplex_state=multiplex_state,
                objects_to_interact=objects_to_interact,
                new_object_masks=new_object_masks,
                new_object_idxs=new_object_idxs,
            )

            add_output_as_cond_frame = (
                stage_id in init_cond_frames
                or (
                    self.add_all_frames_to_correct_as_cond
                    and stage_id in frames_to_add_correction_pt
                )
                or (
                    getattr(self, "add_all_transition_frames_as_cond", False)
                    and stage_id in transition_points
                )
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        output_dict["multiplex_state"] = multiplex_state
        if return_dict:
            return output_dict

        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])

        dynamic_vos_eval = getattr(self, "is_dynamic_vos_evaluation", False)
        if dynamic_vos_eval:
            frame_outputs = [all_frame_outputs.get(t) for t in range(num_frames)]
        else:
            frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        frame_outputs = [
            {k: v for k, v in frame_out.items() if k != "obj_ptr"}
            if frame_out is not None
            else None
            for frame_out in frame_outputs
        ]

        if dynamic_vos_eval:
            object_appearance_order = backbone_out["object_appearance_order"]
            num_objects = len(input.find_metadatas[0].coco_image_id)
            inverse_object_appearance_order: list[int | None] = [
                None for _ in object_appearance_order
            ]
            for idx, obj_id in enumerate(object_appearance_order):
                inverse_object_appearance_order[int(obj_id)] = idx
            if any(idx is None for idx in inverse_object_appearance_order):
                raise ValueError("object_appearance_order must be a dense remapping.")
            if len(inverse_object_appearance_order) < num_objects:
                inverse_object_appearance_order.extend(
                    range(len(inverse_object_appearance_order), num_objects)
                )
            row_order = [int(idx) for idx in inverse_object_appearance_order]

            last_output = frame_outputs[-1]
            if last_output is None:
                raise ValueError(
                    "dynamic VOS evaluation requires a final frame output."
                )
            last_mask = last_output["pred_masks"]
            pad_shape = last_mask.shape[1:]
            pad_dtype = last_mask.dtype
            for stage_i, frame_out in enumerate(frame_outputs):
                if frame_out is None:
                    if _is_mlx_array(last_mask):
                        zero_masks = mx.zeros(
                            (num_objects, *pad_shape),
                            dtype=pad_dtype,
                        )
                    else:
                        zero_masks = np.zeros(
                            (num_objects, *pad_shape),
                            dtype=pad_dtype,
                        )
                    frame_outputs[stage_i] = {"pred_masks": zero_masks}
                    continue

                pred_mask = frame_out["pred_masks"]
                if pred_mask.shape[0] < num_objects:
                    mask_shape = pred_mask.shape[1:]
                    if _is_mlx_array(pred_mask):
                        zero_pad = mx.zeros(
                            (num_objects - pred_mask.shape[0], *mask_shape),
                            dtype=pred_mask.dtype,
                        )
                    else:
                        zero_pad = np.zeros(
                            (num_objects - pred_mask.shape[0], *mask_shape),
                            dtype=pred_mask.dtype,
                        )
                    padded_mask = _concat([pred_mask, zero_pad], axis=0)
                    frame_out["pred_masks"] = _take_rows(padded_mask, row_order)

        return frame_outputs

    def add_new_masks_to_existing_state(
        self,
        *,
        interactive_pix_feat: Any,
        interactive_high_res_features: list[Any],
        propagation_vision_feats: list[Any] | None,
        propagation_feat_sizes: list[tuple[int, int]] | None,
        new_masks: Any,
        obj_idxs_in_mask: list[int],
        obj_ids_in_mask: list[int] | None,
        prev_output: StageOutput,
        multiplex_state: MultiplexState,
        add_mask_to_memory: bool = True,
        are_masks_from_pts: bool = False,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> None:
        """
        Add new objects to an existing output/multiplex state.

        New object entries are appended in data space, then muxed back to the
        updated multiplex assignment just like the official helper.
        """
        if not self.use_mask_input_as_output_without_sam:
            raise_unsupported_multiplex_runtime(
                "VideoTrackingMultiplex.add_new_masks_to_existing_state"
            )

        num_new_objects = new_masks.shape[0]
        if num_new_objects != len(obj_idxs_in_mask):
            raise ValueError(
                "new_masks batch must match obj_idxs_in_mask length: "
                f"{num_new_objects} != {len(obj_idxs_in_mask)}"
            )
        if obj_ids_in_mask is not None and len(obj_ids_in_mask) != num_new_objects:
            raise ValueError(
                "obj_ids_in_mask length must match new_masks batch: "
                f"{len(obj_ids_in_mask)} != {num_new_objects}"
            )

        existing_pointers = None
        if self.use_obj_ptrs_in_encoder:
            existing_pointers = multiplex_state.demux(prev_output["obj_ptr"])

        new_object_idx = multiplex_state.find_next_batch_of_available_indices(
            num_objects=num_new_objects,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )
        multiplex_state.add_objects(
            object_indices=new_object_idx,
            object_ids=obj_ids_in_mask,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )

        mask_output = self._use_mask_as_output(
            backbone_features=interactive_pix_feat,
            high_res_features=interactive_high_res_features,
            mask_inputs=new_masks,
            multiplex_state=multiplex_state,
            objects_in_mask=new_object_idx,
        )

        if "pred_masks" not in prev_output or prev_output["pred_masks"] is None:
            prev_output["pred_masks"] = mx.zeros(
                (0, *mask_output["low_res_masks"].shape[1:]),
                dtype=mask_output["low_res_masks"].dtype,
            )
        if (
            "pred_masks_high_res" not in prev_output
            or prev_output["pred_masks_high_res"] is None
        ):
            prev_output["pred_masks_high_res"] = mx.zeros(
                (0, *mask_output["high_res_masks"].shape[1:]),
                dtype=mask_output["high_res_masks"].dtype,
            )
        if (
            "object_score_logits" not in prev_output
            or prev_output["object_score_logits"] is None
        ):
            prev_output["object_score_logits"] = mx.zeros(
                (0, *mask_output["object_score_logits"].shape[1:]),
                dtype=mask_output["object_score_logits"].dtype,
            )

        interactive_resolution = mask_output["high_res_masks"].shape[-1]
        if (
            "pred_masks_high_res" in prev_output
            and prev_output["pred_masks_high_res"] is not None
        ):
            existing_resolution = prev_output["pred_masks_high_res"].shape[-1]
            if existing_resolution != interactive_resolution:
                prev_output["pred_masks_high_res"] = interpolate(
                    prev_output["pred_masks_high_res"],
                    size=(interactive_resolution, interactive_resolution),
                    mode="bilinear",
                    align_corners=False,
                )

        h, w = prev_output["pred_masks"].shape[-2:]
        mask_output["low_res_masks"] = interpolate(
            mask_output["low_res_masks"],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        _append(prev_output, mask_output, "pred_masks", "low_res_masks")
        _append(
            prev_output,
            mask_output,
            "pred_masks_high_res",
            "high_res_masks",
            strict=False,
        )
        _append(prev_output, mask_output, "object_score_logits", "object_score_logits")

        if self.use_memory_selection:
            ious = mask_output["ious"]
            if len(ious.shape) > 1 and ious.shape[-1] == 1:
                ious = ious.reshape(ious.shape[:-1])
            mask_output["ious"] = ious
            _append(prev_output, mask_output, "iou_score", "ious")

        if "input_masks" in prev_output:
            prev_output["input_masks"] = _concat(
                [prev_output["input_masks"], new_masks],
                axis=0,
            )

        if self.use_obj_ptrs_in_encoder:
            assert existing_pointers is not None
            new_pointers = mask_output["obj_ptr"].astype(existing_pointers.dtype)
            combined_pointers = _concat([existing_pointers, new_pointers], axis=0)
            prev_output["obj_ptr"] = multiplex_state.mux(combined_pointers)

        prev_output["conditioning_objects"].update(new_object_idx)

        if add_mask_to_memory:
            if (
                prev_output["pred_masks_high_res"].shape[0]
                != multiplex_state.total_valid_entries
            ):
                raise ValueError(
                    "pred_masks_high_res must have one row per valid multiplex "
                    "entry before memory encoding"
                )
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=None,
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=prev_output["pred_masks_high_res"],
                object_score_logits=prev_output["object_score_logits"],
                conditioning_objects=prev_output["conditioning_objects"],
                is_mask_from_pts=are_masks_from_pts,
                multiplex_state=multiplex_state,
            )
            prev_output["maskmem_features"] = maskmem_features
            prev_output["maskmem_pos_enc"] = maskmem_pos_enc

            if self.save_image_features:
                if "image_features" not in prev_output:
                    raise ValueError("image_features missing from saved state")
                if "image_pos_enc" not in prev_output:
                    raise ValueError("image_pos_enc missing from saved state")

    def recondition_masks_in_existing_state(
        self,
        *,
        interactive_pix_feat: Any,
        interactive_high_res_features: list[Any],
        propagation_vision_feats: list[Any] | None,
        propagation_feat_sizes: list[tuple[int, int]] | None,
        new_masks: Any,
        obj_idxs_in_mask: list[int],
        obj_ids_in_mask: list[int] | None,
        prev_output: StageOutput,
        multiplex_state: MultiplexState,
        add_mask_to_memory: bool = True,
        are_masks_from_pts: bool = False,
    ) -> None:
        """
        Recondition existing objects in an existing output/multiplex state.

        This mirrors the official helper's state mutation contract while keeping
        the full video forward path behind the existing fail-fast boundary.
        """
        if not self.use_mask_input_as_output_without_sam:
            raise_unsupported_multiplex_runtime(
                "VideoTrackingMultiplex.recondition_masks_in_existing_state"
            )

        num_new_objects = new_masks.shape[0]
        if num_new_objects != len(obj_idxs_in_mask):
            raise ValueError(
                "new_masks batch must match obj_idxs_in_mask length: "
                f"{num_new_objects} != {len(obj_idxs_in_mask)}"
            )
        if obj_ids_in_mask is not None and len(obj_ids_in_mask) != num_new_objects:
            raise ValueError(
                "obj_ids_in_mask length must match new_masks batch: "
                f"{len(obj_ids_in_mask)} != {num_new_objects}"
            )

        existing_pointers = None
        if self.use_obj_ptrs_in_encoder:
            existing_pointers = multiplex_state.demux(prev_output["obj_ptr"])

        mask_output = self._use_mask_as_output(
            backbone_features=interactive_pix_feat,
            high_res_features=interactive_high_res_features,
            mask_inputs=new_masks,
            multiplex_state=multiplex_state,
            objects_in_mask=obj_idxs_in_mask,
        )

        h, w = prev_output["pred_masks"].shape[-2:]
        mask_output["low_res_masks"] = interpolate(
            mask_output["low_res_masks"],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        _merge(
            prev_output,
            mask_output,
            "pred_masks",
            "low_res_masks",
            obj_idxs_in_mask,
        )
        _merge(
            prev_output,
            mask_output,
            "pred_masks_high_res",
            "high_res_masks",
            obj_idxs_in_mask,
            strict=False,
        )
        _merge(
            prev_output,
            mask_output,
            "object_score_logits",
            "object_score_logits",
            obj_idxs_in_mask,
        )

        if self.use_memory_selection:
            ious = mask_output["ious"]
            if len(ious.shape) > 1 and ious.shape[-1] == 1:
                ious = ious.reshape(ious.shape[:-1])
            mask_output["ious"] = ious
            _merge(prev_output, mask_output, "iou_score", "ious", obj_idxs_in_mask)

        if "input_masks" in prev_output:
            prev_output["input_masks"] = _replace_rows(
                prev_output["input_masks"],
                obj_idxs_in_mask,
                new_masks,
            )

        if self.use_obj_ptrs_in_encoder:
            assert existing_pointers is not None
            new_pointers = mask_output["obj_ptr"].astype(existing_pointers.dtype)
            existing_pointers = _replace_rows(
                existing_pointers,
                obj_idxs_in_mask,
                new_pointers,
            )
            prev_output["obj_ptr"] = multiplex_state.mux(existing_pointers)

        prev_output["conditioning_objects"].update(obj_idxs_in_mask)

        if add_mask_to_memory:
            if (
                prev_output["pred_masks_high_res"].shape[0]
                != multiplex_state.total_valid_entries
            ):
                raise ValueError(
                    "pred_masks_high_res must have one row per valid multiplex "
                    "entry before memory encoding"
                )
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=None,
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=prev_output["pred_masks_high_res"],
                object_score_logits=prev_output["object_score_logits"],
                conditioning_objects=prev_output["conditioning_objects"],
                is_mask_from_pts=are_masks_from_pts,
                multiplex_state=multiplex_state,
            )
            prev_output["maskmem_features"] = maskmem_features
            prev_output["maskmem_pos_enc"] = maskmem_pos_enc

            if self.save_image_features:
                if "image_features" not in prev_output:
                    raise ValueError("image_features missing from saved state")
                if "image_pos_enc" not in prev_output:
                    raise ValueError("image_pos_enc missing from saved state")

    def forward(
        self, input: Any = None, is_inference: bool = False
    ) -> tuple[Any, None]:
        del is_inference
        if input is None:
            raise_unsupported_multiplex_runtime("VideoTrackingMultiplex.forward(input)")
        if (
            getattr(self, "training", False)
            or not self.forward_backbone_per_frame_for_eval
        ):
            backbone_out = self.forward_image(
                input.img_batch,
                need_interactive_out=True,
                need_propagation_out=True,
            )
        else:
            backbone_out = {}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)
        return previous_stages_out, None


class VideoTrackingDynamicMultiplex(VideoTrackingMultiplex):
    def __init__(
        self,
        enable_dynamic_training: bool = True,
        rand_num_transition_points: bool = True,
        max_num_transition_points: int = 3,
        add_all_transition_frames_as_cond: bool = True,
        max_trans_frames_in_attn: int = 4,
        is_dynamic_model: bool = True,
        is_dynamic_vos_evaluation: bool = False,
        **kwargs: Any,
    ):
        super().__init__(is_dynamic_model=is_dynamic_model, **kwargs)
        self.enable_dynamic_training = enable_dynamic_training
        self.rand_num_transition_points = rand_num_transition_points
        self.max_num_transition_points = max_num_transition_points
        self.add_all_transition_frames_as_cond = add_all_transition_frames_as_cond
        self.max_trans_frames_in_attn = max_trans_frames_in_attn
        self.is_dynamic_vos_evaluation = is_dynamic_vos_evaluation


__all__ = [
    "MLP",
    "MultiplexController",
    "MultiplexMaskDecoder",
    "MultiplexState",
    "NO_OBJ_SCORE",
    "SAMOutput",
    "StageOutput",
    "VideoTrackingDynamicMultiplex",
    "VideoTrackingMultiplex",
    "_append",
    "_merge",
    "_replace_rows",
    "concat_points",
    "neck_outs",
]
