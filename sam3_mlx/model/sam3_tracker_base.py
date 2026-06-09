# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of the official SAM3 SAM2-style memory tracker base.

Ported from the official SAM3 ``sam3_tracker_base`` at upstream commit
``2814fa619404a722d03e9a012e083e4f293a4e53``.

Tracker port status:

- Construction layer, SAM-head forward, memory encoder, and memory-conditioning
  tracker slices are real MLX ports.
- Image-backbone execution through ``sam2_backbone_out`` and direct mask-output
  prompts are ported for the single-object predictor slice. Training remains an
  explicit fail-fast boundary.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.model.sam3_tracker_utils import (
    get_1d_sine_pe,
    select_closest_cond_frames,
)
from sam3_mlx.sam.mask_decoder import MaskDecoder, MLP
from sam3_mlx.sam.prompt_encoder import PromptEncoder
from sam3_mlx.sam.transformer import TwoWayTransformer


# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


def _is_mlx_array(value: Any) -> bool:
    return type(value).__module__.startswith("mlx.")


def _trunc_normal(shape, std: float = 0.02) -> mx.array:
    """timm ``trunc_normal_(std=...)`` equivalent (mean 0, truncated at +/-2 std).

    These tensors are learned parameters that the checkpoint overwrites at load
    time; the init only matters when the module is exercised without weights.
    """

    return mx.random.truncated_normal(lower=-2.0, upper=2.0, shape=shape) * std


def _unsupported_tracker_base(method: str):
    raise_unsupported(
        f"sam3_mlx.model.sam3_tracker_base.Sam3TrackerBase.{method}",
        reason="video-tracker",
        detail=(
            "Several Sam3TrackerBase slices are ported, but this tracker "
            "method remains outside the current MLX runtime increment."
        ),
        alternative=(
            "sam3_mlx.model.sam3_tracker_utils or "
            "sam3_mlx.model.sam3_video_inference.Sam3VideoInference"
        ),
    )


class Sam3TrackerBase(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        maskmem_backbone,
        num_maskmem=7,  # default: 1 current frame + 6 previous frames
        image_size=1008,
        backbone_stride=14,  # stride of the image backbone output
        max_cond_frames_in_attn=-1,
        keep_first_cond_frame=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        forward_backbone_per_frame_for_eval=False,
        memory_temporal_stride_for_eval=1,
        offload_output_to_cpu_for_eval=False,
        trim_past_non_cond_mem_for_eval=False,
        non_overlap_masks_for_mem_enc=False,
        max_obj_ptrs_in_encoder=16,
        sam_mask_decoder_extra_args=None,
        compile_all_components=False,
        use_memory_selection=False,
        mf_threshold=0.01,
    ):
        super().__init__()

        # Part 1: the image backbone
        self.backbone = backbone
        self.num_feature_levels = 3
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        # A conv layer to downsample the GT mask prompt to stride 4 (the same
        # stride as low-res SAM mask logits) and to change its scales from 0~1 to
        # SAM logit scale, so it can be fed into the SAM mask decoder.
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)

        # Part 2: encoder-only transformer fusing current frame visual features
        # with memories from past frames
        assert transformer.decoder is None, "transformer should be encoder-only"
        self.transformer = transformer
        self.hidden_dim = transformer.d_model

        # Part 3: memory encoder for the previous frame's outputs
        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(self.maskmem_backbone, "out_proj") and hasattr(
            self.maskmem_backbone.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.maskmem_backbone.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible

        # Temporal encoding of the memories
        self.maskmem_tpos_enc = _trunc_normal((num_maskmem, 1, 1, self.mem_dim))

        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = _trunc_normal((1, 1, self.hidden_dim))
        self.no_mem_pos_enc = _trunc_normal((1, 1, self.hidden_dim))
        # Apply sigmoid to the output raw mask logits before feeding them into
        # the memory encoder
        self.sigmoid_scale_for_mem_enc = 20.0
        self.sigmoid_bias_for_mem_enc = -10.0
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask
        # without using a SAM prompt encoder + mask decoder
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking

        # Part 4: SAM-style prompt encoder and mask decoder
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        # we resize the mask if it doesn't match `self.input_mask_size`
        self.input_mask_size = self.low_res_mask_size * 4
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.offload_output_to_cpu_for_eval = offload_output_to_cpu_for_eval
        self.trim_past_non_cond_mem_for_eval = trim_past_non_cond_mem_for_eval
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.no_obj_ptr = _trunc_normal((1, self.hidden_dim))
        self.no_obj_embed_spatial = _trunc_normal((1, self.mem_dim))

        self._build_sam_heads()
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.keep_first_cond_frame = keep_first_cond_frame

        # Use frame filtering according to SAM2Long
        self.use_memory_selection = use_memory_selection
        self.mf_threshold = mf_threshold

        # Training-only teacher forcing flag; eval inference never reads it.
        self.teacher_force_obj_scores_for_mem = False

        # Compile all components of the model
        self.compile_all_components = compile_all_components
        if self.compile_all_components:
            self._compile_all_components()

    @property
    def device(self):
        return "mlx"

    def _build_sam_heads(self):
        """Build SAM-style prompt encoder and mask decoder."""
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride

        # build PromptEncoder and MaskDecoder from SAM
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.sam_mask_decoder = MaskDecoder(
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
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            **(self.sam_mask_decoder_extra_args or {}),
        )
        # a linear projection on SAM output tokens to turn them into object
        # pointers (upstream overwrites the Linear with an MLP)
        self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        # a linear projection on temporal positional encoding in object pointers
        self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.mem_dim)

    def _get_tpos_enc(self, rel_pos_list, device=None, max_abs_pos=None, dummy=False):
        del device  # MLX runtime is explicit; no torch-style device movement.
        if dummy:
            return mx.zeros((len(rel_pos_list), self.mem_dim))

        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        pos_enc = mx.array(rel_pos_list, dtype=mx.float32) / t_diff_max
        pos_enc = get_1d_sine_pe(pos_enc, dim=self.hidden_dim)
        return self.obj_ptr_tpos_proj(pos_enc)

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
        gt_masks=None,
    ):
        """Forward SAM prompt encoder + mask decoder for one frame.

        ``backbone_features`` is [B, C, H, W] with C=hidden_dim and
        H=W=image_size/backbone_stride.
        Returns the 7-tuple (low_res_multimasks, high_res_multimasks, ious,
        low_res_masks, high_res_masks, obj_ptr, object_score_logits).
        """
        del gt_masks  # training-only teacher forcing is out of scope
        B = backbone_features.shape[0]
        assert backbone_features.shape[1] == self.sam_prompt_embed_dim
        assert backbone_features.shape[2] == self.sam_image_embedding_size
        assert backbone_features.shape[3] == self.sam_image_embedding_size

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.shape[0] == B and sam_point_labels.shape[0] == B
        else:
            # pad with an empty point (label -1) when no points are provided
            sam_point_coords = mx.zeros((B, 1, 2))
            sam_point_labels = -mx.ones((B, 1), dtype=mx.int32)

        # b) Handle mask prompts
        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4 and tuple(mask_inputs.shape[:2]) == (
                B,
                1,
            )
            if tuple(mask_inputs.shape[-2:]) != tuple(
                self.sam_prompt_encoder.mask_input_size
            ):
                sam_mask_prompt = interpolate(
                    mask_inputs.astype(mx.float32),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            # SAM's prompt encoder adds a learned no_mask_embed in this case
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        image_pe = self.sam_prompt_encoder.get_dense_pe()
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )

        is_obj_appearing = object_score_logits > 0
        # spatial-memory mask is a hard obj/no-obj choice, matching the prediction
        low_res_multimasks = mx.where(
            is_obj_appearing.reshape(B, 1, 1, 1),
            low_res_multimasks,
            NO_OBJ_SCORE,
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
            # take the best mask prediction (highest IoU estimate)
            best_iou_inds = mx.argmax(ious, axis=-1)
            batch_inds = mx.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]
            if sam_output_tokens.shape[1] > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        lambda_is_obj_appearing = is_obj_appearing.astype(mx.float32)
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs):
        """Convert binary mask prompts directly into SAM-style mask outputs."""
        if len(mask_inputs.shape) != 4 or mask_inputs.shape[1] != 1:
            raise ValueError("mask_inputs must have shape [B, 1, H, W].")

        out_scale, out_bias = 20.0, -10.0
        mask_inputs_float = mask_inputs.astype(mx.float32)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = interpolate(
            high_res_masks,
            size=(
                high_res_masks.shape[-2] // self.backbone_stride * 4,
                high_res_masks.shape[-1] // self.backbone_stride * 4,
            ),
            align_corners=False,
            mode="bilinear",
        )
        ious = mx.ones((mask_inputs.shape[0], 1), dtype=mx.float32)

        # ``mask_downsample`` is a raw MLX Conv2d to preserve checkpoint key
        # names, so call it at its NHWC boundary and return to NCHW for SAM.
        mask_prompt = self.mask_downsample(
            mask_inputs_float.transpose(0, 2, 3, 1)
        ).transpose(0, 3, 1, 2)
        _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
            backbone_features=backbone_features,
            mask_inputs=mask_prompt,
            high_res_features=high_res_features,
            gt_masks=mask_inputs,
        )

        is_obj_appearing = mx.any(
            mask_inputs.reshape(mask_inputs.shape[0], -1).astype(mx.float32) > 0.0,
            axis=1,
        )[:, None]
        lambda_is_obj_appearing = is_obj_appearing.astype(mx.float32)
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward(self, *args, **kwargs):
        _unsupported_tracker_base("forward")

    def __call__(self, *args, **kwargs):
        _unsupported_tracker_base("__call__")

    def forward_image(self, img_batch):
        """Get tracker visual features from the SAM3 image backbone."""
        if self.backbone is None:
            raise RuntimeError(
                "Sam3TrackerBase.forward_image requires a backbone. Build the "
                "tracker with with_backbone=True or provide cached frame features."
            )

        backbone_all = self.backbone.forward_image(img_batch)
        if "tracker_backbone_out" in backbone_all:
            backbone_out = backbone_all["tracker_backbone_out"]
        else:
            backbone_out = backbone_all.get("sam2_backbone_out")
        if backbone_out is None:
            raise ValueError(
                "Tracker forward_image requires backbone.forward_image(...) to "
                "return sam2_backbone_out or tracker_backbone_out. Build the "
                "SAM3 backbone with enable_inst_interactivity=True."
            )
        if "backbone_fpn" not in backbone_out or "vision_pos_enc" not in backbone_out:
            raise KeyError(
                "tracker backbone output must contain backbone_fpn and vision_pos_enc."
            )
        if len(backbone_out["backbone_fpn"]) < 2:
            raise ValueError("Expected at least two high-resolution feature levels.")

        backbone_out = backbone_out.copy()
        backbone_out["backbone_fpn"] = list(backbone_out["backbone_fpn"])
        backbone_out["vision_pos_enc"] = list(backbone_out["vision_pos_enc"])
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features into official ``(HW, B, C)`` form."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [
            x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
            for x in feature_maps
        ]
        vision_pos_embeds = [
            x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 0, 1)
            for x in vision_pos_embeds
        ]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute backbone features for the requested image ids."""
        img_ids_np = np.asarray(img_ids).reshape(-1).astype(np.int64)
        if img_ids_np.size == 0:
            raise ValueError("img_ids must contain at least one image id.")
        unique_img_ids, inv_ids = np.unique(img_ids_np, return_inverse=True)
        unique_ids_mx = mx.array(unique_img_ids, dtype=mx.int32)

        image = img_batch[unique_ids_mx]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)

        if unique_img_ids.size != img_ids_np.size:
            inv_ids_mx = mx.array(inv_ids, dtype=mx.int32)
            image = image[inv_ids_mx]
            vision_feats = [x[:, inv_ids_mx] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids_mx] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def cal_mem_score(self, object_score_logits, iou_score):
        object_score_norm = mx.where(
            object_score_logits > 0,
            mx.sigmoid(object_score_logits) * 2 - 1,  # rescale to [0, 1]
            mx.zeros_like(object_score_logits),
        )
        score_per_frame = (object_score_norm * iou_score).mean()
        return score_per_frame

    def frame_filter(self, output_dict, track_in_reverse, frame_idx, num_frames, r):
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

        valid_indices = []
        for i in range(start, end, step):
            out = output_dict["non_cond_frame_outputs"].get(i)
            if out is None or "eff_iou_score" not in out:
                continue

            score_per_frame = out["eff_iou_score"]
            score_value = (
                float(score_per_frame)
                if _is_mlx_array(score_per_frame)
                else score_per_frame
            )
            if score_value > self.mf_threshold:
                valid_indices.insert(0, i)

            if len(valid_indices) >= max_num - 1:
                break

        if must_include not in valid_indices:
            valid_indices.append(must_include)

        return valid_indices

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        use_prev_mem_frame=True,
    ):
        """Fuse current-frame visual features with previous spatial memories."""
        B = current_vision_feats[-1].shape[1]
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        training = bool(getattr(self, "training", False))

        if self.num_maskmem == 0:
            return current_vision_feats[-1].transpose(1, 2, 0).reshape(B, C, H, W)

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if not is_init_cond_frame and use_prev_mem_frame:
            to_cat_prompt = []
            to_cat_prompt_mask = []
            to_cat_prompt_pos_embed = []

            assert len(output_dict["cond_frame_outputs"]) > 0
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

            r = 1 if training else self.memory_temporal_stride_for_eval
            if self.use_memory_selection:
                valid_indices = self.frame_filter(
                    output_dict, track_in_reverse, frame_idx, num_frames, r
                )

            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                if self.use_memory_selection:
                    if t_rel > len(valid_indices):
                        continue
                    prev_frame_idx = valid_indices[-t_rel]
                else:
                    if t_rel == 1:
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
                feats = prev["maskmem_features"]
                seq_len = feats.shape[-2] * feats.shape[-1]
                to_cat_prompt.append(
                    feats.reshape(B, self.mem_dim, seq_len).transpose(2, 0, 1)
                )
                to_cat_prompt_mask.append(mx.zeros((B, seq_len), dtype=mx.bool_))

                maskmem_enc = prev["maskmem_pos_enc"][-1]
                maskmem_enc = maskmem_enc.reshape(B, self.mem_dim, seq_len).transpose(
                    2, 0, 1
                )
                if (
                    is_selected_cond_frame
                    and getattr(self, "cond_frame_spatial_embedding", None) is not None
                ):
                    maskmem_enc = maskmem_enc + self.cond_frame_spatial_embedding

                t = t_pos if not is_selected_cond_frame else 0
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t - 1]
                )
                to_cat_prompt_pos_embed.append(maskmem_enc)

            prob_drop = getattr(self, "prob_to_dropout_spatial_mem", 0.0)
            if training and prob_drop > 0:
                _unsupported_tracker_base(
                    "_prepare_memory_conditioned_features(training_spatial_mem_dropout)"
                )

            max_obj_ptrs_in_encoder = (
                self.max_obj_ptrs_in_encoder
                if num_frames is None
                else min(num_frames, self.max_obj_ptrs_in_encoder)
            )
            if not training:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs
            pos_and_ptrs = [
                ((frame_idx - t) * tpos_sign_mul, out["obj_ptr"], True)
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
                    t, unselected_cond_outputs.get(t)
                )
                if out is not None:
                    pos_and_ptrs.append((t_diff, out["obj_ptr"], False))

            if len(pos_and_ptrs) > 0:
                pos_list, ptrs_list, is_selected_cond_frame_list = zip(*pos_and_ptrs)
                obj_ptrs = mx.stack(ptrs_list, axis=0)
                if getattr(self, "cond_frame_obj_ptr_embedding", None) is not None:
                    obj_ptrs = obj_ptrs + self.cond_frame_obj_ptr_embedding * mx.array(
                        is_selected_cond_frame_list, dtype=mx.float32
                    ).reshape(-1, 1, 1)

                obj_pos = self._get_tpos_enc(
                    pos_list,
                    max_abs_pos=max_obj_ptrs_in_encoder,
                )
                obj_pos = mx.broadcast_to(
                    obj_pos[:, None, :], (obj_pos.shape[0], B, obj_pos.shape[1])
                )

                if self.mem_dim < C:
                    tokens_per_ptr = C // self.mem_dim
                    obj_ptrs = obj_ptrs.reshape(-1, B, tokens_per_ptr, self.mem_dim)
                    obj_ptrs = obj_ptrs.transpose(0, 2, 1, 3).reshape(
                        -1, B, self.mem_dim
                    )
                    obj_pos = mx.repeat(obj_pos, tokens_per_ptr, axis=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_mask.append(None)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]
            else:
                num_obj_ptr_tokens = 0
        else:
            pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
            return pix_feat_with_mem.transpose(1, 2, 0).reshape(B, C, H, W)

        prompt = mx.concat(to_cat_prompt, axis=0)
        prompt_mask = None
        prompt_pos_embed = mx.concat(to_cat_prompt_pos_embed, axis=0)
        encoder_out = self.transformer.encoder(
            src=current_vision_feats,
            src_key_padding_mask=[None],
            src_pos=current_vision_pos_embeds,
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=feat_sizes,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        return encoder_out["memory"].transpose(1, 2, 0).reshape(B, C, H, W)

    def _encode_new_memory(
        self,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        output_dict=None,
        is_init_cond_frame=False,
    ):
        """Encode the current image + prediction into a spatial memory feature."""
        del image, output_dict, is_init_cond_frame
        B = current_vision_feats[-1].shape[1]  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].transpose(1, 2, 0).reshape(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc:
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        if is_mask_from_pts:
            mask_for_mem = (pred_masks_high_res > 0).astype(mx.float32)
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = mx.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc

        # our tracker always uses the SimpleMaskEncoder memory backbone
        maskmem_out = self.maskmem_backbone(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True
        )
        maskmem_features = self._maybe_clone(maskmem_out["vision_features"])
        maskmem_pos_enc = [self._maybe_clone(m) for m in maskmem_out["vision_pos_enc"]]
        # add a no-object embedding to the spatial memory where the frame is
        # predicted to be occluded (no object appearing)
        is_obj_appearing = (object_score_logits > 0).astype(mx.float32)
        maskmem_features = maskmem_features + (
            1 - is_obj_appearing.reshape(B, 1, 1, 1)
        ) * self.no_obj_embed_spatial.reshape(1, self.mem_dim, 1, 1)

        return maskmem_features, maskmem_pos_enc

    def forward_tracking(self, backbone_out, input, return_dict=False):
        """Forward tracking over precomputed frame features."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            _, vision_feats, vision_pos_embeds, feat_sizes = (
                self._prepare_backbone_features(backbone_out)
            )

        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        for stage_id in processing_order:
            img_ids = input.find_inputs[stage_id].img_ids
            if img_feats_already_computed:
                current_image = input.img_batch[img_ids]
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                (
                    current_image,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(input.img_batch, img_ids)

            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                image=current_image,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id),
                output_dict=output_dict,
                num_frames=num_frames,
            )
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                getattr(self, "add_all_frames_to_correct_as_cond", False)
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        return [
            {k: v for k, v in frame_out.items() if k != "obj_ptr"}
            for frame_out in all_frame_outputs
        ]

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        image,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        use_prev_mem_frame=True,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.transpose(1, 2, 0).reshape(x.shape[1], x.shape[2], *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        if mask_inputs is not None:
            sam_outputs = self._use_mask_as_output(
                current_vision_feats[-1]
                .transpose(1, 2, 0)
                .reshape(-1, self.hidden_dim, *feat_sizes[-1]),
                high_res_features,
                mask_inputs,
            )
        else:
            pix_feat_with_mem = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
            )
            if prev_sam_mask_logits is not None:
                assert getattr(self, "iter_use_prev_mask_pred", False)
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=self._use_multimask(is_init_cond_frame, point_inputs),
            )

        (
            _,
            _high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        if self.use_memory_selection:
            current_out["object_score_logits"] = object_score_logits
            iou_score = mx.max(ious, axis=-1)
            current_out["iou_score"] = iou_score
            current_out["eff_iou_score"] = self.cal_mem_score(
                object_score_logits, iou_score
            )
        if not getattr(self, "training", False):
            current_out["object_score_logits"] = object_score_logits

        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                output_dict=output_dict,
                is_init_cond_frame=is_init_cond_frame,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        if self.offload_output_to_cpu_for_eval and not getattr(self, "training", False):
            _unsupported_tracker_base("track_step(offload_output_to_cpu_for_eval=True)")

        def _trim_past_out(past_out):
            if past_out is None:
                return None
            return {
                "pred_masks": past_out["pred_masks"],
                "obj_ptr": past_out["obj_ptr"],
                "object_score_logits": past_out["object_score_logits"],
            }

        if self.trim_past_non_cond_mem_for_eval and not getattr(
            self, "training", False
        ):
            r = self.memory_temporal_stride_for_eval
            past_frame_idx = frame_idx - r * self.num_maskmem
            past_out = output_dict["non_cond_frame_outputs"].get(past_frame_idx)
            if past_out is not None:
                eff_iou = past_out.get("eff_iou_score", 0)
                eff_iou_value = float(eff_iou) if _is_mlx_array(eff_iou) else eff_iou
                if (
                    self.use_memory_selection and eff_iou_value < self.mf_threshold
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

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].shape[1]
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """Suppress non-winning object logits at overlapping spatial locations."""
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

    def _compile_all_components(self):
        _unsupported_tracker_base("_compile_all_components")

    def _maybe_clone(self, x):
        return mx.array(x) if _is_mlx_array(x) else np.array(x, copy=True)


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new point coordinates and labels to previous point inputs."""

    if old_point_inputs is None:
        points, labels = new_points, new_labels
    elif _is_mlx_array(new_points):
        points = mx.concat([old_point_inputs["point_coords"], new_points], axis=1)
        labels = mx.concat([old_point_inputs["point_labels"], new_labels], axis=1)
    else:
        points = np.concatenate([old_point_inputs["point_coords"], new_points], axis=1)
        labels = np.concatenate([old_point_inputs["point_labels"], new_labels], axis=1)

    return {"point_coords": points, "point_labels": labels}


__all__ = [
    "NO_OBJ_SCORE",
    "Sam3TrackerBase",
    "concat_points",
]
