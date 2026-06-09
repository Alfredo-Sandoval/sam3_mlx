import math
from functools import partial
from typing import Any, Dict, Optional, Union
import mlx.core as mx
import mlx.nn as nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.sam.rope import apply_rotary_enc, apply_rotary_enc_real, compute_axial_cis
from sam3_mlx.sam.transformer import RoPEAttention
from sam3_mlx.model.act_ckpt_utils import activation_ckpt_wrapper
from sam3_mlx.model.box_ops import box_cxcywh_to_xyxy

from sam3_mlx.model.model_misc import (
    MLP,
    get_activation_fn,
    get_clones,
    gen_sineembed_for_position,
    inverse_sigmoid,
    MultiheadAttentionWrapper as MultiHeadAttention,
)


def _raise_decoder_unsupported(feature: str, *, reason: str, detail: str) -> None:
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
    )


def _attention_output(result):
    return result[0] if isinstance(result, (tuple, list)) else result


def _with_pos_embed(array, pos):
    return array if pos is None else array + pos


def _call_short_or_mha_attention(module, q, k, v, **kwargs):
    """Call either SAM q/k/v attention or the local MHA wrapper."""
    kwargs = {name: value for name, value in kwargs.items() if value is not None}
    try:
        return _attention_output(module(q=q, k=k, v=v, **kwargs))
    except TypeError as short_name_error:
        try:
            return _attention_output(module(query=q, key=k, value=v, **kwargs))
        except TypeError:
            raise short_name_error


def _dropout(array: mx.array, p: float, training: bool) -> mx.array:
    if p == 0.0 or not training:
        return array
    keep_prob = 1.0 - p
    if keep_prob <= 0.0:
        return mx.zeros_like(array)
    keep = mx.random.bernoulli(p=keep_prob, shape=array.shape)
    return array * keep.astype(array.dtype) / keep_prob


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        cross_attention: nn.Module,
        n_heads: int,
        use_text_cross_attention: bool = False,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = cross_attention
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention text
        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = MultiHeadAttention(d_model, n_heads)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(array, pos):
        return array if pos is None else array + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[mx.array],  # nq, bs, d_model
        tgt_query_pos: Optional[mx.array] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[mx.array] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[mx.array] = None,
        tgt_reference_points: Optional[mx.array] = None,  # nq, bs, 4
        memory_text: Optional[mx.array] = None,  # num_token, bs, d_model
        text_attention_mask: Optional[mx.array] = None,  # bs, num_token
        # for memory
        memory: Optional[mx.array] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[mx.array] = None,
        memory_level_start_index: Optional[mx.array] = None,  # num_levels
        memory_spatial_shapes: Optional[mx.array] = None,  # bs, num_levels, 2
        memory_pos: Optional[mx.array] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[mx.array] = None,  # mask used for self-attention
        cross_attn_mask: Optional[mx.array] = None,  # mask used for cross-attention
        # dac
        dac=False,
        dac_use_selfatt_ln=True,
        presence_token=None,
        # skip inside deformable attn
        identity=0.0,
        **kwargs,  # additional kwargs for compatibility
    ):
        # self attention
        if self.self_attn is not None:
            if dac:
                assert tgt.shape[0] % 2 == 0
                num_o2o_queries = tgt.shape[0] // 2
                tgt_o2o = tgt[:num_o2o_queries]
                tgt_query_pos_o2o = tgt_query_pos[:num_o2o_queries]
                tgt_o2m = tgt[num_o2o_queries:]
            else:
                tgt_o2o = tgt
                tgt_query_pos_o2o = tgt_query_pos

            if presence_token is not None:
                tgt_o2o = mx.concat([presence_token, tgt_o2o], axis=0)
                tgt_query_pos_o2o = mx.concat(
                    [mx.zeros_like(presence_token), tgt_query_pos], axis=0
                )
                tgt_query_pos = mx.concat(
                    [mx.zeros_like(presence_token), tgt_query_pos], axis=0
                )

            q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o).transpose(1, 0, 2)
            tgt2 = self.self_attn(
                q, k, tgt_o2o.transpose(1, 0, 2), attn_mask=self_attn_mask
            ).transpose(1, 0, 2)
            tgt_o2o = tgt_o2o + self.dropout2(tgt2)
            if dac:
                if not dac_use_selfatt_ln:
                    tgt_o2o = self.norm2(tgt_o2o)
                tgt = mx.concat((tgt_o2o, tgt_o2m), axis=0)  # Recombine
                if dac_use_selfatt_ln:
                    tgt = self.norm2(tgt)
            else:
                tgt = tgt_o2o
                tgt = self.norm2(tgt)

        if self.use_text_cross_attention:
            memory_text = memory_text.transpose(1, 0, 2)
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos).transpose(1, 0, 2),
                memory_text,
                memory_text,
                key_padding_mask=text_attention_mask,
            ).transpose(1, 0, 2)
            tgt = tgt + self.catext_dropout(tgt2)
            tgt = self.catext_norm(tgt)

        if presence_token is not None:
            presence_token_mask = mx.zeros_like(cross_attn_mask[:, :1, :])
            cross_attn_mask = mx.concat(
                [presence_token_mask, cross_attn_mask], axis=1
            )  # (bs*nheads, 1+nq, hw)

        # Cross attention to image
        tgt2 = self.cross_attn(
            queries=self.with_pos_embed(tgt, tgt_query_pos).transpose(1, 0, 2),
            keys=self.with_pos_embed(memory, memory_pos).transpose(1, 0, 2),
            values=memory.transpose(1, 0, 2),
            attn_mask=cross_attn_mask,
            key_padding_mask=(
                memory_key_padding_mask.transpose(0, 1)
                if memory_key_padding_mask is not None
                else None
            ),
        ).transpose(1, 0, 2)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        presence_token_out = None
        if presence_token is not None:
            presence_token_out = tgt[:1]
            tgt = tgt[1:]

        return tgt, presence_token_out

    def __call__(
        self,
        # for tgt
        tgt: Optional[mx.array],  # nq, bs, d_model
        tgt_query_pos: Optional[mx.array] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[mx.array] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[mx.array] = None,
        tgt_reference_points: Optional[mx.array] = None,  # nq, bs, 4
        memory_text: Optional[mx.array] = None,  # num_token, bs, d_model
        text_attention_mask: Optional[mx.array] = None,  # bs, num_token
        # for memory
        memory: Optional[mx.array] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[mx.array] = None,
        memory_level_start_index: Optional[mx.array] = None,  # num_levels
        memory_spatial_shapes: Optional[mx.array] = None,  # bs, num_levels, 2
        memory_pos: Optional[mx.array] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[mx.array] = None,  # mask used for self-attention
        cross_attn_mask: Optional[mx.array] = None,  # mask used for cross-attention
        # dac
        dac=False,
        dac_use_selfatt_ln=True,
        presence_token=None,
        # skip inside deformable attn
        identity=0.0,
        **kwargs,  # additional kwargs for compatibility
    ):
        return self.forward(
            tgt=tgt,
            tgt_query_pos=tgt_query_pos,
            tgt_query_sine_embed=tgt_query_sine_embed,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_reference_points=tgt_reference_points,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_level_start_index=memory_level_start_index,
            memory_spatial_shapes=memory_spatial_shapes,
            memory_pos=memory_pos,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
            dac=dac,
            dac_use_selfatt_ln=dac_use_selfatt_ln,
            presence_token=presence_token,
            identity=identity,
            **kwargs,
        )


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        interaction_layer,
        layer,
        num_layers: int,
        num_queries: int,
        return_intermediate: bool,
        box_refine: bool = False,
        num_o2m_queries: int = 0,
        dac: bool = False,
        boxRPB: str = "none",
        # Experimental: An object query for SAM 2 tasks
        instance_query: bool = False,
        # Defines the number of additional instance queries,
        # 1 or 4 are the most likely for single vs multi mask support
        num_instances: int = 1,  # Irrelevant if instance_query is False
        dac_use_selfatt_ln: bool = True,
        use_act_checkpoint: bool = False,
        compile_mode=None,
        presence_token: bool = False,
        clamp_presence_logits: bool = True,
        clamp_presence_logit_max_val: float = 10.0,
        use_normed_output_consistently: bool = True,
        separate_box_head_instance: bool = False,
        separate_norm_instance: bool = False,
        resolution: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        super().__init__()
        if compile_mode not in (None, False):
            _raise_decoder_unsupported(
                "sam3_mlx.model.decoder.TransformerDecoder(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.fine_layers = (
            get_clones(interaction_layer, num_layers)
            if interaction_layer is not None
            else [None] * num_layers
        )
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.dac = dac
        if dac:
            self.num_o2m_queries = num_queries
            tot_num_queries = num_queries
        else:
            self.num_o2m_queries = num_o2m_queries
            tot_num_queries = num_queries + num_o2m_queries
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.query_embed = nn.Embedding(tot_num_queries, d_model)
        self.instance_query_embed = None
        self.instance_query_reference_points = None
        self.use_instance_query = instance_query
        self.num_instances = num_instances
        self.use_normed_output_consistently = use_normed_output_consistently

        self.instance_norm = nn.LayerNorm(d_model) if separate_norm_instance else None
        self.instance_bbox_embed = None
        if separate_box_head_instance:
            self.instance_bbox_embed = MLP(d_model, d_model, 4, 3)
        if instance_query:
            self.instance_query_embed = nn.Embedding(num_instances, d_model)
        self.box_refine = box_refine

        if box_refine:
            init_fn = nn.init.constant(0.0)
            self.bbox_embed.layers[-1].weight = init_fn(
                self.bbox_embed.layers[-1].weight
            )
            self.bbox_embed.layers[-1].bias = init_fn(self.bbox_embed.layers[-1].bias)

            self.reference_points = nn.Embedding(num_queries, 4)
            if instance_query:
                self.instance_reference_points = nn.Embedding(num_instances, 4)

        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB = boxRPB
        if boxRPB != "none":
            try:
                nheads = self.layers[0].cross_attn_image.num_heads
            except AttributeError:
                nheads = self.layers[0].cross_attn.num_heads

            n_input = 4 if boxRPB == "both" else 2
            self.boxRPB_embed_x = MLP(n_input, d_model, nheads, 2)
            self.boxRPB_embed_y = MLP(n_input, d_model, nheads, 2)
            self.compilable_cord_cache = None
            self.compilable_stored_size = None
            self.coord_cache = {}

            if resolution is not None and stride is not None:
                feat_size = resolution // stride
                coords_h, coords_w = self._get_coords(feat_size, feat_size)
                self.compilable_cord_cache = (coords_h, coords_w)
                self.compilable_stored_size = (feat_size, feat_size)

        self.roi_pooler = None

        self.frozen = frozen

        self.presence_token = None
        self.clamp_presence_logits = clamp_presence_logits
        self.clamp_presence_logit_max_val = clamp_presence_logit_max_val
        if presence_token:
            self.presence_token = nn.Embedding(1, d_model)
            self.presence_token_head = MLP(d_model, d_model, 1, 3)
            self.presence_token_out_norm = nn.LayerNorm(d_model)

        self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)
        self.dac_use_selfatt_ln = dac_use_selfatt_ln
        self.use_act_checkpoint = use_act_checkpoint

        init_normal = nn.init.normal()
        self.query_embed.weight = init_normal(self.query_embed.weight)
        if self.instance_query_embed is not None:
            self.instance_query_embed.weight = init_normal(
                self.instance_query_embed.weight
            )

        assert self.roi_pooler is None
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"
        if frozen:
            self.freeze()

        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    @staticmethod
    def _get_coords(H, W):
        if isinstance(H, mx.array):
            H = H.item()
        if isinstance(W, mx.array):
            W = W.item()

        coords_h = mx.arange(0, H, dtype=mx.float32) / H
        coords_w = mx.arange(0, W, dtype=mx.float32) / W
        return coords_h, coords_w

    def _get_rpb_matrix(self, reference_boxes, feat_size):
        H, W = feat_size
        boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(1, 0, 2)
        bs, num_queries, _ = boxes_xyxy.shape

        if feat_size not in self.coord_cache:
            self.coord_cache[feat_size] = self._get_coords(H, W)
        coords_h, coords_w = self.coord_cache[feat_size]

        assert coords_h.shape == (H,)
        assert coords_w.shape == (W,)

        deltas_y = (
            coords_h.reshape(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
        )
        deltas_y = deltas_y.reshape(bs, num_queries, -1, 2)
        deltas_x = (
            coords_w.reshape(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
        )
        deltas_x = deltas_x.reshape(bs, num_queries, -1, 2)

        if self.boxRPB in ["log", "both"]:
            deltas_x_log = deltas_x * 8  # normalize to -8, 8
            deltas_x_log = (
                mx.sign(deltas_x_log)
                * mx.log2(mx.abs(deltas_x_log) + 1.0)
                / math.log2(8)
            )

            deltas_y_log = deltas_y * 8  # normalize to -8, 8
            deltas_y_log = (
                mx.sign(deltas_y_log)
                * mx.log2(mx.abs(deltas_y_log) + 1.0)
                / math.log2(8)
            )
            if self.boxRPB == "log":
                deltas_x = deltas_x_log
                deltas_y = deltas_y_log
            else:
                deltas_x = mx.concat([deltas_x, deltas_x_log], axis=-1)
                deltas_y = mx.concat([deltas_y, deltas_y_log], axis=-1)

        deltas_x = self.boxRPB_embed_x(
            x=deltas_x,
        )  # bs, num_queries, W, n_heads
        deltas_y = self.boxRPB_embed_y(
            x=deltas_y,
        )  # bs, num_queries, H, n_heads

        B = mx.expand_dims(deltas_y, axis=3) + mx.expand_dims(deltas_x, axis=2)
        # bs, num_queries, H, W, n_heads
        B = B.flatten(2, 3)  # bs, num_queries, H*W, n_heads
        B = B.transpose(0, 3, 1, 2)  # bs, n_heads, num_queries, H*W
        return B

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        reference_boxes: Optional[mx.array] = None,  # num_queries, bs, 4
        # for memory
        level_start_index: Optional[mx.array] = None,
        spatial_shapes: Optional[mx.array] = None,
        valid_ratios: Optional[mx.array] = None,
        # for text
        memory_text: Optional[mx.array] = None,
        text_attention_mask: Optional[mx.array] = None,
        # if `apply_dac` is None, it will default to `self.dac`
        apply_dac: Optional[bool] = None,
        is_instance_prompt=False,
        decoder_extra_kwargs: Optional[Dict] = None,
        # ROI memory bank
        obj_roi_memory_feat=None,
        obj_roi_memory_mask=None,
        box_head_trk=None,
    ):
        if memory_mask is not None:
            assert self.boxRPB == "none", (
                "inputting a memory_mask in the presence of boxRPB is unexpected/not implemented"
            )

        apply_dac = apply_dac if apply_dac is not None else self.dac
        if apply_dac:
            assert (tgt.shape[0] == self.num_queries) or (
                self.use_instance_query
                and (tgt.shape[0] == self.instance_query_embed.weight.shape[0])
            )

            tgt = mx.repeat(tgt, repeats=2, axis=0)
            # note that we don't tile tgt_mask, since DAC doesn't
            # use self-attention in o2m queries
            if reference_boxes is not None:
                assert (reference_boxes.shape[0] == self.num_queries) or (
                    self.use_instance_query
                    and (
                        reference_boxes.shape[0]
                        == self.instance_query_embed.weight.shape[0]
                    )
                )
                reference_boxes = mx.repeat(reference_boxes, repeats=2, axis=0)

        bs = tgt.shape[1]
        intermediate = []
        intermediate_presence_logits = []
        presence_feats = None

        if self.box_refine:
            if reference_boxes is None:
                reference_boxes = self.reference_points.weight[:, None]
                reference_boxes = (
                    mx.tile(reference_boxes, (2, bs, 1))
                    if apply_dac
                    else mx.tile(reference_boxes, (1, bs, 1))
                )
                reference_boxes = mx.sigmoid(reference_boxes)
            intermediate_ref_boxes = [reference_boxes]
        else:
            reference_boxes = None
            intermediate_ref_boxes = None

        output = tgt
        presence_out = None
        if self.presence_token is not None and is_instance_prompt is False:
            presence_out = mx.broadcast_to(
                self.presence_token.weight[None], (1, bs, self.d_model)
            )

        box_head = self.bbox_embed
        if is_instance_prompt and self.instance_bbox_embed is not None:
            box_head = self.instance_bbox_embed

        out_norm = self.norm
        if is_instance_prompt and self.instance_norm is not None:
            out_norm = self.instance_norm

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = (
                reference_boxes[:, :, None]
                * mx.concat([valid_ratios, valid_ratios], -1)[None, :]
            )  # nq, bs, nlevel, 4

            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )  # nq, bs, d_model * 2

            # conditional query
            query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, d_model

            if self.boxRPB != "none" and reference_boxes is not None:
                assert spatial_shapes.shape[0] == 1, (
                    "only single scale support implemented"
                )
                memory_mask = self._get_rpb_matrix(
                    reference_boxes,
                    (spatial_shapes[0, 0], spatial_shapes[0, 1]),
                )
                memory_mask = memory_mask.flatten(0, 1)
            output, presence_out = activation_ckpt_wrapper(layer)(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                dac=apply_dac,
                dac_use_selfatt_ln=self.dac_use_selfatt_ln,
                presence_token=presence_out,
                **(decoder_extra_kwargs or {}),
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                # ROI memory bank
                obj_roi_memory_feat=obj_roi_memory_feat,
                obj_roi_memory_mask=obj_roi_memory_mask,
            )

            if self.box_refine:
                reference_before_sigmoid = inverse_sigmoid(reference_boxes)
                if box_head_trk is None:
                    if not self.use_normed_output_consistently:
                        delta_unsig = box_head(output)
                    else:
                        delta_unsig = box_head(out_norm(output))
                else:
                    Q_det = decoder_extra_kwargs["Q_det"]
                    assert output.shape[0] >= Q_det
                    delta_unsig_det = self.bbox_embed(output[:Q_det])
                    delta_unsig_trk = box_head_trk(output[Q_det:])
                    delta_unsig = mx.concat([delta_unsig_det, delta_unsig_trk], axis=0)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = mx.sigmoid(outputs_unsig)

                reference_boxes = new_reference_points
                if layer_idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_points)
            else:
                _raise_decoder_unsupported(
                    "sam3_mlx.model.decoder.TransformerDecoder(untied_box_head)",
                    reason="video-multiplex",
                    detail="The untied box head path is not implemented yet.",
                )

            intermediate.append(out_norm(output))
            if self.presence_token is not None and is_instance_prompt is False:
                intermediate_layer_presence_logits = self.presence_token_head(
                    self.presence_token_out_norm(presence_out)
                ).squeeze(-1)

                # clamp to mitigate numerical issues
                if self.clamp_presence_logits:
                    intermediate_layer_presence_logits = mx.clip(
                        intermediate_layer_presence_logits,
                        a_min=-self.clamp_presence_logit_max_val,
                        a_max=self.clamp_presence_logit_max_val,
                    )

                intermediate_presence_logits.append(intermediate_layer_presence_logits)
                presence_feats = mx.array(presence_out)

        return (
            mx.stack(intermediate),
            mx.stack(intermediate_ref_boxes),
            (
                mx.stack(intermediate_presence_logits)
                if self.presence_token is not None and is_instance_prompt is False
                else None
            ),
            presence_feats,
        )

    def __call__(
        self,
        tgt,
        memory,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        reference_boxes: Optional[mx.array] = None,
        level_start_index: Optional[mx.array] = None,
        spatial_shapes: Optional[mx.array] = None,
        valid_ratios: Optional[mx.array] = None,
        memory_text: Optional[mx.array] = None,
        text_attention_mask: Optional[mx.array] = None,
        apply_dac: Optional[bool] = None,
        is_instance_prompt=False,
        decoder_extra_kwargs: Optional[Dict] = None,
        obj_roi_memory_feat=None,
        obj_roi_memory_mask=None,
        box_head_trk=None,
    ):
        return self.forward(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            reference_boxes=reference_boxes,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            apply_dac=apply_dac,
            is_instance_prompt=is_instance_prompt,
            decoder_extra_kwargs=decoder_extra_kwargs,
            obj_roi_memory_feat=obj_roi_memory_feat,
            obj_roi_memory_mask=obj_roi_memory_mask,
            box_head_trk=box_head_trk,
        )


class TransformerEncoderCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        pos_enc_at_input: bool,
        layer,
        num_layers: int,
        use_act_checkpoint: bool = False,
        batch_first: bool = False,
        remove_cross_attention_layers: Optional[list] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_act_checkpoint = use_act_checkpoint
        self.batch_first = batch_first
        self.frozen = frozen

        self.remove_cross_attention_layers = [False] * self.num_layers
        if remove_cross_attention_layers is not None:
            for i in remove_cross_attention_layers:
                self.remove_cross_attention_layers[i] = True
        assert len(self.remove_cross_attention_layers) == len(self.layers)

        for i, remove_cross_attention in enumerate(self.remove_cross_attention_layers):
            if remove_cross_attention:
                self.layers[i].cross_attn_image = None
                self.layers[i].norm2 = None
                self.layers[i].dropout2 = None
        if frozen:
            self.freeze()

    def forward(
        self,
        src,
        prompt,
        src_mask: Optional[mx.array] = None,
        prompt_mask: Optional[mx.array] = None,
        src_key_padding_mask: Optional[mx.array] = None,
        prompt_key_padding_mask: Optional[mx.array] = None,
        src_pos: Optional[mx.array] = None,
        prompt_pos: Optional[mx.array] = None,
        feat_sizes: Optional[list] = None,
        num_obj_ptr_tokens: int = 0,
    ):
        del feat_sizes
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src, src_key_padding_mask, src_pos = (
                src[0],
                src_key_padding_mask[0],
                src_pos[0],
            )

        assert src.shape[1] == prompt.shape[1], (
            "Batch size must be the same for src and prompt"
        )

        output = src
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            output = output.transpose(1, 0, 2)
            src_pos = src_pos.transpose(1, 0, 2) if src_pos is not None else None
            prompt = prompt.transpose(1, 0, 2)
            prompt_pos = (
                prompt_pos.transpose(1, 0, 2) if prompt_pos is not None else None
            )

        for layer in self.layers:
            kwds = {}
            cross_attn_image = getattr(layer, "cross_attn_image", None)
            if isinstance(cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = activation_ckpt_wrapper(layer)(
                tgt=output,
                memory=prompt,
                tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos,
                query_pos=src_pos,
                dac=False,
                attn_bias=None,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                **kwds,
            )

        normed_output = self.norm(output)
        if self.batch_first:
            normed_output = normed_output.transpose(1, 0, 2)
            src_pos = src_pos.transpose(1, 0, 2) if src_pos is not None else None

        return {
            "memory": normed_output,
            "pos_embed": src_pos,
            "padding_mask": src_key_padding_mask,
        }

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TransformerDecoderLayerv1(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def __getstate__(self):
        state = self.__dict__.copy()
        state["activation"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.activation = get_activation_fn(self.activation_str)

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        **kwargs,
    ):
        del kwargs
        q = k = _with_pos_embed(tgt, query_pos) if self.pos_enc_at_attn else tgt
        tgt2 = _call_short_or_mha_attention(
            self.self_attn,
            q,
            k,
            tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = _call_short_or_mha_attention(
            self.cross_attn_image,
            _with_pos_embed(tgt, query_pos)
            if self.pos_enc_at_cross_attn_queries
            else tgt,
            _with_pos_embed(memory, pos) if self.pos_enc_at_cross_attn_keys else memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        dac: bool = False,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        attn_bias: Optional[mx.array] = None,
        **kwargs,
    ):
        del kwargs
        if dac:
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
            query_pos_self = (
                query_pos[: query_pos.shape[0] // 2] if query_pos is not None else None
            )
        else:
            other_tgt = None
            query_pos_self = query_pos

        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos_self) if self.pos_enc_at_attn else tgt2
        tgt2 = _call_short_or_mha_attention(
            self.self_attn,
            q,
            k,
            tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            tgt = mx.concat((tgt, other_tgt), axis=0)

        tgt2 = self.norm2(tgt)
        tgt2 = _call_short_or_mha_attention(
            self.cross_attn_image,
            _with_pos_embed(tgt2, query_pos)
            if self.pos_enc_at_cross_attn_queries
            else tgt2,
            _with_pos_embed(memory, pos) if self.pos_enc_at_cross_attn_keys else memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        dac: bool = False,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        attn_bias: Optional[mx.array] = None,
        **kwds: Any,
    ) -> mx.array:
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            attn_bias=attn_bias,
            **kwds,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TransformerDecoderLayerv2(TransformerDecoderLayerv1):
    def __init__(self, cross_attention_first=False, *args: Any, **kwds: Any):
        super().__init__(*args, **kwds)
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos) if self.pos_enc_at_attn else tgt2
        tgt2 = _call_short_or_mha_attention(self.self_attn, q, k, tgt2)
        return tgt + self.dropout1(tgt2)

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        if self.cross_attn_image is None:
            return tgt

        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        tgt2 = _call_short_or_mha_attention(
            self.cross_attn_image,
            _with_pos_embed(tgt2, query_pos)
            if self.pos_enc_at_cross_attn_queries
            else tgt2,
            _with_pos_embed(memory, pos) if self.pos_enc_at_cross_attn_keys else memory,
            memory,
            **kwds,
        )
        return tgt + self.dropout2(tgt2)

    def forward_pre(
        self,
        tgt,
        memory,
        dac: bool,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        attn_bias: Optional[mx.array] = None,
        num_k_exclude_rope: int = 0,
    ):
        assert dac is False
        assert tgt_mask is None
        assert memory_mask is None
        assert tgt_key_padding_mask is None
        assert memory_key_padding_mask is None
        assert attn_bias is None

        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)

    def forward(self, *args: Any, **kwds: Any) -> mx.array:
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        _raise_decoder_unsupported(
            "sam3_mlx.model.decoder.TransformerDecoderLayerv2(pre_norm=False)",
            reason="video-multiplex",
            detail="TransformerDecoderLayerv2 only ports the official pre_norm path.",
        )


def functional_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    dropout: float,
    num_heads: int,
    num_k_exclude_rope: int = 0,
    freqs_cis: Optional[mx.array] = None,
    freqs_cis_real: Optional[mx.array] = None,
    freqs_cis_imag: Optional[mx.array] = None,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    rope_k_repeat: bool,
) -> Union[mx.array, tuple[mx.array, mx.array]]:
    if use_fa3:
        _raise_decoder_unsupported(
            "sam3_mlx.model.decoder.functional_attention(use_fa3=True)",
            reason="flash-attn-3",
            detail="FlashAttention 3 is non-MLX and not ported to MLX.",
        )

    b, n, cq = q.shape
    _, m, ck = k.shape
    _, _, cv = v.shape
    if b > 1:
        assert k.shape[0] == v.shape[0] == b
    else:
        assert k.shape[0] == b == 1, f"{q.shape=} {k.shape=} {v.shape=}"
    assert v.shape[1] == m
    assert cq % num_heads == 0
    assert ck % num_heads == 0
    assert cv % num_heads == 0

    q = q.reshape(b, n, num_heads, cq // num_heads).transpose(0, 2, 1, 3)
    k = k.reshape(k.shape[0], m, num_heads, ck // num_heads).transpose(0, 2, 1, 3)
    v = v.reshape(v.shape[0], m, num_heads, cv // num_heads).transpose(0, 2, 1, 3)

    if freqs_cis is not None:
        num_k_rope = k.shape[-2] - num_k_exclude_rope
        if num_k_rope < 0:
            raise AssertionError("num_k_exclude_rope cannot exceed key length.")
        k_rope = k[:, :, :num_k_rope]
        if use_rope_real:
            if num_k_rope == 0:
                _raise_decoder_unsupported(
                    "sam3_mlx.model.decoder.functional_attention(use_rope_real=True,num_k_exclude_rope=all)",
                    reason="video-multiplex",
                    detail="Real RoPE with all keys excluded is not ported.",
                )
            q, k_rope = apply_rotary_enc_real(
                q,
                k_rope,
                freqs_cis_real=freqs_cis_real,
                freqs_cis_imag=freqs_cis_imag,
                repeat_freqs_k=rope_k_repeat,
            )
        else:
            q, k_rope = apply_rotary_enc(
                q,
                k_rope,
                freqs_cis,
                repeat_freqs_k=rope_k_repeat,
            )
        k = mx.concat([k_rope, k[:, :, num_k_rope:]], axis=-2)

    scale = q.shape[-1] ** -0.5
    if dropout == 0.0:
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    else:
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        weights = _dropout(weights, dropout, training=True)
        out = mx.matmul(weights, v)

    return out.transpose(0, 2, 1, 3).reshape(b, n, cv)


class SimpleRoPEAttention(nn.Module):
    """
    Attention with rotary position encoding and no q/k/v/out projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),
        use_fa3: bool = False,
        use_rope_real: bool = False,
    ):
        super().__init__()
        if use_fa3:
            _raise_decoder_unsupported(
                "sam3_mlx.model.decoder.SimpleRoPEAttention(use_fa3=True)",
                reason="flash-attn-3",
                detail="FlashAttention 3 is non-MLX and not ported to MLX.",
            )

        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.compute_cis = partial(
            compute_axial_cis, dim=d_model // num_heads, theta=rope_theta
        )
        self.freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis_real = self.freqs_cis[..., 0]
        self.freqs_cis_imag = self.freqs_cis[..., 1]
        self.use_fa3 = use_fa3
        self.use_rope_real = use_rope_real
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> Union[mx.array, tuple[mx.array, mx.array]]:
        side = int(math.sqrt(q.shape[-2]))
        if side * side != q.shape[-2]:
            raise ValueError("SimpleRoPEAttention expects square spatial query tokens.")
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=side, end_y=side)
            self.freqs_cis_real = self.freqs_cis[..., 0]
            self.freqs_cis_imag = self.freqs_cis[..., 1]
        if q.shape[-2] != k.shape[-2] and not self.rope_k_repeat:
            raise AssertionError(
                "rope_k_repeat=True is required when q and k lengths differ."
            )

        dropout_p = self.dropout_p if self.training else 0.0
        return functional_attention(
            q,
            k,
            v,
            dropout=dropout_p,
            num_heads=self.num_heads,
            num_k_exclude_rope=num_k_exclude_rope,
            freqs_cis=self.freqs_cis,
            freqs_cis_real=self.freqs_cis_real if self.use_rope_real else None,
            freqs_cis_imag=self.freqs_cis_imag if self.use_rope_real else None,
            use_fa3=self.use_fa3,
            use_rope_real=self.use_rope_real,
            rope_k_repeat=self.rope_k_repeat,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DecoupledTransformerDecoderLayerv2(nn.Module):
    def __init__(
        self,
        *,
        activation: str,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        cross_attention_first: bool = False,
        self_attention_rope: SimpleRoPEAttention,
        cross_attention_rope: SimpleRoPEAttention,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout

        self.self_attn_q_proj = nn.Linear(d_model, d_model)
        self.self_attn_k_proj = nn.Linear(d_model, d_model)
        self.self_attn_v_proj = nn.Linear(d_model, d_model)
        self.self_attn_out_proj = nn.Linear(d_model, d_model)

        self.cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.cross_attn_k_proj = nn.Linear(d_model, d_model)
        self.cross_attn_v_proj = nn.Linear(d_model, d_model)
        self.cross_attn_out_proj = nn.Linear(d_model, d_model)

        self.image_cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.image_cross_attn_k_proj = nn.Linear(d_model, d_model)

        self.self_attention_rope = self_attention_rope
        self.cross_attention_rope = cross_attention_rope

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys
        self.cross_attention_first = cross_attention_first

    def __getstate__(self):
        state = self.__dict__.copy()
        state["activation"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.activation = get_activation_fn(self.activation_str)

    def _forward_sa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = _with_pos_embed(tgt2, query_pos) if self.pos_enc_at_attn else tgt2

        q = self.self_attn_q_proj(q)
        k = self.self_attn_k_proj(k)
        v = self.self_attn_v_proj(tgt2)
        out = self.self_attention_rope(q, k, v)
        tgt2 = self.self_attn_out_proj(out)
        return tgt + self.dropout1(tgt2)

    def _forward_ca(
        self,
        *,
        image,
        tgt,
        memory_image,
        memory,
        query_pos,
        memory_image_pos,
        num_k_exclude_rope=0,
    ):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attention_rope, SimpleRoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        tgt2 = self.norm2(tgt)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(tgt2)
        if self.pos_enc_at_cross_attn_queries:
            q = _with_pos_embed(q, query_pos)
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if self.pos_enc_at_cross_attn_keys:
            k = _with_pos_embed(k, memory_image_pos)
        v = self.cross_attn_v_proj(memory)

        out = self.cross_attention_rope(q, k, v, **kwds)
        tgt2 = self.cross_attn_out_proj(out)
        return tgt + self.dropout2(tgt2)

    def forward_pre(
        self,
        *,
        image,
        tgt,
        memory_image,
        memory,
        image_pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        memory_image_pos: Optional[mx.array] = None,
        memory_pos: Optional[mx.array] = None,
        num_k_exclude_rope: int = 0,
    ):
        del image_pos, memory_pos
        if self.cross_attention_first:
            tgt = self._forward_ca(
                image=image,
                tgt=tgt,
                memory_image=memory_image,
                memory=memory,
                query_pos=query_pos,
                memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(
                image=image,
                tgt=tgt,
                memory_image=memory_image,
                memory=memory,
                query_pos=query_pos,
                memory_image_pos=memory_image_pos,
                num_k_exclude_rope=num_k_exclude_rope,
            )

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return image, tgt + self.dropout3(tgt2)

    def forward(self, *args: Any, **kwds: Any) -> mx.array:
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        _raise_decoder_unsupported(
            "sam3_mlx.model.decoder.DecoupledTransformerDecoderLayerv2(pre_norm=False)",
            reason="video-multiplex",
            detail="DecoupledTransformerDecoderLayerv2 only ports the official pre_norm path.",
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TransformerEncoderDecoupledCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        pos_enc_at_input: bool,
        layer,
        num_layers: int,
        use_act_checkpoint: bool = False,
        batch_first: bool = False,
        use_image_in_output: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_act_checkpoint = use_act_checkpoint
        self.use_image_in_output = use_image_in_output
        self.batch_first = batch_first
        self.frozen = frozen
        if frozen:
            self.freeze()

    def forward(
        self,
        image: mx.array,
        src: mx.array,
        memory_image: mx.array,
        memory: mx.array,
        image_pos: Optional[mx.array] = None,
        src_pos: Optional[mx.array] = None,
        memory_image_pos: Optional[mx.array] = None,
        memory_pos: Optional[mx.array] = None,
        num_obj_ptr_tokens: int = 0,
    ):
        assert src.shape[1] == memory.shape[1], (
            "Batch size must be the same for src and memory"
        )
        assert image.shape[1] == memory_image.shape[1], (
            "Batch size must be the same for image and memory_image"
        )

        output = src
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            output = output.transpose(1, 0, 2)
            src_pos = src_pos.transpose(1, 0, 2) if src_pos is not None else None
            image = image.transpose(1, 0, 2)
            image_pos = image_pos.transpose(1, 0, 2) if image_pos is not None else None
            memory = memory.transpose(1, 0, 2)
            memory_pos = (
                memory_pos.transpose(1, 0, 2) if memory_pos is not None else None
            )
            memory_image = memory_image.transpose(1, 0, 2)
            memory_image_pos = (
                memory_image_pos.transpose(1, 0, 2)
                if memory_image_pos is not None
                else None
            )

        if memory_image.shape[1] != memory.shape[1]:
            assert (memory.shape[1] - memory_image.shape[1]) == num_obj_ptr_tokens, (
                f"{memory.shape[1]} - {memory_image.shape[1]} != {num_obj_ptr_tokens}"
            )
            memory_image = mx.concat(
                [
                    memory_image,
                    mx.zeros(
                        (memory_image.shape[0], num_obj_ptr_tokens)
                        + memory_image.shape[2:],
                        dtype=memory_image.dtype,
                    ),
                ],
                axis=1,
            )
            if memory_image_pos is not None:
                assert memory_pos is not None
                assert (
                    memory_pos.shape[1] - memory_image_pos.shape[1]
                ) == num_obj_ptr_tokens, (
                    f"{memory_pos.shape[1]} - {memory_image_pos.shape[1]} != {num_obj_ptr_tokens}"
                )
                memory_image_pos = mx.concat(
                    [memory_image_pos, memory_pos[0:1, -num_obj_ptr_tokens:]],
                    axis=1,
                )

        for layer in self.layers:
            image, output = activation_ckpt_wrapper(layer)(
                image=image,
                tgt=output,
                memory_image=memory_image,
                memory=memory,
                image_pos=image_pos,
                query_pos=src_pos,
                memory_image_pos=memory_image_pos,
                memory_pos=memory_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
            )

        normed_output = (
            self.norm(output + image) if self.use_image_in_output else self.norm(output)
        )

        if self.batch_first:
            normed_output = normed_output.transpose(1, 0, 2)
            src_pos = src_pos.transpose(1, 0, 2) if src_pos is not None else None

        return {
            "memory": normed_output,
            "pos_embed": src_pos,
        }

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
