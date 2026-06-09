from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.act_ckpt_utils import activation_ckpt_wrapper
from sam3_mlx.model.model_misc import get_activation_fn, get_clones, get_valid_ratio


class TransformerEncoderLayer(nn.Module):
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

        # Feedforward Model
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

        self.layer_idx = None

    def forward_post(
        self,
        tgt: mx.array,
        memory: mx.array,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        # self attention
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attn to image
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt: mx.array,
        memory: mx.array,
        dac: bool = False,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        if dac:
            # we only apply self attention to the first half of the queries
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(
            q, k, values=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            # Recombine
            tgt = mx.concat((tgt, other_tgt), axis=0)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            queries=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            keys=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            values=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            # attn_bias=attn_bias,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt: mx.array,
        memory: mx.array,
        dac: bool = False,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        **kwargs,
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
        )

    def __call__(
        self,
        tgt: mx.array,
        memory: mx.array,
        dac: bool = False,
        tgt_mask: Optional[mx.array] = None,
        memory_mask: Optional[mx.array] = None,
        tgt_key_padding_mask: Optional[mx.array] = None,
        memory_key_padding_mask: Optional[mx.array] = None,
        pos: Optional[mx.array] = None,
        query_pos: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        return self.forward(
            tgt=tgt,
            memory=memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            **kwargs,
        )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        frozen: bool = False,
        use_act_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.use_act_checkpoint = use_act_checkpoint

        self.num_feature_levels = num_feature_levels
        self.level_embed = None
        if num_feature_levels > 1:
            self.level_embed = mx.zeros((num_feature_levels, d_model))

        if frozen:
            self.freeze()

        # assign layer index to each layer so that some layers can decide what to do
        # based on which layer index they are (e.g. cross attention to memory bank only
        # in selected layers)
        for layer_idx, layer in enumerate(self.layers):
            layer.layer_idx = layer_idx

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device=None):
        del spatial_shapes, valid_ratios, device
        return None

    def _prepare_multilevel_features(self, srcs, masks, pos_embeds):
        assert len(srcs) == self.num_feature_levels, (
            "mismatch between expected and received * of feature levels"
        )

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        has_mask = masks is not None and masks[0] is not None
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(
                0, 2, 1
            )  # bs, c, h, w -> bs, c, hw -> bs, hw, c
            if has_mask:
                mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(0, 2, 1)
            if self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].reshape(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            if has_mask:
                mask_flatten.append(mask)
        src_flatten = mx.concat(src_flatten, axis=1)  # bs, \sum{hxw}, c
        mask_flatten = (
            mx.concat(mask_flatten, axis=1) if has_mask else None
        )  # bs, \sum{hxw}
        lvl_pos_embed_flatten = mx.concat(
            lvl_pos_embed_flatten, axis=1
        )  # bs, \sum{hxw}, c
        spatial_shapes = mx.array(spatial_shapes, dtype=mx.int64)
        level_start_index = mx.concat(
            (
                mx.zeros((1,), dtype=mx.int64),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        if has_mask:
            valid_ratios = mx.stack([get_valid_ratio(m) for m in masks], axis=1)
        else:
            valid_ratios = mx.ones(
                (src_flatten.shape[0], self.num_feature_levels, 2),
            )

        return (
            src_flatten,
            mask_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        )

    def forward(
        self,
        src: List[mx.array],
        src_key_padding_masks: Optional[List[mx.array]] = None,
        pos: Optional[List[mx.array]] = None,
        prompt: Optional[mx.array] = None,
        prompt_key_padding_mask: Optional[mx.array] = None,
        encoder_extra_kwargs: Optional[Dict] = None,
    ) -> Tuple[mx.array, Optional[mx.array], mx.array, mx.array, mx.array, mx.array]:
        assert len(src) == self.num_feature_levels, (
            "must be equal to num_feature_levels"
        )
        if src_key_padding_masks is not None:
            assert len(src_key_padding_masks) == self.num_feature_levels
        if pos is not None:
            assert len(pos) == self.num_feature_levels

        # Flatten multilevel feats and add level pos embeds
        (
            src_flatten,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)

        output = src_flatten
        for layer in self.layers:
            layer_kwargs = {}

            assert isinstance(layer, TransformerEncoderLayer)
            layer_kwargs["memory"] = prompt
            layer_kwargs["memory_key_padding_mask"] = prompt_key_padding_mask
            layer_kwargs["query_pos"] = lvl_pos_embed_flatten
            layer_kwargs["tgt"] = output
            layer_kwargs["tgt_key_padding_mask"] = key_padding_masks_flatten

            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = activation_ckpt_wrapper(layer)(
                **layer_kwargs,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
            )

        return (
            output.transpose(1, 0, 2),  # b, hw, c -> hw, b, c
            (
                key_padding_masks_flatten.transpose(1, 0)
                if key_padding_masks_flatten is not None
                else None
            ),
            lvl_pos_embed_flatten.transpose(1, 0, 2),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )

    def __call__(
        self,
        src: List[mx.array],
        src_key_padding_masks: Optional[List[mx.array]] = None,
        pos: Optional[List[mx.array]] = None,
        prompt: Optional[mx.array] = None,
        prompt_key_padding_mask: Optional[mx.array] = None,
        encoder_extra_kwargs: Optional[Dict] = None,
    ) -> Tuple[mx.array, Optional[mx.array], mx.array, mx.array, mx.array, mx.array]:
        return self.forward(
            src=src,
            src_key_padding_masks=src_key_padding_masks,
            pos=pos,
            prompt=prompt,
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )


class TransformerEncoderFusion(TransformerEncoder):
    def __init__(
        self,
        layer: nn.Module,
        num_layers: int,
        d_model: int,
        num_feature_levels: int,
        add_pooled_text_to_img_feat: bool = True,
        pool_text_with_mask: bool = False,
        compile_mode: Optional[str] = None,
        **kwargs,
    ):
        if compile_mode not in (None, False):
            raise_unsupported(
                "sam3_mlx.model.encoder.TransformerEncoderFusion(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        super().__init__(layer, num_layers, d_model, num_feature_levels, **kwargs)

        self.add_pooled_text_to_img_feat = add_pooled_text_to_img_feat
        if self.add_pooled_text_to_img_feat:
            self.text_pooling_proj = nn.Linear(d_model, d_model)
        self.pool_text_with_mask = pool_text_with_mask
        # compile mode

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device=None):
        del spatial_shapes, valid_ratios, device
        return None

    def forward(
        self,
        src: List[mx.array],
        prompt: mx.array,
        src_key_padding_mask: Optional[List[mx.array]] = None,
        src_pos: Optional[List[mx.array]] = None,
        prompt_key_padding_mask: Optional[mx.array] = None,
        prompt_pos: Optional[mx.array] = None,
        feat_sizes: Optional[List[int]] = None,
        encoder_extra_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        if "src_key_padding_masks" in kwargs:
            if src_key_padding_mask is not None:
                raise TypeError(
                    "Pass only one of src_key_padding_mask or src_key_padding_masks."
                )
            src_key_padding_mask = kwargs.pop("src_key_padding_masks")
        if "pos" in kwargs:
            if src_pos is not None:
                raise TypeError("Pass only one of src_pos or pos.")
            src_pos = kwargs.pop("pos")
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected TransformerEncoderFusion keyword(s): {names}")

        bs = src[0].shape[1]
        if feat_sizes is not None:
            assert len(feat_sizes) == len(src)
            if src_key_padding_mask is None:
                src_key_padding_mask = [None] * len(src)
            for i, (h, w) in enumerate(feat_sizes):
                src[i] = src[i].reshape(h, w, bs, -1).transpose(2, 3, 0, 1)
                src_pos[i] = src_pos[i].reshape(h, w, bs, -1).transpose(2, 3, 0, 1)
                src_key_padding_mask[i] = (
                    src_key_padding_mask[i].reshape(h, w, bs).transpose(2, 0, 1)
                    if src_key_padding_mask[i] is not None
                    else None
                )
        else:
            assert all(x.ndim == 4 for x in src), (
                "expected list of (bs, c, h, w) arrays"
            )

        if self.add_pooled_text_to_img_feat:
            pooled_text = pool_text_feat(
                prompt, prompt_key_padding_mask, self.pool_text_with_mask
            )
            pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]
            src = [x + pooled_text for x in src]

        (
            out,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
        ) = TransformerEncoder.forward(
            self,
            src=src,
            src_key_padding_masks=src_key_padding_mask,
            pos=src_pos,
            prompt=prompt.transpose(1, 0, 2),
            prompt_key_padding_mask=prompt_key_padding_mask,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )

        return {
            "memory": out,
            "padding_mask": key_padding_masks_flatten,
            "pos_embed": lvl_pos_embed_flatten,
            "memory_text": prompt,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
        }

    def __call__(
        self,
        src: List[mx.array],
        prompt: mx.array,
        src_key_padding_mask: Optional[List[mx.array]] = None,
        src_pos: Optional[List[mx.array]] = None,
        prompt_key_padding_mask: Optional[mx.array] = None,
        prompt_pos: Optional[mx.array] = None,
        feat_sizes: Optional[List[int]] = None,
        encoder_extra_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        return self.forward(
            src=src,
            prompt=prompt,
            src_key_padding_mask=src_key_padding_mask,
            src_pos=src_pos,
            prompt_key_padding_mask=prompt_key_padding_mask,
            prompt_pos=prompt_pos,
            feat_sizes=feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
            **kwargs,
        )


def pool_text_feat(prompt, prompt_mask, pool_with_mask):
    # prompt has shape (seq, bs, dim)
    if not pool_with_mask:
        return prompt.mean(axis=0)

    # prompt_mask has shape (bs, seq), where False is valid and True is padding
    assert prompt_mask.ndim == 2
    # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
    is_valid = (~prompt_mask).astype(mx.float32).transpose(1, 0)[..., None]
    # num_valid has shape (bs, 1)
    num_valid = mx.clip(mx.sum(is_valid, axis=0), 1.0, None)

    # mean pool over all the valid tokens
    pooled_text = mx.sum(prompt * is_valid, axis=0) / num_valid
    return pooled_text
