from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NoReturn, Protocol, cast

import math
import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.data_misc import (
    NestedTensor,
    reshape_array,
    transpose_array,
)
from sam3_mlx.model.model_misc import MLP


UpsampleMode = Literal["nearest", "linear", "cubic"]


class _MaskModule(Protocol):
    def __call__(self, *values: mx.array) -> mx.array: ...


class _PresenceModule(Protocol):
    def __call__(
        self,
        hs: mx.array,
        *,
        prompt: mx.array | None,
        prompt_mask: mx.array | None,
    ) -> mx.array: ...


class CrossAttention(Protocol):
    def __call__(
        self,
        *,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        key_padding_mask: mx.array | None,
    ) -> mx.array: ...


def _einsum(subscripts: str, *operands: mx.array) -> mx.array:
    einsum = cast(Callable[..., mx.array], getattr(mx, "einsum"))
    return einsum(subscripts, *operands)


class LinearPresenceHead(nn.Sequential):
    def __init__(self, d_model: int) -> None:
        # a hack to make `LinearPresenceHead` compatible with old checkpoints
        initialize = cast(Callable[..., None], getattr(super(), "__init__"))
        initialize(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(
        self,
        hs: mx.array,
        prompt: mx.array | None,
        prompt_mask: mx.array | None,
    ) -> mx.array:
        del prompt, prompt_mask
        call_sequential = cast(
            Callable[[mx.array], mx.array], getattr(super(), "__call__")
        )
        return call_sequential(hs)

    def __call__(self, hs: mx.array, *args: object, **kwargs: object) -> mx.array:
        if len(args) > 2:
            raise TypeError("LinearPresenceHead accepts hs, prompt, and prompt_mask")
        prompt_value = args[0] if args else kwargs.pop("prompt", None)
        prompt_mask_value = (
            args[1] if len(args) == 2 else kwargs.pop("prompt_mask", None)
        )
        if kwargs:
            raise TypeError("LinearPresenceHead accepts hs, prompt, and prompt_mask")
        prompt = cast(mx.array | None, prompt_value)
        prompt_mask = cast(mx.array | None, prompt_mask_value)
        return self.forward(hs, prompt, prompt_mask)


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim: int, mask_dim: int) -> None:
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries: mx.array, pixel_embed: mx.array) -> mx.array:
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = _einsum(
                    "bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = _einsum(
                    "bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
        else:
            # Assumed to have aux masks
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = _einsum(
                    "lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = _einsum(
                    "lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )

        return mask_preds

    def __call__(self, obj_queries: mx.array, pixel_embed: mx.array) -> mx.array:
        return self.forward(obj_queries, pixel_embed)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        upsampling_stages: int,
        use_encoder_inputs: bool = False,
        aux_masks: bool = False,
        no_dec: bool = False,
        pixel_decoder: PixelDecoder | None = None,
        act_ckpt: bool = False,
        shared_conv: bool = False,
        compile_mode_pixel_decoder: str | bool | None = None,
    ) -> None:
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor: _MaskModule = cast(
                _MaskModule,
                nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.mask_predictor = cast(
                _MaskModule, MaskPredictor(hidden_dim, mask_dim=hidden_dim)
            )

        self.act_ckpt = act_ckpt

        # used to update the output dictionary
        self.instance_keys = ["pred_masks"]

    @property
    def device(self) -> Literal["mlx"]:
        return "mlx"

    def to(self, *args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise_unsupported(
            "sam3_mlx.model.maskformer_segmentation.SegmentationHead.to",
            reason="unsupported-device",
            detail="SegmentationHead.to() is a PyTorch device API and is not supported.",
            alternative="Keep tensors on the explicit MLX runtime.",
        )

    def _embed_pixels(
        self,
        backbone_feats: list[mx.array],
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None,
    ) -> mx.array:
        backbone_feats = [_unwrap_nested(feat) for feat in backbone_feats]
        image_ids_ = image_ids
        if self.use_encoder_inputs:
            if backbone_feats[0].shape[0] > 1:
                # For bs > 1, we construct the per query backbone features
                backbone_visual_feats: list[mx.array] = []
                for feat in backbone_feats:
                    # Copy the img features per query (pixel decoder won't share img feats)
                    backbone_visual_feats.append(feat[image_ids_, ...])
            else:
                # Bs=1, we rely on broadcasting for query-based processing
                backbone_visual_feats = [bb_feat for bb_feat in backbone_feats]
            # Extract visual embeddings
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states are required")
            encoder_hidden_states = transpose_array(encoder_hidden_states, 1, 2, 0)
            spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
            encoder_visual_embed = reshape_array(
                encoder_hidden_states[..., :spatial_dim],
                -1,
                *backbone_feats[-1].shape[1:],
            )

            backbone_visual_feats[-1] = encoder_visual_embed
            pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            pixel_embed = self.pixel_decoder(backbone_feats)
            if pixel_embed.shape[0] == 1:
                # For batch_size=1 training, we can avoid the indexing to save memory
                pixel_embed = mx.squeeze(pixel_embed, axis=0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]
        return pixel_embed

    def forward(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        **kwargs: object,
    ) -> dict[str, mx.array | None]:
        del kwargs
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.no_dec:
            mask_pred = transpose_array(
                self.mask_predictor(transpose_array(pixel_embed, 0, 2, 3, 1)),
                0,
                3,
                1,
                2,
            )
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}

    def __call__(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        **kwargs: object,
    ) -> dict[str, mx.array | None]:
        return self.forward(
            backbone_feats=backbone_feats,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )


class PixelDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_upsampling_stages: int,
        interpolation_mode: UpsampleMode = "nearest",
        shared_conv: bool = False,
        compile_mode: str | bool | None = None,
    ) -> None:
        super().__init__()
        if compile_mode not in (None, False):
            raise_unsupported(
                "sam3_mlx.model.maskformer_segmentation.PixelDecoder(compile_mode)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode: UpsampleMode = interpolation_mode
        conv_layers: list[nn.Conv2d] = []
        norms: list[nn.GroupNorm] = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim, pytorch_compatible=True))

        self.conv_layers = conv_layers
        self.norms = norms
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].weight.shape[0]

    def forward(self, backbone_feats: list[mx.array]) -> mx.array:
        # Assumes backbone features are already projected (C == hidden dim)

        backbone_feats = [_unwrap_nested(x) for x in backbone_feats]
        backbone_feats = [transpose_array(x, 0, 2, 3, 1) for x in backbone_feats]

        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat

            current_h, current_w = prev_fpn.shape[-3:-1]
            h, w = curr_fpn.shape[-3:-1]
            scale_h = h / current_h
            scale_w = w / current_w

            upsample_fn = nn.Upsample(
                scale_factor=(scale_h, scale_w),
                mode=self.interpolation_mode,
                align_corners=False,
            )
            prev_fpn = curr_fpn + upsample_fn(prev_fpn)
            if self.shared_conv:
                # only one conv layer
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            relu = cast(Callable[[mx.array], mx.array], getattr(nn, "relu"))
            norm = cast(Callable[[mx.array], mx.array], self.norms[layer_idx])
            prev_fpn = relu(norm(prev_fpn))

        return transpose_array(prev_fpn, 0, 3, 1, 2)

    def __call__(self, backbone_feats: list[mx.array]) -> mx.array:
        return self.forward(backbone_feats)


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation"""

    def __init__(
        self,
        hidden_dim: int,
        upsampling_stages: int,
        pixel_decoder: PixelDecoder,
        aux_masks: bool = False,
        no_dec: bool = False,
        act_ckpt: bool = False,
        presence_head: bool = False,
        dot_product_scorer: _PresenceModule | None = None,
        cross_attend_prompt: CrossAttention | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, (
                "Specifying a dot product scorer without a presence head is likely a mistake"
            )

        self.presence_head: _PresenceModule | None = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer
                if dot_product_scorer is not None
                else cast(_PresenceModule, LinearPresenceHead(self.d_model))
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(
            self.pixel_decoder.out_dim, self.d_model, kernel_size=1
        )

    def forward(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        prompt: mx.array | None = None,
        prompt_mask: mx.array | None = None,
        **kwargs: object,
    ) -> dict[str, mx.array | None]:
        del kwargs
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]

        if self.cross_attend_prompt is not None:
            if prompt is None:
                raise ValueError("prompt is required when cross attention is enabled")
            t_encoder_hidden_states = transpose_array(encoder_hidden_states, 1, 0, 2)
            t_prompt = transpose_array(prompt, 1, 0, 2)

            tgt2 = self.cross_attn_norm(t_encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                queries=tgt2,
                keys=t_prompt,
                values=t_prompt,
                key_padding_mask=prompt_mask,
            )
            tgt2 = transpose_array(tgt2, 1, 0, 2)
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    reshape_array(pooled_enc, 1, bs, 1, self.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        instance_embeds = transpose_array(
            self.instance_seg_head(transpose_array(pixel_embed, 0, 2, 3, 1)),
            0,
            3,
            1,
            2,
        )

        if self.no_dec:
            mask_pred = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        return {
            "pred_masks": mask_pred,
            "semantic_seg": transpose_array(
                self.semantic_seg_head(transpose_array(pixel_embed, 0, 2, 3, 1)),
                0,
                3,
                1,
                2,
            ),
            "presence_logit": presence_logit,
        }

    def __call__(
        self,
        backbone_feats: list[mx.array],
        obj_queries: mx.array,
        image_ids: mx.array,
        encoder_hidden_states: mx.array | None = None,
        prompt: mx.array | None = None,
        prompt_mask: mx.array | None = None,
        **kwargs: object,
    ) -> dict[str, mx.array | None]:
        return self.forward(
            backbone_feats=backbone_feats,
            obj_queries=obj_queries,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
            **kwargs,
        )


def _unwrap_nested(value: mx.array | NestedTensor) -> mx.array:
    return value.tensors if isinstance(value, NestedTensor) else value
