from __future__ import annotations

from typing import List, Optional, Tuple, Type

import mlx.core as mx
import mlx.nn as nn

from sam3_mlx.sam.common import Conv2dNCHW, ConvTranspose2dNCHW, LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = [
            ConvTranspose2dNCHW(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            ConvTranspose2dNCHW(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        ]
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = Conv2dNCHW(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = Conv2dNCHW(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = [
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ]
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def __call__(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        sparse_prompt_embeddings: mx.array,
        dense_prompt_embeddings: mx.array,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not getattr(
            self, "training", False
        ):
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def _upscale(self, src: mx.array, high_res_features: Optional[List[mx.array]]):
        if not self.use_high_res_features:
            x = src
            for layer in self.output_upscaling:
                x = layer(x)
            return x
        if high_res_features is None:
            raise ValueError(
                "high_res_features are required when use_high_res_features=True."
            )
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        x = act1(ln1(dc1(src) + feat_s1))
        return act2(dc2(x) + feat_s0)

    def predict_masks(
        self,
        image_embeddings: mx.array,
        image_pe: mx.array,
        sparse_prompt_embeddings: mx.array,
        dense_prompt_embeddings: mx.array,
        repeat_image: bool,
        high_res_features: Optional[List[mx.array]] = None,
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        offset = 0
        if self.pred_obj_scores:
            output_tokens = mx.concat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                axis=0,
            )
            offset = 1
        else:
            output_tokens = mx.concat(
                [self.iou_token.weight, self.mask_tokens.weight], axis=0
            )
        output_tokens = mx.broadcast_to(
            output_tokens[None, :, :],
            (
                sparse_prompt_embeddings.shape[0],
                output_tokens.shape[0],
                output_tokens.shape[1],
            ),
        )
        tokens = mx.concat((output_tokens, sparse_prompt_embeddings), axis=1)

        if repeat_image:
            src = mx.repeat(image_embeddings, tokens.shape[0], axis=0)
        else:
            if image_embeddings.shape[0] != tokens.shape[0]:
                raise AssertionError(
                    "image batch and token batch must match when repeat_image=False."
                )
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != 1:
            raise AssertionError("image_pe should have batch size 1.")
        pos_src = mx.repeat(image_pe, tokens.shape[0], axis=0)
        batch_size, channels, height, width = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, offset, :]
        mask_tokens_out = hs[:, offset + 1 : offset + 1 + self.num_mask_tokens, :]

        src = src.transpose(0, 2, 1).reshape(batch_size, channels, height, width)
        upscaled_embedding = self._upscale(src, high_res_features)

        hyper_in = mx.stack(
            [
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                for i in range(self.num_mask_tokens)
            ],
            axis=1,
        )
        batch_size, channels, height, width = upscaled_embedding.shape
        masks = (
            hyper_in @ upscaled_embedding.reshape(batch_size, channels, height * width)
        ).reshape(batch_size, -1, height, width)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * mx.ones(
                (iou_pred.shape[0], 1), dtype=iou_pred.dtype
            )
        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        mask_logits = mask_logits.reshape(*mask_logits.shape[:-2], -1)
        delta = self.dynamic_multimask_stability_delta
        area_i = mx.sum(mask_logits > delta, axis=-1).astype(mx.float32)
        area_u = mx.sum(mask_logits > -delta, axis=-1).astype(mx.float32)
        return mx.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_indices = mx.argmax(multimask_iou_scores, axis=-1)
        best_mask = (
            mx.arange(multimask_iou_scores.shape[1])[None, :] == best_indices[:, None]
        )
        best_iou_scores = mx.sum(
            mx.where(
                best_mask, multimask_iou_scores, mx.zeros_like(multimask_iou_scores)
            ),
            axis=1,
            keepdims=True,
        )
        best_logits = mx.sum(
            mx.where(
                best_mask[:, :, None, None],
                multimask_logits,
                mx.zeros_like(multimask_logits),
            ),
            axis=1,
            keepdims=True,
        )

        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        is_stable = (
            self._get_stability_scores(singlemask_logits)
            >= self.dynamic_multimask_stability_thresh
        )
        return (
            mx.where(is_stable[..., None, None], singlemask_logits, best_logits),
            mx.where(is_stable, singlemask_iou_scores, best_iou_scores),
        )


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = [
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip([input_dim] + hidden, hidden + [output_dim])
        ]
        self.sigmoid_output = sigmoid_output

    def __call__(self, x):
        for idx, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if idx < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = mx.sigmoid(x)
        return x
