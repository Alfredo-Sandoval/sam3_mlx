from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Literal, NoReturn, NotRequired, Protocol, TypedDict, cast, overload
import mlx.core as mx
from mlx import nn
import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model import box_ops
from sam3_mlx.model.act_ckpt_utils import activation_ckpt_wrapper
from sam3_mlx.model.data_misc import BatchedDatapoint
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.model_misc import SAM3Output, Sam3StepDict, inverse_sigmoid


class _ArrayCallable(Protocol):
    def __call__(self, value: mx.array) -> mx.array: ...


class _ArrayTransforms(Protocol):
    def reshape(self, *shape: int) -> mx.array: ...

    def swapaxes(self, axis1: int, axis2: int) -> mx.array: ...

    def transpose(self, *axes: int) -> mx.array: ...


class _BoxConverter(Protocol):
    def __call__(self, value: mx.array) -> mx.array: ...


class _Backbone(Protocol):
    def forward_image(self, samples: object) -> _BackboneOutput: ...

    def forward_text(
        self,
        captions: object,
        input_boxes: object | None = None,
        additional_text: object | None = None,
        device: str | None = None,
    ) -> dict[str, mx.array]: ...


class _MaskEncoder(Protocol):
    @property
    def mask_downsampler(self) -> _ArrayCallable: ...


class _GeometryEncoder(Protocol):
    @property
    def mask_encoder(self) -> _MaskEncoder: ...

    def __call__(
        self,
        *,
        geo_prompt: Prompt,
        img_feats: Sequence[mx.array],
        img_sizes: Sequence[tuple[int, int]],
        img_pos_embeds: Sequence[mx.array],
    ) -> tuple[mx.array, mx.array]: ...


class _EncoderMemory(TypedDict):
    memory: mx.array
    memory_text: NotRequired[mx.array]
    pos_embed: mx.array
    padding_mask: mx.array | None
    level_start_index: mx.array
    spatial_shapes: mx.array
    valid_ratios: mx.array


class _EncoderOutput(TypedDict):
    encoder_hidden_states: mx.array
    pos_embed: mx.array
    padding_mask: mx.array | None
    level_start_index: mx.array
    spatial_shapes: mx.array
    valid_ratios: mx.array
    vis_feat_sizes: list[tuple[int, int]]
    prompt_before_enc: mx.array
    prompt_after_enc: mx.array
    prompt_mask: mx.array


class _Encoder(Protocol):
    def __call__(
        self,
        *,
        src: list[mx.array],
        src_key_padding_mask: list[mx.array | None] | None,
        src_pos: list[mx.array],
        prompt: mx.array,
        prompt_pos: mx.array,
        prompt_key_padding_mask: mx.array,
        feat_sizes: list[tuple[int, int]],
        encoder_extra_kwargs: Mapping[str, object] | None,
    ) -> _EncoderMemory: ...


class _Embedding(Protocol):
    @property
    def weight(self) -> mx.array: ...


class _Decoder(Protocol):
    @property
    def num_queries(self) -> int: ...

    @property
    def num_o2m_queries(self) -> int: ...

    @property
    def dac(self) -> bool: ...

    @property
    def query_embed(self) -> _Embedding: ...

    @property
    def bbox_embed(self) -> _ArrayCallable: ...

    @property
    def instance_bbox_embed(self) -> _ArrayCallable | None: ...

    def __call__(
        self,
        *,
        tgt: mx.array,
        memory: mx.array,
        memory_key_padding_mask: mx.array | None,
        pos: mx.array,
        reference_boxes: mx.array | None,
        level_start_index: mx.array,
        spatial_shapes: mx.array,
        valid_ratios: mx.array,
        tgt_mask: mx.array | None,
        memory_text: mx.array,
        text_attention_mask: mx.array,
        apply_dac: bool,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None]: ...


class _Transformer(Protocol):
    @property
    def d_model(self) -> int: ...

    @property
    def encoder(self) -> _Encoder: ...

    @property
    def decoder(self) -> _Decoder: ...


class _Scorer(Protocol):
    def __call__(
        self, hs: mx.array, prompt: mx.array, prompt_mask: mx.array
    ) -> mx.array: ...


class _SegmentationHead(Protocol):
    @property
    def instance_keys(self) -> Sequence[str]: ...

    def __call__(
        self, *args: object, **kwargs: object
    ) -> dict[str, mx.array | None]: ...


class _Matcher(Protocol):
    def __call__(
        self, out: dict[str, object], targets: dict[str, object]
    ) -> object: ...


class _FindInput(Protocol):
    @property
    def img_ids(self) -> mx.array: ...

    @property
    def text_ids(self) -> mx.array: ...

    @property
    def input_boxes(self) -> mx.array: ...

    @property
    def input_boxes_mask(self) -> mx.array: ...

    @property
    def input_boxes_label(self) -> mx.array: ...

    @property
    def input_points(self) -> mx.array | None: ...

    @property
    def input_boxes_before_embed(self) -> mx.array | None: ...

    @property
    def input_points_before_embed(self) -> mx.array | None: ...

    @property
    def input_points_mask(self) -> mx.array | None: ...


class _FindTarget(Protocol):
    @property
    def boxes(self) -> mx.array: ...

    @property
    def boxes_padded(self) -> mx.array: ...

    @property
    def num_boxes(self) -> mx.array: ...

    @property
    def segments(self) -> mx.array | None: ...

    @property
    def semantic_segments(self) -> mx.array | None: ...

    @property
    def is_valid_segment(self) -> mx.array | None: ...

    @property
    def is_exhaustive(self) -> mx.array: ...

    @property
    def object_ids(self) -> mx.array: ...

    @property
    def object_ids_padded(self) -> mx.array: ...


class _Datapoint(Protocol):
    @property
    def img_batch(self) -> mx.array | list[object]: ...

    @property
    def find_text_batch(self) -> Sequence[str]: ...

    @property
    def find_inputs(self) -> Sequence[_FindInput]: ...

    @property
    def find_targets(self) -> Sequence[_FindTarget]: ...


class _InteractiveBackboneModel(Protocol):
    @property
    def no_mem_embed(self) -> mx.array: ...

    def _prepare_backbone_features(
        self, backbone_out: Mapping[str, object]
    ) -> tuple[object, list[mx.array], object, object]: ...


class _PredictorFeatures(TypedDict):
    image_embed: mx.array
    high_res_feats: list[mx.array]


class _InteractivePredictor(Protocol):
    @property
    def model(self) -> _InteractiveBackboneModel: ...

    @property
    def _bb_feat_sizes(self) -> list[tuple[int, int]]: ...

    @property
    def _features(self) -> Mapping[str, object] | None: ...

    @property
    def _is_image_set(self) -> bool: ...

    @property
    def _is_batch(self) -> bool: ...

    @property
    def _orig_hw(self) -> list[tuple[int, int]]: ...

    def predict(
        self, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def predict_batch(
        self, *args: object, **kwargs: object
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]: ...


class _FeatureWrapper(Protocol):
    """Structural view of the repository's NestedTensor feature wrapper."""

    tensors: mx.array
    mask: mx.array | None


class _IndexableBatch(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


Feature = mx.array | _FeatureWrapper


class _BackboneOutput(TypedDict, total=False):
    """Heterogeneous image/text backbone fields consumed by Sam3Image."""

    img_batch_all_stages: object
    backbone_fpn: list[Feature]
    vision_pos_enc: list[mx.array]
    id_mapping: mx.array | None
    language_features: mx.array
    language_mask: mx.array


class _PrecomputedBackboneOutput(TypedDict):
    backbone_fpn: list[Feature]
    vision_pos_enc: list[mx.array]
    id_mapping: NotRequired[mx.array | None]


class _LanguageBackboneOutput(TypedDict):
    language_features: mx.array
    language_mask: mx.array


ImageFeatures = tuple[
    _BackboneOutput,
    list[mx.array],
    list[mx.array],
    list[tuple[int, int]],
]


class _PreviousEncoderOutput(TypedDict):
    encoder_out: _EncoderOutput
    backbone_out: _BackboneOutput


class Output(dict[str, object]):
    @overload
    def __getitem__(
        self, key: Literal["prev_encoder_out"]
    ) -> _PreviousEncoderOutput: ...

    @overload
    def __getitem__(self, key: str) -> object: ...

    def __getitem__(self, key: str) -> object:
        return super().__getitem__(key)


ArrayInput = (
    mx.array | np.ndarray | int | float | bool | list[object] | tuple[object, ...]
)


_box_cxcywh_to_xyxy = cast(_BoxConverter, getattr(box_ops, "box_cxcywh_to_xyxy"))


def _feature_tensor(value: Feature) -> mx.array:
    if isinstance(value, mx.array):
        return value
    return value.tensors


def _array_transforms(value: mx.array) -> _ArrayTransforms:
    return cast(_ArrayTransforms, value)


def _raise_image_unsupported(
    feature: str,
    *,
    reason: str,
    detail: str,
    alternative: str | None = None,
) -> NoReturn:
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
        alternative=alternative,
    )


def _to_index_list(value: mx.array) -> list[int]:
    values = np.asarray(to_numpy(value), dtype=np.int64).reshape(-1)
    return [int(item) for item in values]


def _update_out(
    out: Output,
    out_name: str,
    out_value: mx.array,
    auxiliary: bool = True,
    update_aux: bool = True,
) -> None:
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            new_aux_outputs = [Output() for _ in range(len(out_value) - 1)]
            out["aux_outputs"] = new_aux_outputs
        aux_outputs = cast(list[Output], out["aux_outputs"])
        assert len(aux_outputs) == len(out_value) - 1
        for aux_output, aux_value in zip(aux_outputs, out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3Image(nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        backbone: object,
        transformer: object,
        input_geometry_encoder: object,
        segmentation_head: object | None = None,
        num_feature_levels: int = 1,
        o2m_mask_predict: bool = True,
        dot_prod_scoring: _Scorer | None = None,
        use_instance_query: bool = True,
        multimask_output: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        interactivity_in_encoder: bool = True,
        matcher: _Matcher | None = None,
        use_dot_prod_scoring: bool = True,
        supervise_joint_box_scores: bool = False,
        detach_presence_in_joint_score: bool = False,
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        inst_interactive_predictor: object | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if "multimask_otuput" in kwargs:
            multimask_output = cast(bool, kwargs.pop("multimask_otuput"))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Sam3Image keyword argument(s): {unexpected}")
        # Builder modules retain broad legacy annotations, so adapt them once at
        # this constructor boundary to the precise contracts used below.
        self.backbone = cast(_Backbone, backbone)
        self.geometry_encoder = cast(_GeometryEncoder, input_geometry_encoder)
        self.transformer = cast(_Transformer, transformer)
        self.hidden_dim = self.transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = (
            None
            if segmentation_head is None
            else cast(_SegmentationHead, segmentation_head)
        )

        self.o2m_mask_predict = o2m_mask_predict

        self.dot_prod_scoring: _Scorer | None = dot_prod_scoring
        self.use_act_checkpoint_seg_head = use_act_checkpoint_seg_head
        self.interactivity_in_encoder = interactivity_in_encoder
        self.matcher: _Matcher | None = matcher

        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring
        self.instance_dot_prod_scoring: _Scorer | None = None
        self.class_embed: _ArrayCallable | None = None
        self.instance_class_embed: _ArrayCallable | None = None

        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.dot_prod_scoring = dot_prod_scoring
            if separate_scorer_for_instance:
                # The concrete scorer is an MLX module.  Deep-copying it keeps
                # its registered parameter tree and exact prompt-MLP state,
                # matching the source model's independent instance scorer.
                self.instance_dot_prod_scoring = deepcopy(self.dot_prod_scoring)
        else:
            self.class_embed = cast(_ArrayCallable, nn.Linear(self.hidden_dim, 1))
            if separate_scorer_for_instance:
                self.instance_class_embed = cast(
                    _ArrayCallable, nn.Linear(self.hidden_dim, 1)
                )

        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score

        # verify the number of queries for O2O and O2M
        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

        self.use_instance_query = use_instance_query
        self.multimask_output = multimask_output

        self.inst_interactive_predictor = (
            None
            if inst_interactive_predictor is None
            else cast(_InteractivePredictor, inst_interactive_predictor)
        )

    @property
    def device(self):
        return "mlx"

    def _validate_interactive_steps_val(self) -> None:
        if not self.training and self.num_interactive_steps_val > 0:
            _raise_image_unsupported(
                "sam3_mlx.model.sam3_image.Sam3Image(interactive-validation)",
                reason="image-interactivity",
                detail=(
                    "Validation interactive prompt sampling is not implemented in "
                    "the MLX image port. Set num_interactive_steps_val=0."
                ),
                alternative="num_interactive_steps_val=0",
            )

    def to(self, *args: object, **kwargs: object) -> None:
        _raise_image_unsupported(
            "sam3_mlx.model.sam3_image.Sam3Image.to",
            reason="unsupported-device",
            detail="Sam3Image.to() is a PyTorch device API and is not supported.",
            alternative="Keep tensors on the explicit MLX runtime.",
        )

    def _get_img_feats(
        self, backbone_out: _BackboneOutput, img_ids: mx.array
    ) -> ImageFeatures:
        """Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            precomputed_out = cast(_PrecomputedBackboneOutput, backbone_out)
            if (
                "id_mapping" in precomputed_out
                and precomputed_out["id_mapping"] is not None
            ):
                id_mapping = precomputed_out["id_mapping"]
                img_ids = id_mapping[img_ids]
                # If this assert fails, it likely means we're requesting different img_ids (perhaps a different frame?)
                # We currently don't expect this to happen. We could technically trigger a recompute here,
                # but likely at the cost of a cpu<->gpu sync point, which would deteriorate perf
                assert (img_ids >= 0).all()

            vis_feats = precomputed_out["backbone_fpn"][-self.num_feature_levels :]
            vis_pos_enc = precomputed_out["vision_pos_enc"][-self.num_feature_levels :]
            vis_feat_sizes = [
                (int(x.shape[-2]), int(x.shape[-1])) for x in vis_pos_enc
            ]  # (H, W) Shapes
            # index and flatten visual features  NxCxHxW => HWxNxC (batch-first => seq-first)
            img_feats = [
                _array_transforms(_feature_tensor(x)[img_ids].flatten(2)).transpose(
                    2, 0, 1
                )
                for x in vis_feats
            ]
            img_pos_embeds = [
                _array_transforms(x[img_ids].flatten(2)).transpose(2, 0, 1)
                for x in vis_pos_enc
            ]
            return backbone_out, img_feats, img_pos_embeds, vis_feat_sizes

        if "img_batch_all_stages" not in backbone_out:
            raise KeyError(
                "backbone_out must contain either precomputed 'backbone_fpn' "
                "features or an 'img_batch_all_stages' image batch."
            )

        img_batch = backbone_out["img_batch_all_stages"]
        requested_ids = _to_index_list(img_ids)
        unique_ids = list(dict.fromkeys(requested_ids))
        if isinstance(img_batch, mx.array):
            image = img_batch[mx.array(unique_ids, dtype=mx.int64)]
            batch_len = img_batch.shape[0]
        else:
            image_batch_items = cast(_IndexableBatch, img_batch)
            image = mx.stack(
                [
                    mx.array(cast(ArrayInput, image_batch_items[index]))
                    for index in unique_ids
                ],
                axis=0,
            )
            batch_len = len(image_batch_items)
        image = image.astype(mx.float32)

        id_mapping_np = np.full((batch_len,), -1, dtype=np.int64)
        for local_index, source_index in enumerate(unique_ids):
            id_mapping_np[source_index] = local_index
        new_backbone_out = cast(
            _BackboneOutput,
            {
                **backbone_out,
                **self.backbone.forward_image(image),
                "id_mapping": mx.array(id_mapping_np, dtype=mx.int64),
            },
        )
        if "backbone_fpn" not in new_backbone_out:
            raise AssertionError("backbone.forward_image must return 'backbone_fpn'.")
        return self._get_img_feats(new_backbone_out, img_ids=img_ids)

    def _encode_prompt(
        self,
        backbone_out: _BackboneOutput,
        find_input: _FindInput,
        geometric_prompt: Prompt,
        visual_prompt_embed: mx.array | None = None,
        visual_prompt_mask: mx.array | None = None,
        encode_text: bool = True,
        prev_mask_pred: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, _BackboneOutput]:
        if (visual_prompt_embed is None) != (visual_prompt_mask is None):
            raise ValueError(
                "visual_prompt_embed and visual_prompt_mask must be provided together."
            )
        if (
            visual_prompt_embed is not None
            and visual_prompt_mask is not None
            and visual_prompt_embed.shape[0] != visual_prompt_mask.shape[1]
        ):
            raise ValueError(
                "visual_prompt_embed and visual_prompt_mask disagree on prompt count."
            )
        # index text features (note that regardless of early or late fusion, the batch size of
        # `txt_feats`  is always the number of *prompts* in the encoder)
        txt_ids = find_input.text_ids
        language_out = cast(_LanguageBackboneOutput, backbone_out)
        language_features = language_out["language_features"]
        language_mask = language_out["language_mask"]
        txt_feats = language_features[:, txt_ids]
        txt_masks = language_mask[txt_ids]

        img_ids = find_input.img_ids
        feat_tuple = self._get_img_feats(backbone_out, img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            img_feats = [img_feats[-1] + prev_mask_pred]

        # Encode geometry
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        if visual_prompt_embed is None and visual_prompt_mask is None:
            visual_prompt_embed = mx.zeros((0, *geo_feats.shape[1:]))
            visual_prompt_mask = mx.zeros(
                (*geo_masks.shape[:-1], 0), dtype=geo_masks.dtype
            )
        assert visual_prompt_embed is not None
        assert visual_prompt_mask is not None

        if encode_text:
            prompt = mx.concat([txt_feats, geo_feats, visual_prompt_embed], axis=0)
            prompt_mask = mx.concat([txt_masks, geo_masks, visual_prompt_mask], axis=1)
        else:
            prompt = mx.concat([geo_feats, visual_prompt_embed], axis=0)
            prompt_mask = mx.concat([geo_masks, visual_prompt_mask], axis=1)

        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out: _BackboneOutput,
        find_input: _FindInput,
        prompt: mx.array,
        prompt_mask: mx.array,
        encoder_extra_kwargs: Mapping[str, object] | None = None,
    ) -> tuple[_BackboneOutput, _EncoderOutput, ImageFeatures]:
        img_ids = find_input.img_ids
        feat_tuple = self._get_img_feats(backbone_out, img_ids)
        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        # Run the encoder
        prompt_pos_embed = mx.zeros_like(prompt)
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out: _EncoderOutput = {
            # encoded image features
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            # encoded text features (or other prompts)
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }

        return backbone_out, encoder_out, feat_tuple

    def _run_decoder(
        self,
        pos_embed: mx.array,
        memory: mx.array,
        src_mask: mx.array | None,
        out: Output,
        prompt: mx.array,
        prompt_mask: mx.array,
        encoder_out: _EncoderOutput,
    ) -> tuple[Output, mx.array]:
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = mx.tile(query_embed[:, None], (1, bs, 1))

        apply_dac = self.transformer.decoder.dac and self.training
        hs, reference_boxes, dec_presence_out, dec_presence_feats = (
            self.transformer.decoder(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=src_mask,
                pos=pos_embed,
                reference_boxes=None,
                level_start_index=encoder_out["level_start_index"],
                spatial_shapes=encoder_out["spatial_shapes"],
                valid_ratios=encoder_out["valid_ratios"],
                tgt_mask=None,
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=apply_dac,
            )
        )
        hs = _array_transforms(hs).transpose(0, 2, 1, 3)  # seq-first to batch_first
        reference_boxes = _array_transforms(reference_boxes).transpose(0, 2, 1, 3)
        if dec_presence_out is not None:
            # seq-first to batch-first
            dec_presence_out = _array_transforms(dec_presence_out).transpose(0, 2, 1)

        out["presence_feats"] = dec_presence_feats
        self._update_scores_and_boxes(
            out,
            hs,
            reference_boxes,
            prompt,
            prompt_mask,
            dec_presence_out=dec_presence_out,
        )

        return out, hs

    def _update_scores_and_boxes(
        self,
        out: Output,
        hs: mx.array,
        reference_boxes: mx.array,
        prompt: mx.array,
        prompt_mask: mx.array,
        dec_presence_out: mx.array | None = None,
        is_instance_prompt: bool = False,
    ) -> None:
        apply_dac = self.transformer.decoder.dac and self.training
        num_o2o = (hs.shape[2] // 2) if apply_dac else hs.shape[2]
        num_o2m = hs.shape[2] - num_o2o
        assert num_o2m == (num_o2o if apply_dac else 0)
        out["queries"] = hs[-1][:, :num_o2o]
        # score prediction
        if self.use_dot_prod_scoring:
            dot_prod_scoring_head = self.dot_prod_scoring
            dot_prod_scoring_head = cast(_Scorer, dot_prod_scoring_head)
            if is_instance_prompt and self.instance_dot_prod_scoring is not None:
                dot_prod_scoring_head = self.instance_dot_prod_scoring
            outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
        else:
            class_embed_head = self.class_embed
            class_embed_head = cast(_ArrayCallable, class_embed_head)
            if is_instance_prompt and self.instance_class_embed is not None:
                class_embed_head = self.instance_class_embed
            outputs_class = class_embed_head(hs)

        # box prediction
        box_head = self.transformer.decoder.bbox_embed
        if (
            is_instance_prompt
            and self.transformer.decoder.instance_bbox_embed is not None
        ):
            box_head = self.transformer.decoder.instance_bbox_embed
        anchor_box_offsets = box_head(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = mx.sigmoid(reference_boxes_inv_sig + anchor_box_offsets)
        outputs_boxes_xyxy = _box_cxcywh_to_xyxy(outputs_coord)

        if dec_presence_out is not None:
            _update_out(
                out, "presence_logit_dec", dec_presence_out, update_aux=self.training
            )

        if self.supervise_joint_box_scores:
            assert dec_presence_out is not None
            prob_dec_presence_out = mx.sigmoid(dec_presence_out)
            if self.detach_presence_in_joint_score:
                prob_dec_presence_out = mx.stop_gradient(prob_dec_presence_out)

            outputs_class = mx.clip(
                inverse_sigmoid(
                    mx.sigmoid(outputs_class) * prob_dec_presence_out[:, :, None]
                ),
                -10.0,
                10.0,
            )

        _update_out(
            out, "pred_logits", outputs_class[:, :, :num_o2o], update_aux=self.training
        )

        _update_out(
            out, "pred_boxes", outputs_coord[:, :, :num_o2o], update_aux=self.training
        )
        _update_out(
            out,
            "pred_boxes_xyxy",
            outputs_boxes_xyxy[:, :, :num_o2o],
            update_aux=self.training,
        )

        if num_o2m > 0 and self.training:
            _update_out(
                out,
                "pred_logits_o2m",
                outputs_class[:, :, num_o2o:],
                update_aux=self.training,
            )

            _update_out(
                out,
                "pred_boxes_o2m",
                outputs_coord[:, :, num_o2o:],
                update_aux=self.training,
            )
            _update_out(
                out,
                "pred_boxes_xyxy_o2m",
                outputs_boxes_xyxy[:, :, num_o2o:],
                update_aux=self.training,
            )

    def _run_segmentation_heads(
        self,
        out: Output,
        backbone_out: _BackboneOutput,
        img_ids: mx.array,
        vis_feat_sizes: list[tuple[int, int]],
        encoder_hidden_states: mx.array,
        prompt: mx.array,
        prompt_mask: mx.array,
        hs: mx.array,
    ) -> None:
        apply_dac = self.transformer.decoder.dac and self.training
        if self.segmentation_head is not None:
            num_o2o = (hs.shape[2] // 2) if apply_dac else hs.shape[2]
            num_o2m = hs.shape[2] - num_o2o
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head = cast(
                _SegmentationHead, activation_ckpt_wrapper(self.segmentation_head)
            )
            instance_keys = self.segmentation_head.instance_keys
            seg_head_outputs = seg_head(
                backbone_feats=cast(_PrecomputedBackboneOutput, backbone_out)[
                    "backbone_fpn"
                ],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                act_ckpt_enable=self.training and self.use_act_checkpoint_seg_head,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            aux_masks = False
            for k, v in seg_head_outputs.items():
                if k in instance_keys:
                    value = cast(mx.array, v)
                    _update_out(out, k, value[:, :num_o2o], auxiliary=aux_masks)
                    if self.o2m_mask_predict and num_o2m > 0:
                        _update_out(
                            out, f"{k}_o2m", value[:, num_o2o:], auxiliary=aux_masks
                        )
                else:
                    out[k] = v
        else:
            backbone_out.pop("backbone_fpn", None)

    def _get_best_mask(self, out: Output) -> mx.array:
        pred_logits = cast(mx.array, out["pred_logits"])
        pred_masks = cast(mx.array, out["pred_masks"])
        prev_mask_idx = mx.argmax(pred_logits, axis=1).squeeze(1)
        batch_idx = mx.arange(pred_logits.shape[0], dtype=mx.int64)
        prev_mask_pred = pred_masks[batch_idx, prev_mask_idx][:, None]
        prev_mask_pred = self.geometry_encoder.mask_encoder.mask_downsampler(
            prev_mask_pred
        )
        return _array_transforms(prev_mask_pred.flatten(-2)).transpose(2, 0, 1)

    def forward_grounding(
        self,
        backbone_out: Mapping[str, object],
        find_input: object,
        find_target: object | None,
        geometric_prompt: Prompt,
    ) -> Output:
        self._validate_interactive_steps_val()
        typed_backbone_out = cast(_BackboneOutput, backbone_out)
        typed_find_input = cast(_FindInput, find_input)
        # profile geometry encoder
        prompt, prompt_mask, typed_backbone_out = self._encode_prompt(
            typed_backbone_out, typed_find_input, geometric_prompt
        )

        # profile encoder
        typed_backbone_out, encoder_out, _ = self._run_encoder(
            typed_backbone_out, typed_find_input, prompt, prompt_mask
        )

        out = Output(
            {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": typed_backbone_out,
                },
            },
        )

        # profile decoder
        out, hs = self._run_decoder(
            memory=cast(mx.array, out["encoder_hidden_states"]),
            pos_embed=encoder_out["pos_embed"],
            src_mask=(
                None
                if encoder_out["padding_mask"] is None
                else encoder_out["padding_mask"]
            ),
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        # profile segmentation heads
        seg_img_ids = typed_find_input.img_ids
        if (
            "id_mapping" in typed_backbone_out
            and typed_backbone_out["id_mapping"] is not None
        ):
            seg_img_ids = typed_backbone_out["id_mapping"][seg_img_ids]
        self._run_segmentation_heads(
            out=out,
            backbone_out=typed_backbone_out,
            img_ids=seg_img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=cast(mx.array, out["encoder_hidden_states"]),
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )

        if self.training or self.num_interactive_steps_val > 0:
            self._compute_matching(
                out, self.back_convert(cast(_FindTarget, find_target))
            )
        return out

    def _postprocess_out(self, out: Output, multimask_output: bool = False) -> Output:
        pred_boxes = cast(mx.array, out["pred_boxes"])
        pred_logits = cast(mx.array, out["pred_logits"])
        pred_boxes_xyxy = cast(mx.array, out["pred_boxes_xyxy"])
        num_mask_boxes = pred_boxes.shape[1]
        if not self.training and multimask_output and num_mask_boxes > 1:
            out["multi_pred_logits"] = pred_logits
            if "pred_masks" in out:
                out["multi_pred_masks"] = out["pred_masks"]
            out["multi_pred_boxes"] = pred_boxes
            out["multi_pred_boxes_xyxy"] = pred_boxes_xyxy

            best_mask_idx = mx.argmax(pred_logits, axis=1).squeeze(1)
            batch_idx = mx.arange(best_mask_idx.shape[0], dtype=mx.int64)

            out["pred_logits"] = pred_logits[batch_idx, best_mask_idx][:, None]
            if "pred_masks" in out:
                pred_masks = cast(mx.array, out["pred_masks"])
                out["pred_masks"] = pred_masks[batch_idx, best_mask_idx][:, None]
            out["pred_boxes"] = pred_boxes[batch_idx, best_mask_idx][:, None]
            out["pred_boxes_xyxy"] = pred_boxes_xyxy[batch_idx, best_mask_idx][:, None]
        return out

    def _get_geo_prompt_from_find_input(self, find_input: object) -> Prompt:
        typed_find_input = cast(_FindInput, find_input)
        point_embeddings: mx.array | None = None
        point_mask: mx.array | None = None
        point_labels: mx.array | None = None
        point_embeddings_before = typed_find_input.input_points_before_embed
        if point_embeddings_before is not None:
            point_embeddings = _array_transforms(point_embeddings_before).swapaxes(0, 1)
            point_labels = point_embeddings[..., -1]
            point_embeddings = point_embeddings[..., :-1]
            point_mask = typed_find_input.input_points_mask

        return Prompt(
            box_embeddings=(
                None
                if typed_find_input.input_boxes_before_embed is None
                else typed_find_input.input_boxes_before_embed
            ),
            box_mask=typed_find_input.input_boxes_mask,
            box_labels=typed_find_input.input_boxes_label,
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels,
        )

    def forward(self, input: BatchedDatapoint) -> SAM3Output | tuple[SAM3Output, None]:
        self._validate_interactive_steps_val()
        typed_input = cast(_Datapoint, input)
        device = self.device
        image_batch = typed_input.img_batch
        backbone_out: _BackboneOutput = {"img_batch_all_stages": image_batch}
        backbone_out.update(self.backbone.forward_image(image_batch))
        num_frames = len(typed_input.find_inputs)
        assert num_frames == 1

        text_outputs = self.backbone.forward_text(
            typed_input.find_text_batch, device=device
        )
        backbone_out["language_features"] = text_outputs["language_features"]
        backbone_out["language_mask"] = text_outputs["language_mask"]

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )
        find_input = typed_input.find_inputs[0]
        find_target = typed_input.find_targets[0]
        input_points = find_input.input_points
        if input_points is not None and input_points.size > 0:
            print("Warning: Point prompts are ignored in PCS.")

        geometric_prompt = Prompt(
            box_embeddings=find_input.input_boxes,
            box_mask=find_input.input_boxes_mask,
            box_labels=find_input.input_boxes_label,
        )

        stage_outs: list[Sam3StepDict] = []
        out = self.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=geometric_prompt.clone(),
        )
        out = self._postprocess_out(out, multimask_output=self.multimask_output)
        stage_outs.append(out)

        previous_stages_out.append(stage_outs)
        return previous_stages_out

    def _compute_matching(self, out: Output, targets: dict[str, object]) -> None:
        if self.matcher is None:
            _raise_image_unsupported(
                "sam3_mlx.model.sam3_image.Sam3Image._compute_matching",
                reason="training-loop",
                detail="Training matching is not configured in the MLX image port.",
            )
        matcher = self.matcher
        out["indices"] = matcher(out, targets)
        aux_outputs = cast(list[Output], out.get("aux_outputs", []))
        for aux_out in aux_outputs:
            aux_out["indices"] = matcher(aux_out, targets)

    def back_convert(self, targets: _FindTarget) -> dict[str, object]:
        boxes = targets.boxes
        return {
            "boxes": _array_transforms(boxes).reshape(-1, 4),
            "boxes_xyxy": _box_cxcywh_to_xyxy(_array_transforms(boxes).reshape(-1, 4)),
            "boxes_padded": targets.boxes_padded,
            "positive_map": mx.ones((len(boxes), 1), dtype=boxes.dtype),
            "num_boxes": targets.num_boxes,
            "masks": targets.segments,
            "semantic_masks": targets.semantic_segments,
            "is_valid_mask": targets.is_valid_segment,
            "is_exhaustive": targets.is_exhaustive,
            "object_ids_packed": targets.object_ids,
            "object_ids_padded": targets.object_ids_padded,
        }

    def _require_interactive_predictor(self) -> _InteractivePredictor:
        predictor = self.inst_interactive_predictor
        if predictor is None:
            _raise_image_unsupported(
                "sam3_mlx.model.sam3_image.Sam3Image.predict_inst",
                reason="image-interactivity",
                detail=(
                    "SAM1-style interactive image prediction requires "
                    "enable_inst_interactivity=True."
                ),
                alternative="build_sam3_image_model(enable_inst_interactivity=True)",
            )
        return predictor

    def _inst_predictor_features_from_state(
        self, inference_state: Mapping[str, object], batch_size: int | None = None
    ) -> _PredictorFeatures:
        predictor = self._require_interactive_predictor()
        if "backbone_out" not in inference_state:
            raise ValueError("inference_state must contain backbone_out.")
        typed_state_backbone_out = cast(
            Mapping[str, object], inference_state["backbone_out"]
        )
        backbone_out = typed_state_backbone_out.get("sam2_backbone_out")
        if backbone_out is None:
            _raise_image_unsupported(
                "sam3_mlx.model.sam3_image.Sam3Image.predict_inst(sam2_backbone_out)",
                reason="image-interactivity",
                detail=(
                    "SAM1-style interactive prediction requires sam2_backbone_out. "
                    "Build the image backbone with enable_inst_interactivity=True."
                ),
                alternative="build_sam3_image_model(enable_inst_interactivity=True)",
            )
        typed_backbone_out = cast(Mapping[str, object], backbone_out)
        prepare_backbone_features = cast(
            Callable[
                [Mapping[str, object]], tuple[object, list[mx.array], object, object]
            ],
            getattr(predictor.model, "_prepare_backbone_features"),
        )
        _, vision_feats, _, _ = prepare_backbone_features(typed_backbone_out)
        vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed
        if batch_size is None:
            batch_size = vision_feats[-1].shape[1]
        feature_sizes = cast(
            list[tuple[int, int]], getattr(predictor, "_bb_feat_sizes")
        )
        feats = [
            _array_transforms(_array_transforms(feat).transpose(1, 2, 0)).reshape(
                batch_size, -1, *feat_size
            )
            for feat, feat_size in zip(vision_feats[::-1], feature_sizes[::-1])
        ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def predict_inst(
        self,
        inference_state: Mapping[str, object],
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_h = cast(int, inference_state["original_height"])
        orig_w = cast(int, inference_state["original_width"])
        predictor = self._require_interactive_predictor()
        features = self._inst_predictor_features_from_state(inference_state, 1)
        previous_features = cast(
            _PredictorFeatures | None, getattr(predictor, "_features")
        )
        previous_image_set = cast(bool, getattr(predictor, "_is_image_set"))
        previous_orig_hw = cast(list[tuple[int, int]], getattr(predictor, "_orig_hw"))
        setattr(predictor, "_features", features)
        setattr(predictor, "_is_image_set", True)
        setattr(predictor, "_orig_hw", [(orig_h, orig_w)])
        try:
            return predictor.predict(**kwargs)
        finally:
            setattr(predictor, "_features", previous_features)
            setattr(predictor, "_is_image_set", previous_image_set)
            setattr(predictor, "_orig_hw", previous_orig_hw)

    def predict_inst_batch(
        self,
        inference_state: Mapping[str, object],
        *args: object,
        **kwargs: object,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        orig_heights = inference_state["original_heights"]
        orig_widths = inference_state["original_widths"]
        typed_orig_heights = cast(_IndexableBatch, orig_heights)
        typed_orig_widths = cast(_IndexableBatch, orig_widths)
        batch_size = len(typed_orig_heights)
        if batch_size != len(typed_orig_widths):
            raise AssertionError(
                "original_heights and original_widths must have the same length."
            )
        predictor = self._require_interactive_predictor()
        features = self._inst_predictor_features_from_state(inference_state, batch_size)
        if features["image_embed"].shape[0] != batch_size:
            raise AssertionError(
                "Batch size mismatch in predict_inst_batch. Got "
                f"{features['image_embed'].shape[0]}, {len(typed_orig_heights)}, "
                f"{len(typed_orig_widths)}"
            )
        typed_heights = [
            cast(int, typed_orig_heights[index]) for index in range(batch_size)
        ]
        typed_widths = [
            cast(int, typed_orig_widths[index]) for index in range(batch_size)
        ]
        previous_features = cast(
            _PredictorFeatures | None, getattr(predictor, "_features")
        )
        previous_image_set = cast(bool, getattr(predictor, "_is_image_set"))
        previous_batch = cast(bool, getattr(predictor, "_is_batch"))
        previous_orig_hw = cast(list[tuple[int, int]], getattr(predictor, "_orig_hw"))
        setattr(predictor, "_features", features)
        setattr(predictor, "_is_image_set", True)
        setattr(predictor, "_is_batch", True)
        setattr(
            predictor,
            "_orig_hw",
            [(orig_h, orig_w) for orig_h, orig_w in zip(typed_heights, typed_widths)],
        )
        try:
            return predictor.predict_batch(*args, **kwargs)
        finally:
            setattr(predictor, "_features", previous_features)
            setattr(predictor, "_is_image_set", previous_image_set)
            setattr(predictor, "_is_batch", previous_batch)
            setattr(predictor, "_orig_hw", previous_orig_hw)

    def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
        geometric_prompt = Prompt(
            box_embeddings=mx.zeros((0, num_prompts, 4)),
            box_mask=mx.zeros((num_prompts, 0), dtype=mx.bool_),
        )
        return geometric_prompt


class Sam3ImageOnVideoMultiGPU(Sam3Image):
    """Official multi-GPU video-grounding wrapper, unavailable in MLX."""

    def __init__(
        self,
        *args: object,
        async_all_gather: bool = True,
        gather_backbone_out: Callable[..., object] | None = None,
        **kwargs: object,
    ) -> None:
        del args, async_all_gather, gather_backbone_out, kwargs
        _raise_image_unsupported(
            "sam3_mlx.model.sam3_image.Sam3ImageOnVideoMultiGPU",
            reason="video-multi-gpu",
            detail=(
                "Sam3ImageOnVideoMultiGPU depends on official Torch distributed "
                "all-gather video grounding. The MLX port supports the image model "
                "and selected-frame video API only."
            ),
        )

    def forward_video_grounding_multigpu(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        _raise_image_unsupported(
            "sam3_mlx.model.sam3_image.Sam3ImageOnVideoMultiGPU.forward_video_grounding_multigpu",
            reason="video-multi-gpu",
            detail=(
                "This is a Torch distributed video-grounding path and is not "
                "ported to MLX."
            ),
        )
