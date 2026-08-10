from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict, cast

import mlx.core as mx
from mlx import nn
import numpy as np
import pytest

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.data_misc import NestedTensor
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.model_misc import DotProductScoring, MLP
from sam3_mlx.model.sam3_image import Sam3Image


class _FakeFindInput:
    @property
    def img_ids(self) -> mx.array:
        return mx.array([0], dtype=mx.int64)

    @property
    def text_ids(self) -> mx.array:
        return mx.array([0], dtype=mx.int64)

    @property
    def input_boxes(self) -> mx.array:
        return mx.zeros((0, 1, 4))

    @property
    def input_boxes_mask(self) -> mx.array:
        return mx.zeros((1, 0), dtype=mx.bool_)

    @property
    def input_boxes_label(self) -> mx.array:
        return mx.zeros((0, 1), dtype=mx.int64)

    @property
    def input_points(self) -> mx.array | None:
        return None

    @property
    def input_boxes_before_embed(self) -> mx.array | None:
        return None

    @property
    def input_points_before_embed(self) -> mx.array | None:
        return None

    @property
    def input_points_mask(self) -> mx.array | None:
        return mx.zeros((1, 0), dtype=mx.bool_)


class _FakeFindTarget:
    @property
    def boxes(self) -> mx.array:
        return mx.zeros((1, 0, 4))

    @property
    def boxes_padded(self) -> mx.array:
        return mx.zeros((1, 0, 4))

    @property
    def num_boxes(self) -> mx.array:
        return mx.zeros((1,), dtype=mx.int64)

    @property
    def segments(self) -> mx.array | None:
        return None

    @property
    def semantic_segments(self) -> mx.array | None:
        return None

    @property
    def is_valid_segment(self) -> mx.array | None:
        return None

    @property
    def is_exhaustive(self) -> mx.array:
        return mx.zeros((1,), dtype=mx.bool_)

    @property
    def object_ids(self) -> mx.array:
        return mx.zeros((1, 0), dtype=mx.int64)

    @property
    def object_ids_padded(self) -> mx.array:
        return mx.zeros((1, 0), dtype=mx.int64)


class _FakeDatapoint:
    @property
    def img_batch(self) -> mx.array:
        return mx.zeros((1, 3, 4, 4))

    @property
    def find_text_batch(self) -> Sequence[str]:
        return ["object"]

    @property
    def find_inputs(self) -> Sequence[_FakeFindInput]:
        return [_FakeFindInput()]

    @property
    def find_targets(self) -> Sequence[_FakeFindTarget]:
        return [_FakeFindTarget()]


class _FakeMaskEncoder:
    @property
    def mask_downsampler(self) -> Callable[[mx.array], mx.array]:
        return lambda value: value


class _Geometry:
    @property
    def mask_encoder(self) -> _FakeMaskEncoder:
        return _FakeMaskEncoder()

    def __call__(
        self,
        *,
        geo_prompt: Prompt,
        img_feats: Sequence[mx.array],
        img_sizes: Sequence[tuple[int, int]],
        img_pos_embeds: Sequence[mx.array],
    ) -> tuple[mx.array, mx.array]:
        del geo_prompt, img_feats, img_sizes, img_pos_embeds
        return mx.zeros((0, 1, 2)), mx.zeros((1, 0), dtype=mx.bool_)


class _ConstructorBackbone:
    def forward_image(self, samples: object) -> dict[str, object]:
        del samples
        return {}

    def forward_text(
        self,
        captions: object,
        input_boxes: object | None = None,
        additional_text: object | None = None,
        device: str | None = None,
    ) -> dict[str, object]:
        del captions, input_boxes, additional_text, device
        return {}


class _ConstructorEmbedding:
    @property
    def weight(self) -> mx.array:
        return mx.zeros((1, 8))


class _ConstructorDecoder:
    @property
    def num_queries(self) -> int:
        return 1

    @property
    def num_o2m_queries(self) -> int:
        return 0

    @property
    def dac(self) -> bool:
        return False

    @property
    def query_embed(self) -> _ConstructorEmbedding:
        return _ConstructorEmbedding()

    @property
    def bbox_embed(self) -> Callable[[mx.array], mx.array]:
        return lambda value: value

    @property
    def instance_bbox_embed(self) -> Callable[[mx.array], mx.array] | None:
        return None

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
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array]:
        del (
            tgt,
            memory,
            memory_key_padding_mask,
            pos,
            reference_boxes,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            tgt_mask,
            memory_text,
            text_attention_mask,
            apply_dac,
        )
        return (
            mx.zeros((1, 1, 1, 8)),
            mx.zeros((1, 1, 1, 4)),
            None,
            mx.zeros((1, 1, 8)),
        )


class _EncoderMemory(TypedDict):
    memory: mx.array
    pos_embed: mx.array
    padding_mask: mx.array | None
    level_start_index: mx.array
    spatial_shapes: mx.array
    valid_ratios: mx.array


class _ConstructorEncoder:
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
    ) -> _EncoderMemory:
        del (
            src,
            src_key_padding_mask,
            src_pos,
            prompt,
            prompt_pos,
            prompt_key_padding_mask,
            feat_sizes,
            encoder_extra_kwargs,
        )
        return {
            "memory": mx.zeros((1, 1, 8)),
            "pos_embed": mx.zeros((1, 1, 8)),
            "padding_mask": None,
            "level_start_index": mx.zeros((1,), dtype=mx.int64),
            "spatial_shapes": mx.zeros((1, 2), dtype=mx.int64),
            "valid_ratios": mx.ones((1, 2)),
        }


class _ConstructorTransformer:
    d_model = 8
    encoder = _ConstructorEncoder()
    decoder = _ConstructorDecoder()


class _BatchBackbone:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def forward_image(self, samples: object) -> dict[str, object]:
        image = np.asarray(cast(np.ndarray, samples))
        self.calls.append(image)
        image_mx = mx.array(image)
        return {
            "backbone_fpn": [image_mx],
            "vision_pos_enc": [mx.zeros_like(image_mx)],
        }

    def forward_text(
        self,
        captions: object,
        input_boxes: object | None = None,
        additional_text: object | None = None,
        device: str | None = None,
    ) -> dict[str, object]:
        del captions, input_boxes, additional_text, device
        return {}


class _PresenceSegmentationHead:
    @property
    def instance_keys(self) -> Sequence[str]:
        return ("pred_masks",)

    def __call__(self, *args: object, **kwargs: object) -> dict[str, mx.array | None]:
        del args, kwargs
        return {
            "pred_masks": mx.ones((1, 1, 1, 1)),
            "presence_logit": None,
        }


class _InteractiveModel:
    @property
    def no_mem_embed(self) -> mx.array:
        return mx.zeros((1, 1, 1))

    def _prepare_backbone_features(
        self, backbone_out: Mapping[str, object]
    ) -> tuple[object, list[mx.array], object, object]:
        del backbone_out
        return None, [mx.zeros((1, 1, 1))], None, None


class _PredictorFeatures(TypedDict):
    image_embed: mx.array
    high_res_feats: list[mx.array]


class _Predictor:
    def __init__(self) -> None:
        self.fail = True
        self.model = _InteractiveModel()
        self._bb_feat_sizes = [(1, 1)]
        self._features: _PredictorFeatures = {
            "image_embed": mx.zeros((1, 1, 1)),
            "high_res_feats": [],
        }
        self._is_image_set = False
        self._is_batch = False
        self._orig_hw = [(9, 11)]

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del kwargs
        if self.fail:
            raise RuntimeError("predictor failed")
        return np.zeros(1), np.zeros(1), np.zeros(1)

    def predict_batch(
        self, *args: object, **kwargs: object
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        del args, kwargs
        if self.fail:
            raise RuntimeError("predictor failed")
        return [np.zeros(1)], [np.zeros(1)], [np.zeros(1)]


def _empty_model() -> Sam3Image:
    model = Sam3Image.__new__(Sam3Image)
    model.num_feature_levels = 1
    setattr(model, "geometry_encoder", _Geometry())
    return model


def test_validation_interactive_steps_raise_canonical_unsupported_error() -> None:
    model = _empty_model()
    model.eval()
    model.num_interactive_steps_val = 1

    with pytest.raises(
        Sam3MlxUnsupportedError,
        match="Validation interactive prompt sampling is not implemented",
    ) as exc_info:
        forward = cast(Callable[[object], object], model.forward)
        forward(_FakeDatapoint())

    assert exc_info.value.reason == "image-interactivity"
    assert exc_info.value.alternative == "num_interactive_steps_val=0"


def test_forward_grounding_rejects_unsupported_validation_interactivity() -> None:
    model = _empty_model()
    model.eval()
    model.num_interactive_steps_val = 1

    with pytest.raises(
        Sam3MlxUnsupportedError,
        match="Validation interactive prompt sampling is not implemented",
    ) as exc_info:
        model.forward_grounding({}, _FakeFindInput(), _FakeFindTarget(), Prompt())

    assert exc_info.value.reason == "image-interactivity"
    assert exc_info.value.alternative == "num_interactive_steps_val=0"


def test_visual_prompt_embedding_and_mask_must_be_provided_together() -> None:
    model = _empty_model()
    backbone_out: dict[str, object] = {
        "language_features": mx.zeros((1, 1, 2)),
        "language_mask": mx.zeros((1, 1), dtype=mx.bool_),
        "backbone_fpn": [mx.zeros((1, 2, 2, 2))],
        "vision_pos_enc": [mx.zeros((1, 2, 2, 2))],
    }

    with pytest.raises(
        ValueError,
        match="visual_prompt_embed and visual_prompt_mask must be provided together",
    ):
        encode_prompt = cast(
            Callable[..., tuple[mx.array, mx.array, dict[str, object]]],
            getattr(model, "_encode_prompt"),
        )
        encode_prompt(
            backbone_out,
            _FakeFindInput(),
            Prompt(),
            visual_prompt_embed=mx.zeros((1, 1, 2)),
        )


def test_visual_prompt_embedding_and_mask_prompt_counts_must_match() -> None:
    model = _empty_model()
    backbone_out: dict[str, object] = {
        "language_features": mx.zeros((1, 1, 2)),
        "language_mask": mx.zeros((1, 1), dtype=mx.bool_),
        "backbone_fpn": [mx.zeros((1, 2, 2, 2))],
        "vision_pos_enc": [mx.zeros((1, 2, 2, 2))],
    }

    with pytest.raises(
        ValueError,
        match="visual_prompt_embed and visual_prompt_mask disagree on prompt count",
    ):
        encode_prompt = cast(
            Callable[..., tuple[mx.array, mx.array, dict[str, object]]],
            getattr(model, "_encode_prompt"),
        )
        encode_prompt(
            backbone_out,
            _FakeFindInput(),
            Prompt(),
            visual_prompt_embed=mx.zeros((2, 1, 2)),
            visual_prompt_mask=mx.zeros((1, 1), dtype=mx.bool_),
        )


def test_nested_tensor_backbone_features_preserve_wrappers_and_layout() -> None:
    model = _empty_model()
    model.num_feature_levels = 2
    low_values = np.arange(2 * 2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2, 2)
    high_values = (np.arange(2 * 2, dtype=np.float32).reshape(2, 2, 1, 1)) + 100
    low = NestedTensor(mx.array(low_values), mx.zeros((2, 2, 2), dtype=mx.bool_))
    high = NestedTensor(mx.array(high_values), mx.zeros((2, 1, 1), dtype=mx.bool_))
    backbone_out: dict[str, object] = {
        "backbone_fpn": [low, high],
        "vision_pos_enc": [
            mx.zeros((2, 2, 2, 2)),
            mx.zeros((2, 2, 1, 1)),
        ],
    }

    get_img_feats = cast(
        Callable[
            [dict[str, object], mx.array],
            tuple[object, list[mx.array], list[mx.array], list[tuple[int, int]]],
        ],
        getattr(model, "_get_img_feats"),
    )
    returned, image_feats, image_pos, sizes = get_img_feats(
        backbone_out, mx.array([1, 0], dtype=mx.int64)
    )

    assert returned is backbone_out
    returned_mapping = cast(Mapping[str, object], returned)
    assert returned_mapping["backbone_fpn"] == [low, high]
    assert sizes == [(2, 2), (1, 1)]
    assert [tuple(value.shape) for value in image_feats] == [
        (4, 2, 2),
        (1, 2, 2),
    ]
    expected_low = np.transpose(low_values[[1, 0]].reshape(2, 2, -1), (2, 0, 1))
    expected_high = np.transpose(high_values[[1, 0]].reshape(2, 2, -1), (2, 0, 1))
    np.testing.assert_allclose(np.asarray(image_feats[0]), expected_low)
    np.testing.assert_allclose(np.asarray(image_feats[1]), expected_high)
    assert [tuple(value.shape) for value in image_pos] == [(4, 2, 2), (1, 2, 2)]


def test_numpy_image_batch_deduplicates_and_remaps_in_source_order() -> None:
    model = _empty_model()
    backbone = _BatchBackbone()
    setattr(model, "backbone", backbone)
    image_batch = np.arange(4 * 1 * 2 * 2, dtype=np.float32).reshape(4, 1, 2, 2)

    get_img_feats = cast(
        Callable[
            [dict[str, object], mx.array],
            tuple[object, list[mx.array], list[mx.array], list[tuple[int, int]]],
        ],
        getattr(model, "_get_img_feats"),
    )
    _, image_feats, _, _ = get_img_feats(
        {"img_batch_all_stages": image_batch},
        mx.array([2, 0, 2, 1], dtype=mx.int64),
    )

    np.testing.assert_allclose(backbone.calls[0][:, 0, 0, 0], [8, 0, 4])
    returned_values = np.asarray(image_feats[0])[0, :, 0]
    np.testing.assert_allclose(returned_values, [8, 0, 8, 4])


def test_recomputed_backbone_requires_fpn_output() -> None:
    model = _empty_model()
    setattr(model, "backbone", _ConstructorBackbone())
    get_img_feats = cast(
        Callable[..., tuple[object, list[mx.array], list[mx.array], object]],
        getattr(model, "_get_img_feats"),
    )

    with pytest.raises(
        AssertionError,
        match=r"backbone\.forward_image must return 'backbone_fpn'",
    ):
        get_img_feats(
            {"img_batch_all_stages": mx.zeros((1, 3, 2, 2))},
            mx.array([0], dtype=mx.int64),
        )


def test_separate_scorer_deep_copies_concrete_module_and_parameters() -> None:
    prompt_mlp = MLP(
        input_dim=8,
        hidden_dim=16,
        output_dim=8,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(8),
    )
    scorer = DotProductScoring(d_model=8, d_proj=4, prompt_mlp=prompt_mlp)
    # The constructor accepts heterogeneous module protocols; these doubles
    # exercise only its concrete scorer-registration path.
    construct_model = cast(Callable[..., Sam3Image], Sam3Image)
    model = construct_model(
        backbone=_ConstructorBackbone(),
        transformer=_ConstructorTransformer(),
        input_geometry_encoder=_Geometry(),
        dot_prod_scoring=scorer,
        separate_scorer_for_instance=True,
    )

    instance_scorer = model.instance_dot_prod_scoring
    assert isinstance(instance_scorer, DotProductScoring)
    assert instance_scorer is not scorer
    assert instance_scorer.prompt_mlp is not scorer.prompt_mlp
    instance_parameters = cast(Mapping[str, object], instance_scorer.parameters())
    scorer_parameters = cast(Mapping[str, object], scorer.parameters())
    assert set(cast(Mapping[str, object], model.parameters())) == {
        "dot_prod_scoring",
        "instance_dot_prod_scoring",
    }
    assert set(instance_parameters) == {
        "prompt_mlp",
        "prompt_proj",
        "hs_proj",
    }
    assert isinstance(instance_scorer.prompt_mlp, MLP)
    np.testing.assert_allclose(
        np.asarray(
            cast(Mapping[str, object], instance_parameters["prompt_proj"])["weight"]
        ),
        np.asarray(
            cast(Mapping[str, object], scorer_parameters["prompt_proj"])["weight"]
        ),
    )


def test_segmentation_head_preserves_optional_none_presence_output() -> None:
    model = _empty_model()
    model.segmentation_head = _PresenceSegmentationHead()
    model.o2m_mask_predict = False
    setattr(model, "transformer", _ConstructorTransformer())
    model.eval()
    out: dict[str, object] = {}
    run_segmentation_heads = cast(
        Callable[..., None], getattr(model, "_run_segmentation_heads")
    )
    run_segmentation_heads(
        out=out,
        backbone_out={"backbone_fpn": [mx.zeros((1, 2, 1, 1))]},
        img_ids=mx.array([0], dtype=mx.int64),
        vis_feat_sizes=[(1, 1)],
        encoder_hidden_states=mx.zeros((1, 1, 2)),
        prompt=mx.zeros((1, 1, 2)),
        prompt_mask=mx.zeros((1, 1), dtype=mx.bool_),
        hs=mx.zeros((1, 1, 1, 2)),
    )

    assert "presence_logit" in out
    assert out["presence_logit"] is None


def test_predict_inst_restores_predictor_state_when_prediction_fails() -> None:
    model = _empty_model()
    predictor = _Predictor()
    model.inst_interactive_predictor = predictor
    original: tuple[object, object, object, object] = (
        getattr(predictor, "_features"),
        getattr(predictor, "_is_image_set"),
        getattr(predictor, "_is_batch"),
        getattr(predictor, "_orig_hw"),
    )

    with pytest.raises(RuntimeError, match="predictor failed"):
        model.predict_inst(
            {
                "original_height": 32,
                "original_width": 48,
                "backbone_out": {"sam2_backbone_out": {}},
            }
        )

    assert getattr(predictor, "_features") is original[0]
    assert getattr(predictor, "_is_image_set") is original[1]
    assert getattr(predictor, "_is_batch") is original[2]
    assert getattr(predictor, "_orig_hw") is original[3]

    with pytest.raises(RuntimeError, match="predictor failed"):
        model.predict_inst_batch(
            {
                "original_heights": [32],
                "original_widths": [48],
                "backbone_out": {"sam2_backbone_out": {}},
            }
        )

    assert getattr(predictor, "_features") is original[0]
    assert getattr(predictor, "_is_image_set") is original[1]
    assert getattr(predictor, "_is_batch") is original[2]
    assert getattr(predictor, "_orig_hw") is original[3]

    predictor.fail = False
    model.predict_inst(
        {
            "original_height": 32,
            "original_width": 48,
            "backbone_out": {"sam2_backbone_out": {}},
        }
    )
    assert getattr(predictor, "_features") is original[0]
    assert getattr(predictor, "_is_image_set") is original[1]
    assert getattr(predictor, "_is_batch") is original[2]
    assert getattr(predictor, "_orig_hw") is original[3]

    model.predict_inst_batch(
        {
            "original_heights": [32],
            "original_widths": [48],
            "backbone_out": {"sam2_backbone_out": {}},
        }
    )
    assert getattr(predictor, "_features") is original[0]
    assert getattr(predictor, "_is_image_set") is original[1]
    assert getattr(predictor, "_is_batch") is original[2]
    assert getattr(predictor, "_orig_hw") is original[3]
