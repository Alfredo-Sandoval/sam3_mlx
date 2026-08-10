import pytest
import mlx.core as mx
from typing import Protocol, cast

from sam3_mlx.model.decoder import (
    DecoupledTransformerDecoderLayerv2,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderCrossAttention,
    TransformerEncoderDecoupledCrossAttention,
    nn,
)


class _DummyCrossAttention:
    num_heads = 2


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = _DummyCrossAttention()
        self.cross_attn_image = _DummyCrossAttention()
        self.proj = nn.Linear(4, 4)

    def __call__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("decoder constructor test must not execute a layer")


class _IdentityRope(nn.Module):
    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        num_k_exclude_rope: int = 0,
    ) -> mx.array:
        del k, v, num_k_exclude_rope
        return q


class _ConstantSelfAttention(nn.Module):
    num_heads = 1

    def __call__(self, *args: object, **kwargs: object) -> mx.array:
        del kwargs
        query = args[0]
        if not isinstance(query, mx.array):
            raise TypeError("attention query must be an MLX array")
        return mx.ones_like(query)


class _ZeroCrossAttention(nn.Module):
    num_heads = 1

    def __call__(self, *args: object, **kwargs: object) -> mx.array:
        del args
        queries = kwargs["queries"]
        if not isinstance(queries, mx.array):
            raise TypeError("cross-attention queries must be an MLX array")
        return mx.zeros_like(queries)


class _ZeroArray(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return mx.zeros_like(x)


class _ParameterModule(Protocol):
    def parameters(self) -> object: ...

    def trainable_parameters(self) -> object: ...


def _count_parameter_leaves(tree: object) -> int:
    if isinstance(tree, dict):
        mapping = cast(dict[object, object], tree)
        return sum(_count_parameter_leaves(value) for value in mapping.values())
    if isinstance(tree, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], tree)
        return sum(_count_parameter_leaves(value) for value in sequence)
    return 1


def _assert_frozen_module_has_parameters(module: _ParameterModule) -> None:
    assert _count_parameter_leaves(module.parameters()) > 0
    assert _count_parameter_leaves(module.trainable_parameters()) == 0


def _assert_unfrozen_module_is_trainable(module: _ParameterModule) -> None:
    total = _count_parameter_leaves(module.parameters())
    assert total > 0
    assert _count_parameter_leaves(module.trainable_parameters()) == total


def test_transformer_decoder_rejects_unknown_boxrpb_without_instance_query():
    with pytest.raises(AssertionError):
        TransformerDecoder(
            d_model=4,
            frozen=False,
            interaction_layer=None,
            layer=_DummyLayer(),
            num_layers=1,
            num_queries=1,
            return_intermediate=True,
            box_refine=True,
            boxRPB="cubic",
            instance_query=False,
        )


def test_transformer_decoder_layer_default_dac_keeps_self_attention_update() -> None:
    layer = TransformerDecoderLayer(
        activation="relu",
        d_model=2,
        dim_feedforward=2,
        dropout=0.0,
        cross_attention=_ZeroCrossAttention(),
        n_heads=1,
    )
    layer.self_attn = _ConstantSelfAttention()
    layer.norm1 = nn.Identity()
    layer.norm2 = nn.Identity()
    layer.norm3 = nn.Identity()
    layer.linear1 = _ZeroArray()
    layer.linear2 = _ZeroArray()
    target = mx.concat([mx.zeros((2, 1, 2)), mx.full((2, 1, 2), 5.0)], axis=0)

    output, presence = layer(
        tgt=target,
        tgt_query_pos=mx.zeros_like(target),
        memory=mx.zeros((1, 1, 2)),
        dac=True,
    )

    assert presence is None
    assert bool(mx.array_equal(output[:2], mx.ones((2, 1, 2))))
    assert bool(mx.array_equal(output[2:], mx.full((2, 1, 2), 5.0)))


def test_transformer_decoder_frozen_constructor_freezes_parameters():
    decoder = TransformerDecoder(
        d_model=4,
        frozen=True,
        interaction_layer=None,
        layer=_DummyLayer(),
        num_layers=1,
        num_queries=1,
        return_intermediate=True,
        box_refine=True,
    )

    _assert_frozen_module_has_parameters(decoder)


def test_transformer_encoder_cross_attention_frozen_constructor_freezes_parameters():
    encoder = TransformerEncoderCrossAttention(
        d_model=4,
        frozen=True,
        pos_enc_at_input=False,
        layer=_DummyLayer(),
        num_layers=1,
    )

    _assert_frozen_module_has_parameters(encoder)


def test_transformer_encoder_decoupled_cross_attention_frozen_constructor_freezes_parameters():
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=4,
        frozen=True,
        pos_enc_at_input=False,
        layer=_DummyLayer(),
        num_layers=1,
    )

    _assert_frozen_module_has_parameters(encoder)


def test_transformer_decoder_unfrozen_constructor_keeps_parameters_trainable():
    decoder = TransformerDecoder(
        d_model=4,
        frozen=False,
        interaction_layer=None,
        layer=_DummyLayer(),
        num_layers=1,
        num_queries=1,
        return_intermediate=True,
        box_refine=True,
    )

    _assert_unfrozen_module_is_trainable(decoder)


def test_transformer_encoder_cross_attention_unfrozen_constructor_keeps_parameters_trainable():
    encoder = TransformerEncoderCrossAttention(
        d_model=4,
        frozen=False,
        pos_enc_at_input=False,
        layer=_DummyLayer(),
        num_layers=1,
    )

    _assert_unfrozen_module_is_trainable(encoder)


def test_transformer_encoder_decoupled_cross_attention_unfrozen_constructor_keeps_parameters_trainable():
    encoder = TransformerEncoderDecoupledCrossAttention(
        d_model=4,
        frozen=False,
        pos_enc_at_input=False,
        layer=_DummyLayer(),
        num_layers=1,
    )

    _assert_unfrozen_module_is_trainable(encoder)


def test_decoupled_decoder_layer_returns_image_and_target() -> None:
    layer = DecoupledTransformerDecoderLayerv2(
        activation="relu",
        d_model=4,
        num_heads=1,
        dim_feedforward=8,
        dropout=0.0,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention_rope=_IdentityRope(),
        cross_attention_rope=_IdentityRope(),
    )
    layer.self_attn_out_proj = _ZeroArray()
    layer.cross_attn_out_proj = _ZeroArray()
    layer.linear1 = _ZeroArray()
    layer.linear2 = _ZeroArray()
    image = mx.array([[[1.0, 2.0, 3.0, 4.0]]])
    target = mx.array([[[9.0, 8.0, 7.0, 6.0]]])

    image_out, target_out = layer(
        image=image,
        tgt=target,
        memory_image=image,
        memory=target,
    )

    assert bool(mx.array_equal(image_out, image))
    assert not bool(mx.array_equal(image_out, target))
    assert bool(mx.array_equal(target_out, target))
    assert not bool(mx.array_equal(target_out, image))
