import pytest
import mlx.nn as nn

from sam3_mlx.model.decoder import (
    TransformerDecoder,
    TransformerEncoderCrossAttention,
    TransformerEncoderDecoupledCrossAttention,
)


class _DummyCrossAttention:
    num_heads = 2


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = _DummyCrossAttention()
        self.cross_attn_image = _DummyCrossAttention()
        self.proj = nn.Linear(4, 4)

    def __call__(self, *args, **kwargs):
        raise AssertionError("decoder constructor test must not execute a layer")


def _count_parameter_leaves(tree):
    if isinstance(tree, dict):
        return sum(_count_parameter_leaves(value) for value in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(_count_parameter_leaves(value) for value in tree)
    return 1


def _assert_frozen_module_has_parameters(module):
    assert _count_parameter_leaves(module.parameters()) > 0
    assert _count_parameter_leaves(module.trainable_parameters()) == 0


def _assert_unfrozen_module_is_trainable(module):
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
