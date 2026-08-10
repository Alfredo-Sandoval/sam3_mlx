import mlx.core as mx
import pytest

from sam3_mlx.model.encoder import (
    TransformerEncoder,
    TransformerEncoderFusion,
    TransformerEncoderLayer,
    pool_text_feat,
)
from sam3_mlx.model.model_misc import MultiheadAttentionWrapper


def _tiny_layer(
    *,
    pre_norm: bool = True,
    pos_enc_at_attn: bool = True,
    pos_enc_at_cross_attn_keys: bool = False,
    pos_enc_at_cross_attn_queries: bool = False,
) -> TransformerEncoderLayer:
    return TransformerEncoderLayer(
        activation="relu",
        cross_attention=MultiheadAttentionWrapper(4, 2, bias=True),
        d_model=4,
        dim_feedforward=8,
        dropout=0.0,
        pos_enc_at_attn=pos_enc_at_attn,
        pos_enc_at_cross_attn_keys=pos_enc_at_cross_attn_keys,
        pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries,
        pre_norm=pre_norm,
        self_attention=MultiheadAttentionWrapper(4, 2, bias=True),
    )


def test_pool_text_feat_mean_and_masked():
    prompt = mx.ones((3, 2, 4))
    mean = pool_text_feat(prompt, prompt_mask=None, pool_with_mask=False)
    assert mean.shape == (2, 4)

    mask = mx.array([[False, False, True], [False, True, True]])
    prompt = mx.arange(24, dtype=mx.float32).reshape(3, 2, 4)
    pooled = pool_text_feat(prompt, prompt_mask=mask, pool_with_mask=True)
    assert pooled.shape == (2, 4)
    # batch 0: tokens 0,1 valid; batch 1: token 0 only
    expected_b0 = (prompt[0, 0] + prompt[1, 0]) / 2
    expected_b1 = prompt[0, 1]
    assert mx.allclose(pooled[0], expected_b0).item()
    assert mx.allclose(pooled[1], expected_b1).item()


def test_transformer_encoder_multilevel_prepare_shapes():
    # Factory path: layer instances hold non-pickleable MLX activation callables.
    encoder = TransformerEncoder(
        layer=_tiny_layer,
        num_layers=1,
        d_model=4,
        num_feature_levels=2,
    )
    srcs = [
        mx.zeros((1, 4, 2, 3)),
        mx.zeros((1, 4, 1, 2)),
    ]
    masks = [
        mx.zeros((1, 2, 3), dtype=mx.bool_),
        mx.zeros((1, 1, 2), dtype=mx.bool_),
    ]
    pos = [mx.zeros_like(src) for src in srcs]
    (
        src_flat,
        mask_flat,
        pos_flat,
        level_start_index,
        valid_ratios,
        spatial_shapes,
    ) = encoder._prepare_multilevel_features(srcs, masks, pos)

    assert src_flat.shape == (1, 2 * 3 + 1 * 2, 4)
    assert mask_flat is not None and mask_flat.shape == (1, 8)
    assert pos_flat.shape == (1, 8, 4)
    assert spatial_shapes.shape == (2, 2)
    assert level_start_index.shape == (2,)
    assert valid_ratios.shape == (1, 2, 2)


def test_transformer_encoder_fusion_output_keys():
    fusion = TransformerEncoderFusion(
        layer=_tiny_layer,
        num_layers=1,
        d_model=4,
        num_feature_levels=1,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=False,
    )
    src = [mx.zeros((1, 4, 2, 2))]
    pos = [mx.zeros_like(src[0])]
    prompt = mx.zeros((3, 1, 4))  # seq, bs, dim
    out = fusion(
        src=src,
        prompt=prompt,
        src_pos=pos,
        src_key_padding_mask=[mx.zeros((1, 2, 2), dtype=mx.bool_)],
        prompt_key_padding_mask=mx.zeros((1, 3), dtype=mx.bool_),
    )
    assert set(out) == {
        "memory",
        "padding_mask",
        "pos_embed",
        "memory_text",
        "level_start_index",
        "spatial_shapes",
        "valid_ratios",
    }
    assert out["memory"].shape[0] == 4  # hw, b, c
    assert out["memory_text"].shape == prompt.shape


def test_encoder_layer_fail_fast_when_pos_flag_true_and_pos_missing():
    layer = _tiny_layer(
        pre_norm=False,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=True,
    )
    tgt = mx.zeros((2, 1, 4))
    memory = mx.zeros((3, 1, 4))
    with pytest.raises(TypeError, match="positional encoding is required"):
        layer.forward_post(tgt=tgt, memory=memory, pos=None, query_pos=None)
