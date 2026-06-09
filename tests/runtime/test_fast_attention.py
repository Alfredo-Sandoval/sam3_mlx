import math

import mlx.core as mx
import numpy as np

from sam3_mlx.model.model_misc import (
    MultiheadAttentionWrapper,
    _merge_attention_masks,
    _scaled_dot_product_attention,
    _to_attention_mask,
    multi_head_attention_forward,
)
from sam3_mlx.perflib.fa3 import flash_attn_func


def _to_numpy(value):
    mx.eval(value)
    return np.asarray(value)


def _softmax(value, axis=-1):
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _flash_attn_reference(q, k, v):
    q_heads = np.transpose(q.astype(np.float32), (0, 2, 1, 3))
    k_heads = np.transpose(k.astype(np.float32), (0, 2, 3, 1))
    v_heads = np.transpose(v.astype(np.float32), (0, 2, 1, 3))
    scores = q_heads @ k_heads / math.sqrt(q.shape[-1])
    probs = _softmax(scores, axis=-1)
    return np.transpose(probs @ v_heads, (0, 2, 1, 3)).astype(q.dtype)


def test_attention_mask_merge_preserves_bool_keep_mask_for_fast_path():
    attn_mask = mx.array([[[[False, True, False]]]], dtype=mx.bool_)
    key_padding_mask = mx.array([[[[False, False, True]]]], dtype=mx.uint8)

    merged = _merge_attention_masks(
        attn_mask,
        key_padding_mask,
        mx.float32,
        preserve_bool=True,
    )

    assert merged.dtype == mx.bool_
    np.testing.assert_array_equal(
        _to_numpy(merged),
        np.array([[[[True, False, False]]]], dtype=bool),
    )


def test_attention_mask_merge_converts_bool_when_additive_bias_is_present():
    attn_mask = mx.array([[[[False, True]]]], dtype=mx.bool_)
    attn_bias = mx.array([[[[0.25, 0.5]]]], dtype=mx.float32)

    merged = _merge_attention_masks(
        attn_mask,
        attn_bias,
        mx.float32,
        preserve_bool=False,
    )

    assert merged.dtype == mx.float32
    np.testing.assert_allclose(
        _to_numpy(merged),
        np.array([[[[0.25, -np.inf]]]], dtype=np.float32),
    )


def test_attention_mask_normalization_keeps_bool_for_fast_path():
    block_mask = mx.array([[False, True, False], [True, False, False]], dtype=mx.bool_)
    uint8_block_mask = mx.array([[0, 1, 0], [1, 0, 0]], dtype=mx.uint8)

    bool_keep_mask = _to_attention_mask(block_mask, mx.float32, preserve_bool=True)
    uint8_keep_mask = _to_attention_mask(
        uint8_block_mask, mx.float32, preserve_bool=True
    )

    assert bool_keep_mask.dtype == mx.bool_
    assert uint8_keep_mask.dtype == mx.bool_
    expected_keep = np.array([[True, False, True], [False, True, True]], dtype=bool)
    np.testing.assert_array_equal(_to_numpy(bool_keep_mask), expected_keep)
    np.testing.assert_array_equal(_to_numpy(uint8_keep_mask), expected_keep)


def test_attention_mask_normalization_converts_bool_for_manual_path():
    block_mask = mx.array([[False, True], [True, False]], dtype=mx.bool_)

    additive = _to_attention_mask(block_mask, mx.float32, preserve_bool=False)

    assert additive.dtype == mx.float32
    np.testing.assert_allclose(
        _to_numpy(additive),
        np.array([[0.0, -np.inf], [-np.inf, 0.0]], dtype=np.float32),
    )


def test_multihead_attention_wrapper_combines_bool_masks_as_fast_keep_mask():
    wrapper = MultiheadAttentionWrapper(4, 2)
    queries = mx.zeros((2, 2, 4), dtype=mx.float32)
    attn_mask = mx.array(
        [
            [False, True, False],
            [False, False, False],
        ],
        dtype=mx.bool_,
    )
    key_padding_mask = mx.array(
        [
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=mx.uint8,
    )

    mask = wrapper._combine_masks(
        attn_mask,
        key_padding_mask,
        base_mask=None,
        queries=queries,
        preserve_bool=True,
        dtype=mx.float32,
    )

    assert mask.dtype == mx.bool_
    np.testing.assert_array_equal(
        _to_numpy(mask),
        np.array(
            [
                [[[True, False, False], [True, True, False]]],
                [[[False, False, True], [False, True, True]]],
            ],
            dtype=bool,
        ),
    )


def test_multihead_attention_wrapper_uses_additive_mask_when_bias_requires_it():
    wrapper = MultiheadAttentionWrapper(4, 2)
    queries = mx.zeros((1, 2, 4), dtype=mx.float32)
    attn_mask = mx.array([[False, True, False], [False, False, False]], dtype=mx.bool_)
    key_padding_mask = mx.array([[0, 0, 1]], dtype=mx.uint8)

    mask = wrapper._combine_masks(
        attn_mask,
        key_padding_mask,
        base_mask=None,
        queries=queries,
        preserve_bool=False,
        dtype=mx.float32,
    )

    assert mask.dtype == mx.float32
    np.testing.assert_allclose(
        _to_numpy(mask),
        np.array([[[[0.0, -np.inf, -np.inf], [0.0, 0.0, -np.inf]]]], dtype=np.float32),
    )


def _split_heads(sequence_first, num_heads):
    seq_len, batch_size, embed_dim = sequence_first.shape
    head_dim = embed_dim // num_heads
    return (
        sequence_first.transpose(1, 0, 2)
        .reshape(batch_size, seq_len, num_heads, head_dim)
        .transpose(0, 2, 1, 3)
    )


def _mha_reference(
    query,
    key,
    value,
    num_heads,
    attn_mask=None,
    key_padding_mask=None,
    attn_bias=None,
):
    q = _split_heads(query, num_heads)
    k = _split_heads(key, num_heads)
    v = _split_heads(value, num_heads)

    scores = q @ np.swapaxes(k, -1, -2) * (q.shape[-1] ** -0.5)
    if attn_mask is not None:
        if attn_mask.dtype == np.bool_:
            scores = np.where(attn_mask[None, None], -np.inf, scores)
        else:
            scores = scores + attn_mask[None, None].astype(np.float32)
    if key_padding_mask is not None:
        scores = np.where(
            key_padding_mask.astype(bool)[:, None, None, :], -np.inf, scores
        )
    if attn_bias is not None:
        scores = scores + attn_bias.astype(np.float32)

    weights = _softmax(scores, axis=-1)
    out = weights @ v
    seq_len, batch_size, embed_dim = query.shape
    out = out.transpose(2, 0, 1, 3).reshape(seq_len, batch_size, embed_dim)
    return out.astype(np.float32), weights.mean(axis=1).astype(np.float32)


def _set_identity_attention_weights(
    wrapper, q_bias=None, k_bias=None, v_bias=None, out_bias=None
):
    wrapper.query_proj.weight = mx.eye(4, dtype=mx.float32)
    wrapper.key_proj.weight = mx.eye(4, dtype=mx.float32)
    wrapper.value_proj.weight = mx.eye(4, dtype=mx.float32)
    wrapper.out_proj.weight = mx.eye(4, dtype=mx.float32)
    wrapper.query_proj.bias = (
        mx.zeros((4,), dtype=mx.float32) if q_bias is None else mx.array(q_bias)
    )
    wrapper.key_proj.bias = (
        mx.zeros((4,), dtype=mx.float32) if k_bias is None else mx.array(k_bias)
    )
    wrapper.value_proj.bias = (
        mx.zeros((4,), dtype=mx.float32) if v_bias is None else mx.array(v_bias)
    )
    wrapper.out_proj.bias = (
        mx.zeros((4,), dtype=mx.float32) if out_bias is None else mx.array(out_bias)
    )


def test_multihead_attention_wrapper_keeps_local_batch_first_contract():
    query = np.array(
        [
            [[0.2, -0.1, 0.4, 0.7], [-0.3, 0.8, 0.1, -0.6]],
            [[0.5, 0.2, -0.4, 0.1], [0.9, -0.2, 0.3, -0.5]],
        ],
        dtype=np.float32,
    )
    key = np.array(
        [
            [[0.5, -0.4, 0.2, 0.0], [-0.2, 0.3, 0.6, -0.1], [0.7, 0.1, -0.5, 0.4]],
            [[-0.3, 0.6, 0.2, 0.5], [0.8, -0.4, 0.1, 0.2], [0.0, 0.7, -0.6, 0.3]],
        ],
        dtype=np.float32,
    )
    value = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [-0.2, 0.5, -0.4, 0.7]],
            [[-0.1, 0.6, 0.2, -0.3], [0.3, -0.7, 0.5, 0.8], [0.9, 0.1, -0.2, 0.4]],
        ],
        dtype=np.float32,
    )
    attn_mask = np.array([[False, True, False], [False, False, False]])
    key_padding_mask = np.array([[False, False, True], [True, False, False]])
    expected, _ = _mha_reference(
        query.transpose(1, 0, 2),
        key.transpose(1, 0, 2),
        value.transpose(1, 0, 2),
        num_heads=2,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )

    wrapper = MultiheadAttentionWrapper(4, 2)
    _set_identity_attention_weights(wrapper)
    out = wrapper(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        attn_mask=mx.array(attn_mask),
        key_padding_mask=mx.array(key_padding_mask),
    )

    np.testing.assert_allclose(
        _to_numpy(out), expected.transpose(1, 0, 2), atol=1e-6, rtol=1e-6
    )


def test_multihead_attention_wrapper_supports_official_embed_dim_sequence_first_api():
    query = np.array(
        [
            [[0.2, -0.1, 0.4, 0.7]],
            [[-0.3, 0.8, 0.1, -0.6]],
        ],
        dtype=np.float32,
    )
    key = np.array(
        [
            [[0.5, -0.4, 0.2, 0.0]],
            [[-0.2, 0.3, 0.6, -0.1]],
            [[0.7, 0.1, -0.5, 0.4]],
        ],
        dtype=np.float32,
    )
    value = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1]],
            [[-0.2, 0.5, -0.4, 0.7]],
        ],
        dtype=np.float32,
    )
    attn_mask = np.array([[False, True, False], [False, False, False]])
    expected_out, expected_weights = _mha_reference(
        query, key, value, num_heads=2, attn_mask=attn_mask
    )

    wrapper = MultiheadAttentionWrapper(embed_dim=4, num_heads=2)
    _set_identity_attention_weights(wrapper)
    out, weights = wrapper(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        attn_mask=mx.array(attn_mask),
        need_weights=True,
    )

    np.testing.assert_allclose(_to_numpy(out), expected_out, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(weights), expected_weights, atol=1e-6, rtol=1e-6
    )


def test_multihead_attention_wrapper_preserves_split_qkv_projection_biases():
    query = np.array([[[0.2, -0.1, 0.4, 0.7]]], dtype=np.float32)
    key = np.array(
        [
            [[0.5, -0.4, 0.2, 0.0]],
            [[-0.2, 0.3, 0.6, -0.1]],
        ],
        dtype=np.float32,
    )
    value = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )
    q_bias = np.array([0.05, -0.02, 0.03, 0.01], dtype=np.float32)
    k_bias = np.array([-0.01, 0.04, -0.03, 0.02], dtype=np.float32)
    v_bias = np.array([0.07, -0.05, 0.02, -0.04], dtype=np.float32)
    out_bias = np.array([0.11, -0.09, 0.03, -0.02], dtype=np.float32)
    expected, _ = _mha_reference(
        query + q_bias, key + k_bias, value + v_bias, num_heads=2
    )
    expected = expected + out_bias

    wrapper = MultiheadAttentionWrapper(embed_dim=4, num_heads=2)
    _set_identity_attention_weights(
        wrapper,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        out_bias=out_bias,
    )
    out = wrapper(mx.array(query), mx.array(key), mx.array(value))

    np.testing.assert_allclose(_to_numpy(out), expected, atol=1e-6, rtol=1e-6)


def test_flash_attn_func_mlx_matches_independent_attention_reference():
    q = (np.arange(1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4) - 7) / 5
    k = (np.arange(1 * 4 * 2 * 4, dtype=np.float32).reshape(1, 4, 2, 4) - 11) / 7
    v = (np.arange(1 * 4 * 2 * 4, dtype=np.float32).reshape(1, 4, 2, 4) - 13) / 9

    out = flash_attn_func(mx.array(q), mx.array(k), mx.array(v))

    np.testing.assert_allclose(
        _to_numpy(out), _flash_attn_reference(q, k, v), atol=1e-6, rtol=1e-6
    )


def test_scaled_dot_product_attention_fast_path_matches_manual_path():
    query = mx.array(
        [
            [
                [[0.1, 0.4, -0.2], [0.3, -0.1, 0.5]],
                [[-0.2, 0.7, 0.1], [0.6, 0.2, -0.3]],
            ]
        ],
        dtype=mx.float32,
    )
    key = mx.array(
        [
            [
                [[0.2, -0.3, 0.8], [0.5, 0.1, -0.4], [-0.2, 0.6, 0.3]],
                [[-0.4, 0.3, 0.2], [0.7, -0.1, 0.5], [0.2, 0.4, -0.6]],
            ]
        ],
        dtype=mx.float32,
    )
    value = mx.array(
        [
            [
                [[0.5, 0.1, -0.2], [0.2, -0.4, 0.3], [0.7, 0.6, -0.1]],
                [[-0.3, 0.2, 0.8], [0.1, -0.5, 0.4], [0.9, 0.3, -0.7]],
            ]
        ],
        dtype=mx.float32,
    )
    attn_mask = mx.array(
        [[[[0.0, -float("inf"), 0.0], [0.0, 0.0, -0.25]]]],
        dtype=mx.float32,
    )

    fast_out, fast_weights = _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask,
        dropout_p=0.0,
        training=True,
        need_weights=False,
    )
    manual_out, manual_weights = _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask,
        dropout_p=0.0,
        training=True,
        need_weights=True,
    )

    assert fast_weights is None
    assert manual_weights.shape == (1, 2, 2, 3)
    np.testing.assert_allclose(
        _to_numpy(manual_weights).sum(axis=-1),
        np.ones((1, 2, 2), dtype=np.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        _to_numpy(fast_out),
        _to_numpy(manual_out),
        atol=1e-5,
        rtol=1e-5,
    )


def test_multi_head_attention_forward_fast_path_matches_masked_reference():
    query = np.array(
        [
            [[0.2, -0.1, 0.4, 0.7]],
            [[-0.3, 0.8, 0.1, -0.6]],
        ],
        dtype=np.float32,
    )
    key = np.array(
        [
            [[0.5, -0.4, 0.2, 0.0]],
            [[-0.2, 0.3, 0.6, -0.1]],
            [[0.7, 0.1, -0.5, 0.4]],
        ],
        dtype=np.float32,
    )
    value = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1]],
            [[-0.2, 0.5, -0.4, 0.7]],
        ],
        dtype=np.float32,
    )
    attn_mask = np.array([[False, True, False], [False, False, False]])
    key_padding_mask = np.array([[False, False, True]])
    expected, _ = _mha_reference(
        query,
        key,
        value,
        num_heads=2,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
    )

    identity = mx.eye(4, dtype=mx.float32)
    out, weights = multi_head_attention_forward(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        embed_dim_to_check=4,
        num_heads=2,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=identity,
        out_proj_bias=None,
        training=False,
        key_padding_mask=mx.array(key_padding_mask),
        need_weights=False,
        attn_mask=mx.array(attn_mask),
        use_separate_proj_weight=True,
        q_proj_weight=identity,
        k_proj_weight=identity,
        v_proj_weight=identity,
    )

    assert weights is None
    np.testing.assert_allclose(_to_numpy(out), expected, atol=1e-6, rtol=1e-6)


def test_multi_head_attention_forward_bool_mask_matches_additive_mask():
    query = np.array(
        [
            [[0.2, -0.1, 0.4, 0.7]],
            [[-0.3, 0.8, 0.1, -0.6]],
        ],
        dtype=np.float32,
    )
    key = np.array(
        [
            [[0.5, -0.4, 0.2, 0.0]],
            [[-0.2, 0.3, 0.6, -0.1]],
            [[0.7, 0.1, -0.5, 0.4]],
        ],
        dtype=np.float32,
    )
    value = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1]],
            [[-0.2, 0.5, -0.4, 0.7]],
        ],
        dtype=np.float32,
    )
    bool_mask = np.array([[False, True, False], [True, False, False]])
    additive_mask = np.where(bool_mask, -np.inf, 0.0).astype(np.float32)

    identity = mx.eye(4, dtype=mx.float32)
    common_kwargs = dict(
        embed_dim_to_check=4,
        num_heads=2,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=identity,
        out_proj_bias=None,
        training=False,
        key_padding_mask=None,
        need_weights=False,
        use_separate_proj_weight=True,
        q_proj_weight=identity,
        k_proj_weight=identity,
        v_proj_weight=identity,
    )
    bool_out, _ = multi_head_attention_forward(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        attn_mask=mx.array(bool_mask),
        **common_kwargs,
    )
    additive_out, _ = multi_head_attention_forward(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        attn_mask=mx.array(additive_mask),
        **common_kwargs,
    )

    np.testing.assert_allclose(
        _to_numpy(bool_out), _to_numpy(additive_out), atol=1e-6, rtol=1e-6
    )


def test_multi_head_attention_forward_combines_key_padding_mask_and_attn_bias():
    query = np.array(
        [
            [[0.2, -0.1, 0.4, 0.7]],
            [[-0.3, 0.8, 0.1, -0.6]],
        ],
        dtype=np.float32,
    )
    key = np.array(
        [
            [[0.5, -0.4, 0.2, 0.0]],
            [[-0.2, 0.3, 0.6, -0.1]],
            [[0.7, 0.1, -0.5, 0.4]],
        ],
        dtype=np.float32,
    )
    value = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1]],
            [[-0.2, 0.5, -0.4, 0.7]],
        ],
        dtype=np.float32,
    )
    key_padding_mask = np.array([[0, 0, 1]], dtype=np.uint8)
    attn_bias = np.array(
        [
            [
                [[0.0, 0.25, 0.0], [0.1, 0.0, 0.0]],
                [[0.0, -0.2, 0.0], [0.0, 0.3, 0.0]],
            ]
        ],
        dtype=np.float32,
    )
    expected, _ = _mha_reference(
        query,
        key,
        value,
        num_heads=2,
        key_padding_mask=key_padding_mask,
        attn_bias=attn_bias,
    )

    identity = mx.eye(4, dtype=mx.float32)
    out, weights = multi_head_attention_forward(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        embed_dim_to_check=4,
        num_heads=2,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=identity,
        out_proj_bias=None,
        training=False,
        key_padding_mask=mx.array(key_padding_mask, dtype=mx.uint8),
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=True,
        q_proj_weight=identity,
        k_proj_weight=identity,
        v_proj_weight=identity,
        attn_bias=mx.array(attn_bias),
    )

    assert weights is None
    np.testing.assert_allclose(_to_numpy(out), expected, atol=1e-6, rtol=1e-6)


def test_scaled_dot_product_attention_keeps_manual_dropout_path_for_training():
    query = mx.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=mx.float32)
    key = mx.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=mx.float32)
    value = mx.array([[[[2.0, -1.0], [0.5, 3.0]]]], dtype=mx.float32)

    out, weights = _scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=1.0,
        training=True,
        need_weights=False,
    )

    assert weights.shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(
        _to_numpy(weights),
        np.zeros((1, 1, 2, 2), dtype=np.float32),
    )
    np.testing.assert_allclose(_to_numpy(out), np.zeros((1, 1, 2, 2), dtype=np.float32))


def test_multihead_attention_wrapper_uint8_mask_uses_pytorch_blocking_semantics():
    mask = mx.array([[0, 1]], dtype=mx.uint8)

    additive = MultiheadAttentionWrapper._to_additive_mask(mask)

    np.testing.assert_allclose(
        _to_numpy(additive), np.array([[0.0, -np.inf]], dtype=np.float32)
    )


def test_multi_head_attention_forward_weight_path_still_returns_weights():
    query = np.array(
        [
            [[0.2, -0.1, 0.4, 0.7]],
            [[-0.3, 0.8, 0.1, -0.6]],
        ],
        dtype=np.float32,
    )
    key = np.array(
        [
            [[0.5, -0.4, 0.2, 0.0]],
            [[-0.2, 0.3, 0.6, -0.1]],
            [[0.7, 0.1, -0.5, 0.4]],
        ],
        dtype=np.float32,
    )
    value = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.4, 0.3, 0.2, 0.1]],
            [[-0.2, 0.5, -0.4, 0.7]],
        ],
        dtype=np.float32,
    )
    attn_mask = np.array([[False, True, False], [False, False, False]])
    expected_out, expected_weights = _mha_reference(
        query,
        key,
        value,
        num_heads=2,
        attn_mask=attn_mask,
    )

    identity = mx.eye(4, dtype=mx.float32)
    out, weights = multi_head_attention_forward(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        embed_dim_to_check=4,
        num_heads=2,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=identity,
        out_proj_bias=None,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=mx.array(attn_mask),
        use_separate_proj_weight=True,
        q_proj_weight=identity,
        k_proj_weight=identity,
        v_proj_weight=identity,
    )

    np.testing.assert_allclose(_to_numpy(out), expected_out, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(weights), expected_weights, atol=1e-6, rtol=1e-6
    )


def test_multi_head_attention_forward_applies_per_row_key_padding_in_batched_input():
    # Boundary: every prior test runs batch=1, which cannot catch per-row mask
    # broadcasting bugs. Build batch=2 with *different* key_padding_masks per row
    # and verify the per-row output differs accordingly, against the independent
    # numpy reference.
    rng = np.random.default_rng(seed=0)
    query = rng.standard_normal((2, 2, 4)).astype(np.float32)
    key = rng.standard_normal((3, 2, 4)).astype(np.float32)
    value = rng.standard_normal((3, 2, 4)).astype(np.float32)
    key_padding_mask = np.array(
        [
            [False, False, True],  # batch row 0: last key masked
            [True, False, False],  # batch row 1: first key masked
        ]
    )
    expected, _ = _mha_reference(
        query,
        key,
        value,
        num_heads=2,
        key_padding_mask=key_padding_mask,
    )

    identity = mx.eye(4, dtype=mx.float32)
    out, _ = multi_head_attention_forward(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        embed_dim_to_check=4,
        num_heads=2,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=identity,
        out_proj_bias=None,
        training=False,
        key_padding_mask=mx.array(key_padding_mask),
        need_weights=False,
        attn_mask=None,
        use_separate_proj_weight=True,
        q_proj_weight=identity,
        k_proj_weight=identity,
        v_proj_weight=identity,
    )

    out_np = _to_numpy(out)
    np.testing.assert_allclose(out_np, expected, atol=1e-6, rtol=1e-6)
    # Guard the *point* of the test: row 0 and row 1 must differ, otherwise a bug
    # that collapses per-row masks would still match the (also collapsed) reference.
    assert not np.allclose(out_np[:, 0, :], out_np[:, 1, :], atol=1e-3), (
        "Per-row key_padding_mask did not affect outputs — broadcasting collapsed."
    )


def test_multi_head_attention_forward_handles_single_token_query_and_key():
    # Smallest valid shape: q_seq=k_seq=1, batch=1. Off-by-one bugs in softmax,
    # reshape, or num_heads splitting tend to fire here.
    query = np.array([[[0.4, -0.2, 0.1, 0.8]]], dtype=np.float32)
    key = np.array([[[0.3, 0.7, -0.5, 0.2]]], dtype=np.float32)
    value = np.array([[[1.0, -1.0, 2.0, -2.0]]], dtype=np.float32)

    # With a single key, softmax over one entry must be exactly 1.0, so the output
    # equals the projected value — a hand-computable invariant that does not depend
    # on any reference implementation.
    expected = value[0, 0]

    identity = mx.eye(4, dtype=mx.float32)
    out, weights = multi_head_attention_forward(
        mx.array(query),
        mx.array(key),
        mx.array(value),
        embed_dim_to_check=4,
        num_heads=2,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=identity,
        out_proj_bias=None,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        use_separate_proj_weight=True,
        q_proj_weight=identity,
        k_proj_weight=identity,
        v_proj_weight=identity,
    )

    np.testing.assert_allclose(_to_numpy(out)[0, 0], expected, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(
        _to_numpy(weights), np.ones((1, 1, 1), dtype=np.float32), atol=1e-6
    )
