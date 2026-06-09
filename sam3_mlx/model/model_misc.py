import math
import weakref
from contextlib import AbstractContextManager
from copy import deepcopy
from enum import Enum, auto
from functools import partial
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path

from sam3_mlx._unsupported import raise_unsupported


def _raise_attention_unsupported(feature: str, *, reason: str, detail: str) -> None:
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
    )


def inverse_sigmoid(x, eps=1e-3):
    x = mx.clip(x, 0, 1)
    x1 = mx.clip(x, eps, None)
    x2 = mx.clip((1 - x), eps, None)
    return mx.log(x1 / x2)


class AttentionType:
    """Type of attention, matching the official SAM3 constants."""

    Vanilla = "Vanilla"
    Xformer = "Xformer"
    Sparse = "Sparse"
    Deformable = "Deformable"


def get_sdpa_settings():
    return True, False, True


OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()


def _linear(input_array, weight, bias=None):
    output = mx.matmul(input_array, weight.transpose(1, 0))
    if bias is not None:
        output = output + bias
    return output


def _in_projection(query, key, value, q_weight, k_weight, v_weight, bias=None):
    if bias is None:
        q_bias = k_bias = v_bias = None
    else:
        q_bias, k_bias, v_bias = mx.split(bias, 3, axis=0)
    return (
        _linear(query, q_weight, q_bias),
        _linear(key, k_weight, k_bias),
        _linear(value, v_weight, v_bias),
    )


def _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias=None):
    q_weight, k_weight, v_weight = mx.split(in_proj_weight, 3, axis=0)
    return _in_projection(
        query,
        key,
        value,
        q_weight,
        k_weight,
        v_weight,
        in_proj_bias,
    )


def _pad_last_dim(array, pad_count=1):
    pad_width = [(0, 0)] * array.ndim
    pad_width[-1] = (0, pad_count)
    return mx.pad(array, pad_width)


def _is_bool_mask(mask):
    return mask is not None and mask.dtype == mx.bool_


def _is_bool_like_mask(mask):
    return mask is not None and mask.dtype in (mx.bool_, mx.uint8)


def _to_block_bool_mask(mask):
    if mask.dtype == mx.bool_:
        return mask
    if mask.dtype == mx.uint8:
        return mask.astype(mx.bool_)
    raise TypeError(f"Expected a bool-like attention mask, got {mask.dtype}.")


def _to_additive_attention_mask(mask, dtype):
    if mask is None:
        return None
    if _is_bool_mask(mask):
        return mx.where(
            mask, mx.array(-float("inf"), dtype=dtype), mx.array(0.0, dtype=dtype)
        )
    if mask.dtype == mx.uint8:
        mask = mask.astype(mx.bool_)
        return mx.where(
            mask, mx.array(-float("inf"), dtype=dtype), mx.array(0.0, dtype=dtype)
        )
    return mask.astype(dtype)


def _to_attention_mask(mask, dtype, *, preserve_bool: bool):
    if mask is None:
        return None
    if _is_bool_like_mask(mask):
        block_mask = _to_block_bool_mask(mask)
        if preserve_bool:
            return mx.logical_not(block_mask)
        return _to_additive_attention_mask(block_mask, dtype)
    return mask.astype(dtype)


def _merge_attention_masks(left, right, dtype, *, preserve_bool: bool):
    if left is None:
        return _to_attention_mask(right, dtype, preserve_bool=preserve_bool)
    if right is None:
        return _to_attention_mask(left, dtype, preserve_bool=preserve_bool)
    if preserve_bool and _is_bool_like_mask(left) and _is_bool_like_mask(right):
        block_mask = mx.logical_or(
            _to_block_bool_mask(left),
            _to_block_bool_mask(right),
        )
        return mx.logical_not(block_mask)
    return _to_additive_attention_mask(left, dtype) + _to_additive_attention_mask(
        right, dtype
    )


def _dropout(array, p, training):
    if p == 0.0 or not training:
        return array
    keep_prob = 1.0 - p
    if keep_prob <= 0.0:
        return mx.zeros_like(array)
    keep = mx.random.bernoulli(p=keep_prob, shape=array.shape)
    return array * keep.astype(array.dtype) / keep_prob


def _scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    training,
    need_weights=True,
):
    scale = query.shape[-1] ** -0.5
    if not need_weights and (dropout_p == 0.0 or not training):
        return (
            mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=scale,
                mask=attn_mask,
            ),
            None,
        )
    scores = mx.matmul(query, key.transpose(0, 1, 3, 2)) * scale
    if attn_mask is not None:
        scores = scores + _to_additive_attention_mask(attn_mask, scores.dtype)
    weights = mx.softmax(scores, axis=-1)
    weights = _dropout(weights, dropout_p, training)
    return mx.matmul(weights, value), weights


def multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[mx.array],
    in_proj_bias: Optional[mx.array],
    bias_k: Optional[mx.array],
    bias_v: Optional[mx.array],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: mx.array,
    out_proj_bias: Optional[mx.array],
    training: bool = True,
    key_padding_mask: Optional[mx.array] = None,
    need_weights: bool = True,
    attn_mask: Optional[mx.array] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[mx.array] = None,
    k_proj_weight: Optional[mx.array] = None,
    v_proj_weight: Optional[mx.array] = None,
    static_k: Optional[mx.array] = None,
    static_v: Optional[mx.array] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    attn_type: AttentionType = AttentionType.Vanilla,
    attn_sparsity: float = 0.0,
    attn_bias: Optional[mx.array] = None,
    use_fa3: bool = False,
) -> Tuple[mx.array, Optional[mx.array]]:
    """MLX implementation of the official vanilla MHA helper.

    The xformers sparse/efficient and FA3 branches remain explicit boundaries:
    those upstream paths depend on PyTorch-only kernels rather than portable MLX
    array operations.
    """
    if is_causal:
        _raise_attention_unsupported(
            "sam3_mlx.model.model_misc.multi_head_attention_forward(is_causal=True)",
            reason="xformers",
            detail="The MLX vanilla MHA helper does not implement the official causal branch.",
        )
    if attn_type != AttentionType.Vanilla:
        _raise_attention_unsupported(
            f"sam3_mlx.model.model_misc.multi_head_attention_forward(attn_type={attn_type!r})",
            reason="xformers",
            detail="Non-vanilla attention depends on PyTorch/xformers kernels.",
        )
    if attn_sparsity != 0.0:
        _raise_attention_unsupported(
            "sam3_mlx.model.model_misc.multi_head_attention_forward(attn_sparsity)",
            reason="xformers",
            detail="Sparse attention is not ported to MLX.",
        )
    if use_fa3:
        _raise_attention_unsupported(
            "sam3_mlx.model.model_misc.multi_head_attention_forward(use_fa3=True)",
            reason="flash-attn-3",
            detail="FlashAttention 3 is non-MLX and not ported to MLX.",
        )
    fast_attention_path = (
        not need_weights and (dropout_p == 0.0 or not training) and attn_bias is None
    )

    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    if embed_dim != embed_dim_to_check:
        raise AssertionError(
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        )
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise AssertionError(f"embed_dim {embed_dim} not divisible by num_heads")

    if use_separate_proj_weight:
        if key.shape[:2] != value.shape[:2]:
            raise AssertionError(
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )
        if q_proj_weight is None or k_proj_weight is None or v_proj_weight is None:
            raise AssertionError(
                "q_proj_weight, k_proj_weight, and v_proj_weight are required"
            )
        q, k, v = _in_projection(
            query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias
        )
    else:
        if key.shape != value.shape:
            raise AssertionError(
                f"key shape {key.shape} does not match value shape {value.shape}"
            )
        if in_proj_weight is None:
            raise AssertionError(
                "use_separate_proj_weight is False but in_proj_weight is None"
            )
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    if attn_mask is not None:
        if attn_mask.ndim == 2:
            if attn_mask.shape != (tgt_len, src_len):
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {(tgt_len, src_len)}."
                )
            attn_mask = attn_mask[None, None, :, :]
        elif attn_mask.ndim == 3:
            if attn_mask.shape != (bsz * num_heads, tgt_len, src_len):
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {(bsz * num_heads, tgt_len, src_len)}."
                )
            attn_mask = attn_mask.reshape(bsz, num_heads, tgt_len, src_len)
        elif attn_mask.ndim == 4:
            if attn_mask.shape != (bsz, num_heads, tgt_len, src_len):
                raise RuntimeError(
                    f"The shape of the 4D attn_mask is {attn_mask.shape}, but should be {(bsz, num_heads, tgt_len, src_len)}."
                )
        else:
            raise RuntimeError(
                f"attn_mask's dimension {attn_mask.ndim} is not supported"
            )
    attn_mask_ready = False

    if bias_k is not None and bias_v is not None:
        if static_k is not None:
            raise AssertionError("bias cannot be added to static key.")
        if static_v is not None:
            raise AssertionError("bias cannot be added to static value.")
        k = mx.concat([k, mx.repeat(bias_k, bsz, axis=1)], axis=0)
        v = mx.concat([v, mx.repeat(bias_v, bsz, axis=1)], axis=0)
        if attn_mask is not None:
            attn_mask = _pad_last_dim(attn_mask)
        if key_padding_mask is not None:
            key_padding_mask = _pad_last_dim(key_padding_mask)
    else:
        if bias_k is not None or bias_v is not None:
            raise AssertionError("bias_k and bias_v must both be set or both be None")

    q = q.reshape(tgt_len, bsz * num_heads, head_dim).transpose(1, 0, 2)
    if static_k is None:
        k = k.reshape(k.shape[0], bsz * num_heads, head_dim).transpose(1, 0, 2)
    else:
        if static_k.shape[0] != bsz * num_heads:
            raise AssertionError(
                f"expecting static_k.shape[0] of {bsz * num_heads}, but got {static_k.shape[0]}"
            )
        if static_k.shape[2] != head_dim:
            raise AssertionError(
                f"expecting static_k.shape[2] of {head_dim}, but got {static_k.shape[2]}"
            )
        k = static_k
    if static_v is None:
        v = v.reshape(v.shape[0], bsz * num_heads, head_dim).transpose(1, 0, 2)
    else:
        if static_v.shape[0] != bsz * num_heads:
            raise AssertionError(
                f"expecting static_v.shape[0] of {bsz * num_heads}, but got {static_v.shape[0]}"
            )
        if static_v.shape[2] != head_dim:
            raise AssertionError(
                f"expecting static_v.shape[2] of {head_dim}, but got {static_v.shape[2]}"
            )
        v = static_v

    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = mx.concat([k, mx.zeros(zero_attn_shape, dtype=k.dtype)], axis=1)
        v = mx.concat([v, mx.zeros(zero_attn_shape, dtype=v.dtype)], axis=1)
        if attn_mask is not None:
            attn_mask = _pad_last_dim(attn_mask)
        if key_padding_mask is not None:
            key_padding_mask = _pad_last_dim(key_padding_mask)

    src_len = k.shape[1]

    if key_padding_mask is not None:
        if key_padding_mask.shape != (bsz, src_len):
            raise AssertionError(
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            )
        key_padding_mask = key_padding_mask.reshape(bsz, 1, 1, src_len)
        attn_mask = _merge_attention_masks(
            attn_mask,
            key_padding_mask,
            q.dtype,
            preserve_bool=fast_attention_path,
        )
        attn_mask_ready = True

    if attn_bias is not None:
        if attn_bias.shape != (bsz, num_heads, tgt_len, src_len):
            raise AssertionError(
                f"expecting attn_bias shape of {(bsz, num_heads, tgt_len, src_len)}, but got {attn_bias.shape}"
            )
        attn_mask = _merge_attention_masks(
            attn_mask,
            attn_bias,
            q.dtype,
            preserve_bool=False,
        )
        attn_mask_ready = True
    elif attn_mask is not None and not attn_mask_ready:
        attn_mask = _to_attention_mask(
            attn_mask,
            q.dtype,
            preserve_bool=fast_attention_path,
        )

    q = q.reshape(bsz, num_heads, tgt_len, head_dim)
    k = k.reshape(bsz, num_heads, src_len, head_dim)
    v = v.reshape(bsz, num_heads, src_len, head_dim)

    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask,
        dropout_p,
        training,
        need_weights=need_weights,
    )
    attn_output = attn_output.transpose(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
    attn_output = _linear(attn_output, out_proj_weight, out_proj_bias)

    if not need_weights:
        return attn_output, None

    if average_attn_weights:
        attn_output_weights = mx.sum(attn_output_weights, axis=1) / num_heads
    return attn_output, attn_output_weights


class MultiheadAttentionWrapper(nn.MultiHeadAttention):
    def __init__(self, *args, **kwargs):
        has_dims_keyword = "dims" in kwargs
        has_embed_dim_keyword = "embed_dim" in kwargs
        dims = kwargs.pop("dims", None)
        embed_dim = kwargs.pop("embed_dim", None)
        num_heads = kwargs.pop("num_heads", None)
        dropout = kwargs.pop("dropout", 0.0)
        bias = kwargs.pop("bias", True)
        add_bias_kv = kwargs.pop("add_bias_kv", False)
        add_zero_attn = kwargs.pop("add_zero_attn", False)
        kdim = kwargs.pop("kdim", None)
        vdim = kwargs.pop("vdim", None)
        batch_first = kwargs.pop("batch_first", None)
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)
        attn_type = kwargs.pop("attn_type", AttentionType.Vanilla)
        sparsity = kwargs.pop("sparsity", 0.0)
        use_act_checkpoint = kwargs.pop("use_act_checkpoint", False)
        use_fa3 = kwargs.pop("use_fa3", False)
        query_input_dims = kwargs.pop("query_input_dims", None)
        key_input_dims = kwargs.pop("key_input_dims", None)
        value_input_dims = kwargs.pop("value_input_dims", None)
        value_dims = kwargs.pop("value_dims", None)
        value_output_dims = kwargs.pop("value_output_dims", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported MultiHeadAttention kwargs: {names}.")

        if args:
            if dims is not None or embed_dim is not None:
                raise TypeError(
                    "Embedding dimension was provided both positionally and by keyword."
                )
            dims = args[0]
            args = args[1:]
        elif dims is None:
            dims = embed_dim
        elif embed_dim is not None and embed_dim != dims:
            raise TypeError("Received conflicting 'dims' and 'embed_dim' values.")
        used_embed_dim_keyword = has_embed_dim_keyword and not has_dims_keyword

        if args:
            if num_heads is not None:
                raise TypeError(
                    "num_heads was provided both positionally and by keyword."
                )
            num_heads = args[0]
            args = args[1:]
        if args:
            raise TypeError(
                f"Expected at most 2 positional arguments, got {len(args) + 2}."
            )
        if dims is None or num_heads is None:
            raise TypeError("dims/embed_dim and num_heads are required.")

        if device is not None or dtype is not None:
            _raise_attention_unsupported(
                "sam3_mlx.model.model_misc.MultiheadAttentionWrapper(device/dtype factory kwargs)",
                reason="port-gap",
                detail="MLX attention parameters are created in the active MLX runtime dtype/device.",
            )
        if add_bias_kv:
            _raise_attention_unsupported(
                "sam3_mlx.model.model_misc.MultiheadAttentionWrapper(add_bias_kv=True)",
                reason="port-gap",
                detail="The MLX wrapper does not implement extra learned key/value bias rows.",
            )
        if value_dims is not None and value_dims != dims:
            _raise_attention_unsupported(
                "sam3_mlx.model.model_misc.MultiheadAttentionWrapper(value_dims)",
                reason="port-gap",
                detail="The official SAM3 MHA path keeps value and embedding dimensions equal.",
            )
        if value_output_dims is not None and value_output_dims != dims:
            _raise_attention_unsupported(
                "sam3_mlx.model.model_misc.MultiheadAttentionWrapper(value_output_dims)",
                reason="port-gap",
                detail="The official SAM3 MHA path keeps output and embedding dimensions equal.",
            )
        if query_input_dims is not None and query_input_dims != dims:
            _raise_attention_unsupported(
                "sam3_mlx.model.model_misc.MultiheadAttentionWrapper(query_input_dims)",
                reason="port-gap",
                detail="The official SAM3 MHA path expects query input dimensions to equal embed_dim.",
            )

        key_input_dims = key_input_dims if key_input_dims is not None else kdim
        value_input_dims = value_input_dims if value_input_dims is not None else vdim
        if batch_first is None:
            batch_first = not used_embed_dim_keyword

        super().__init__(
            dims,
            num_heads,
            query_input_dims=query_input_dims,
            key_input_dims=key_input_dims,
            value_input_dims=value_input_dims,
            value_dims=dims,
            value_output_dims=dims,
            bias=bias,
        )
        self.embed_dim = dims
        self.kdim = key_input_dims if key_input_dims is not None else dims
        self.vdim = value_input_dims if value_input_dims is not None else dims
        self._qkv_same_embed_dim = self.kdim == dims and self.vdim == dims
        self.dropout = dropout
        self.batch_first = bool(batch_first)
        self.head_dim = dims // num_heads
        if self.head_dim * num_heads != dims:
            raise AssertionError("embed_dim must be divisible by num_heads")
        self.use_act_checkpoint = use_act_checkpoint
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.attn_type = attn_type
        self.sparsity = sparsity
        self.use_fa3 = use_fa3

    @staticmethod
    def _to_additive_mask(mask):
        return _to_additive_attention_mask(mask, mx.float32)

    def _reshape_attn_mask(self, mask, batch_size):
        if mask is None:
            return None
        if mask.ndim == 2:
            return mask[None, None, :, :]
        if mask.ndim == 3:
            tgt_len, src_len = mask.shape[-2], mask.shape[-1]
            if mask.shape[0] == batch_size * self.num_heads:
                return mask.reshape(batch_size, self.num_heads, tgt_len, src_len)
            if mask.shape[0] == batch_size:
                return mask[:, None, :, :]
        return mask

    def _normalize_attn_mask(
        self, mask, batch_size, *, preserve_bool=False, dtype=mx.float32
    ):
        mask = _to_attention_mask(mask, dtype, preserve_bool=preserve_bool)
        return self._reshape_attn_mask(mask, batch_size)

    def _combine_masks(
        self,
        attn_mask,
        key_padding_mask,
        base_mask,
        queries,
        *,
        preserve_bool=False,
        dtype=mx.float32,
    ):
        masks = [
            self._reshape_attn_mask(base_mask, queries.shape[0]),
            self._reshape_attn_mask(attn_mask, queries.shape[0]),
        ]
        if key_padding_mask is not None:
            masks.append(key_padding_mask[:, None, None, :])
        masks = [mask for mask in masks if mask is not None]
        if not masks:
            return None

        if preserve_bool and all(_is_bool_like_mask(mask) for mask in masks):
            block_mask = _to_block_bool_mask(masks[0])
            for mask in masks[1:]:
                block_mask = mx.logical_or(block_mask, _to_block_bool_mask(mask))
            return mx.logical_not(block_mask)

        final_mask = _to_additive_attention_mask(masks[0], dtype)
        for mask in masks[1:]:
            final_mask = final_mask + _to_additive_attention_mask(mask, dtype)
        return final_mask

    @staticmethod
    def _pop_alias(kwargs, canonical, *aliases):
        canonical_value = kwargs.pop(canonical, None)
        chosen_value = canonical_value
        chosen_name = canonical if canonical_value is not None else None
        for alias in aliases:
            alias_value = kwargs.pop(alias, None)
            if alias_value is not None:
                if chosen_value is not None:
                    raise TypeError(f"Received both {chosen_name!r} and {alias!r}.")
                chosen_value = alias_value
                chosen_name = alias
        return chosen_value

    def _combined_in_proj_bias(self):
        biases = (self.query_proj.bias, self.key_proj.bias, self.value_proj.bias)
        if all(bias is None for bias in biases):
            return None
        if any(bias is None for bias in biases):
            raise AssertionError("q/k/v projection biases must be all set or all None.")
        return mx.concat(biases, axis=0)

    def __call__(self, *args, **kwargs):
        queries = self._pop_alias(kwargs, "queries", "query", "q")
        keys = self._pop_alias(kwargs, "keys", "key", "k")
        values = self._pop_alias(kwargs, "values", "value", "v")

        if len(args) > 3:
            raise TypeError(f"Expected at most 3 positional arrays, got {len(args)}.")
        if len(args) > 0:
            if queries is not None:
                raise TypeError(
                    "queries was provided both positionally and by keyword."
                )
            queries = args[0]
        if len(args) > 1:
            if keys is not None:
                raise TypeError("keys was provided both positionally and by keyword.")
            keys = args[1]
        if len(args) > 2:
            if values is not None:
                raise TypeError("values was provided both positionally and by keyword.")
            values = args[2]
        if queries is None or keys is None or values is None:
            raise TypeError("queries, keys, and values are required.")

        attn_mask = kwargs.pop("attn_mask", None)
        key_padding_mask = kwargs.pop("key_padding_mask", None)
        base_mask = kwargs.pop("mask", None)
        need_weights = kwargs.pop("need_weights", False)
        average_attn_weights = kwargs.pop("average_attn_weights", True)
        attn_bias = kwargs.pop("attn_bias", None)
        is_causal = kwargs.pop("is_causal", False)
        attn_type = kwargs.pop("attn_type", self.attn_type)
        attn_sparsity = kwargs.pop("attn_sparsity", self.sparsity)
        use_fa3 = kwargs.pop("use_fa3", self.use_fa3)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported MultiHeadAttention kwargs: {names}.")
        if is_causal:
            _raise_attention_unsupported(
                "sam3_mlx.model.model_misc.MultiheadAttentionWrapper(is_causal=True)",
                reason="xformers",
                detail="The MLX wrapper does not implement the official causal branch.",
            )
        if attn_type != AttentionType.Vanilla:
            _raise_attention_unsupported(
                f"sam3_mlx.model.model_misc.MultiheadAttentionWrapper(attn_type={attn_type!r})",
                reason="xformers",
                detail="Non-vanilla attention depends on PyTorch/xformers kernels.",
            )
        if attn_sparsity != 0.0:
            _raise_attention_unsupported(
                "sam3_mlx.model.model_misc.MultiheadAttentionWrapper(attn_sparsity)",
                reason="xformers",
                detail="Sparse attention is not ported to MLX.",
            )
        if use_fa3:
            _raise_attention_unsupported(
                "sam3_mlx.model.model_misc.MultiheadAttentionWrapper(use_fa3=True)",
                reason="flash-attn-3",
                detail="FlashAttention 3 is non-MLX and not ported to MLX.",
            )

        is_batched = queries.ndim == 3
        if self.batch_first and is_batched:
            queries, keys, values = [
                array.transpose(1, 0, 2) for array in (queries, keys, values)
            ]

        if base_mask is not None:
            attn_mask = _merge_attention_masks(
                attn_mask,
                base_mask,
                queries.dtype,
                preserve_bool=False,
            )

        attn_output, attn_output_weights = multi_head_attention_forward(
            queries,
            keys,
            values,
            self.embed_dim,
            self.num_heads,
            in_proj_weight=None,
            in_proj_bias=self._combined_in_proj_bias(),
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.query_proj.weight,
            k_proj_weight=self.key_proj.weight,
            v_proj_weight=self.value_proj.weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            attn_type=attn_type,
            attn_sparsity=attn_sparsity,
            attn_bias=attn_bias,
            use_fa3=use_fa3,
        )
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0, 2)
        if need_weights:
            return attn_output, attn_output_weights
        return attn_output


class MultiheadAttention(MultiheadAttentionWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _reset_parameters(self):
        return None

    def forward(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class DotProductScoring(nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp  # an optional MLP projection for prompt
        self.prompt_proj = nn.Linear(d_model, d_proj)
        self.hs_proj = nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / math.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
        is_valid = (~prompt_mask).astype(mx.float32).transpose(1, 0)[..., None]
        # num_valid has shape (bs, 1)
        num_valid = mx.clip(mx.sum(is_valid, axis=0), 1.0, None)
        # mean pool over all the valid tokens -- pooled_prompt has shape (bs, proj_dim)
        pooled_prompt = mx.sum(prompt * is_valid, axis=0) / num_valid
        return pooled_prompt

    def forward(self, hs, prompt, prompt_mask):
        # hs has shape (num_layer, bs, num_query, d_model)
        # prompt has shape (seq, bs, d_model)
        # prompt_mask has shape (bs, seq), where True/1 is padding and 0 is a valid token
        assert hs.ndim == 4 and prompt.ndim == 3 and prompt_mask.ndim == 2

        # apply MLP on prompt if specified
        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        # first, get the mean-pooled version of the prompt
        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)

        # then, project pooled_prompt and hs to d_proj dimensions
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # (bs, d_proj)
        proj_hs = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # finally, get dot-product scores of shape (num_layer, bs, num_query, 1)
        scores = mx.matmul(proj_hs, proj_pooled_prompt[..., None])
        scores *= self.scale

        # clamp scores to a max value to avoid numerical issues in loss or matcher
        if self.clamp_logits:
            scores = mx.clip(scores, -self.clamp_max_val, self.clamp_max_val)

        return scores

    def __call__(self, hs, prompt, prompt_mask):
        return self.forward(hs, prompt, prompt_mask)


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    # Stochastic-depth behavior follows timm DropPath semantics.
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff-dim arrays, not just 2D ConvNets
    random_tensor = mx.random.bernoulli(p=keep_prob, shape=shape)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def __call__(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",
        pos_enc_at_input_dec=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec

        # for two stage
        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        def _init_fn(path, params):
            if params.ndim > 1:
                if (
                    "box_embed" not in path
                    and "query_embed" not in path
                    and "reference_points" not in path
                ):
                    return nn.init.glorot_uniform()(params, 1.0)
            return params

        self.update(tree_map_with_path(_init_fn, self.parameters()))


class MLP(nn.Module):
    """Multi-layer perceptron (also called FFN)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        ]
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # whether to add the output as a residual connection to the input
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(self.act(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x

    def __call__(self, x):
        return self.forward(x)


class Mlp(nn.Module):
    # ViT MLP block shape follows timm Mlp semantics.
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Optional[Type[nn.Module]] = None,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
        use_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias if isinstance(bias, tuple) else (bias, bias)
        drop_probs = drop if isinstance(drop, tuple) else (drop, drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Identity() if drop_probs[0] == 0 else nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Identity() if drop_probs[1] == 0 else nn.Dropout(drop_probs[1])

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, mx.array] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = init_values * mx.ones(dim)

    def forward(self, x: mx.array) -> mx.array:
        # Note: MLX arrays are immutable, so "inplace" operations still create new arrays.
        # The inplace flag is kept for API compatibility with PyTorch but doesn't change behavior.
        # Both paths return a new array.
        return x * self.gamma

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.eps = eps

    def forward(self, x: mx.array) -> mx.array:
        u = mx.mean(x, axis=1, keepdims=True)
        s = mx.mean((x - u) ** 2, axis=1, keepdims=True)
        x = (x - u) / mx.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)


def get_clones(module, N):
    if isinstance(module, nn.Module):
        return [deepcopy(module) for _ in range(N)]
    if callable(module):
        return [module() for _ in range(N)]
    raise TypeError(
        "get_clones expects an MLX module instance or zero-argument factory."
    )


def get_clones_seq(module, N):
    return nn.Sequential(*get_clones(module, N))


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.relu
    if activation == "gelu":
        return nn.gelu
    if activation == "glu":
        return nn.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class _GLUModule(nn.Module):
    def __call__(self, x):
        return nn.glu(x)


def get_activation_module(activation):
    """Return an activation module class given a string."""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return getattr(nn, "GLU", _GLUModule)
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = mx.sum(~mask[:, :, 0], 1)
    valid_W = mx.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.astype(mx.float32) / H
    valid_ratio_w = valid_W.astype(mx.float32) / W
    valid_ratio = mx.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(pos_array, num_feats=256):
    assert num_feats % 2 == 0
    num_feats = num_feats // 2

    scale = 2 * math.pi
    dim_t = mx.arange(num_feats, dtype=mx.float32)
    dim_t = 10000 ** (2 * mx.floor(mx.divide(dim_t, 2)) / num_feats)
    x_embed = pos_array[:, :, 0] * scale
    y_embed = pos_array[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = mx.stack(
        (mx.sin(pos_x[:, :, 0::2]), mx.cos(pos_x[:, :, 1::2])), axis=3
    ).flatten(2)
    pos_y = mx.stack(
        (mx.sin(pos_y[:, :, 0::2]), mx.cos(pos_y[:, :, 1::2])), axis=3
    ).flatten(2)
    if pos_array.shape[-1] == 2:
        pos = mx.concat([pos_y, pos_x], axis=2)
    elif pos_array.shape[-1] == 4:
        w_embed = pos_array[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = mx.stack(
            (mx.sin(pos_w[:, :, 0::2]), mx.cos(pos_w[:, :, 1::2])), axis=3
        ).flatten(2)

        h_embed = pos_array[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = mx.stack(
            (mx.sin(pos_h[:, :, 0::2]), mx.cos(pos_h[:, :, 1::2])), axis=3
        ).flatten(2)

        pos = mx.concat((pos_y, pos_x, pos_w, pos_h), axis=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_array.shape[-1]))
    return pos


class SAM3Output(list):
    """Official-style SAM3 output container with selectable iteration modes."""

    class IterMode(Enum):
        ALL_STEPS_PER_STAGE = auto()
        LAST_STEP_PER_STAGE = auto()
        FLATTENED = auto()

    def __init__(
        self,
        output: Optional[List[List[Dict]]] = None,
        iter_mode: IterMode = IterMode.ALL_STEPS_PER_STAGE,
        loss_stages: Optional[List[int]] = None,
    ):
        super().__init__()
        if output is not None:
            assert (
                isinstance(output, list)
                and len(output) > 0
                and isinstance(output[0], list)
            ), "Expected output to be a list of lists"
            self.output = output
        else:
            self.output = []
        assert isinstance(iter_mode, SAM3Output.IterMode), (
            f"iter_mode should be of enum type 'SAM3Output.IterMode'. Got {type(iter_mode)}"
        )
        self.iter_mode = iter_mode
        self.loss_stages = loss_stages
        self_ref = weakref.ref(self)
        self._mode2iter = {
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE: lambda: iter(self_ref().output),
            SAM3Output.IterMode.LAST_STEP_PER_STAGE: lambda: (
                inner_list[-1] for inner_list in self_ref().output
            ),
            SAM3Output.IterMode.FLATTENED: lambda: (
                element for inner_list in self_ref().output for element in inner_list
            ),
        }

    def __iter__(self) -> Iterator:
        return self._mode2iter[self.iter_mode]()

    def __getitem__(self, index):
        assert isinstance(index, int), f"index should be an integer. Got {type(index)}"
        if self.iter_mode == SAM3Output.IterMode.ALL_STEPS_PER_STAGE:
            return self.output[index]
        if self.iter_mode == SAM3Output.IterMode.LAST_STEP_PER_STAGE:
            return self.output[index][-1]
        if index == -1:
            return self.output[-1][-1]
        flattened_output = sum(self.output, [])
        return flattened_output[index]

    class _IterationMode(AbstractContextManager):
        def __init__(
            self, model_output: "SAM3Output", iter_mode: "SAM3Output.IterMode"
        ):
            self._model_output = model_output
            self._orig_iter_mode = model_output.iter_mode
            self._new_iter_mode = iter_mode

        def __enter__(self) -> "SAM3Output":
            self._model_output.iter_mode = self._new_iter_mode
            return self._model_output

        def __exit__(self, exc_type, exc_value, traceback):
            self._model_output.iter_mode = self._orig_iter_mode
            return super().__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def iteration_mode(
        model_output: "SAM3Output", iter_mode: "SAM3Output.IterMode"
    ) -> "SAM3Output._IterationMode":
        return SAM3Output._IterationMode(model_output=model_output, iter_mode=iter_mode)

    def append(self, item: list):
        assert isinstance(item, list), (
            f"Only list items are supported. Got {type(item)}"
        )
        self.output.append(item)

    def __repr__(self):
        return self.output.__repr__()

    def __len__(self):
        if self.iter_mode in (
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
            SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        ):
            return len(self.output)
        flattened_output = sum(self.output, [])
        return len(flattened_output)
