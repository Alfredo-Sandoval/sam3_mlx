from __future__ import annotations

import math

import numpy as np

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported


_UNSUPPORTED_FA3_MESSAGE = (
    "Official SAM3 FlashAttention 3 custom-op behavior is not implemented in "
    "sam3_mlx. The official fa3.py at commit "
    f"{UPSTREAM_COMMIT} depends on Torch custom ops and CUDA-specific "
    "FlashAttention 3 kernels."
)


def _raise_fa3_unsupported(feature: str) -> None:
    raise_unsupported(
        feature,
        reason="flash-attn-3",
        detail=_UNSUPPORTED_FA3_MESSAGE,
    )


def _is_mlx_array(value) -> bool:
    return type(value).__module__.startswith("mlx.")


def flash_attn_func_op(q, k, v):
    _raise_fa3_unsupported("flash_attn_func_op")


def _flash_attn_mlx(q, k, v):
    import mlx.core as mx

    q = q if _is_mlx_array(q) else mx.array(q)
    k = k if _is_mlx_array(k) else mx.array(k)
    v = v if _is_mlx_array(v) else mx.array(v)

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("flash_attn_func expects q, k, v with shape (B, S, H, D).")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v batch dimensions must match.")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("q, k, and v head dimensions must match.")
    if q.shape[3] != k.shape[3]:
        raise ValueError("q and k head dimensions must match.")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v sequence dimensions must match.")

    q_heads = q.transpose(0, 2, 1, 3)
    k_heads = k.transpose(0, 2, 1, 3)
    v_heads = v.transpose(0, 2, 1, 3)
    out = mx.fast.scaled_dot_product_attention(
        q_heads,
        k_heads,
        v_heads,
        scale=q.shape[-1] ** -0.5,
    ).transpose(0, 2, 1, 3)
    return out.astype(q.dtype)


def _flash_attn_numpy(q, k, v):
    q_np = np.asarray(q)
    k_np = np.asarray(k)
    v_np = np.asarray(v)
    if q_np.ndim != 4 or k_np.ndim != 4 or v_np.ndim != 4:
        raise ValueError("flash_attn_func expects q, k, v with shape (B, S, H, D).")
    if q_np.shape[0] != k_np.shape[0] or q_np.shape[0] != v_np.shape[0]:
        raise ValueError("q, k, and v batch dimensions must match.")
    if q_np.shape[2] != k_np.shape[2] or q_np.shape[2] != v_np.shape[2]:
        raise ValueError("q, k, and v head dimensions must match.")
    if q_np.shape[3] != k_np.shape[3]:
        raise ValueError("q and k head dimensions must match.")
    if k_np.shape[1] != v_np.shape[1]:
        raise ValueError("k and v sequence dimensions must match.")

    q_heads = np.transpose(q_np.astype(np.float32, copy=False), (0, 2, 1, 3))
    k_heads = np.transpose(k_np.astype(np.float32, copy=False), (0, 2, 3, 1))
    v_heads = np.transpose(v_np.astype(np.float32, copy=False), (0, 2, 1, 3))
    scores = q_heads @ k_heads / math.sqrt(q_np.shape[-1])
    scores = scores - scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out = np.transpose(probs @ v_heads, (0, 2, 1, 3))
    return out.astype(q_np.dtype, copy=False)


def flash_attn_func(q, k, v):
    if _is_mlx_array(q) or _is_mlx_array(k) or _is_mlx_array(v):
        return _flash_attn_mlx(q, k, v)
    return _flash_attn_numpy(q, k, v)


def _(q, k, v, **kwargs):
    raise_unsupported(
        "flash_attn_func_op.register_fake",
        reason="torch-autograd",
        detail=_UNSUPPORTED_FA3_MESSAGE,
    )
