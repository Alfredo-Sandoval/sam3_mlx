from __future__ import annotations

import math
from typing import TYPE_CHECKING, NoReturn, TypeAlias, TypeGuard, cast

import numpy as np
import numpy.typing as npt

from sam3_mlx._unsupported import UPSTREAM_COMMIT, raise_unsupported

if TYPE_CHECKING:
    import mlx.core as mx


NumpyArray = npt.NDArray[np.generic]
NumericArray = npt.NDArray[np.number]
if TYPE_CHECKING:
    AttentionArray: TypeAlias = NumpyArray | mx.array
else:
    AttentionArray: TypeAlias = object

_UNSUPPORTED_FA3_MESSAGE = (
    "Official SAM3 FlashAttention 3 custom-op behavior is not implemented in "
    "sam3_mlx. The official fa3.py at commit "
    f"{UPSTREAM_COMMIT} depends on Torch custom ops and CUDA-specific "
    "FlashAttention 3 kernels."
)


def _raise_fa3_unsupported(feature: str) -> NoReturn:
    raise_unsupported(
        feature,
        reason="flash-attn-3",
        detail=_UNSUPPORTED_FA3_MESSAGE,
    )


def _is_mlx_array(value: object) -> TypeGuard["mx.array"]:
    return type(value).__module__.startswith("mlx.")


def _as_mlx_array(value: object) -> "mx.array":
    import mlx.core as mx

    if _is_mlx_array(value):
        return value
    return mx.array(cast(NumpyArray, value))


def flash_attn_func_op(q: object, k: object, v: object) -> NoReturn:
    del q, k, v
    _raise_fa3_unsupported("flash_attn_func_op")


def _flash_attn_mlx(q: object, k: object, v: object) -> "mx.array":
    import mlx.core as mx

    q_mx = _as_mlx_array(q)
    k_mx = _as_mlx_array(k)
    v_mx = _as_mlx_array(v)

    if q_mx.ndim != 4 or k_mx.ndim != 4 or v_mx.ndim != 4:
        raise ValueError("flash_attn_func expects q, k, v with shape (B, S, H, D).")
    if q_mx.shape[0] != k_mx.shape[0] or q_mx.shape[0] != v_mx.shape[0]:
        raise ValueError("q, k, and v batch dimensions must match.")
    if q_mx.shape[2] != k_mx.shape[2] or q_mx.shape[2] != v_mx.shape[2]:
        raise ValueError("q, k, and v head dimensions must match.")
    if q_mx.shape[3] != k_mx.shape[3]:
        raise ValueError("q and k head dimensions must match.")
    if k_mx.shape[1] != v_mx.shape[1]:
        raise ValueError("k and v sequence dimensions must match.")

    q_heads = mx.transpose(q_mx, (0, 2, 1, 3))
    k_heads = mx.transpose(k_mx, (0, 2, 1, 3))
    v_heads = mx.transpose(v_mx, (0, 2, 1, 3))
    attention = mx.fast.scaled_dot_product_attention(
        q_heads,
        k_heads,
        v_heads,
        scale=q_mx.shape[-1] ** -0.5,
    )
    return mx.transpose(attention, (0, 2, 1, 3)).astype(q_mx.dtype)


def _flash_attn_numpy(q: object, k: object, v: object) -> NumpyArray:
    q_np = cast(NumericArray, np.asarray(q))
    k_np = cast(NumericArray, np.asarray(k))
    v_np = cast(NumericArray, np.asarray(v))
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
    return cast(NumpyArray, out.astype(q_np.dtype, copy=False))


def flash_attn_func(q: object, k: object, v: object) -> AttentionArray:
    if _is_mlx_array(q) or _is_mlx_array(k) or _is_mlx_array(v):
        return _flash_attn_mlx(q, k, v)
    return _flash_attn_numpy(q, k, v)


def _(q: object, k: object, v: object, **kwargs: object) -> NoReturn:
    del q, k, v, kwargs
    raise_unsupported(
        "flash_attn_func_op.register_fake",
        reason="torch-autograd",
        detail=_UNSUPPORTED_FA3_MESSAGE,
    )
