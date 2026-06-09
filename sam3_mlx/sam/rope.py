from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from sam3_mlx._unsupported import raise_unsupported


def _raise_rope_device_unsupported(feature: str, device) -> None:
    raise_unsupported(
        f"{feature}(device={device!r})",
        reason="unsupported-device",
        detail="RoPE helpers only support the explicit MLX runtime.",
        alternative="device='mlx'",
    )


def init_t_xy(end_x: int, end_y: int, scale: float = 1.0, offset: int = 0, device=None):
    if device not in (None, "mlx"):
        _raise_rope_device_unsupported("sam3_mlx.sam.rope.init_t_xy", device)
    t = mx.arange(end_x * end_y, dtype=mx.float32)
    t_x = mx.remainder(t, end_x)
    t_y = mx.floor(t / end_x)
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device=None,
):
    if device not in (None, "mlx"):
        _raise_rope_device_unsupported("sam3_mlx.sam.rope.compute_axial_cis", device)
    freq_idx = mx.arange(0, dim, 4, dtype=mx.float32)[: dim // 4]
    freqs_x = 1.0 / (theta ** (freq_idx / dim))
    freqs_y = 1.0 / (theta ** (freq_idx / dim))

    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = t_x[:, None] * freqs_x[None, :]
    freqs_y = t_y[:, None] * freqs_y[None, :]
    freqs = mx.concat([freqs_x, freqs_y], axis=-1)
    return mx.stack([mx.cos(freqs), mx.sin(freqs)], axis=-1)


def reshape_for_broadcast(freqs: mx.array, x: mx.array):
    if freqs.shape != (x.shape[-2], x.shape[-1]):
        raise AssertionError(
            f"freqs shape {freqs.shape} does not match {x.shape[-2:]}."
        )
    shape = [1] * x.ndim
    shape[-2], shape[-1] = freqs.shape
    return freqs.reshape(shape)


def complex_mult(x_real, x_imag, freqs_real, freqs_imag):
    real = x_real * freqs_real - x_imag * freqs_imag
    imag = x_real * freqs_imag + x_imag * freqs_real
    return mx.stack([real, imag], axis=-1)


def _real_imag(freqs_cis: mx.array) -> tuple[mx.array, mx.array]:
    if freqs_cis.shape[-1] != 2:
        raise ValueError("MLX RoPE frequencies must have trailing real/imag dimension.")
    return freqs_cis[..., 0], freqs_cis[..., 1]


def _apply_real_rotary(x: mx.array, freqs_real: mx.array, freqs_imag: mx.array):
    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    real = pairs[..., 0]
    imag = pairs[..., 1]
    freqs_real = reshape_for_broadcast(freqs_real, real)
    freqs_imag = reshape_for_broadcast(freqs_imag, imag)
    return (
        complex_mult(real, imag, freqs_real, freqs_imag)
        .reshape(x.shape)
        .astype(x.dtype)
    )


def _tile_freqs_for_repeated_keys(freqs: mx.array, repeat: int) -> mx.array:
    if repeat <= 1:
        return freqs
    return mx.tile(freqs, (repeat, 1))


def apply_rotary_enc(
    xq: mx.array,
    xk: mx.array,
    freqs_cis: mx.array,
    repeat_freqs_k: bool = False,
):
    freqs_real, freqs_imag = _real_imag(freqs_cis)
    xq_out = _apply_real_rotary(xq, freqs_real, freqs_imag)
    if xk.shape[-2] == 0:
        return xq_out, xk
    if repeat_freqs_k:
        repeat = xk.shape[-2] // xq.shape[-2]
        freqs_real = _tile_freqs_for_repeated_keys(freqs_real, repeat)
        freqs_imag = _tile_freqs_for_repeated_keys(freqs_imag, repeat)
    xk_out = _apply_real_rotary(xk, freqs_real, freqs_imag)
    return xq_out, xk_out


def apply_rotary_enc_real(
    xq: mx.array,
    xk: mx.array,
    freqs_cis_real: mx.array,
    freqs_cis_imag: mx.array,
    repeat_freqs_k: bool = False,
):
    if xk is None or xk.shape[-2] == 0:
        raise AssertionError("apply_rotary_enc_real requires non-empty keys.")
    xq_out = _apply_real_rotary(xq, freqs_cis_real, freqs_cis_imag)
    if repeat_freqs_k:
        repeat = xk.shape[-2] // xq.shape[-2]
        freqs_cis_real = _tile_freqs_for_repeated_keys(freqs_cis_real, repeat)
        freqs_cis_imag = _tile_freqs_for_repeated_keys(freqs_cis_imag, repeat)
    xk_out = _apply_real_rotary(xk, freqs_cis_real, freqs_cis_imag)
    return xq_out, xk_out


def broadcat(tensors, dim=-1):
    shape = []
    for axes in zip(*(tensor.shape for tensor in tensors), strict=True):
        shape.append(max(axes))
    return mx.concat([mx.broadcast_to(tensor, shape) for tensor in tensors], axis=dim)


def rotate_half(x: mx.array):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    return mx.stack([-x2, x1], axis=-1).reshape(*x.shape[:-2], -1)


class VisionRotaryEmbeddingVE(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        pt_seq_len: Optional[int] = None,
        theta: float = 10000.0,
        offset: int = 1,
    ):
        super().__init__()
        freqs = 1.0 / (
            theta ** (mx.arange(0, dim, 2, dtype=mx.float32)[: dim // 2] / dim)
        )
        scale = 1.0 if pt_seq_len is None else pt_seq_len / seq_len
        t = mx.arange(seq_len, dtype=mx.float32) * scale + offset
        freqs = t[:, None] * freqs[None, :]
        freqs = mx.repeat(freqs, 2, axis=-1)
        freqs = broadcat((freqs[None, :, :], freqs[:, None, :]), dim=-1)
        freqs = freqs.reshape(-1, freqs.shape[-1])
        self.freqs_cos = mx.cos(freqs)
        self.freqs_sin = mx.sin(freqs)

    def __call__(self, t: mx.array):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin
