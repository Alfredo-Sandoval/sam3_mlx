# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX/NumPy Euclidean distance transform helpers.

The upstream SAM3 module provides a Torch/Triton kernel. This port keeps
the public ``edt_triton`` name for call-site compatibility, but implements the
same binary EDT contract with NumPy and returns an MLX array for MLX inputs.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.mlx_runtime import to_numpy


_INF = 1.0e18


def _is_mlx_array(value: Any) -> bool:
    return type(value).__module__.startswith("mlx.")


def _to_host_edt_input(value: Any) -> np.ndarray:
    """Synchronize and export masks at the explicit CPU EDT boundary."""

    if isinstance(value, np.ndarray):
        return value
    return to_numpy(value)


def _edt_1d_squared(f: np.ndarray) -> np.ndarray:
    """Felzenszwalb-Huttenlocher squared distance transform for one row."""

    n = f.shape[0]
    v = np.zeros(n, dtype=np.int64)
    z = np.empty(n + 1, dtype=np.float64)
    d = np.empty(n, dtype=np.float64)

    k = 0
    v[0] = 0
    z[0] = -_INF
    z[1] = _INF

    for q in range(1, n):
        q_sq = q * q
        while True:
            r = v[k]
            s = ((f[q] + q_sq) - (f[r] + r * r)) / (2 * q - 2 * r)
            if s > z[k] or k == 0:
                break
            k -= 1
        if s <= z[k]:
            s = z[k]
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = _INF

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        r = v[k]
        d[q] = (q - r) * (q - r) + f[r]
    return d


def edt_numpy(data: Any) -> np.ndarray:
    """
    Compute the Euclidean distance transform of a batch of binary images.

    Args:
        data: Boolean or numeric array with shape ``(B, H, W)``. Nonzero pixels
            are treated as foreground and receive distance to the nearest zero.

    Returns:
        ``float32`` NumPy array with shape ``(B, H, W)``.
    """

    data_np = _to_host_edt_input(data)
    if data_np.ndim != 3:
        raise AssertionError("edt_numpy expects an array with shape (B, H, W).")

    bsz, height, width = data_np.shape
    work = np.where(data_np.astype(bool), _INF, 0.0).astype(np.float64, copy=False)
    row_pass = np.empty_like(work)
    col_pass = np.empty_like(work)

    for batch_idx in range(bsz):
        for y in range(height):
            row_pass[batch_idx, y, :] = _edt_1d_squared(work[batch_idx, y, :])
        for x in range(width):
            col_pass[batch_idx, :, x] = _edt_1d_squared(row_pass[batch_idx, :, x])

    return np.sqrt(col_pass, dtype=np.float64).astype(np.float32, copy=False)


def edt_mlx(data: Any) -> mx.array:
    """Compute binary EDT at the named CPU boundary and return an MLX array."""

    return mx.array(edt_numpy(data), dtype=mx.float32)


def edt_triton(data: Any):
    """
    Compatibility wrapper for upstream ``edt_triton``.

    The official implementation requires Torch + Triton. The MLX port is
    intentionally backend-explicit: NumPy inputs produce NumPy outputs, while MLX
    inputs produce MLX outputs after an explicit host-side EDT calculation.
    """

    if _is_mlx_array(data):
        return edt_mlx(data)
    return edt_numpy(data)


def edt_kernel(*args, **kwargs):
    del args, kwargs
    raise_unsupported(
        "sam3_mlx.model.edt.edt_kernel",
        reason="triton-kernel",
        detail="sam3_mlx.model.edt does not port the upstream Triton kernel.",
        alternative="sam3_mlx.model.edt.edt_triton",
    )


__all__ = [
    "edt_kernel",
    "edt_mlx",
    "edt_numpy",
    "edt_triton",
]
