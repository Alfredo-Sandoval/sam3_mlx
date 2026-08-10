"""Shared MLX runtime boundaries for synchronization and host export."""

from __future__ import annotations

import importlib.metadata
import platform
from dataclasses import dataclass
from typing import Any, Protocol, cast

import mlx.core as mx
import numpy as np
from numpy.typing import DTypeLike, NDArray

from sam3_mlx._unsupported import raise_unsupported


@dataclass(frozen=True)
class MlxRuntimeInfo:
    python: str
    mlx_version: str
    default_device: str
    metal_available: bool
    platform: str


class _MlxEval(Protocol):
    def __call__(self, *values: mx.array) -> None: ...


_mlx_eval = cast(_MlxEval, getattr(mx, "eval"))


def runtime_info() -> MlxRuntimeInfo:
    """Return stable runtime metadata for smoke and parity artifacts."""

    return MlxRuntimeInfo(
        python=platform.python_version(),
        mlx_version=importlib.metadata.version("mlx"),
        default_device=str(mx.default_device()),
        metal_available=bool(mx.metal.is_available()),
        platform=platform.platform(),
    )


def require_mlx_runtime(feature: str = "sam3_mlx MLX runtime") -> MlxRuntimeInfo:
    """Fail fast when the expected Apple MLX runtime is unavailable."""

    info = runtime_info()
    if not info.metal_available:
        raise_unsupported(
            feature,
            reason="unsupported-device",
            detail="MLX Metal support is required for sam3_mlx runtime execution.",
            alternative="run on macOS Apple Silicon with MLX Metal support",
        )
    return info


def evaluate_boundary(*values: mx.array) -> None:
    """Synchronize intentionally at a named MLX runtime boundary."""

    if values:
        _mlx_eval(*values)


def to_numpy(
    value: object,
    *,
    dtype: DTypeLike | None = None,
    copy: bool = False,
) -> NDArray[Any]:
    """Synchronize MLX arrays and convert to NumPy at an explicit host boundary."""

    array: NDArray[Any]
    if isinstance(value, np.ndarray):
        array = cast(NDArray[Any], value)
    else:
        if isinstance(value, mx.array):
            evaluate_boundary(value)
        array = np.asarray(value)
    if dtype is not None:
        return array.astype(dtype, copy=copy)
    if copy:
        return array.copy()
    return array


def shape_dtype(value: mx.array) -> dict[str, object]:
    """Return synchronized shape/dtype metadata for smoke reports."""

    evaluate_boundary(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def finite_all(value: object) -> bool:
    """Return whether a value is finite after explicit host export."""

    return bool(np.isfinite(to_numpy(value)).all())
