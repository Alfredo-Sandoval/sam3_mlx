"""Typed adapters for dynamic MLX utilities used by tests."""

from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
from mlx import nn
from mlx import utils as mlx_utils


class _SaveSafetensors(Protocol):
    def __call__(
        self,
        file: str | Path,
        arrays: dict[str, mx.array],
        metadata: dict[str, str] | None = None,
    ) -> None: ...


class _TreeFlatten(Protocol):
    def __call__(
        self,
        tree: object,
        prefix: str = "",
        is_leaf: object | None = None,
        destination: dict[str, object] | None = None,
    ) -> object: ...


_save_safetensors = cast(_SaveSafetensors, getattr(mx, "save_safetensors"))
_tree_flatten = cast(_TreeFlatten, getattr(mlx_utils, "tree_flatten"))


def save_safetensors(path: Path, arrays: dict[str, mx.array]) -> None:
    _save_safetensors(path, arrays)


def flat_parameters(model: nn.Module) -> dict[str, mx.array]:
    payload = _tree_flatten(cast(object, model.parameters()), destination={})
    if not isinstance(payload, dict):
        raise AssertionError("MLX tree_flatten must return a mapping destination.")
    raw_payload = cast(dict[object, object], payload)
    flattened: dict[str, mx.array] = {}
    for key, value in raw_payload.items():
        if not isinstance(key, str) or not isinstance(value, mx.array):
            raise AssertionError("MLX parameters must map strings to arrays.")
        flattened[key] = value
    return flattened
