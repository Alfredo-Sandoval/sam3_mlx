# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of ``sam3.model.utils.misc`` from the official SAM3 tree."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, Protocol, runtime_checkable

from sam3_mlx._unsupported import raise_unsupported


def _is_named_tuple(value) -> bool:
    return (
        isinstance(value, tuple)
        and hasattr(value, "_asdict")
        and hasattr(value, "_fields")
    )


def _is_mlx_array(value) -> bool:
    return type(value).__module__.startswith("mlx.")


def _is_torch_object(value) -> bool:
    return type(value).__module__.startswith("torch")


@runtime_checkable
class _CopyableData(Protocol):
    def to(self, device, *args: Any, **kwargs: Any): ...


def copy_data_to_device(data, device=None, *args: Any, **kwargs: Any):
    """Recursively copy data to the explicit MLX runtime.

    The official helper recursively calls PyTorch ``.to(device)``. MLX arrays do
    not expose that API, so MLX leaves are returned as-is unless an explicit
    ``dtype=...`` conversion is requested. PyTorch leaves fail fast instead of
    silently taking a CPU/non-MLX fallback path inside the MLX port.
    """
    if device not in (None, "mlx"):
        raise_unsupported(
            f"sam3_mlx.model.utils.misc.copy_data_to_device(device={device!r})",
            reason="unsupported-device",
            detail="sam3_mlx targets the explicit MLX runtime; pass device='mlx' or None.",
        )
    if args:
        raise_unsupported(
            "sam3_mlx.model.utils.misc.copy_data_to_device(positional torch args)",
            reason="unsupported-device",
            detail="Only keyword dtype conversion is supported on the MLX port.",
        )
    unsupported_kwargs = set(kwargs) - {"dtype"}
    if unsupported_kwargs:
        names = ", ".join(sorted(unsupported_kwargs))
        raise_unsupported(
            f"sam3_mlx.model.utils.misc.copy_data_to_device(kwargs={names})",
            reason="unsupported-device",
            detail="Only the dtype kwarg is supported on the MLX port.",
        )
    if _is_named_tuple(data):
        return type(data)(
            **copy_data_to_device(data._asdict(), device, *args, **kwargs)
        )
    if isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(v, device, *args, **kwargs) for v in data)
    if isinstance(data, defaultdict):
        return type(data)(
            data.default_factory,
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            },
        )
    if isinstance(data, Mapping):
        return type(data)(
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            }
        )
    if is_dataclass(data) and not isinstance(data, type):
        copied = type(data)(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    copied,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return copied
    if _is_mlx_array(data):
        dtype = kwargs.get("dtype")
        return data.astype(dtype) if dtype is not None else data
    if _is_torch_object(data):
        raise_unsupported(
            "sam3_mlx.model.utils.misc.copy_data_to_device(torch object)",
            reason="unsupported-device",
            detail="PyTorch tensors/modules cannot be copied through the MLX port.",
        )
    if isinstance(data, _CopyableData):
        return data.to(device, *args, **kwargs)
    return data
