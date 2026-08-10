# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of ``sam3.model.utils.misc`` from the official SAM3 tree."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping
from copy import copy
from dataclasses import Field, is_dataclass
from typing import Literal, Protocol, TypeGuard, TypeVar, cast, runtime_checkable

from sam3_mlx._unsupported import raise_unsupported


_MlxDevice = Literal["mlx"] | None
_DataT = TypeVar("_DataT")


class _NamedTupleInstance(Protocol):
    _fields: tuple[str, ...]

    def _asdict(self) -> dict[str, object]: ...


class _DataclassInstance(Protocol):
    __dataclass_fields__: dict[str, Field[object]]


class _MlxArrayLike(Protocol):
    def astype(self, dtype: object) -> object: ...


def _is_named_tuple(value: object) -> TypeGuard[_NamedTupleInstance]:
    if not isinstance(value, tuple):
        return False
    tuple_value = cast(object, value)
    return callable(getattr(tuple_value, "_asdict", None)) and isinstance(
        getattr(tuple_value, "_fields", None), tuple
    )


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    return is_dataclass(value) and not isinstance(value, type)


def _is_mlx_array(value: object) -> TypeGuard[_MlxArrayLike]:
    return type(value).__module__.startswith("mlx.")


def _is_torch_object(value: object) -> bool:
    return type(value).__module__.startswith("torch")


@runtime_checkable
class _CopyableData(Protocol):
    def to(self, device: _MlxDevice, *args: object, **kwargs: object) -> object: ...


def _copy_named_tuple(
    data: _NamedTupleInstance,
    device: _MlxDevice,
    *args: object,
    **kwargs: object,
) -> object:
    constructor = cast(Callable[..., object], type(data))
    as_dict = cast(Callable[[], dict[str, object]], getattr(data, "_asdict"))
    return constructor(**copy_data_to_device(as_dict(), device, *args, **kwargs))


def _copy_dataclass_instance(
    data: _DataclassInstance,
    device: _MlxDevice,
    *args: object,
    **kwargs: object,
) -> object:
    constructor = cast(Callable[..., object], type(data))
    init_values = {
        field.name: copy_data_to_device(
            getattr(data, field.name), device, *args, **kwargs
        )
        for field in data.__dataclass_fields__.values()
        if field.init
    }
    copied = constructor(**init_values)
    for field in data.__dataclass_fields__.values():
        if not field.init:
            setattr(
                copied,
                field.name,
                copy_data_to_device(getattr(data, field.name), device, *args, **kwargs),
            )
    return copied


def copy_data_to_device(
    data: _DataT,
    device: _MlxDevice = None,
    *args: object,
    **kwargs: object,
) -> _DataT:
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
        return cast(_DataT, _copy_named_tuple(data, device, *args, **kwargs))
    if isinstance(data, list):
        typed_items = cast(list[object], data)
        copied_items = [
            copy_data_to_device(v, device, *args, **kwargs) for v in typed_items
        ]
        return cast(_DataT, copied_items)
    if isinstance(data, tuple):
        typed_items = cast(tuple[object, ...], data)
        return cast(
            _DataT,
            tuple(copy_data_to_device(v, device, *args, **kwargs) for v in typed_items),
        )
    if isinstance(data, defaultdict):
        typed_data = cast(defaultdict[object, object], data)
        return cast(
            _DataT,
            defaultdict(
                typed_data.default_factory,
                {
                    k: copy_data_to_device(v, device, *args, **kwargs)
                    for k, v in typed_data.items()
                },
            ),
        )
    if isinstance(data, MutableMapping):
        typed_data = cast(MutableMapping[object, object], data)
        copied_mapping: dict[object, object] = {
            k: copy_data_to_device(v, device, *args, **kwargs)
            for k, v in typed_data.items()
        }
        copied_mutable = copy(typed_data)
        copied_mutable.clear()
        copied_mutable.update(copied_mapping)
        return cast(_DataT, copied_mutable)
    if isinstance(data, Mapping):
        typed_data = cast(Mapping[object, object], data)
        copied_mapping: dict[object, object] = {
            k: copy_data_to_device(v, device, *args, **kwargs)
            for k, v in typed_data.items()
        }
        return cast(_DataT, copied_mapping)
    if _is_dataclass_instance(data):
        return cast(_DataT, _copy_dataclass_instance(data, device, *args, **kwargs))
    if _is_mlx_array(data):
        dtype = kwargs.get("dtype")
        return cast(_DataT, data.astype(dtype) if dtype is not None else data)
    if _is_torch_object(data):
        raise_unsupported(
            "sam3_mlx.model.utils.misc.copy_data_to_device(torch object)",
            reason="unsupported-device",
            detail="PyTorch tensors/modules cannot be copied through the MLX port.",
        )
    if isinstance(data, _CopyableData):
        return cast(_DataT, data.to(device, *args, **kwargs))
    return data
