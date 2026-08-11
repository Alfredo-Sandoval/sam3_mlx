from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps
from typing import ParamSpec, Protocol, TypeGuard, TypeVar, cast

import numpy as np
import numpy.typing as npt


class _NestedTensorLike(Protocol):
    tensors: object
    mask: object | None


class _NestedTensorConstructor(Protocol):
    def __call__(self, *, tensors: object, mask: object | None) -> object: ...


P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
NumpyArray = npt.NDArray[np.generic]


class _ObjectCallable(Protocol[R_co]):
    def __call__(self, *args: object, **kwargs: object) -> R_co: ...


class _LoggingSetter(Protocol):
    def __call__(self, enabled: bool = False) -> None: ...


class ShapeLoggedCallable(Protocol[P, R_co]):
    enable_logging: bool
    set_logging: _LoggingSetter

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


def _is_nested_tensor(value: object) -> TypeGuard[_NestedTensorLike]:
    return value.__class__.__name__ == "NestedTensor" and hasattr(value, "tensors")


def _copy_method(value: object) -> Callable[[], object] | None:
    method: object = getattr(value, "copy", None)
    return cast(Callable[[], object], method) if callable(method) else None


def recursive_fn_factory(
    fn: Callable[[object], object],
) -> Callable[[object], object]:
    def recursive_fn(value: object) -> object:
        if isinstance(value, dict):
            source = cast(dict[object, object], value)
            return {key: recursive_fn(item) for key, item in source.items()}
        if isinstance(value, list):
            return [recursive_fn(item) for item in cast(list[object], value)]
        if isinstance(value, tuple):
            return tuple(recursive_fn(item) for item in cast(tuple[object, ...], value))
        if _is_nested_tensor(value):
            mask = None if value.mask is None else recursive_fn(value.mask)
            constructor = cast(_NestedTensorConstructor, type(value))
            return constructor(tensors=recursive_fn(value.tensors), mask=mask)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return fn(value)

    return recursive_fn


def _contiguous_leaf(value: object) -> object:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(cast(NumpyArray, value))
    return value


def _clone_leaf(value: object) -> object:
    if hasattr(value, "clone"):
        clone: object = getattr(value, "clone")
        return cast(Callable[[], object], clone)()
    if isinstance(value, np.ndarray):
        return cast(NumpyArray, value.copy())
    if type(value).__module__.startswith("mlx."):
        import mlx.core as mx

        return mx.array(cast(npt.NDArray[np.generic], value))
    copy = _copy_method(value)
    if copy is not None:
        try:
            return copy()
        except TypeError:
            return value
    return value


recursive_contiguous = recursive_fn_factory(_contiguous_leaf)
recursive_clone = recursive_fn_factory(_clone_leaf)


def clone_output_wrapper(fn: Callable[P, R]) -> Callable[P, R]:
    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        return cast(R, recursive_clone(fn(*args, **kwargs)))

    return wrapped


def compile_wrapper(
    fn: Callable[P, R],
    *,
    mode: str = "max-autotune",
    fullgraph: bool = True,
    dynamic: bool = False,
    name: str | None = None,
) -> Callable[P, R]:
    del fullgraph, dynamic, name

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        contiguous_args = cast(tuple[object, ...], recursive_contiguous(args))
        contiguous_kwargs = cast(dict[str, object], recursive_contiguous(kwargs))
        dynamic_fn = cast(_ObjectCallable[R], fn)
        result = dynamic_fn(*contiguous_args, **contiguous_kwargs)
        if mode in {"max-autotune", "reduce-overhead"}:
            return cast(R, recursive_clone(result))
        return result

    return wrapped


def shape_logging_wrapper(
    fn: Callable[P, R],
    keep_kwargs: Iterable[str] | None,
    enable_logging: bool = False,
) -> ShapeLoggedCallable[P, R]:
    kept_names = set(keep_kwargs or ())
    seen_shapes: set[tuple[object, ...]] = set()

    def get_shape(obj: object) -> object:
        if hasattr(obj, "shape"):
            shape: object = getattr(obj, "shape")
            return tuple(cast(Iterable[object], shape))
        if isinstance(obj, (list, tuple)):
            return tuple(get_shape(item) for item in cast(Iterable[object], obj))
        if isinstance(obj, dict):
            source = cast(dict[object, object], obj)
            return tuple(
                sorted((key, get_shape(value)) for key, value in source.items())
            )
        return type(obj).__name__

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        shapes = tuple(get_shape(arg) for arg in args) + tuple(
            (key, get_shape(value))
            for key, value in kwargs.items()
            if key in kept_names
        )
        if shapes not in seen_shapes:
            seen_shapes.add(shapes)
            if enable_logging:
                qualname = cast(str, getattr(fn, "__qualname__"))
                print(f"[ShapeLogger] New input shapes for {qualname}: {shapes}")
        return fn(*args, **kwargs)

    typed_wrapper = cast(ShapeLoggedCallable[P, R], wrapper)

    def set_logging(enabled: bool = False) -> None:
        nonlocal enable_logging
        enable_logging = enabled
        typed_wrapper.enable_logging = enabled

    typed_wrapper.enable_logging = enable_logging
    typed_wrapper.set_logging = set_logging
    return typed_wrapper
