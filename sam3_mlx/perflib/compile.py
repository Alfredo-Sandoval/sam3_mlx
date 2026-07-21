from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import update_wrapper, wraps
from typing import Any

import numpy as np


def recursive_fn_factory(fn):
    def recursive_fn(value):
        if isinstance(value, dict):
            return {k: recursive_fn(v) for k, v in value.items()}
        if isinstance(value, list):
            return [recursive_fn(v) for v in value]
        if isinstance(value, tuple):
            return tuple(recursive_fn(v) for v in value)
        if value.__class__.__name__ == "NestedTensor" and hasattr(value, "tensors"):
            mask = None if value.mask is None else recursive_fn(value.mask)
            return type(value)(tensors=recursive_fn(value.tensors), mask=mask)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return fn(value)

    return recursive_fn


def _contiguous_leaf(value):
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return value


def _clone_leaf(value):
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if type(value).__module__.startswith("mlx."):
        import mlx.core as mx

        return mx.array(value)
    if hasattr(value, "copy"):
        try:
            return value.copy()
        except TypeError:
            return value
    return value


recursive_contiguous = recursive_fn_factory(_contiguous_leaf)
recursive_clone = recursive_fn_factory(_clone_leaf)


def clone_output_wrapper(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        return recursive_clone(fn(*args, **kwargs))

    return wrapped


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    del fullgraph, dynamic, name

    @wraps(fn)
    def wrapped(*args, **kwargs):
        contiguous_args = recursive_contiguous(args)
        contiguous_kwargs = recursive_contiguous(kwargs)
        if not isinstance(contiguous_args, tuple):
            raise TypeError("recursive_contiguous(args) must preserve tuple shape.")
        if not isinstance(contiguous_kwargs, dict):
            raise TypeError("recursive_contiguous(kwargs) must preserve mapping shape.")
        result = fn(*contiguous_args, **contiguous_kwargs)
        if mode in {"max-autotune", "reduce-overhead"}:
            return recursive_clone(result)
        return result

    return wrapped


class _ShapeLoggingWrapper:
    def __init__(
        self,
        fn: Callable[..., Any],
        keep_kwargs: Iterable[str] | None,
        enable_logging: bool,
    ) -> None:
        self.fn = fn
        self.keep_kwargs = set(keep_kwargs or ())
        self.enable_logging = enable_logging
        self.seen_shapes: set[tuple[Any, ...]] = set()
        update_wrapper(self, fn)

    def _get_shape(self, obj: Any) -> Any:
        if hasattr(obj, "shape"):
            return tuple(obj.shape)
        if isinstance(obj, (list, tuple)):
            return tuple(self._get_shape(value) for value in obj)
        if isinstance(obj, dict):
            return tuple(
                sorted((key, self._get_shape(value)) for key, value in obj.items())
            )
        return type(obj).__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        shapes = tuple(self._get_shape(arg) for arg in args) + tuple(
            (key, self._get_shape(value))
            for key, value in kwargs.items()
            if key in self.keep_kwargs
        )
        if shapes not in self.seen_shapes:
            self.seen_shapes.add(shapes)
            if self.enable_logging:
                print(
                    f"[ShapeLogger] New input shapes for {self.fn.__qualname__}: {shapes}"
                )
        return self.fn(*args, **kwargs)

    def set_logging(self, enabled: bool = False) -> None:
        self.enable_logging = enabled


def shape_logging_wrapper(
    fn: Callable[..., Any],
    keep_kwargs: Iterable[str] | None,
    enable_logging: bool = False,
) -> _ShapeLoggingWrapper:
    return _ShapeLoggingWrapper(fn, keep_kwargs, enable_logging)
