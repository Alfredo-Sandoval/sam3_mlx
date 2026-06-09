from __future__ import annotations

from functools import wraps

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
        result = fn(*recursive_contiguous(args), **recursive_contiguous(kwargs))
        if mode in {"max-autotune", "reduce-overhead"}:
            return recursive_clone(result)
        return result

    return wrapped


def shape_logging_wrapper(fn, keep_kwargs, enable_logging=False):
    keep_kwargs = set(keep_kwargs or [])
    seen_shapes = set()

    def get_shape(obj):
        if hasattr(obj, "shape"):
            return tuple(obj.shape)
        if isinstance(obj, (list, tuple)):
            return tuple(get_shape(v) for v in obj)
        if isinstance(obj, dict):
            return tuple(sorted((k, get_shape(v)) for k, v in obj.items()))
        return type(obj).__name__

    @wraps(fn)
    def wrapper(*args, **kwargs):
        shapes = tuple(get_shape(arg) for arg in args) + tuple(
            (k, get_shape(v)) for k, v in kwargs.items() if k in keep_kwargs
        )
        if shapes not in seen_shapes:
            seen_shapes.add(shapes)
            if enable_logging:
                print(f"[ShapeLogger] New input shapes for {fn.__qualname__}: {shapes}")
        return fn(*args, **kwargs)

    def set_logging(enabled=False):
        nonlocal enable_logging
        enable_logging = enabled
        wrapper.enable_logging = enabled

    wrapper.enable_logging = enable_logging
    wrapper.set_logging = set_logging
    return wrapper
