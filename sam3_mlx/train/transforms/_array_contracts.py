"""Shared typed MLX/NumPy boundaries for training transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias, TypeGuard, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from sam3_mlx.model import box_ops as _box_ops

ArrayData: TypeAlias = mx.array | NDArray[np.generic]
NumericSequence: TypeAlias = Sequence[int] | Sequence[float]
ArrayInput: TypeAlias = ArrayData | NumericSequence
ImageInput: TypeAlias = PILImage.Image | ArrayData
Padding: TypeAlias = tuple[int, int] | tuple[int, int, int, int]
ImageSize: TypeAlias = tuple[int, int]
CropRegion: TypeAlias = tuple[int, int, int, int]


class _MxArrayCtor(Protocol):
    def __call__(self, val: ArrayInput, dtype: object | None = None) -> mx.array: ...


class _MxEvalFn(Protocol):
    def __call__(self, *values: mx.array) -> None: ...


class MxArrayOps(Protocol):
    dtype: object
    shape: tuple[int, ...]
    ndim: int

    def astype(self, dtype: object) -> mx.array: ...

    def transpose(
        self, axis0: int, axis1: int, axis2: int, /, stream: object | None = None
    ) -> mx.array: ...

    def reshape(self, *shape: int) -> mx.array: ...


class _ArrayTransform(Protocol):
    def __call__(self, value: mx.array) -> mx.array: ...


mx_array = cast(_MxArrayCtor, getattr(mx, "array"))
_mx_eval = cast(_MxEvalFn, getattr(mx, "eval"))
box_xyxy_to_cxcywh = cast(
    _ArrayTransform,
    getattr(_box_ops, "box_xyxy_to_cxcywh"),
)
masks_to_boxes = cast(_ArrayTransform, getattr(_box_ops, "masks_to_boxes"))


def mx_ops(array: mx.array) -> MxArrayOps:
    return cast(MxArrayOps, array)


def is_mlx_array(value: object) -> TypeGuard[mx.array]:
    return isinstance(value, mx.array)


def to_numpy(value: object) -> NDArray[np.generic]:
    if isinstance(value, np.ndarray):
        return cast(NDArray[np.generic], value)
    if is_mlx_array(value):
        _mx_eval(value)
    return cast(NDArray[np.generic], np.asarray(value))


def as_float_array(value: ArrayInput) -> mx.array:
    if is_mlx_array(value):
        return mx_ops(value).astype(mx.float32)
    return mx_array(value, dtype=mx.float32)


def restore_array(
    value: NDArray[np.generic],
    like: ArrayData,
    *,
    preserve_dtype: bool,
) -> ArrayData:
    """Restore container type with the caller's established dtype semantics."""

    if is_mlx_array(like):
        if preserve_dtype:
            return mx_array(value, dtype=mx_ops(like).dtype)
        return mx_array(value)
    if isinstance(like, np.ndarray):
        if preserve_dtype:
            return value.astype(like.dtype, copy=False)
        return value
    raise TypeError(f"Unsupported array type: {type(like)!r}")


def mx_shape(array: mx.array) -> tuple[int, ...]:
    return mx_ops(array).shape


def mx_dtype(array: mx.array) -> object:
    return mx_ops(array).dtype


def transpose_hwc_to_chw(array: mx.array) -> mx.array:
    return mx_ops(array).transpose(2, 0, 1)


def image_size(image: ImageInput) -> ImageSize:
    if isinstance(image, PILImage.Image):
        return image.size
    array = to_numpy(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        return int(array.shape[2]), int(array.shape[1])
    if array.ndim >= 2:
        return int(array.shape[1]), int(array.shape[0])
    raise TypeError(f"Unsupported image shape: {array.shape}.")
