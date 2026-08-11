from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol, TypeAlias, cast, get_args, get_origin

import mlx.core as mx
from mlx import nn
import numpy as np
import numpy.typing as npt

from sam3_mlx._unsupported import raise_unsupported


MyTensor: TypeAlias = mx.array | list[Any]
_INT32 = getattr(mx, "int32", mx.int64)

IndexArray = npt.NDArray[np.int64]
WeightArray = npt.NDArray[np.float32]
ResizeWeights = tuple[tuple[IndexArray, WeightArray], ...]


class _ArrayMethods(Protocol):
    def reshape(self, *shape: int) -> mx.array: ...

    def transpose(self, *axes: int) -> mx.array: ...


def _reshape(array: mx.array, *shape: int) -> mx.array:
    return cast(_ArrayMethods, array).reshape(*shape)


def _transpose(array: mx.array, *axes: int) -> mx.array:
    return cast(_ArrayMethods, array).transpose(*axes)


class NestedTensor:
    def __init__(self, tensors: mx.array, mask: mx.array | None) -> None:
        self.tensors = tensors
        self.mask = mask

    def to(self, *args: object, **kwargs: object) -> "NestedTensor":
        device_value = kwargs.pop("device", None)
        dtype_value = kwargs.pop("dtype", None)
        device = cast(str | None, device_value)
        dtype = cast(mx.Dtype | None, dtype_value)
        if len(args) > 1:
            raise TypeError(
                "NestedTensor.to() accepts at most one positional argument."
            )
        if args:
            arg = args[0]
            if arg is None or isinstance(arg, str):
                device = arg
            else:
                dtype = cast(mx.Dtype, arg)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported NestedTensor.to() kwargs: {names}.")
        if device not in (None, "mlx"):
            raise_unsupported(
                f"sam3_mlx.model.data_misc.NestedTensor.to(device={device!r})",
                reason="unsupported-device",
                detail="NestedTensor.to() only supports the explicit MLX device.",
                alternative="device='mlx'",
            )
        tensors = self.tensors.astype(dtype) if dtype is not None else self.tensors
        mask = self.mask
        if dtype is not None and mask is not None:
            mask = mask.astype(dtype)
        return type(self)(tensors, mask)

    def clone(self) -> "NestedTensor":
        new_tensors = mx.array(self.tensors)
        new_mask = None if self.mask is None else mx.array(self.mask)
        return NestedTensor(new_tensors, new_mask)

    def decompose(self) -> tuple[mx.array, mx.array | None]:
        return self.tensors, self.mask

    def __getitem__(
        self,
        idx: int | slice | tuple[int | slice, ...] | mx.array,
    ) -> mx.array:
        return self.tensors[idx]

    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def device(self) -> Literal["mlx"]:
        return "mlx"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensors.shape

    def pin_memory(self, device: str | None = None) -> "NestedTensor":
        if device not in (None, "mlx"):
            raise_unsupported(
                f"sam3_mlx.model.data_misc.NestedTensor.pin_memory(device={device!r})",
                reason="training-loop",
                detail=(
                    "NestedTensor.pin_memory() is a PyTorch CPU-pinning API "
                    "and is not supported in the MLX port."
                ),
                alternative="device=None",
            )
        return self


@lru_cache(maxsize=128)
def _resize_weights_1d(
    in_size: int,
    out_size: int,
) -> ResizeWeights:
    scale = in_size / out_size
    weights_by_output: list[tuple[IndexArray, WeightArray]] = []
    if out_size < in_size:
        support = scale
        for out_index in range(out_size):
            center = (out_index + 0.5) * scale
            start = max(int(np.floor(center - support + 0.5)), 0)
            stop = min(int(np.floor(center + support + 0.5)), in_size)
            indices = np.arange(start, stop, dtype=np.int64)
            weights = 1.0 - np.abs((indices + 0.5 - center) / scale)
            weights = np.maximum(weights, 0.0).astype(np.float32)
            weights /= weights.sum(dtype=np.float32)
            indices.setflags(write=False)
            weights.setflags(write=False)
            weights_by_output.append((indices, weights))
        return tuple(weights_by_output)

    for out_index in range(out_size):
        source = (out_index + 0.5) * scale - 0.5
        left_raw = int(np.floor(source))
        right_raw = left_raw + 1
        weight_right = np.float32(source - left_raw)
        indices = np.array(
            [
                np.clip(left_raw, 0, in_size - 1),
                np.clip(right_raw, 0, in_size - 1),
            ],
            dtype=np.int64,
        )
        weights = np.array([1.0 - weight_right, weight_right], dtype=np.float32)
        indices.setflags(write=False)
        weights.setflags(write=False)
        weights_by_output.append((indices, weights))
    return tuple(weights_by_output)


def _interpolate_bilinear_antialias_nchw(
    input: mx.array,
    size: tuple[int, int],
) -> mx.array:
    if input.ndim != 4:
        raise ValueError(
            "antialiased bilinear interpolation expects a 4D NCHW tensor, "
            f"got shape {input.shape}."
        )

    out_h, out_w = size
    in_h, in_w = input.shape[-2:]
    if (out_h < in_h or out_w < in_w) and (out_h <= 1 or out_w <= 1):
        raise ValueError(
            "antialiased bilinear interpolation is currently supported for "
            "non-singleton output grids only."
        )
    if (in_h, in_w) == (out_h, out_w):
        return input

    y_weights = _resize_weights_1d(int(in_h), int(out_h))
    x_weights = _resize_weights_1d(int(in_w), int(out_w))

    rows: list[mx.array] = []
    for indices_np, weights_np in y_weights:
        indices = mx.array(indices_np, dtype=mx.int64)
        weights = _reshape(mx.array(weights_np, dtype=input.dtype), 1, 1, -1, 1)
        rows.append(mx.sum(mx.take(input, indices, axis=2) * weights, axis=2))
    resized_h = mx.stack(rows, axis=2)

    cols: list[mx.array] = []
    for indices_np, weights_np in x_weights:
        indices = mx.array(indices_np, dtype=mx.int64)
        weights = _reshape(mx.array(weights_np, dtype=input.dtype), 1, 1, 1, -1)
        cols.append(mx.sum(mx.take(resized_h, indices, axis=3) * weights, axis=3))
    return mx.stack(cols, axis=3)


_INTERPOLATE_MODE_MAP: dict[
    str,
    Literal["nearest", "linear", "cubic"],
] = {
    "nearest": "nearest",
    "bilinear": "linear",
    "linear": "linear",
    "bicubic": "cubic",
    "cubic": "cubic",
}


def _resolve_output_size(
    spatial_shape: tuple[int, ...],
    *,
    size: int | tuple[int, int] | None = None,
    scale_factor: float | tuple[float, float] | None = None,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Normalize Torch-style size/scale_factor into (out_h, out_w) and scale pair."""
    current_h, current_w = int(spatial_shape[0]), int(spatial_shape[1])

    if size is not None:
        if isinstance(size, int):
            out_h = out_w = int(size)
        else:
            if len(size) != 2:
                raise ValueError("size must be an int or a length-2 sequence.")
            out_h, out_w = int(size[0]), int(size[1])
        if current_h == 0 or current_w == 0:
            final_scale = (1.0, 1.0)
        else:
            final_scale = (out_h / current_h, out_w / current_w)
        return (out_h, out_w), final_scale

    if scale_factor is not None:
        if isinstance(scale_factor, (float, int)):
            scale_h = scale_w = float(scale_factor)
        else:
            if len(scale_factor) != 2:
                raise ValueError(
                    "scale_factor must be a float/int or a length-2 sequence."
                )
            scale_h, scale_w = float(scale_factor[0]), float(scale_factor[1])
        out_h = int(current_h * scale_h)
        out_w = int(current_w * scale_w)
        return (out_h, out_w), (scale_h, scale_w)

    raise ValueError("Either size or scale_factor must be defined")


def _map_interpolate_mode(mode: str) -> Literal["nearest", "linear", "cubic"]:
    try:
        return _INTERPOLATE_MODE_MAP[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(_INTERPOLATE_MODE_MAP))
        raise ValueError(
            f"Unsupported interpolate mode {mode!r}; expected one of: {supported}."
        ) from exc


def interpolate(
    input: mx.array,
    size: int | tuple[int, int] | None = None,
    scale_factor: float | tuple[float, float] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    antialias: bool = False,
) -> mx.array:
    mlx_mode = _map_interpolate_mode(mode)
    out_hw, final_scale = _resolve_output_size(
        input.shape[-2:],
        size=size,
        scale_factor=scale_factor,
    )

    if input.size == 0:
        if input.ndim == 4 and input.shape[0] == 0 and input.shape[1] == 0:
            raise ValueError(
                "interpolate does not support tensors with both empty batch "
                "and channel dimensions."
            )
        out_shape = list(input.shape)
        out_shape[-2] = out_hw[0]
        out_shape[-1] = out_hw[1]
        return mx.zeros(out_shape, dtype=input.dtype)

    if antialias:
        if mlx_mode != "linear" or align_corners not in (False, None):
            raise ValueError(
                "antialias=True is only supported for bilinear interpolation "
                "with align_corners=False."
            )
        return _interpolate_bilinear_antialias_nchw(input, out_hw)

    x = _transpose(input, 0, 2, 3, 1)
    upsample_layer = nn.Upsample(
        scale_factor=final_scale,
        mode=mlx_mode,
        align_corners=False if align_corners is None else align_corners,
    )
    x = upsample_layer(x)
    return _transpose(x, 0, 3, 1, 2)


@dataclass
class BatchedPointer:
    stage_ids: MyTensor
    stage_ids__type = mx.int64
    query_ids: MyTensor
    query_ids__type = mx.int64
    object_ids: MyTensor
    object_ids__type = mx.int64
    ptr_mask: MyTensor
    ptr_mask__type = mx.bool_
    ptr_types: MyTensor
    ptr_types__type = mx.int64


@dataclass
class FindStage:
    img_ids: MyTensor
    img_ids__type = mx.int64
    text_ids: MyTensor
    text_ids__type = mx.int64

    input_boxes: MyTensor
    input_boxes__type = mx.float32
    input_boxes_mask: MyTensor
    input_boxes_mask__type = mx.bool_
    input_boxes_label: MyTensor
    input_boxes_label__type = mx.int64

    input_points: MyTensor
    input_points__type = mx.float32
    input_points_mask: MyTensor
    input_points_mask__type = mx.bool_

    # We track the object ids referred to by this query.

    # This is beneficial for tracking in videos without the need for pointers.
    object_ids: list[list[object]] | None = None  # List of objects per query

    # Official SAM3 prompt fields used by Sam3Image.forward. Multiplex pointer
    # fields are kept as opaque placeholders until that path is ported.
    img_ids_np: Any | None = None
    input_boxes_before_embed: MyTensor | None = None
    input_boxes_before_embed__type = mx.float32
    input_points_before_embed: MyTensor | None = None
    input_points_before_embed__type = mx.float32
    ptrs: Any | None = None
    ptrs_seg: Any | None = None


@dataclass
class BatchedFindTarget:
    num_boxes: MyTensor
    num_boxes__type = mx.int64

    boxes: MyTensor
    boxes__type = mx.float32
    boxes_padded: MyTensor
    boxes_padded__type = mx.float32
    repeated_boxes: MyTensor
    repeated_boxes__type = mx.float32

    segments: MyTensor | None
    segments__type = mx.bool_
    semantic_segments: MyTensor | None
    semantic_segments__type = mx.bool_
    is_valid_segment: MyTensor | None
    is_valid_segment__type = mx.bool_
    is_exhaustive: MyTensor
    is_exhaustive__type = mx.bool_

    object_ids: MyTensor
    object_ids__type = mx.int64
    object_ids_padded: MyTensor
    object_ids_padded__type = mx.int64


@dataclass
class BatchedInferenceMetadata:
    coco_image_id: MyTensor
    coco_image_id__type = mx.int64
    original_image_id: MyTensor
    original_image_id__type = mx.int64
    original_category_id: MyTensor
    original_category_id__type = _INT32
    original_size: MyTensor
    original_size__type = mx.int64
    object_id: MyTensor
    object_id__type = mx.int64
    frame_index: MyTensor
    frame_index__type = mx.int64
    is_conditioning_only: list[bool | None]


@dataclass
class BatchedDatapoint:
    img_batch: MyTensor
    find_text_batch: list[str]
    find_inputs: list[FindStage]
    find_targets: list[BatchedFindTarget | None]
    find_metadatas: list[BatchedInferenceMetadata | None]
    raw_images: list[Any] | None = None
    get_queries: Any | None = None


def convert_my_tensors(obj: object) -> object:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError("convert_my_tensors expects a dataclass instance")
    for field in fields(obj):
        value = getattr(obj, field.name)
        if is_dataclass(value):
            convert_my_tensors(value)
            continue
        if not _is_mytensor_field(field.type) or value is None:
            continue
        dtype = cast(mx.Dtype, getattr(obj, field.name + "__type"))
        if isinstance(value, mx.array):
            setattr(obj, field.name, value.astype(dtype))
        elif (
            isinstance(value, list)
            and value
            and isinstance(value[0], mx.array)
        ):
            array_values = cast(list[mx.array], value)
            stack_dim = (
                1
                if field.name
                in {"input_boxes_before_embed", "input_boxes", "input_boxes_label"}
                else 0
            )
            setattr(
                obj,
                field.name,
                mx.stack(array_values, axis=stack_dim).astype(dtype),
            )
        else:
            setattr(obj, field.name, mx.array(value, dtype=dtype))
    return obj


def _is_mytensor_field(field_type: object) -> bool:
    if field_type == MyTensor:
        return True
    field_args = get_args(field_type)
    return mx.array in field_args and any(
        get_origin(arg) is list for arg in field_args
    )
