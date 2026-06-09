from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from typing import Any, get_args, get_origin, List, Optional, Union

from sam3_mlx._unsupported import raise_unsupported


MyTensor = Union[mx.array, List[Any]]
_INT32 = getattr(mx, "int32", mx.int64)


class NestedTensor:
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)
        if len(args) > 1:
            raise TypeError(
                "NestedTensor.to() accepts at most one positional argument."
            )
        if args:
            arg = args[0]
            if arg is None or isinstance(arg, str):
                device = arg
            else:
                dtype = arg
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

    def clone(self):
        new_tensors = mx.array(self.tensors)
        new_mask = None if self.mask is None else mx.array(self.mask)
        return NestedTensor(new_tensors, new_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __getitem__(self, idx):
        return self.tensors[idx]

    def __len__(self):
        return len(self.tensors)

    @property
    def device(self):
        return "mlx"

    @property
    def shape(self):
        return self.tensors.shape

    def pin_memory(self, device=None):
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
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    scale = in_size / out_size
    weights_by_output = []
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


def _interpolate_bilinear_antialias_nchw(input: mx.array, size: tuple[int, int]):
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

    rows = []
    for indices_np, weights_np in y_weights:
        indices = mx.array(indices_np, dtype=mx.int64)
        weights = mx.array(weights_np, dtype=input.dtype).reshape(1, 1, -1, 1)
        rows.append(mx.sum(mx.take(input, indices, axis=2) * weights, axis=2))
    resized_h = mx.stack(rows, axis=2)

    cols = []
    for indices_np, weights_np in x_weights:
        indices = mx.array(indices_np, dtype=mx.int64)
        weights = mx.array(weights_np, dtype=input.dtype).reshape(1, 1, 1, -1)
        cols.append(mx.sum(mx.take(resized_h, indices, axis=3) * weights, axis=3))
    return mx.stack(cols, axis=3)


def interpolate(
    input,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    antialias=False,
):
    if input.size == 0:
        out_shape = list(input.shape)
        if size is not None:
            # size is usually (H, W)
            out_shape[2] = size[0]
            out_shape[3] = size[1]
        elif scale_factor is not None:
            out_shape[2] = int(out_shape[2] * scale_factor)
            out_shape[3] = int(out_shape[3] * scale_factor)
        return mx.zeros(out_shape, dtype=input.dtype)

    x = input.transpose(0, 2, 3, 1)
    if mode == "bilinear" or mode == "bicubic":
        mode = "linear"

    current_h, current_w = x.shape[1], x.shape[2]

    if size is not None:
        if isinstance(size, int):
            size = (size, size)

        scale_h = size[0] / current_h
        scale_w = size[1] / current_w
        final_scale = (scale_h, scale_w)

    elif scale_factor is not None:
        if isinstance(scale_factor, (float, int)):
            final_scale = (float(scale_factor), float(scale_factor))
        else:
            final_scale = scale_factor
        size = (
            int(current_h * final_scale[0]),
            int(current_w * final_scale[1]),
        )

    else:
        raise ValueError("Either size or scale_factor must be defined")

    if antialias:
        if mode != "linear" or align_corners not in (False, None):
            raise ValueError(
                "antialias=True is only supported for bilinear interpolation "
                "with align_corners=False."
            )
        return _interpolate_bilinear_antialias_nchw(input, size)

    upsample_layer = nn.Upsample(
        scale_factor=final_scale, mode=mode, align_corners=align_corners
    )

    x = upsample_layer(x)

    return x.transpose(0, 3, 1, 2)


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
    object_ids: Optional[List[List]] = None  # List of objects per query

    # Official SAM3 prompt fields used by Sam3Image.forward. Multiplex pointer
    # fields are kept as opaque placeholders until that path is ported.
    img_ids_np: Optional[Any] = None
    input_boxes_before_embed: Optional[MyTensor] = None
    input_boxes_before_embed__type = mx.float32
    input_points_before_embed: Optional[MyTensor] = None
    input_points_before_embed__type = mx.float32
    ptrs: Optional[Any] = None
    ptrs_seg: Optional[Any] = None


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

    segments: Optional[MyTensor]
    segments__type = mx.bool_
    semantic_segments: Optional[MyTensor]
    semantic_segments__type = mx.bool_
    is_valid_segment: Optional[MyTensor]
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
    is_conditioning_only: List[Optional[bool]]


@dataclass
class BatchedDatapoint:
    img_batch: MyTensor
    find_text_batch: List[str]
    find_inputs: List[FindStage]
    find_targets: List[BatchedFindTarget]
    find_metadatas: List[BatchedInferenceMetadata]
    raw_images: Optional[List[Any]] = None
    get_queries: Optional[Any] = None


def convert_my_tensors(obj):
    for field in fields(obj):
        value = getattr(obj, field.name)
        if is_dataclass(value):
            convert_my_tensors(value)
            continue
        if not _is_mytensor_field(field.type) or value is None:
            continue
        dtype = getattr(obj, field.name + "__type")
        if isinstance(value, mx.array):
            setattr(obj, field.name, value.astype(dtype))
        elif (
            isinstance(value, list)
            and len(value) > 0
            and isinstance(value[0], mx.array)
        ):
            stack_dim = (
                1
                if field.name
                in {"input_boxes_before_embed", "input_boxes", "input_boxes_label"}
                else 0
            )
            setattr(obj, field.name, mx.stack(value, axis=stack_dim).astype(dtype))
        else:
            setattr(obj, field.name, mx.array(value, dtype=dtype))
    return obj


def _is_mytensor_field(field_type) -> bool:
    if field_type == MyTensor:
        return True
    if get_origin(field_type) is Union:
        return any(arg == MyTensor for arg in get_args(field_type))
    return False
