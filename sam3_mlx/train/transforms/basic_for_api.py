# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Minimal MLX image transform surface for SAM3 API datapoints.

Ported from ``third_party/facebook-sam3/sam3/train/transforms/basic_for_api.py``.
Only image input shaping that can run without Torch/Torchvision is active.
Video augmentation and Torchvision v2 behavior fail fast.
"""

from __future__ import annotations

import random
from typing import Iterable

import mlx.core as mx
import numpy as np
from PIL import Image as PILImage
from PIL import ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps

from sam3_mlx.model.box_ops import box_xyxy_to_cxcywh, masks_to_boxes
from sam3_mlx.train._unsupported import raise_unsupported
from sam3_mlx.train.data.sam3_image_dataset import Datapoint


def _check_no_v2(v2: bool, feature: str) -> None:
    if v2:
        raise_unsupported(f"{feature} with torchvision v2")


def _as_float_array(value) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    return mx.array(value, dtype=mx.float32)


def _is_mlx_array(value) -> bool:
    return isinstance(value, mx.array)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if _is_mlx_array(value):
        mx.eval(value)
    return np.asarray(value)


def _restore_array(value: np.ndarray, like):
    if _is_mlx_array(like):
        return mx.array(value, dtype=like.dtype)
    return value.astype(getattr(like, "dtype", value.dtype), copy=False)


def _image_size(data) -> tuple[int, int]:
    if isinstance(data, PILImage.Image):
        return data.size
    array = _to_numpy(data)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        return int(array.shape[2]), int(array.shape[1])
    if array.ndim >= 2:
        return int(array.shape[1]), int(array.shape[0])
    raise TypeError(f"Unsupported image shape: {array.shape}.")


def _is_chw(array: np.ndarray) -> bool:
    return array.ndim == 3 and array.shape[0] in (1, 3, 4)


def _array_to_pil_image(value):
    array = _to_numpy(value)
    chw = _is_chw(array)
    if chw:
        array = array.transpose(1, 2, 0)
    was_unit_float = np.issubdtype(array.dtype, np.floating) and (
        array.size == 0 or (array.min() >= 0.0 and array.max() <= 1.0)
    )
    if was_unit_float:
        array = array * 255.0
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    return PILImage.fromarray(array), chw, was_unit_float


def _restore_pil_image(image: PILImage.Image, like, chw: bool, was_unit_float: bool):
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if was_unit_float:
        array = array.astype(np.float32) / 255.0
    if chw:
        array = array.transpose(2, 0, 1)
    return _restore_array(array, like)


def _apply_pil_transform(data, transform):
    if isinstance(data, PILImage.Image):
        return transform(data)
    image, chw, was_unit_float = _array_to_pil_image(data)
    return _restore_pil_image(transform(image), data, chw, was_unit_float)


def _crop_data(data, top: int, left: int, height: int, width: int):
    if isinstance(data, PILImage.Image):
        return data.crop((left, top, left + width, top + height))
    array = _to_numpy(data)
    if _is_chw(array):
        cropped = array[:, top : top + height, left : left + width]
    else:
        cropped = array[top : top + height, left : left + width, ...]
    return _restore_array(cropped, data)


def _hflip_data(data):
    if isinstance(data, PILImage.Image):
        return ImageOps.mirror(data)
    array = _to_numpy(data)
    flipped = np.flip(array, axis=2 if _is_chw(array) else 1)
    return _restore_array(flipped, data)


def _resize_data(data, height: int, width: int):
    if isinstance(data, PILImage.Image):
        return data.resize((width, height), resample=PILImage.Resampling.BILINEAR)
    image, chw, was_unit_float = _array_to_pil_image(data)
    resized = image.resize((width, height), resample=PILImage.Resampling.BILINEAR)
    return _restore_pil_image(resized, data, chw, was_unit_float)


def _pad_data(data, padding):
    if len(padding) == 2:
        left, top, right, bottom = 0, 0, padding[0], padding[1]
    else:
        left, top, right, bottom = padding
    if isinstance(data, PILImage.Image):
        return ImageOps.expand(data, border=(left, top, right, bottom), fill=0)
    array = _to_numpy(data)
    if _is_chw(array):
        pad_width = ((0, 0), (top, bottom), (left, right))
    else:
        pad_width = ((top, bottom), (left, right)) + tuple(
            (0, 0) for _ in range(max(array.ndim - 2, 0))
        )
    return _restore_array(np.pad(array, pad_width), data)


def _crop_mask(mask, top: int, left: int, height: int, width: int):
    array = _to_numpy(mask)
    cropped = array[top : top + height, left : left + width]
    return _restore_array(cropped, mask)


def _hflip_mask(mask):
    return _restore_array(np.flip(_to_numpy(mask), axis=-1), mask)


def _resize_mask(mask, height: int, width: int):
    array = _to_numpy(mask).astype(np.uint8, copy=False)
    image = PILImage.fromarray(array)
    resized = image.resize((width, height), resample=PILImage.Resampling.NEAREST)
    return _restore_array(np.asarray(resized, dtype=np.uint8), mask)


def _pad_mask(mask, padding):
    if len(padding) == 2:
        left, top, right, bottom = 0, 0, padding[0], padding[1]
    else:
        left, top, right, bottom = padding
    return _restore_array(np.pad(_to_numpy(mask), ((top, bottom), (left, right))), mask)


def _scale_boxes_xyxy(boxes, ratio_width: float, ratio_height: float) -> mx.array:
    boxes = _as_float_array(boxes)
    scale = mx.array([ratio_width, ratio_height, ratio_width, ratio_height])
    return boxes * scale


def _offset_boxes_xyxy(boxes, offset_x: float, offset_y: float) -> mx.array:
    boxes = _as_float_array(boxes)
    offset = mx.array([offset_x, offset_y, offset_x, offset_y])
    return boxes + offset


def _hflip_boxes_xyxy(boxes, width: int) -> mx.array:
    boxes = _as_float_array(boxes)
    x0 = boxes[..., 0]
    y0 = boxes[..., 1]
    x1 = boxes[..., 2]
    y1 = boxes[..., 3]
    return mx.stack([width - x1, y0, width - x0, y1], axis=-1)


def _hflip_points(points, width: int) -> mx.array:
    points = _as_float_array(points)
    x = width - points[..., 0]
    y = points[..., 1]
    label = points[..., 2]
    return mx.stack([x, y, label], axis=-1)


def _scale_points(points, ratio_width: float, ratio_height: float) -> mx.array:
    points = _as_float_array(points)
    scale = mx.array([ratio_width, ratio_height, 1.0])
    return points * scale


def _offset_points(points, offset_x: float, offset_y: float) -> mx.array:
    points = _as_float_array(points)
    offset = mx.array([offset_x, offset_y, 0.0])
    return points + offset


def _crop_boxes_xyxy(boxes, top: int, left: int, height: int, width: int) -> mx.array:
    boxes = _as_float_array(boxes)
    cropped = boxes - mx.array([left, top, left, top], dtype=mx.float32)
    max_size = mx.array([width, height], dtype=mx.float32)
    reshaped = cropped.reshape(-1, 2, 2)
    reshaped = mx.minimum(reshaped, max_size)
    reshaped = mx.maximum(reshaped, mx.zeros_like(reshaped))
    return reshaped.reshape(-1, 4)


def _crop_points(points, top: int, left: int, height: int, width: int) -> mx.array:
    points = _as_float_array(points)
    cropped = points - mx.array([left, top, 0.0], dtype=mx.float32)
    max_size = mx.array([width - 1, height - 1], dtype=mx.float32)
    xy = mx.minimum(cropped[..., :2], max_size)
    xy = mx.maximum(xy, mx.zeros_like(xy))
    return mx.concat([xy, cropped[..., 2:]], axis=-1)


def _transform_boxes_xyxy(boxes, matrix, width: int, height: int) -> mx.array:
    boxes_np = _to_numpy(boxes).astype(np.float32, copy=False).reshape(-1, 4)
    corners = np.stack(
        [
            boxes_np[:, [0, 1]],
            boxes_np[:, [2, 1]],
            boxes_np[:, [2, 3]],
            boxes_np[:, [0, 3]],
        ],
        axis=1,
    )
    ones = np.ones((*corners.shape[:2], 1), dtype=np.float32)
    homogeneous = np.concatenate([corners, ones], axis=-1)
    transformed = homogeneous @ matrix.T
    xy = transformed[..., :2]
    xy[..., 0] = np.clip(xy[..., 0], 0, width)
    xy[..., 1] = np.clip(xy[..., 1], 0, height)
    mins = xy.min(axis=1)
    maxs = xy.max(axis=1)
    return mx.array(np.concatenate([mins, maxs], axis=1), dtype=mx.float32)


def _transform_points(points, matrix, width: int, height: int) -> mx.array:
    points_np = _to_numpy(points).astype(np.float32, copy=False)
    xy = points_np[..., :2]
    ones = np.ones((*xy.shape[:-1], 1), dtype=np.float32)
    transformed = np.concatenate([xy, ones], axis=-1) @ matrix.T
    transformed_xy = transformed[..., :2]
    transformed_xy[..., 0] = np.clip(transformed_xy[..., 0], 0, width - 1)
    transformed_xy[..., 1] = np.clip(transformed_xy[..., 1], 0, height - 1)
    out = np.concatenate([transformed_xy, points_np[..., 2:]], axis=-1)
    return mx.array(out, dtype=mx.float32)


def _affine_matrices(width, height, angle, translate, scale, shear):
    center_x = width * 0.5
    center_y = height * 0.5
    angle_rad = np.deg2rad(angle)
    shear_x = np.deg2rad(shear[0]) if shear is not None else 0.0
    shear_y = np.deg2rad(shear[1]) if shear is not None and len(shear) > 1 else 0.0
    cos_a = np.cos(angle_rad) * scale
    sin_a = np.sin(angle_rad) * scale
    rotation = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    shear_matrix = np.array(
        [[1.0, np.tan(shear_x), 0.0], [np.tan(shear_y), 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    from_origin = np.array(
        [
            [1.0, 0.0, center_x + translate[0]],
            [0.0, 1.0, center_y + translate[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    forward = from_origin @ shear_matrix @ rotation @ to_origin
    inverse = np.linalg.inv(forward)
    return forward, inverse


def _apply_affine_image(data, inverse_matrix, interpolation, fill):
    data_tuple = tuple(float(v) for v in inverse_matrix[:2].reshape(-1))

    def transform(image):
        return image.transform(
            image.size,
            PILImage.Transform.AFFINE,
            data_tuple,
            resample=interpolation,
            fillcolor=fill,
        )

    return _apply_pil_transform(data, transform)


def _apply_affine_mask(mask, inverse_matrix):
    data_tuple = tuple(float(v) for v in inverse_matrix[:2].reshape(-1))
    array = _to_numpy(mask).astype(np.uint8, copy=False)
    image = PILImage.fromarray(array)
    transformed = image.transform(
        image.size,
        PILImage.Transform.AFFINE,
        data_tuple,
        resample=PILImage.Resampling.NEAREST,
        fillcolor=0,
    )
    return _restore_array(np.asarray(transformed, dtype=np.uint8), mask)


def hflip(datapoint, index):
    img = datapoint.images[index]
    width, _height = _image_size(img.data)
    img.data = _hflip_data(img.data)

    for obj in img.objects:
        obj.bbox = _hflip_boxes_xyxy(obj.bbox, width)
        if obj.segment is not None:
            obj.segment = _hflip_mask(obj.segment)

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = _hflip_mask(query.semantic_target)
        if query.image_id == index and query.input_bbox is not None:
            query.input_bbox = _hflip_boxes_xyxy(query.input_bbox, width)
        if query.image_id == index and query.input_points is not None:
            query.input_points = _hflip_points(query.input_points, width)
    return datapoint


def get_size_with_aspect_ratio(image_size, size, max_size=None):
    width, height = image_size
    if max_size is not None:
        min_original_size = float(min((width, height)))
        max_original_size = float(max((width, height)))
        if max_original_size / min_original_size * size > max_size:
            size = max_size * min_original_size / max_original_size

    if (width <= height and width == size) or (height <= width and height == size):
        return height, width

    if width < height:
        out_width = int(round(size))
        out_height = int(round(size * height / width))
    else:
        out_height = int(round(size))
        out_width = int(round(size * width / height))

    return out_height, out_width


def resize(datapoint, index, size, max_size=None, square=False, v2=False):
    _check_no_v2(v2, "resize")
    img = datapoint.images[index]

    old_width, old_height = _image_size(img.data)
    if square:
        new_height, new_width = size, size
    elif isinstance(size, (list, tuple)):
        new_height, new_width = size[::-1]
    else:
        new_height, new_width = get_size_with_aspect_ratio(
            _image_size(img.data), size, max_size
        )

    new_height, new_width = int(new_height), int(new_width)
    img.data = _resize_data(img.data, new_height, new_width)
    ratio_width = float(new_width) / float(old_width)
    ratio_height = float(new_height) / float(old_height)

    for obj in img.objects:
        obj.bbox = _scale_boxes_xyxy(obj.bbox, ratio_width, ratio_height)
        obj.area *= ratio_width * ratio_height
        if obj.segment is not None:
            obj.segment = _resize_mask(obj.segment, new_height, new_width)

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = _resize_mask(
                query.semantic_target, new_height, new_width
            )
        if query.image_id == index and query.input_bbox is not None:
            query.input_bbox = _scale_boxes_xyxy(
                query.input_bbox, ratio_width, ratio_height
            )
        if query.image_id == index and query.input_points is not None:
            query.input_points = _scale_points(
                query.input_points, ratio_width, ratio_height
            )

    img.size = (int(new_height), int(new_width))
    return datapoint


def pad(datapoint, index, padding, v2=False):
    _check_no_v2(v2, "pad")
    img = datapoint.images[index]

    if len(padding) == 2:
        left, top, right, bottom = 0, 0, padding[0], padding[1]
    else:
        left, top, right, bottom = padding

    img.data = _pad_data(img.data, padding)
    width, height = _image_size(img.data)
    img.size = (height, width)

    for obj in img.objects:
        if left or top:
            obj.bbox = _offset_boxes_xyxy(obj.bbox, left, top)
        if obj.segment is not None:
            obj.segment = _pad_mask(obj.segment, padding)
    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = _pad_mask(query.semantic_target, padding)
        if left or top and query.image_id == index and query.input_bbox is not None:
            query.input_bbox = _offset_boxes_xyxy(query.input_bbox, left, top)
        if left or top and query.image_id == index and query.input_points is not None:
            query.input_points = _offset_points(query.input_points, left, top)

    return datapoint


def crop(
    datapoint,
    index,
    region,
    v2=False,
    check_validity=True,
    check_input_validity=True,
    recompute_box_from_mask=False,
):
    _check_no_v2(v2, "crop")
    del check_input_validity
    top, left, height, width = [int(round(v)) for v in region]
    img = datapoint.images[index]
    img.data = _crop_data(img.data, top, left, height, width)
    img.size = (height, width)

    for obj in img.objects:
        if obj.segment is not None:
            obj.segment = _crop_mask(obj.segment, top, left, height, width)
        if recompute_box_from_mask and obj.segment is not None:
            obj.bbox, obj.area = get_bbox_xyxy_abs_coords_from_mask(obj.segment)
        else:
            obj.bbox = _crop_boxes_xyxy(obj.bbox, top, left, height, width)
            cropped = obj.bbox.reshape(-1, 2, 2)
            obj.area = mx.prod(cropped[:, 1, :] - cropped[:, 0, :], axis=1)

    for query in datapoint.find_queries:
        if query.semantic_target is not None:
            query.semantic_target = _crop_mask(
                query.semantic_target, top, left, height, width
            )
        if query.image_id == index and query.input_bbox is not None:
            query.input_bbox = _crop_boxes_xyxy(
                query.input_bbox, top, left, height, width
            )
        if query.image_id == index and query.input_points is not None:
            query.input_points = _crop_points(
                query.input_points, top, left, height, width
            )

    if check_validity:
        for obj in img.objects:
            area = _to_numpy(obj.area)
            if not np.all(area > 0):
                raise AssertionError(f"Box {obj.bbox} has no area")

    return datapoint


class RandomHorizontalFlip:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.images)):
                    datapoint = hflip(datapoint, i)
            return datapoint
        for i in range(len(datapoint.images)):
            if random.random() < self.p:
                datapoint = hflip(datapoint, i)
        return datapoint


class RandomResizeAPI:
    def __init__(
        self, sizes, consistent_transform, max_size=None, square=False, v2=False
    ):
        if isinstance(sizes, int):
            sizes = (sizes,)
        if not isinstance(sizes, Iterable):
            raise TypeError("sizes must be an int or iterable of ints")
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(datapoint.images)):
                datapoint = resize(
                    datapoint, i, size, self.max_size, square=self.square, v2=self.v2
                )
            return datapoint
        for i in range(len(datapoint.images)):
            size = random.choice(self.sizes)
            datapoint = resize(
                datapoint, i, size, self.max_size, square=self.square, v2=self.v2
            )
        return datapoint


class ScheduledRandomResizeAPI(RandomResizeAPI):
    def __init__(self, size_scheduler, consistent_transform, square=False):
        self.size_scheduler = size_scheduler
        params = self.size_scheduler(epoch_num=0)
        sizes, max_size = params["sizes"], params["max_size"]
        super().__init__(sizes, consistent_transform, max_size=max_size, square=square)

    def __call__(self, datapoint, **kwargs):
        if "epoch" not in kwargs:
            raise ValueError("Param scheduler needs to know the current epoch")
        params = self.size_scheduler(kwargs["epoch"])
        self.sizes = params["sizes"]
        self.max_size = params["max_size"]
        return super().__call__(datapoint, **kwargs)


class RandomPadAPI:
    def __init__(self, max_pad, consistent_transform):
        self.max_pad = max_pad
        self.consistent_transform = consistent_transform

    def _sample_pad(self):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad_x, pad_y

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            padding = self._sample_pad()
            for i in range(len(datapoint.images)):
                datapoint = pad(datapoint, i, padding)
            return datapoint
        for i in range(len(datapoint.images)):
            datapoint = pad(datapoint, i, self._sample_pad())
        return datapoint


class PadToSizeAPI:
    def __init__(self, size, consistent_transform, bottom_right=False, v2=False):
        self.size = size
        self.consistent_transform = consistent_transform
        self.v2 = v2
        self.bottom_right = bottom_right

    def _sample_pad(self, width, height):
        pad_x = self.size - width
        pad_y = self.size - height
        if pad_x < 0 or pad_y < 0:
            raise ValueError("PadToSizeAPI size must be >= current image dimensions")
        pad_left = random.randint(0, pad_x)
        pad_right = pad_x - pad_left
        pad_top = random.randint(0, pad_y)
        pad_bottom = pad_y - pad_top
        return pad_left, pad_top, pad_right, pad_bottom

    def __call__(self, datapoint, **kwargs):
        if self.consistent_transform:
            width, height = _image_size(datapoint.images[0].data)
            for image in datapoint.images:
                if _image_size(image.data) != (width, height):
                    raise ValueError(
                        "consistent PadToSizeAPI requires equal image sizes"
                    )
            padding = (
                (self.size - width, self.size - height)
                if self.bottom_right
                else self._sample_pad(width, height)
            )
            for i in range(len(datapoint.images)):
                datapoint = pad(datapoint, i, padding, v2=self.v2)
            return datapoint

        for i, img in enumerate(datapoint.images):
            width, height = _image_size(img.data)
            padding = (
                (self.size - width, self.size - height)
                if self.bottom_right
                else self._sample_pad(width, height)
            )
            datapoint = pad(datapoint, i, padding, v2=self.v2)
        return datapoint


class ScheduledPadToSizeAPI(PadToSizeAPI):
    def __init__(self, size_scheduler, consistent_transform):
        self.size_scheduler = size_scheduler
        size = self.size_scheduler(epoch_num=0)
        super().__init__(size, consistent_transform)

    def __call__(self, datapoint, **kwargs):
        if "epoch" not in kwargs:
            raise ValueError("Param scheduler needs to know the current epoch")
        self.size = self.size_scheduler(kwargs["epoch"])
        return super().__call__(datapoint, **kwargs)


class IdentityAPI:
    def __call__(self, datapoint, **kwargs):
        return datapoint


class RandomSelectAPI:
    def __init__(self, transforms1=None, transforms2=None, p=0.5):
        self.transforms1 = transforms1 or IdentityAPI()
        self.transforms2 = transforms2 or IdentityAPI()
        self.p = p

    def __call__(self, datapoint, **kwargs):
        if random.random() < self.p:
            return self.transforms1(datapoint, **kwargs)
        return self.transforms2(datapoint, **kwargs)


class ToTensorAPI:
    def __init__(self, v2=False):
        _check_no_v2(v2, "ToTensorAPI")
        self.v2 = v2

    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            if isinstance(img.data, PILImage.Image):
                array = np.asarray(img.data)
                if array.ndim == 2:
                    array = array[:, :, None]
                if array.ndim != 3:
                    raise ValueError("Expected a HWC image array")
                img.data = mx.array(array.transpose(2, 0, 1), dtype=mx.float32) / 255.0
            elif isinstance(img.data, mx.array):
                data = img.data
                if data.ndim == 3 and data.shape[-1] in (1, 3, 4):
                    data = data.transpose(2, 0, 1)
                uint8_dtype = getattr(mx, "uint8", None)
                data = data.astype(mx.float32)
                if uint8_dtype is not None and img.data.dtype == uint8_dtype:
                    data = data / 255.0
                img.data = data
            else:
                raise TypeError(f"Unsupported image type: {type(img.data)!r}")
        return datapoint


class NormalizeAPI:
    def __init__(self, mean, std, v2=False):
        _check_no_v2(v2, "NormalizeAPI")
        self.mean = mx.array(mean, dtype=mx.float32).reshape(-1, 1, 1)
        self.std = mx.array(std, dtype=mx.float32).reshape(-1, 1, 1)
        self.v2 = v2

    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            if not isinstance(img.data, mx.array):
                raise TypeError(
                    "NormalizeAPI expects MLX arrays; call ToTensorAPI first"
                )
            img.data = (img.data.astype(mx.float32) - self.mean) / self.std
            cur_h, cur_w = img.data.shape[-2:]
            norm = mx.array([cur_w, cur_h, cur_w, cur_h], dtype=mx.float32)
            for obj in img.objects:
                obj.bbox = box_xyxy_to_cxcywh(_as_float_array(obj.bbox)) / norm

        for query in datapoint.find_queries:
            cur_h, cur_w = datapoint.images[query.image_id].data.shape[-2:]
            norm = mx.array([cur_w, cur_h, cur_w, cur_h], dtype=mx.float32)
            if query.input_bbox is not None:
                query.input_bbox = (
                    box_xyxy_to_cxcywh(_as_float_array(query.input_bbox)) / norm
                )
            if query.input_points is not None:
                query.input_points = _as_float_array(query.input_points) / mx.array(
                    [cur_w, cur_h, 1.0], dtype=mx.float32
                )

        return datapoint


class ComposeAPI:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, datapoint, **kwargs):
        for transform in self.transforms:
            datapoint = transform(datapoint, **kwargs)
        return datapoint

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(transform)
        format_string += "\n)"
        return format_string


class RandomSizeCropAPI:
    def __init__(
        self,
        min_size: int,
        max_size: int,
        respect_boxes: bool,
        consistent_transform: bool,
        respect_input_boxes: bool = True,
        v2: bool = False,
        recompute_box_from_mask: bool = False,
    ):
        _check_no_v2(v2, "RandomSizeCropAPI")
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes
        self.respect_input_boxes = respect_input_boxes
        self.consistent_transform = consistent_transform
        self.v2 = v2
        self.recompute_box_from_mask = recompute_box_from_mask

    def _sample_no_respect_boxes(self, image_data):
        width, height = _image_size(image_data)
        crop_width = random.randint(self.min_size, min(width, self.max_size))
        crop_height = random.randint(self.min_size, min(height, self.max_size))
        top = random.randint(0, max(height - crop_height, 0))
        left = random.randint(0, max(width - crop_width, 0))
        return top, left, crop_height, crop_width

    def _sample_respect_boxes(self, image_data, boxes, points, min_box_size=10.0):
        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        points_np = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(boxes_np) == 0 and len(points_np) == 0:
            return self._sample_no_respect_boxes(image_data)

        img_width, img_height = _image_size(image_data)
        min_width = min(img_width, self.min_size)
        min_height = min(img_height, self.min_size)
        max_width = min(img_width, self.max_size)
        max_height = min(img_height, self.max_size)

        left_requirements = []
        top_requirements = []
        right_requirements = []
        bottom_requirements = []
        if len(boxes_np):
            left_requirements.append(boxes_np[:, 0] + min_box_size)
            top_requirements.append(boxes_np[:, 1] + min_box_size)
            right_requirements.append(boxes_np[:, 2] - min_box_size)
            bottom_requirements.append(boxes_np[:, 3] - min_box_size)
        if len(points_np):
            left_requirements.append(points_np[:, 0] + 1.0)
            top_requirements.append(points_np[:, 1] + 1.0)
            right_requirements.append(points_np[:, 0])
            bottom_requirements.append(points_np[:, 1])

        min_x = min(img_width, float(np.concatenate(left_requirements).max()))
        min_y = min(img_height, float(np.concatenate(top_requirements).max()))
        max_x = max(0.0, float(np.concatenate(right_requirements).min()))
        max_y = max(0.0, float(np.concatenate(bottom_requirements).min()))

        min_width = max(min_width, min_x - max_x)
        min_height = max(min_height, min_y - max_y)
        crop_width = int(round(random.uniform(min_width, max(min_width, max_width))))
        crop_height = int(
            round(random.uniform(min_height, max(min_height, max_height)))
        )
        if min_x > max_x:
            left = random.uniform(
                max(0, min_x - crop_width), max(max_x, max(0, min_x - crop_width))
            )
        else:
            left = random.uniform(
                max(0, min_x - crop_width + 1),
                max(max_x - 1, max(0, min_x - crop_width + 1)),
            )
        if min_y > max_y:
            top = random.uniform(
                max(0, min_y - crop_height), max(max_y, max(0, min_y - crop_height))
            )
        else:
            top = random.uniform(
                max(0, min_y - crop_height + 1),
                max(max_y - 1, max(0, min_y - crop_height + 1)),
            )
        return top, left, crop_height, crop_width

    def _collect_boxes_and_points(self, datapoint, image_index=None):
        boxes = []
        points = []
        image_indices = (
            range(len(datapoint.images)) if image_index is None else [image_index]
        )
        if self.respect_boxes:
            for idx in image_indices:
                boxes.extend(
                    _to_numpy(obj.bbox).reshape(-1, 4)
                    for obj in datapoint.images[idx].objects
                )
        if self.respect_input_boxes:
            boxes.extend(
                _to_numpy(query.input_bbox).reshape(-1, 4)
                for query in datapoint.find_queries
                if query.input_bbox is not None
                and (image_index is None or query.image_id == image_index)
            )
        points.extend(
            _to_numpy(query.input_points).reshape(-1, 3)[:, :2]
            for query in datapoint.find_queries
            if query.input_points is not None
            and (image_index is None or query.image_id == image_index)
        )
        boxes_np = (
            np.concatenate(boxes, axis=0)
            if boxes
            else np.empty((0, 4), dtype=np.float32)
        )
        points_np = (
            np.concatenate(points, axis=0)
            if points
            else np.empty((0, 2), dtype=np.float32)
        )
        return boxes_np, points_np

    def __call__(self, datapoint, **kwargs):
        del kwargs
        respect_any = self.respect_boxes or self.respect_input_boxes
        if self.consistent_transform:
            width, height = _image_size(datapoint.images[0].data)
            for image in datapoint.images:
                if _image_size(image.data) != (width, height):
                    raise AssertionError(
                        "consistent RandomSizeCropAPI requires equal image sizes."
                    )
            if respect_any:
                boxes, points = self._collect_boxes_and_points(datapoint)
                crop_param = self._sample_respect_boxes(
                    datapoint.images[0].data, boxes, points
                )
            else:
                crop_param = self._sample_no_respect_boxes(datapoint.images[0].data)
            for index in range(len(datapoint.images)):
                datapoint = crop(
                    datapoint,
                    index,
                    crop_param,
                    v2=self.v2,
                    check_validity=self.respect_boxes,
                    check_input_validity=self.respect_input_boxes,
                    recompute_box_from_mask=self.recompute_box_from_mask,
                )
            return datapoint

        for index, image in enumerate(datapoint.images):
            if respect_any:
                boxes, points = self._collect_boxes_and_points(datapoint, index)
                crop_param = self._sample_respect_boxes(image.data, boxes, points)
            else:
                crop_param = self._sample_no_respect_boxes(image.data)
            datapoint = crop(
                datapoint,
                index,
                crop_param,
                v2=self.v2,
                check_validity=self.respect_boxes,
                check_input_validity=self.respect_input_boxes,
                recompute_box_from_mask=self.recompute_box_from_mask,
            )
        return datapoint


class CenterCropAPI:
    def __init__(self, size, consistent_transform, recompute_box_from_mask=False):
        self.size = size
        self.consistent_transform = consistent_transform
        self.recompute_box_from_mask = recompute_box_from_mask

    def _sample_crop(self, image_width, image_height):
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop_top, crop_left, crop_height, crop_width

    def __call__(self, datapoint, **kwargs):
        del kwargs
        if self.consistent_transform:
            width, height = _image_size(datapoint.images[0].data)
            for image in datapoint.images:
                if _image_size(image.data) != (width, height):
                    raise AssertionError(
                        "consistent CenterCropAPI requires equal image sizes."
                    )
            crop_param = self._sample_crop(width, height)
            for index in range(len(datapoint.images)):
                datapoint = crop(
                    datapoint,
                    index,
                    crop_param,
                    recompute_box_from_mask=self.recompute_box_from_mask,
                )
            return datapoint

        for index, image in enumerate(datapoint.images):
            width, height = _image_size(image.data)
            datapoint = crop(
                datapoint,
                index,
                self._sample_crop(width, height),
                recompute_box_from_mask=self.recompute_box_from_mask,
            )
        return datapoint


class RandomMosaicVideoAPI:
    def __init__(self, *args, **kwargs):
        raise_unsupported("RandomMosaicVideoAPI")


def random_mosaic_frame(*args, **kwargs):
    raise_unsupported("random_mosaic_frame")


class RandomGrayscale:
    def __init__(self, consistent_transform, p=0.5):
        self.p = p
        self.consistent_transform = consistent_transform

    @staticmethod
    def _grayscale(data):
        return _apply_pil_transform(
            data, lambda image: ImageOps.grayscale(image).convert("RGB")
        )

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        if self.consistent_transform:
            if random.random() < self.p:
                for image in datapoint.images:
                    image.data = self._grayscale(image.data)
            return datapoint
        for image in datapoint.images:
            if random.random() < self.p:
                image.data = self._grayscale(image.data)
        return datapoint


class ColorJitter:
    def __init__(self, consistent_transform, brightness, contrast, saturation, hue):
        self.consistent_transform = consistent_transform
        self.brightness = self._range(brightness, lower_bound=0.0)
        self.contrast = self._range(contrast, lower_bound=0.0)
        self.saturation = self._range(saturation, lower_bound=0.0)
        self.hue = hue if isinstance(hue, list) or hue is None else [-hue, hue]

    @staticmethod
    def _range(value, lower_bound=None):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        low, high = 1 - value, 1 + value
        if lower_bound is not None:
            low = max(lower_bound, low)
        return [low, high]

    def _params(self):
        order = [0, 1, 2, 3]
        random.shuffle(order)
        return (
            order,
            random.uniform(*self.brightness) if self.brightness is not None else None,
            random.uniform(*self.contrast) if self.contrast is not None else None,
            random.uniform(*self.saturation) if self.saturation is not None else None,
            random.uniform(*self.hue) if self.hue is not None else None,
        )

    @staticmethod
    def _adjust_hue(image: PILImage.Image, hue_factor: float):
        if hue_factor == 0:
            return image
        hsv = np.asarray(image.convert("HSV")).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(hue_factor * 255)) % 255
        return PILImage.fromarray(hsv, mode="HSV").convert(image.mode)

    def _apply(self, data, params):
        order, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
            params
        )

        def transform(image):
            out = image
            for fn_id in order:
                if fn_id == 0 and brightness_factor is not None:
                    out = ImageEnhance.Brightness(out).enhance(brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    out = ImageEnhance.Contrast(out).enhance(contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    out = ImageEnhance.Color(out).enhance(saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    out = self._adjust_hue(out, hue_factor)
            return out

        return _apply_pil_transform(data, transform)

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        params = self._params() if self.consistent_transform else None
        for image in datapoint.images:
            image.data = self._apply(image.data, params or self._params())
        return datapoint


class RandomAffine:
    def __init__(
        self,
        degrees,
        consistent_transform,
        scale=None,
        translate=None,
        shear=None,
        image_mean=(123, 116, 103),
        log_warning=True,
        num_tentatives=1,
        image_interpolation="bicubic",
    ):
        self.degrees = degrees if isinstance(degrees, list) else [-degrees, degrees]
        self.scale = scale
        self.shear = (
            shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        )
        self.translate = translate
        self.fill_img = (
            tuple(image_mean) if isinstance(image_mean, (list, tuple)) else image_mean
        )
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives
        if image_interpolation == "bicubic":
            self.image_interpolation = PILImage.Resampling.BICUBIC
        elif image_interpolation == "bilinear":
            self.image_interpolation = PILImage.Resampling.BILINEAR
        else:
            raise ValueError(f"Unsupported image_interpolation={image_interpolation!r}")

    def _sample_params(self, image_data):
        width, height = _image_size(image_data)
        angle = random.uniform(*self.degrees)
        scale = random.uniform(*self.scale) if self.scale is not None else 1.0
        if self.translate is None:
            translations = (0.0, 0.0)
        else:
            max_dx = self.translate[0] * width
            max_dy = self.translate[1] * height
            translations = (
                random.uniform(-max_dx, max_dx),
                random.uniform(-max_dy, max_dy),
            )
        if self.shear is None:
            shear = None
        elif len(self.shear) == 2:
            shear = (random.uniform(*self.shear), 0.0)
        else:
            shear = (
                random.uniform(self.shear[0], self.shear[1]),
                random.uniform(self.shear[2], self.shear[3]),
            )
        return angle, translations, scale, shear

    def _apply_one(self, datapoint, image_index, params):
        image = datapoint.images[image_index]
        width, height = _image_size(image.data)
        forward, inverse = _affine_matrices(width, height, *params)
        image.data = _apply_affine_image(
            image.data, inverse, self.image_interpolation, self.fill_img
        )

        for obj in image.objects:
            if obj.segment is not None:
                obj.segment = _apply_affine_mask(obj.segment, inverse)
                obj.bbox, obj.area = get_bbox_xyxy_abs_coords_from_mask(obj.segment)
            else:
                obj.bbox = _transform_boxes_xyxy(obj.bbox, forward, width, height)

        for query in datapoint.find_queries:
            if query.semantic_target is not None:
                query.semantic_target = _apply_affine_mask(
                    query.semantic_target, inverse
                )
            if query.image_id == image_index and query.input_bbox is not None:
                query.input_bbox = _transform_boxes_xyxy(
                    query.input_bbox, forward, width, height
                )
            if query.image_id == image_index and query.input_points is not None:
                query.input_points = _transform_points(
                    query.input_points, forward, width, height
                )
        return datapoint

    def transform_datapoint(self, datapoint: Datapoint):
        params = (
            self._sample_params(datapoint.images[0].data)
            if self.consistent_transform
            else None
        )
        for image_index, image in enumerate(datapoint.images):
            datapoint = self._apply_one(
                datapoint,
                image_index,
                params or self._sample_params(image.data),
            )
        return datapoint

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        for _ in range(self.num_tentatives):
            return self.transform_datapoint(datapoint)
        return datapoint


class RandomResizedCrop:
    def __init__(
        self,
        consistent_transform,
        size,
        scale=None,
        ratio=None,
        log_warning=True,
        num_tentatives=4,
        keep_aspect_ratio=False,
    ):
        if isinstance(size, int):
            self.size = (size, size)
        elif len(size) == 1:
            self.size = (size[0], size[0])
        elif len(size) != 2:
            raise ValueError("Please provide only two dimensions (h, w) for size.")
        else:
            self.size = tuple(size)
        self.scale = scale if scale is not None else (0.08, 1.0)
        self.ratio = ratio if ratio is not None else (3.0 / 4.0, 4.0 / 3.0)
        self.consistent_transform = consistent_transform
        self.log_warning = log_warning
        self.num_tentatives = num_tentatives
        self.keep_aspect_ratio = keep_aspect_ratio

    def _sample_crop(self, image):
        width, height = _image_size(image)
        area = width * height
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect = float(np.exp(random.uniform(*log_ratio)))
            crop_width = int(round((target_area * aspect) ** 0.5))
            crop_height = int(round((target_area / aspect) ** 0.5))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = random.randint(0, height - crop_height)
                left = random.randint(0, width - crop_width)
                return top, left, crop_height, crop_width
        side = min(width, height)
        top = (height - side) // 2
        left = (width - side) // 2
        return top, left, side, side

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        crop_param = (
            self._sample_crop(datapoint.images[0].data)
            if self.consistent_transform
            else None
        )
        for index, image in enumerate(datapoint.images):
            params = crop_param or self._sample_crop(image.data)
            datapoint = crop(datapoint, index, params, recompute_box_from_mask=True)
            datapoint = resize(datapoint, index, self.size)
        return datapoint


class ResizeToMaxIfAbove:
    def __init__(self, max_size=None):
        self.max_size = max_size

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        if self.max_size is None:
            return datapoint
        width, height = _image_size(datapoint.images[0].data)
        if height <= self.max_size and width <= self.max_size:
            return datapoint
        if height >= width:
            new_height = self.max_size
            new_width = int(round(self.max_size * width / height))
        else:
            new_height = int(round(self.max_size * height / width))
            new_width = self.max_size
        for index in range(len(datapoint.images)):
            datapoint = resize(datapoint, index, (new_width, new_height))
        return datapoint


def get_bbox_xyxy_abs_coords_from_mask(mask):
    mask_array = mx.array(mask, dtype=mx.bool_)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, :, :]
    boxes = masks_to_boxes(mask_array)
    area = mx.sum(mask_array.astype(mx.float32), axis=(1, 2))
    return boxes.reshape(-1, 4), area


class MotionBlur:
    def __init__(self, kernel_size=5, consistent_transform=True, p=0.5):
        if kernel_size % 2 != 1:
            raise AssertionError("Kernel size must be odd.")
        self.kernel_size = kernel_size
        self.consistent_transform = consistent_transform
        self.p = p

    def _filter(self):
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        direction = random.choice(["horizontal", "vertical", "diagonal"])
        center = self.kernel_size // 2
        if direction == "horizontal":
            kernel[center, :] = 1.0
        elif direction == "vertical":
            kernel[:, center] = 1.0
        else:
            np.fill_diagonal(kernel, 1.0)
        kernel /= kernel.sum()
        return ImageFilter.Kernel(
            (self.kernel_size, self.kernel_size),
            kernel.reshape(-1).tolist(),
            scale=1.0,
        )

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        if random.random() >= self.p:
            return datapoint
        image_filter = self._filter() if self.consistent_transform else None
        for image in datapoint.images:
            filt = image_filter or self._filter()
            image.data = _apply_pil_transform(image.data, lambda img: img.filter(filt))
        return datapoint


class LargeScaleJitter:
    def __init__(
        self,
        scale_range=(0.1, 2.0),
        aspect_ratio_range=(0.75, 1.33),
        crop_size=(640, 640),
        consistent_transform=True,
        p=0.5,
    ):
        self.scale_range = scale_range
        self.aspect_ratio_range = aspect_ratio_range
        self.crop_size = crop_size
        self.consistent_transform = consistent_transform
        self.p = p

    def _sample(self, image_data):
        width, height = _image_size(image_data)
        target_area = width * height * random.uniform(*self.scale_range)
        aspect = float(
            np.exp(
                random.uniform(
                    np.log(self.aspect_ratio_range[0]),
                    np.log(self.aspect_ratio_range[1]),
                )
            )
        )
        crop_width = min(width, max(1, int(round((target_area * aspect) ** 0.5))))
        crop_height = min(height, max(1, int(round((target_area / aspect) ** 0.5))))
        left = random.randint(0, max(0, width - crop_width))
        top = random.randint(0, max(0, height - crop_height))
        return top, left, crop_height, crop_width

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        if random.random() >= self.p:
            return datapoint
        crop_param = (
            self._sample(datapoint.images[0].data)
            if self.consistent_transform
            else None
        )
        for index, image in enumerate(datapoint.images):
            params = crop_param or self._sample(image.data)
            datapoint = crop(datapoint, index, params, recompute_box_from_mask=True)
            datapoint = resize(datapoint, index, self.crop_size)
        return datapoint
