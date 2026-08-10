# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Target-dict transforms ported from official SAM3 to PIL/NumPy/MLX."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Protocol, TypeAlias, overload

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageOps

from sam3_mlx.train.transforms._array_contracts import (
    ArrayData,
    ImageInput,
    ImageSize,
    Padding,
    as_float_array as _as_float_array,
    box_xyxy_to_cxcywh as _box_xyxy_to_cxcywh,
    image_size as _image_size,
    is_mlx_array as _is_mlx_array,
    mx_array as _mx_array,
    mx_dtype as _mx_dtype,
    mx_ops as _mx_ops,
    mx_shape as _mx_shape,
    restore_array,
    to_numpy as _to_numpy,
)

MLX_BASIC_TRANSFORMS_BASE_COMMIT = "629029d376426710c263b606aa137ec17dc55a94"

TargetValue: TypeAlias = ArrayData | str
TargetDict: TypeAlias = dict[str, TargetValue]
ResizeArg: TypeAlias = int | Sequence[int]


class _ImageTransform(Protocol):
    def __call__(
        self, image: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]: ...


@overload
def _restore_array(value: NDArray[np.generic], like: mx.array) -> mx.array: ...


@overload
def _restore_array(
    value: NDArray[np.generic], like: NDArray[np.generic]
) -> NDArray[np.generic]: ...


def _restore_array(value: NDArray[np.generic], like: ArrayData) -> ArrayData:
    return restore_array(value, like, preserve_dtype=False)


def _target_array(target: TargetDict, key: str) -> ArrayData:
    value = target[key]
    if isinstance(value, str):
        raise TypeError(f"Target field {key!r} must be array-like.")
    return value


def _pil_from_array_image(image: ArrayData) -> tuple[PILImage.Image, bool]:
    array = _to_numpy(image)
    chw = array.ndim == 3 and array.shape[0] in (1, 3, 4)
    if chw:
        array = array.transpose(1, 2, 0)
    if np.issubdtype(array.dtype, np.floating):
        float_array = array.astype(np.float32, copy=False)
        if array.size and array.min() >= 0.0 and array.max() <= 1.0:
            float_array = float_array * 255.0
        array = np.clip(float_array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    return PILImage.fromarray(array), chw


@overload
def _restore_image(
    pil_image: PILImage.Image, like: PILImage.Image, chw: bool | None = None
) -> PILImage.Image: ...


@overload
def _restore_image(
    pil_image: PILImage.Image, like: ArrayData, chw: bool | None = None
) -> ArrayData: ...


def _restore_image(
    pil_image: PILImage.Image, like: ImageInput, chw: bool | None = None
) -> ImageInput:
    if isinstance(like, PILImage.Image):
        return pil_image
    array = np.asarray(pil_image)
    if array.ndim == 2:
        array = array[:, :, None]
    if chw:
        array = array.transpose(2, 0, 1)
    if _is_mlx_array(like):
        return _mx_array(array, dtype=_mx_dtype(like))
    if isinstance(like, np.ndarray):
        return array.astype(like.dtype, copy=False)
    raise TypeError(f"Unsupported image type: {type(like)!r}")


def _crop_image(
    image: ImageInput, top: int, left: int, height: int, width: int
) -> ImageInput:
    if isinstance(image, PILImage.Image):
        return image.crop((left, top, left + width, top + height))
    pil_image, chw = _pil_from_array_image(image)
    cropped = pil_image.crop((left, top, left + width, top + height))
    return _restore_image(cropped, image, chw)


def _resize_image(image: ImageInput, size_hw: ImageSize) -> ImageInput:
    height, width = size_hw
    if isinstance(image, PILImage.Image):
        return image.resize((width, height), resample=PILImage.Resampling.BILINEAR)
    pil_image, chw = _pil_from_array_image(image)
    resized = pil_image.resize((width, height), resample=PILImage.Resampling.BILINEAR)
    return _restore_image(resized, image, chw)


def _pad_image(image: ImageInput, padding: Padding) -> ImageInput:
    if len(padding) == 2:
        border = (0, 0, padding[0], padding[1])
    else:
        border = (padding[0], padding[1], padding[2], padding[3])
    if isinstance(image, PILImage.Image):
        return ImageOps.expand(image, border=border, fill=0)
    pil_image, chw = _pil_from_array_image(image)
    padded = ImageOps.expand(pil_image, border=border, fill=0)
    return _restore_image(padded, image, chw)


def _hflip_image(image: ImageInput) -> ImageInput:
    if isinstance(image, PILImage.Image):
        return ImageOps.mirror(image)
    pil_image, chw = _pil_from_array_image(image)
    flipped = ImageOps.mirror(pil_image)
    return _restore_image(flipped, image, chw)


def _resize_masks(masks: ArrayData, size_hw: ImageSize) -> ArrayData:
    masks_np = _to_numpy(masks).astype(np.uint8, copy=False)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    height, width = size_hw
    resized: list[NDArray[np.uint8]] = []
    for mask in masks_np:
        mask_img = PILImage.fromarray(mask)
        resized.append(
            np.asarray(
                mask_img.resize((width, height), resample=PILImage.Resampling.NEAREST),
                dtype=np.uint8,
            )
        )
    return _restore_array(np.stack(resized, axis=0).astype(bool), masks)


def _pad_masks(masks: ArrayData, padding: Padding) -> ArrayData:
    masks_np = _to_numpy(masks)
    if len(padding) == 2:
        left, top, right, bottom = 0, 0, padding[0], padding[1]
    else:
        left, top, right, bottom = padding
    padded = np.pad(masks_np, ((0, 0), (top, bottom), (left, right)))
    return _restore_array(padded, masks)


def _take_first_dim(value: ArrayData, keep: ArrayData) -> ArrayData:
    keep_np = _to_numpy(keep).astype(bool, copy=False)
    return _restore_array(_to_numpy(value)[keep_np], value)


def _filter_fields(target: TargetDict, fields: Sequence[str], keep: ArrayData) -> None:
    for field in fields:
        if field in target:
            target[field] = _take_first_dim(_target_array(target, field), keep)


def crop(
    image: ImageInput,
    target: TargetDict | None,
    region: Sequence[int | float],
) -> tuple[ImageInput, TargetDict | None]:
    top, left, height, width = [int(round(v)) for v in region]
    cropped_image = _crop_image(image, top, left, height, width)

    if target is None:
        return cropped_image, None
    target = target.copy()
    target["size"] = mx.array([height, width], dtype=mx.int64)

    fields = ["labels", "area", "iscrowd", "positive_map"]

    if "boxes" in target:
        boxes_like = _target_array(target, "boxes")
        boxes = _to_numpy(boxes_like).astype(np.float32, copy=False)
        max_size = np.array([width, height], dtype=np.float32)
        cropped_boxes = boxes - np.array([left, top, left, top], dtype=np.float32)
        cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = np.maximum(cropped_boxes, 0)
        area = np.prod(cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :], axis=1)
        target["boxes"] = _restore_array(cropped_boxes.reshape(-1, 4), boxes_like)
        area_like = _target_array(target, "area") if "area" in target else area
        target["area"] = _restore_array(area.astype(np.float32), area_like)
        fields.append("boxes")

    if "input_boxes" in target:
        boxes_like = _target_array(target, "input_boxes")
        boxes = _to_numpy(boxes_like).astype(np.float32, copy=False)
        max_size = np.array([width, height], dtype=np.float32)
        cropped_boxes = boxes - np.array([left, top, left, top], dtype=np.float32)
        cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = np.maximum(cropped_boxes, 0)
        target["input_boxes"] = _restore_array(cropped_boxes.reshape(-1, 4), boxes_like)

    if "masks" in target:
        masks_like = _target_array(target, "masks")
        masks = _to_numpy(masks_like)
        cropped = masks[:, top : top + height, left : left + width]
        target["masks"] = _restore_array(cropped, masks_like)
        fields.append("masks")

    if "boxes" in target:
        cropped_boxes = (
            _to_numpy(_target_array(target, "boxes"))
            .astype(np.float32, copy=False)
            .reshape(-1, 2, 2)
        )
        keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
        _filter_fields(target, fields, keep)
    elif "masks" in target:
        masks = _target_array(target, "masks")
        masks_np = _to_numpy(masks)
        keep = np.asarray(
            masks_np.reshape(masks_np.shape[0], -1).any(axis=1),
            dtype=np.bool_,
        )
        _filter_fields(target, fields, keep)

    return cropped_image, target


def hflip(
    image: ImageInput, target: TargetDict | None
) -> tuple[ImageInput, TargetDict | None]:
    flipped_image = _hflip_image(image)
    width, _height = _image_size(image)

    if target is None:
        return flipped_image, None
    target = target.copy()

    if "boxes" in target:
        boxes_like = _target_array(target, "boxes")
        boxes = _to_numpy(boxes_like).astype(np.float32, copy=False)
        flipped = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1])
        flipped = flipped + np.array([width, 0, width, 0], dtype=np.float32)
        target["boxes"] = _restore_array(flipped, boxes_like)

    if "input_boxes" in target:
        boxes_like = _target_array(target, "input_boxes")
        boxes = _to_numpy(boxes_like).astype(np.float32, copy=False)
        flipped = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1])
        flipped = flipped + np.array([width, 0, width, 0], dtype=np.float32)
        target["input_boxes"] = _restore_array(flipped, boxes_like)

    if "masks" in target:
        masks_like = _target_array(target, "masks")
        target["masks"] = _restore_array(np.flip(_to_numpy(masks_like), -1), masks_like)

    text_input = target.get("text_input")
    if isinstance(text_input, str):
        target["text_input"] = (
            text_input.replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )

    return flipped_image, target


def resize(
    image: ImageInput,
    target: TargetDict | None,
    size: ResizeArg,
    max_size: int | None = None,
    square: bool = False,
) -> tuple[ImageInput, TargetDict | None]:
    def get_size_with_aspect_ratio(
        image_size: ImageSize, requested_size: int, max_size: int | None = None
    ) -> ImageSize:
        width, height = image_size
        if max_size is not None:
            min_original_size = float(min((width, height)))
            max_original_size = float(max((width, height)))
            if max_original_size / min_original_size * requested_size > max_size:
                requested_size = int(
                    round(max_size * min_original_size / max_original_size)
                )

        if (width <= height and width == requested_size) or (
            height <= width and height == requested_size
        ):
            return height, width
        if width < height:
            out_width = requested_size
            out_height = int(requested_size * height / width)
        else:
            out_height = requested_size
            out_width = int(requested_size * width / height)
        return out_height, out_width

    if square:
        if not isinstance(size, int):
            raise TypeError("square resize expects an integer size.")
        size_hw = (size, size)
    elif isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
        dims = list(size)
        size_hw = (int(dims[1]), int(dims[0]))
    else:
        size_hw = get_size_with_aspect_ratio(_image_size(image), int(size), max_size)

    old_width, old_height = _image_size(image)
    rescaled_image = _resize_image(image, size_hw)
    new_height, new_width = size_hw

    if target is None:
        return rescaled_image, None
    target = target.copy()
    ratio_width = float(new_width) / float(old_width)
    ratio_height = float(new_height) / float(old_height)
    scale = np.array(
        [ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32
    )

    if "boxes" in target:
        boxes_like = _target_array(target, "boxes")
        target["boxes"] = _restore_array(
            _to_numpy(boxes_like).astype(np.float32, copy=False) * scale,
            boxes_like,
        )
    if "input_boxes" in target:
        boxes_like = _target_array(target, "input_boxes")
        target["input_boxes"] = _restore_array(
            _to_numpy(boxes_like).astype(np.float32, copy=False) * scale, boxes_like
        )
    if "area" in target:
        area_like = _target_array(target, "area")
        target["area"] = _restore_array(
            _to_numpy(area_like).astype(np.float32, copy=False)
            * (ratio_width * ratio_height),
            area_like,
        )
    target["size"] = mx.array([new_height, new_width], dtype=mx.int64)

    if "masks" in target:
        target["masks"] = _resize_masks(_target_array(target, "masks"), size_hw)

    return rescaled_image, target


def pad(
    image: ImageInput, target: TargetDict | None, padding: Padding
) -> tuple[ImageInput, TargetDict | None]:
    padded_image = _pad_image(image, padding)

    if target is None:
        return padded_image, None
    target = target.copy()
    width, height = _image_size(padded_image)
    target["size"] = mx.array([height, width], dtype=mx.int64)

    if len(padding) == 4:
        offset = np.array(
            [padding[0], padding[1], padding[0], padding[1]], dtype=np.float32
        )
        if "boxes" in target:
            boxes_like = _target_array(target, "boxes")
            target["boxes"] = _restore_array(
                _to_numpy(boxes_like).astype(np.float32, copy=False) + offset,
                boxes_like,
            )
        if "input_boxes" in target:
            boxes_like = _target_array(target, "input_boxes")
            target["input_boxes"] = _restore_array(
                _to_numpy(boxes_like).astype(np.float32, copy=False) + offset,
                boxes_like,
            )

    if "masks" in target:
        target["masks"] = _pad_masks(_target_array(target, "masks"), padding)
    return padded_image, target


class RandomCrop:
    def __init__(self, size: Sequence[int]) -> None:
        self.size = (int(size[0]), int(size[1]))

    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        height, width = self.size
        image_width, image_height = _image_size(img)
        top = random.randint(0, max(image_height - height, 0))
        left = random.randint(0, max(image_width - width, 0))
        return crop(
            img, target, (top, left, min(height, image_height), min(width, image_width))
        )


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int, respect_boxes: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.respect_boxes = respect_boxes

    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        image_width, image_height = _image_size(img)
        if self.respect_boxes and target is not None and "boxes" in target:
            boxes = _to_numpy(_target_array(target, "boxes"))
            if len(boxes) > 0:
                min_width = min(image_width, self.min_size)
                min_height = min(image_height, self.min_size)
                max_width = min(image_width, self.max_size)
                max_height = min(image_height, self.max_size)
                min_left = min(image_width, float(boxes[:, 0].max()) + 10.0)
                min_top = min(image_height, float(boxes[:, 1].max()) + 10.0)
                max_left = max(0.0, float(boxes[:, 2].min()) - 10.0)
                max_top = max(0.0, float(boxes[:, 3].min()) - 10.0)
                width = int(round(random.uniform(min_width, max(min_width, max_width))))
                height = int(
                    round(random.uniform(min_height, max(min_height, max_height)))
                )
                left = int(
                    round(random.uniform(max(0, min_left - width), max(max_left, 0)))
                )
                top = int(
                    round(random.uniform(max(0, min_top - height), max(max_top, 0)))
                )
                return crop(img, target, (top, left, height, width))

        width = random.randint(self.min_size, min(image_width, self.max_size))
        height = random.randint(self.min_size, min(image_height, self.max_size))
        top = random.randint(0, max(image_height - height, 0))
        left = random.randint(0, max(image_width - width, 0))
        return crop(img, target, (top, left, height, width))


class CenterCrop:
    def __init__(self, size: Sequence[int]) -> None:
        self.size = (int(size[0]), int(size[1]))

    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        image_width, image_height = _image_size(img)
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    def __init__(
        self,
        sizes: int | Iterable[int],
        max_size: int | None = None,
        square: bool = False,
    ) -> None:
        if isinstance(sizes, int):
            sizes = (sizes,)
        self.sizes = [int(size) for size in sizes]
        self.max_size = max_size
        self.square = square

    def __call__(
        self, img: ImageInput, target: TargetDict | None = None
    ) -> tuple[ImageInput, TargetDict | None]:
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, square=self.square)


class RandomPad:
    def __init__(self, max_pad: int) -> None:
        self.max_pad = max_pad

    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class PadToSize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        width, height = _image_size(img)
        pad_x = self.size - width
        pad_y = self.size - height
        if pad_x < 0 or pad_y < 0:
            raise AssertionError("PadToSize size must be >= image dimensions.")
        pad_left = random.randint(0, pad_x)
        pad_right = pad_x - pad_left
        pad_top = random.randint(0, pad_y)
        pad_bottom = pad_y - pad_top
        return pad(img, target, (pad_left, pad_top, pad_right, pad_bottom))


class Identity:
    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        return img, target


class RandomSelect:
    def __init__(
        self,
        transforms1: _ImageTransform | None = None,
        transforms2: _ImageTransform | None = None,
        p: float = 0.5,
    ) -> None:
        self.transforms1 = transforms1 or Identity()
        self.transforms2 = transforms2 or Identity()
        self.p = p

    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[mx.array, TargetDict | None]:
        if isinstance(img, PILImage.Image):
            array = np.asarray(img)
            if array.ndim == 2:
                array = array[:, :, None]
            image = _mx_array(array.transpose(2, 0, 1), dtype=mx.float32) / 255.0
            return image, target
        array = _to_numpy(img)
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            array = array.transpose(2, 0, 1)
        image = _mx_array(array, dtype=mx.float32)
        if array.dtype == np.uint8:
            image = image / 255.0
        return image, target


class RandomErasing:
    def __init__(
        self,
        p: float = 0.5,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
        inplace: bool = False,
    ) -> None:
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(
        self, img: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        if random.random() >= self.p:
            return img, target
        image = _mx_array(_to_numpy(img), dtype=mx.float32)
        if _mx_ops(image).ndim != 3:
            raise ValueError("RandomErasing expects a CHW image array.")
        _channels, height, width = _mx_shape(image)
        area = height * width
        for _ in range(10):
            erase_area = random.uniform(*self.scale) * area
            aspect = math.exp(
                random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]))
            )
            erase_h = int(round(math.sqrt(erase_area * aspect)))
            erase_w = int(round(math.sqrt(erase_area / aspect)))
            if 0 < erase_h < height and 0 < erase_w < width:
                top = random.randint(0, height - erase_h)
                left = random.randint(0, width - erase_w)
                image_np = _to_numpy(image).copy()
                image_np[:, top : top + erase_h, left : left + erase_w] = self.value
                return _mx_array(image_np, dtype=_mx_dtype(image)), target
        return img, target


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = _mx_ops(_mx_array(mean, dtype=mx.float32)).reshape(-1, 1, 1)
        self.std = _mx_ops(_mx_array(std, dtype=mx.float32)).reshape(-1, 1, 1)

    def __call__(
        self, image: ImageInput, target: TargetDict | None = None
    ) -> tuple[mx.array, TargetDict | None]:
        if isinstance(image, PILImage.Image):
            image, target = ToTensor()(image, target)
        image = (_mx_array(_to_numpy(image), dtype=mx.float32) - self.mean) / self.std
        if target is None:
            return image, None
        target = target.copy()
        height, width = _mx_shape(image)[-2:]
        norm = _mx_array([width, height, width, height], dtype=mx.float32)
        if "boxes" in target:
            target["boxes"] = (
                _box_xyxy_to_cxcywh(_as_float_array(_target_array(target, "boxes")))
                / norm
            )
        if "input_boxes" in target:
            target["input_boxes"] = (
                _box_xyxy_to_cxcywh(
                    _as_float_array(_target_array(target, "input_boxes"))
                )
                / norm
            )
        return image, target


class RemoveDifficult:
    def __init__(self, enabled: bool = False) -> None:
        self.remove_difficult = enabled

    def __call__(
        self, image: ImageInput, target: TargetDict | None = None
    ) -> tuple[ImageInput, TargetDict | None]:
        if target is None:
            return image, None
        target = target.copy()
        if "iscrowd" not in target:
            return image, target
        keep = ~_to_numpy(_target_array(target, "iscrowd")).astype(bool) | (
            not self.remove_difficult
        )
        for field in ("boxes", "labels", "iscrowd"):
            if field in target:
                target[field] = _take_first_dim(_target_array(target, field), keep)
        return image, target


class Compose:
    def __init__(self, transforms: Sequence[_ImageTransform]) -> None:
        self.transforms = transforms

    def __call__(
        self, image: ImageInput, target: TargetDict | None
    ) -> tuple[ImageInput, TargetDict | None]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(transform)
        format_string += "\n)"
        return format_string


def get_random_resize_scales(size: int, min_size: int, rounded: bool) -> list[int]:
    stride = 128 if rounded else 32
    min_size = int(stride * math.ceil(min_size / stride))
    return list(range(min_size, size + 1, stride))


def get_random_resize_max_size(size: int, ratio: float = 5 / 3) -> int:
    return round(ratio * size)


__all__ = [
    "CenterCrop",
    "Compose",
    "Identity",
    "MLX_BASIC_TRANSFORMS_BASE_COMMIT",
    "Normalize",
    "PadToSize",
    "RandomCrop",
    "RandomErasing",
    "RandomHorizontalFlip",
    "RandomPad",
    "RandomResize",
    "RandomSelect",
    "RandomSizeCrop",
    "RemoveDifficult",
    "ToTensor",
    "crop",
    "get_random_resize_max_size",
    "get_random_resize_scales",
    "hflip",
    "pad",
    "resize",
]
