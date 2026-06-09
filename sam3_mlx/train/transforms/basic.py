# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Target-dict transforms ported from official SAM3 to PIL/NumPy/MLX."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable

import mlx.core as mx
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from sam3_mlx.model.box_ops import box_xyxy_to_cxcywh

MLX_BASIC_TRANSFORMS_BASE_COMMIT = "629029d376426710c263b606aa137ec17dc55a94"


def _is_mlx_array(value) -> bool:
    return isinstance(value, mx.array)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if _is_mlx_array(value):
        mx.eval(value)
    return np.asarray(value)


def _as_float_array(value) -> mx.array:
    if _is_mlx_array(value):
        return value.astype(mx.float32)
    return mx.array(value, dtype=mx.float32)


def _restore_array(value: np.ndarray, like):
    if _is_mlx_array(like):
        return mx.array(value)
    return value


def _image_size(image) -> tuple[int, int]:
    if isinstance(image, PILImage.Image):
        return image.size
    array = _to_numpy(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        return int(array.shape[2]), int(array.shape[1])
    if array.ndim >= 2:
        return int(array.shape[1]), int(array.shape[0])
    raise TypeError(f"Unsupported image shape: {array.shape}.")


def _pil_from_array_image(image):
    array = _to_numpy(image)
    chw = array.ndim == 3 and array.shape[0] in (1, 3, 4)
    if chw:
        array = array.transpose(1, 2, 0)
    if np.issubdtype(array.dtype, np.floating):
        if array.size and array.min() >= 0.0 and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    return PILImage.fromarray(array), chw


def _restore_image(pil_image: PILImage.Image, like, chw: bool | None = None):
    if isinstance(like, PILImage.Image):
        return pil_image
    array = np.asarray(pil_image)
    if array.ndim == 2:
        array = array[:, :, None]
    if chw:
        array = array.transpose(2, 0, 1)
    if _is_mlx_array(like):
        return mx.array(array, dtype=like.dtype)
    return array.astype(getattr(like, "dtype", array.dtype), copy=False)


def _crop_image(image, top: int, left: int, height: int, width: int):
    if isinstance(image, PILImage.Image):
        return ImageOps.crop(
            image,
            border=(left, top, image.width - left - width, image.height - top - height),
        )
    pil_image, chw = _pil_from_array_image(image)
    cropped = ImageOps.crop(
        pil_image,
        border=(
            left,
            top,
            pil_image.width - left - width,
            pil_image.height - top - height,
        ),
    )
    return _restore_image(cropped, image, chw)


def _resize_image(image, size_hw: tuple[int, int]):
    height, width = size_hw
    if isinstance(image, PILImage.Image):
        return image.resize((width, height), resample=PILImage.Resampling.BILINEAR)
    pil_image, chw = _pil_from_array_image(image)
    resized = pil_image.resize((width, height), resample=PILImage.Resampling.BILINEAR)
    return _restore_image(resized, image, chw)


def _pad_image(image, padding):
    if len(padding) == 2:
        border = (0, 0, padding[0], padding[1])
    else:
        border = (padding[0], padding[1], padding[2], padding[3])
    if isinstance(image, PILImage.Image):
        return ImageOps.expand(image, border=border, fill=0)
    pil_image, chw = _pil_from_array_image(image)
    padded = ImageOps.expand(pil_image, border=border, fill=0)
    return _restore_image(padded, image, chw)


def _hflip_image(image):
    if isinstance(image, PILImage.Image):
        return ImageOps.mirror(image)
    pil_image, chw = _pil_from_array_image(image)
    flipped = ImageOps.mirror(pil_image)
    return _restore_image(flipped, image, chw)


def _resize_masks(masks, size_hw: tuple[int, int]):
    masks_np = _to_numpy(masks).astype(np.uint8, copy=False)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    height, width = size_hw
    resized = []
    for mask in masks_np:
        mask_img = PILImage.fromarray(mask)
        resized.append(
            np.asarray(
                mask_img.resize((width, height), resample=PILImage.Resampling.NEAREST),
                dtype=np.uint8,
            )
        )
    return _restore_array(np.stack(resized, axis=0).astype(bool), masks)


def _pad_masks(masks, padding):
    masks_np = _to_numpy(masks)
    if len(padding) == 2:
        left, top, right, bottom = 0, 0, padding[0], padding[1]
    else:
        left, top, right, bottom = padding
    padded = np.pad(masks_np, ((0, 0), (top, bottom), (left, right)))
    return _restore_array(padded, masks)


def _take_first_dim(value, keep):
    keep_np = _to_numpy(keep).astype(bool, copy=False)
    return _restore_array(_to_numpy(value)[keep_np], value)


def _filter_fields(target: dict, fields: list[str], keep):
    for field in fields:
        if field in target:
            target[field] = _take_first_dim(target[field], keep)


def crop(image, target, region):
    top, left, height, width = [int(round(v)) for v in region]
    cropped_image = _crop_image(image, top, left, height, width)

    if target is None:
        return cropped_image, None
    target = target.copy()
    target["size"] = mx.array([height, width], dtype=mx.int64)

    fields = ["labels", "area", "iscrowd", "positive_map"]

    if "boxes" in target:
        boxes = _to_numpy(target["boxes"]).astype(np.float32, copy=False)
        max_size = np.array([width, height], dtype=np.float32)
        cropped_boxes = boxes - np.array([left, top, left, top], dtype=np.float32)
        cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = np.maximum(cropped_boxes, 0)
        area = np.prod(cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :], axis=1)
        target["boxes"] = _restore_array(cropped_boxes.reshape(-1, 4), target["boxes"])
        target["area"] = _restore_array(
            area.astype(np.float32), target.get("area", area)
        )
        fields.append("boxes")

    if "input_boxes" in target:
        boxes = _to_numpy(target["input_boxes"]).astype(np.float32, copy=False)
        max_size = np.array([width, height], dtype=np.float32)
        cropped_boxes = boxes - np.array([left, top, left, top], dtype=np.float32)
        cropped_boxes = np.minimum(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = np.maximum(cropped_boxes, 0)
        target["input_boxes"] = _restore_array(
            cropped_boxes.reshape(-1, 4), target["input_boxes"]
        )

    if "masks" in target:
        masks = _to_numpy(target["masks"])
        cropped = masks[:, top : top + height, left : left + width]
        target["masks"] = _restore_array(cropped, target["masks"])
        fields.append("masks")

    if "boxes" in target:
        cropped_boxes = _to_numpy(target["boxes"]).reshape(-1, 2, 2)
        keep = np.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1)
        _filter_fields(target, fields, keep)
    elif "masks" in target:
        keep = _to_numpy(target["masks"]).reshape(target["masks"].shape[0], -1).any(1)
        _filter_fields(target, fields, keep)

    return cropped_image, target


def hflip(image, target):
    flipped_image = _hflip_image(image)
    width, _height = _image_size(image)

    if target is None:
        return flipped_image, None
    target = target.copy()

    if "boxes" in target:
        boxes = _to_numpy(target["boxes"]).astype(np.float32, copy=False)
        flipped = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1])
        flipped = flipped + np.array([width, 0, width, 0], dtype=np.float32)
        target["boxes"] = _restore_array(flipped, target["boxes"])

    if "input_boxes" in target:
        boxes = _to_numpy(target["input_boxes"]).astype(np.float32, copy=False)
        flipped = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1])
        flipped = flipped + np.array([width, 0, width, 0], dtype=np.float32)
        target["input_boxes"] = _restore_array(flipped, target["input_boxes"])

    if "masks" in target:
        target["masks"] = _restore_array(
            np.flip(_to_numpy(target["masks"]), -1), target["masks"]
        )

    if "text_input" in target:
        target["text_input"] = (
            target["text_input"]
            .replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        )

    return flipped_image, target


def resize(image, target, size, max_size=None, square=False):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        width, height = image_size
        if max_size is not None:
            min_original_size = float(min((width, height)))
            max_original_size = float(max((width, height)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (width <= height and width == size) or (height <= width and height == size):
            return height, width
        if width < height:
            out_width = size
            out_height = int(size * height / width)
        else:
            out_height = size
            out_width = int(size * width / height)
        return out_height, out_width

    if square:
        size_hw = (size, size)
    elif isinstance(size, (list, tuple)):
        size_hw = tuple(size[::-1])
    else:
        size_hw = get_size_with_aspect_ratio(_image_size(image), size, max_size)

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
        target["boxes"] = _restore_array(
            _to_numpy(target["boxes"]) * scale, target["boxes"]
        )
    if "input_boxes" in target:
        target["input_boxes"] = _restore_array(
            _to_numpy(target["input_boxes"]) * scale, target["input_boxes"]
        )
    if "area" in target:
        target["area"] = _restore_array(
            _to_numpy(target["area"]) * (ratio_width * ratio_height), target["area"]
        )
    target["size"] = mx.array([new_height, new_width], dtype=mx.int64)

    if "masks" in target:
        target["masks"] = _resize_masks(target["masks"], size_hw)

    return rescaled_image, target


def pad(image, target, padding):
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
            target["boxes"] = _restore_array(
                _to_numpy(target["boxes"]) + offset, target["boxes"]
            )
        if "input_boxes" in target:
            target["input_boxes"] = _restore_array(
                _to_numpy(target["input_boxes"]) + offset, target["input_boxes"]
            )

    if "masks" in target:
        target["masks"] = _pad_masks(target["masks"], padding)
    return padded_image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
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

    def __call__(self, img: PILImage.Image, target: dict):
        image_width, image_height = _image_size(img)
        if self.respect_boxes and target is not None and "boxes" in target:
            boxes = _to_numpy(target["boxes"])
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
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = _image_size(img)
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    def __init__(self, sizes, max_size=None, square=False):
        if isinstance(sizes, int):
            sizes = (sizes,)
        if not isinstance(sizes, Iterable):
            raise AssertionError("sizes must be an int or iterable.")
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, square=self.square)


class RandomPad:
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class PadToSize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
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
    def __call__(self, img, target):
        return img, target


class RandomSelect:
    def __init__(self, transforms1=None, transforms2=None, p=0.5):
        self.transforms1 = transforms1 or Identity()
        self.transforms2 = transforms2 or Identity()
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor:
    def __call__(self, img, target):
        if isinstance(img, PILImage.Image):
            array = np.asarray(img)
            if array.ndim == 2:
                array = array[:, :, None]
            image = mx.array(array.transpose(2, 0, 1), dtype=mx.float32) / 255.0
            return image, target
        array = _to_numpy(img)
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            array = array.transpose(2, 0, 1)
        image = mx.array(array, dtype=mx.float32)
        if array.dtype == np.uint8:
            image = image / 255.0
        return image, target


class RandomErasing:
    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=0.0,
        inplace=False,
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def __call__(self, img, target):
        if random.random() >= self.p:
            return img, target
        image = mx.array(img, dtype=mx.float32)
        if image.ndim != 3:
            raise ValueError("RandomErasing expects a CHW image array.")
        channels, height, width = image.shape
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
                return mx.array(image_np, dtype=image.dtype), target
        return img, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mx.array(mean, dtype=mx.float32).reshape(-1, 1, 1)
        self.std = mx.array(std, dtype=mx.float32).reshape(-1, 1, 1)

    def __call__(self, image, target=None):
        if isinstance(image, PILImage.Image):
            image, target = ToTensor()(image, target)
        image = (mx.array(image, dtype=mx.float32) - self.mean) / self.std
        if target is None:
            return image, None
        target = target.copy()
        height, width = image.shape[-2:]
        norm = mx.array([width, height, width, height], dtype=mx.float32)
        if "boxes" in target:
            target["boxes"] = (
                box_xyxy_to_cxcywh(_as_float_array(target["boxes"])) / norm
            )
        if "input_boxes" in target:
            target["input_boxes"] = (
                box_xyxy_to_cxcywh(_as_float_array(target["input_boxes"])) / norm
            )
        return image, target


class RemoveDifficult:
    def __init__(self, enabled=False):
        self.remove_difficult = enabled

    def __call__(self, image, target=None):
        if target is None:
            return image, None
        target = target.copy()
        if "iscrowd" not in target:
            return image, target
        keep = ~_to_numpy(target["iscrowd"]).astype(bool) | (not self.remove_difficult)
        for field in ("boxes", "labels", "iscrowd"):
            if field in target:
                target[field] = _take_first_dim(target[field], keep)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for transform in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(transform)
        format_string += "\n)"
        return format_string


def get_random_resize_scales(size, min_size, rounded):
    stride = 128 if rounded else 32
    min_size = int(stride * math.ceil(min_size / stride))
    return list(range(min_size, size + 1, stride))


def get_random_resize_max_size(size, ratio=5 / 3):
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
