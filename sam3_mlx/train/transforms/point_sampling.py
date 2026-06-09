# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Point-sampling transforms ported from official SAM3 to NumPy/MLX."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import mlx.core as mx
import numpy as np
from PIL import Image as PILImage


MLX_POINT_SAMPLING_BASE_COMMIT = "629029d376426710c263b606aa137ec17dc55a94"


def _is_mlx_array(value) -> bool:
    return isinstance(value, mx.array)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if _is_mlx_array(value):
        mx.eval(value)
    return np.asarray(value)


def _from_numpy(value: np.ndarray, like):
    if _is_mlx_array(like):
        return mx.array(value)
    return value


def _as_rank3_points(value) -> mx.array:
    points = mx.array(value, dtype=mx.float32)
    if points.ndim == 2:
        return points[None, :, :]
    if points.ndim == 3:
        return points
    raise ValueError(
        f"input points must have shape (N, 3) or (1, N, 3), got {points.shape}."
    )


def _decode_compressed_rle_counts(counts: str | bytes) -> list[int]:
    """Decode COCO's compact ASCII RLE count format.

    This mirrors the algorithm used by pycocotools. Counts are delta encoded
    after the first two runs.
    """

    if isinstance(counts, bytes):
        counts = counts.decode("ascii")

    decoded: list[int] = []
    index = 0
    while index < len(counts):
        shift = 0
        value = 0
        while True:
            char_value = ord(counts[index]) - 48
            index += 1
            value |= (char_value & 0x1F) << shift
            shift += 5
            if (char_value & 0x20) == 0:
                break
        if char_value & 0x10:
            value |= -1 << shift
        if len(decoded) > 1:
            value += decoded[-2]
        decoded.append(value)
    return decoded


def _decode_coco_rle(rle) -> np.ndarray:
    """Decode uncompressed or compressed COCO RLE to a boolean mask."""

    if isinstance(rle, np.ndarray):
        return np.asarray(rle, dtype=bool)
    if not isinstance(rle, dict):
        raise TypeError("COCO RLE must be a dict with 'size' and 'counts'.")
    if "size" not in rle or "counts" not in rle:
        raise ValueError("COCO RLE must contain 'size' and 'counts'.")

    height, width = [int(v) for v in rle["size"]]
    counts = rle["counts"]
    if isinstance(counts, (str, bytes)):
        counts = _decode_compressed_rle_counts(counts)
    elif isinstance(counts, Sequence):
        counts = [int(v) for v in counts]
    else:
        raise TypeError("COCO RLE counts must be a sequence, str, or bytes.")

    total = height * width
    flat = np.zeros((total,), dtype=np.uint8)
    offset = 0
    value = 0
    for count in counts:
        next_offset = min(offset + count, total)
        if value == 1 and next_offset > offset:
            flat[offset:next_offset] = 1
        offset = next_offset
        value = 1 - value
        if offset >= total:
            break
    if offset != total:
        raise ValueError(
            f"COCO RLE decoded to {offset} pixels, expected {total} for size {(height, width)}."
        )
    return flat.reshape((width, height)).T.astype(bool, copy=False)


def _distance_transform_inside_mask(mask: np.ndarray) -> np.ndarray:
    """Exact but simple Euclidean distance transform for data-loader masks."""

    mask_bool = mask.astype(bool, copy=False)
    background = np.argwhere(~mask_bool)
    foreground = np.argwhere(mask_bool)
    dist = np.zeros(mask_bool.shape, dtype=np.float32)
    if len(background) == 0:
        dist[mask_bool] = math.sqrt(mask_bool.shape[0] ** 2 + mask_bool.shape[1] ** 2)
        return dist
    if len(foreground) == 0:
        return dist

    # Keep the implementation dependency-free. This runs in data transforms and
    # centered sampling is usually requested for sparse prompt generation.
    for y, x in foreground:
        diff = background - np.array([y, x])
        dist[y, x] = float(np.sqrt(np.min(np.sum(diff * diff, axis=1))))
    return dist


def _mask_to_box_np(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def sample_points_from_rle(rle, n_points, mode, box=None, normalize=True):
    """Sample points from a COCO RLE mask using the official mode contract."""

    mask = np.ascontiguousarray(_decode_coco_rle(rle))
    points = sample_points_from_mask(mask, n_points, mode, box)
    if normalize:
        height, width = mask.shape
        points = points / np.array([width, height, 1.0], dtype=np.float32)[None, :]
    return points


def sample_points_from_mask(mask, n_points, mode, box=None):
    if mode == "centered":
        return center_positive_sample(mask, n_points)
    if mode == "random_mask":
        return uniform_positive_sample(mask, n_points)
    if mode == "random_box":
        if box is None:
            raise AssertionError("'random_box' mode requires a provided box.")
        return uniform_sample_from_box(mask, box, n_points)
    raise ValueError(f"Unknown point sampling mode {mode}.")


def uniform_positive_sample(mask, n_points):
    """Sample positive integer-pixel points uniformly from a binary mask."""

    mask_np = _to_numpy(mask).astype(bool, copy=False)
    mask_points = np.stack(np.nonzero(mask_np), axis=0).transpose(1, 0)
    if len(mask_points) == 0:
        raise AssertionError("Can't sample positive points from an empty mask.")
    selected_idxs = np.random.randint(low=0, high=len(mask_points), size=n_points)
    selected_points = mask_points[selected_idxs][:, ::-1]
    labels = np.ones((len(selected_points), 1), dtype=selected_points.dtype)
    return np.concatenate([selected_points, labels], axis=1)


def center_positive_sample(mask, n_points):
    """Sample points farthest from mask edges and previously sampled points."""

    padded_mask = np.pad(_to_numpy(mask).astype(np.uint8, copy=False), 1)
    points = []
    for _ in range(n_points):
        if np.max(padded_mask) <= 0:
            raise AssertionError("Can't sample positive points from an empty mask.")
        dist = _distance_transform_inside_mask(padded_mask)
        point = np.unravel_index(dist.argmax(), dist.shape)
        padded_mask[point[0], point[1]] = 0
        points.append(point[::-1])
    points = np.stack(points, axis=0) - 1
    labels = np.ones((len(points), 1), dtype=points.dtype)
    return np.concatenate([points, labels], axis=1)


def uniform_sample_from_box(mask, box, n_points):
    """Sample integer points uniformly from an unnormalized XYXY box."""

    mask_np = _to_numpy(mask)
    int_box = np.ceil(_to_numpy(box)).astype(np.int64)
    x0, y0, x1, y1 = int_box.tolist()
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Cannot sample from an empty box: {box!r}")
    x = np.random.randint(low=x0, high=x1, size=n_points)
    y = np.random.randint(low=y0, high=y1, size=n_points)
    y = np.clip(y, 0, mask_np.shape[0] - 1)
    x = np.clip(x, 0, mask_np.shape[1] - 1)
    labels = mask_np[y, x]
    return np.stack([x, y, labels], axis=1)


def rescale_box_xyxy(box, factor, imsize=None):
    """Rescale an unnormalized XYXY box around its center."""

    box_np = _to_numpy(box).astype(np.float32, copy=False)
    cx = (box_np[0] + box_np[2]) / 2
    cy = (box_np[1] + box_np[3]) / 2
    width = box_np[2] - box_np[0]
    height = box_np[3] - box_np[1]
    new_width = factor * width
    new_height = factor * height
    new_box = np.array(
        [
            cx - new_width / 2,
            cy - new_height / 2,
            cx + new_width / 2,
            cy + new_height / 2,
        ],
        dtype=np.float32,
    )
    if imsize is not None:
        new_box[[0, 2]] = np.clip(new_box[[0, 2]], 0, imsize[1])
        new_box[[1, 3]] = np.clip(new_box[[1, 3]], 0, imsize[0])
    return _from_numpy(new_box, box)


def noise_box(box, im_size, box_noise_std, box_noise_max, min_box_area):
    """Apply official-style relative Gaussian box noise."""

    if box_noise_std <= 0.0:
        return box
    box_np = _to_numpy(box).astype(np.float32, copy=False)
    width = box_np[2] - box_np[0]
    height = box_np[3] - box_np[1]
    noise = box_noise_std * np.random.normal(size=(4,)).astype(np.float32)
    noise = noise * np.array([width, height, width, height], dtype=np.float32)
    if box_noise_max is not None:
        noise = np.clip(noise, -box_noise_max, box_noise_max)
    input_box = box_np + noise
    img_clamp = np.array(
        [im_size[1], im_size[0], im_size[1], im_size[0]], dtype=np.float32
    )
    input_box = np.maximum(input_box, np.zeros_like(input_box))
    input_box = np.minimum(input_box, img_clamp)
    if (input_box[2] - input_box[0]) * (input_box[3] - input_box[1]) <= min_box_area:
        return box
    return _from_numpy(input_box, box)


class RandomGeometricInputsAPI:
    """Replace geometric query prompts with sampled points or boxes."""

    def __init__(
        self,
        num_points,
        box_chance,
        box_noise_std=0.0,
        box_noise_max=None,
        minimum_box_area=0.0,
        resample_box_from_mask=False,
        point_sample_mode="random_mask",
        sample_box_scale_factor=1.0,
        geometric_query_str="geometric",
        concat_points=False,
    ):
        self.num_points = num_points
        if not isinstance(self.num_points, int):
            low, high = self.num_points
            self.num_points = (low, high + 1)
        self.box_chance = box_chance
        self.box_noise_std = box_noise_std
        self.box_noise_max = box_noise_max
        self.minimum_box_area = minimum_box_area
        self.resample_box_from_mask = resample_box_from_mask
        if point_sample_mode not in {"centered", "random_mask", "random_box"}:
            raise AssertionError("Unknown point sample mode.")
        self.point_sample_mode = point_sample_mode
        self.geometric_query_str = geometric_query_str
        self.concat_points = concat_points
        self.sample_box_scale_factor = sample_box_scale_factor

    def _sample_num_points_and_if_box(self):
        if isinstance(self.num_points, tuple):
            n_points = random.randrange(self.num_points[0], self.num_points[1])
        else:
            n_points = self.num_points
        use_box = self.box_chance > 0.0 and random.random() < self.box_chance
        n_points -= int(use_box)
        return n_points, use_box

    def _get_original_box(self, target_object):
        if not self.resample_box_from_mask:
            return target_object.bbox
        if target_object.segment is None:
            raise ValueError("resample_box_from_mask requires target_object.segment.")
        return _from_numpy(
            _mask_to_box_np(_to_numpy(target_object.segment)),
            target_object.bbox,
        )

    def _get_target_object(self, datapoint, query):
        img = datapoint.images[query.image_id]
        targets = query.object_ids_output
        if len(targets) != 1:
            raise AssertionError(
                "Geometric queries only support a single target object."
            )
        return img.objects[targets[0]]

    def __call__(self, datapoint, **kwargs):
        del kwargs
        for query in datapoint.find_queries:
            if query.query_text != self.geometric_query_str:
                continue

            target_object = self._get_target_object(datapoint, query)
            n_points, use_box = self._sample_num_points_and_if_box()
            box = self._get_original_box(target_object)
            mask = target_object.segment
            if mask is None:
                raise ValueError(
                    "Geometric point sampling requires target_object.segment."
                )

            if n_points > 0:
                sample_box = (
                    rescale_box_xyxy(
                        box,
                        self.sample_box_scale_factor,
                        _to_numpy(mask).shape,
                    )
                    if self.sample_box_scale_factor != 1.0
                    else box
                )
                input_points = sample_points_from_mask(
                    _to_numpy(mask),
                    n_points,
                    self.point_sample_mode,
                    _to_numpy(sample_box),
                )
                input_points = mx.array(input_points, dtype=mx.float32)[None, :, :]
                if self.concat_points and query.input_points is not None:
                    input_points = mx.concat(
                        [_as_rank3_points(query.input_points), input_points],
                        axis=1,
                    )
            else:
                input_points = query.input_points if self.concat_points else None

            if use_box:
                height, width = datapoint.images[query.image_id].size
                input_box = noise_box(
                    box,
                    (height, width),
                    box_noise_std=self.box_noise_std,
                    box_noise_max=self.box_noise_max,
                    min_box_area=self.minimum_box_area,
                )
                input_box = mx.array(input_box, dtype=mx.float32)[None, :]
                query.input_bbox_label = mx.ones((1,), dtype=mx.bool_)
            else:
                input_box = query.input_bbox if self.concat_points else None
                if input_box is None:
                    query.input_bbox_label = None
                else:
                    num_boxes = mx.array(input_box).reshape(-1, 4).shape[0]
                    if (
                        query.input_bbox_label is None
                        or np.size(_to_numpy(query.input_bbox_label)) != num_boxes
                    ):
                        query.input_bbox_label = mx.ones((num_boxes,), dtype=mx.bool_)

            query.input_points = input_points
            query.input_bbox = input_box
        return datapoint


class RandomizeInputBbox:
    """Apply official-style noise to existing input boxes."""

    def __init__(
        self,
        box_noise_std=0.0,
        box_noise_max=None,
        minimum_box_area=0.0,
    ):
        self.box_noise_std = box_noise_std
        self.box_noise_max = box_noise_max
        self.minimum_box_area = minimum_box_area

    def __call__(self, datapoint, **kwargs):
        del kwargs
        for query in datapoint.find_queries:
            if query.input_bbox is None:
                continue
            img = datapoint.images[query.image_id].data
            if isinstance(img, PILImage.Image):
                width, height = img.size
            elif _is_mlx_array(img):
                height, width = img.shape[-2:]
            else:
                raise TypeError(f"Unsupported image type: {type(img)!r}")

            boxes = mx.array(query.input_bbox, dtype=mx.float32).reshape(-1, 4)
            if boxes.shape[0] == 0:
                continue
            noised_boxes = [
                mx.array(
                    noise_box(
                        boxes[box_id],
                        (height, width),
                        box_noise_std=self.box_noise_std,
                        box_noise_max=self.box_noise_max,
                        min_box_area=self.minimum_box_area,
                    ),
                    dtype=mx.float32,
                )
                for box_id in range(boxes.shape[0])
            ]
            query.input_bbox = mx.stack(noised_boxes, axis=0).reshape(
                query.input_bbox.shape
            )
        return datapoint


__all__ = [
    "MLX_POINT_SAMPLING_BASE_COMMIT",
    "RandomGeometricInputsAPI",
    "RandomizeInputBbox",
    "center_positive_sample",
    "noise_box",
    "rescale_box_xyxy",
    "sample_points_from_mask",
    "sample_points_from_rle",
    "uniform_positive_sample",
    "uniform_sample_from_box",
]
