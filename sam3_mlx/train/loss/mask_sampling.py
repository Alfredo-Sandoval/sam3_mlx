# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""PointRend-style mask point sampling ported from official SAM3 to MLX."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias, cast

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from sam3_mlx.model.grid_sample_mlx import grid_sample
from sam3_mlx.sam.tensor_protocols import ArrayConstructor, ArrayMethods

MLX_MASK_SAMPLING_BASE_COMMIT = "dc33741d86020f34c73f9534deabff1007cdd886"
ArrayValue: TypeAlias = (
    mx.array
    | NDArray[np.generic]
    | bool
    | int
    | float
    | Sequence[int]
    | Sequence[float]
)
_mx_array = cast(ArrayConstructor, getattr(mx, "array"))


def _as_float_array(value: ArrayValue) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    return _mx_array(value, dtype=mx.float32)


def _validate_grid_sample_kwargs(kwargs: Mapping[str, object]) -> None:
    unsupported = sorted(set(kwargs) - {"align_corners", "mode", "padding_mode"})
    if unsupported:
        raise TypeError(f"Unsupported point_sample kwargs: {', '.join(unsupported)}.")
    if kwargs.get("align_corners", False) not in (False, None):
        raise NotImplementedError("point_sample only supports align_corners=False.")
    if kwargs.get("mode", "bilinear") != "bilinear":
        raise NotImplementedError("point_sample only supports bilinear sampling.")
    if kwargs.get("padding_mode", "zeros") != "zeros":
        raise NotImplementedError("point_sample only supports zero padding.")


def point_sample(
    input: ArrayValue, point_coords: ArrayValue, **kwargs: object
) -> mx.array:
    """Sample ``NCHW`` features at normalized ``[0, 1]`` point coordinates."""

    _validate_grid_sample_kwargs(kwargs)
    input = _as_float_array(input)
    point_coords = _as_float_array(point_coords)
    if input.ndim != 4:
        raise ValueError(f"input must have shape (N, C, H, W), got {input.shape}.")
    if point_coords.shape[0] != input.shape[0] or point_coords.shape[-1] != 2:
        raise ValueError(
            "point_coords must share the input batch dimension and end in XY pairs."
        )

    add_dim = False
    if point_coords.ndim == 3:
        add_dim = True
        point_coords = point_coords[:, :, None, :]
    elif point_coords.ndim != 4:
        raise ValueError(
            "point_coords must have shape (N, P, 2) or (N, Hgrid, Wgrid, 2)."
        )

    normalized_point_coords = 2.0 * point_coords - 1.0
    output = grid_sample(
        cast(ArrayMethods, input).transpose(0, 2, 3, 1), normalized_point_coords
    )
    output = cast(ArrayMethods, output).transpose(0, 3, 1, 2)
    if add_dim:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    logits: mx.array,
    uncertainty_func: Callable[[mx.array], mx.array],
    num_points: int,
    oversample_ratio: int,
    importance_sample_ratio: float,
) -> mx.array:
    """Sample coordinates according to uncertainty plus random exploration."""

    if isinstance(num_points, bool):
        raise TypeError("num_points must be an integer, not bool.")
    if isinstance(oversample_ratio, bool):
        raise TypeError("oversample_ratio must be an integer, not bool.")
    if oversample_ratio < 1:
        raise AssertionError("oversample_ratio must be >= 1.")
    if not 0 <= importance_sample_ratio <= 1:
        raise AssertionError("importance_sample_ratio must be in [0, 1].")

    logits = _as_float_array(logits)
    if logits.ndim != 4:
        raise ValueError(f"logits must have shape (N, C, H, W), got {logits.shape}.")
    num_boxes = logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    point_coords = mx.random.uniform(shape=(num_boxes, num_sampled, 2))
    if num_uncertain_points > 0:
        point_logits = point_sample(logits, point_coords, align_corners=False)
        point_uncertainties = uncertainty_func(point_logits)
        if point_uncertainties.shape[:2] != (num_boxes, 1):
            raise ValueError(
                "uncertainty_func must return an array with shape (N, 1, P)."
            )
        uncertain_order = mx.argsort(point_uncertainties[:, 0, :], axis=1)[
            :, -num_uncertain_points:
        ]
        shifts = num_sampled * mx.arange(num_boxes, dtype=mx.int64)
        flat_indices = mx.reshape(uncertain_order + shifts[:, None], (-1,))
        uncertain_coords = mx.reshape(
            mx.reshape(point_coords, (-1, 2))[flat_indices],
            (num_boxes, num_uncertain_points, 2),
        )
    else:
        uncertain_coords = mx.zeros((num_boxes, 0, 2), dtype=logits.dtype)

    if num_random_points > 0:
        random_coords = mx.random.uniform(shape=(num_boxes, num_random_points, 2))
        return mx.concat([uncertain_coords, random_coords], axis=1)
    return uncertain_coords


def calculate_uncertainty(logits: mx.array) -> mx.array:
    """Official class-agnostic mask uncertainty: negative absolute logit."""

    logits = _as_float_array(logits)
    if logits.shape[1] != 1:
        raise AssertionError("calculate_uncertainty expects logits.shape[1] == 1.")
    return -mx.abs(logits)


__all__ = [
    "MLX_MASK_SAMPLING_BASE_COMMIT",
    "calculate_uncertainty",
    "get_uncertain_point_coords_with_randomness",
    "point_sample",
]
