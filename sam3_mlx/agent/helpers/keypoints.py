"""Keypoint helper compatibility surface backed by NumPy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Never, cast

import numpy as np
from numpy.typing import NDArray

from sam3_mlx.agent._unsupported import raise_unsupported
from sam3_mlx.agent.helpers.boxes import Boxes


type FloatArray = NDArray[np.float32]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
type KeypointIndex = int | slice | Sequence[int] | NDArray[np.integer] | BoolArray


def _float_array(value: object) -> FloatArray:
    return np.asarray(value, dtype=np.float32)


def _heatmap_size(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("heatmap_size must be an integer, not bool.")
    return value


class Keypoints:
    def __init__(self, keypoints: object):
        keypoints = _float_array(keypoints)
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise AssertionError(keypoints.shape)
        self.tensor = keypoints

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def to(self, *args: object, **kwargs: object) -> "Keypoints":
        del args, kwargs
        return Keypoints(self.tensor.copy())

    @property
    def device(self) -> str:
        return "cpu"

    def to_heatmap(
        self, boxes: Boxes | object, heatmap_size: int
    ) -> tuple[IntArray, BoolArray]:
        return _keypoints_to_heatmap(self.tensor, boxes, heatmap_size)

    def __getitem__(self, item: KeypointIndex) -> "Keypoints":
        if isinstance(item, int):
            return Keypoints(self.tensor[item][None, :, :])
        return Keypoints(self.tensor[item])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self)})"

    @staticmethod
    def cat(keypoints_list: Sequence[object]) -> "Keypoints":
        if not keypoints_list:
            raise AssertionError("keypoints_list must be non-empty")
        if not all(isinstance(item, Keypoints) for item in keypoints_list):
            raise AssertionError("All entries must be Keypoints")
        keypoints = cast(Sequence[Keypoints], keypoints_list)
        return Keypoints(np.concatenate([item.tensor for item in keypoints], axis=0))


def _keypoints_to_heatmap(
    keypoints: object, rois: Boxes | object, heatmap_size: int
) -> tuple[IntArray, BoolArray]:
    keypoints = _float_array(keypoints)
    rois = (
        cast(FloatArray, rois.tensor) if isinstance(rois, Boxes) else _float_array(rois)
    )
    heatmap_size = _heatmap_size(heatmap_size)
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / np.maximum(rois[:, 2] - rois[:, 0], 1e-6)
    scale_y = heatmap_size / np.maximum(rois[:, 3] - rois[:, 1], 1e-6)
    x = np.floor((keypoints[:, :, 0] - offset_x[:, None]) * scale_x[:, None]).astype(
        np.int64
    )
    y = np.floor((keypoints[:, :, 1] - offset_y[:, None]) * scale_y[:, None]).astype(
        np.int64
    )
    valid = (
        (keypoints[:, :, 2] > 0)
        & (x >= 0)
        & (x < heatmap_size)
        & (y >= 0)
        & (y < heatmap_size)
    )
    linear = y * heatmap_size + x
    linear = np.where(valid, linear, 0)
    return cast(IntArray, linear), valid


def heatmaps_to_keypoints(maps: object, rois: object) -> Never:
    del maps, rois
    raise_unsupported("agent.helpers.keypoints.heatmaps_to_keypoints")
