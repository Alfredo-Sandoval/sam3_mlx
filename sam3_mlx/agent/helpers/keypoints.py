"""Keypoint helper compatibility surface backed by NumPy."""

from __future__ import annotations

import numpy as np

from sam3_mlx.agent._unsupported import raise_unsupported


class Keypoints:
    def __init__(self, keypoints):
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[2] != 3:
            raise AssertionError(keypoints.shape)
        self.tensor = keypoints

    def __len__(self):
        return int(self.tensor.shape[0])

    def to(self, *args, **kwargs):
        return Keypoints(self.tensor.copy())

    @property
    def device(self):
        return "cpu"

    def to_heatmap(self, boxes, heatmap_size):
        return _keypoints_to_heatmap(self.tensor, boxes, heatmap_size)

    def __getitem__(self, item):
        if isinstance(item, int):
            return Keypoints(self.tensor[item][None, :, :])
        return Keypoints(self.tensor[item])

    def __repr__(self):
        return f"{self.__class__.__name__}(num_instances={len(self)})"

    @staticmethod
    def cat(keypoints_list):
        if not keypoints_list:
            raise AssertionError("keypoints_list must be non-empty")
        return Keypoints(np.concatenate([k.tensor for k in keypoints_list], axis=0))


def _keypoints_to_heatmap(keypoints, rois, heatmap_size):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    rois = (
        rois.tensor if hasattr(rois, "tensor") else np.asarray(rois, dtype=np.float32)
    )
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
    return linear, valid


def heatmaps_to_keypoints(maps, rois):
    raise_unsupported("agent.helpers.keypoints.heatmaps_to_keypoints")
