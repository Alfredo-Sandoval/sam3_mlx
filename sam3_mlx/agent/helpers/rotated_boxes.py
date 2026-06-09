"""Rotated box compatibility helpers."""

from __future__ import annotations

import numpy as np

from sam3_mlx.agent.helpers.boxes import BoxMode, Boxes


def pairwise_iou_rotated(boxes1, boxes2):
    """Approximate rotated IoU through enclosing XYXY boxes."""
    b1 = Boxes(BoxMode.convert(np.asarray(boxes1), BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS))
    b2 = Boxes(BoxMode.convert(np.asarray(boxes2), BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS))
    from .boxes import pairwise_iou as pairwise_iou_axis_aligned

    return pairwise_iou_axis_aligned(b1, b2)


class RotatedBoxes(Boxes):
    """NumPy-backed rotated boxes in ``(xc, yc, w, h, angle_degrees)`` format."""

    def __init__(self, tensor):
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.size == 0:
            tensor = tensor.reshape((-1, 5)).astype(np.float32)
        if tensor.ndim != 2 or tensor.shape[-1] != 5:
            raise AssertionError(tensor.shape)
        self.tensor = tensor

    def clone(self) -> "RotatedBoxes":
        return RotatedBoxes(self.tensor.copy())

    def to(self, device):
        if str(device) not in ("cpu", "None"):
            raise ValueError("NumPy RotatedBoxes only support CPU storage")
        return self.clone()

    def area(self) -> np.ndarray:
        return self.tensor[:, 2] * self.tensor[:, 3]

    def normalize_angles(self) -> None:
        self.tensor[:, 4] = ((self.tensor[:, 4] + 180.0) % 360.0) - 180.0

    def clip(self, box_size, clip_angle_threshold: float = 1.0) -> None:
        height, width = box_size
        near_axis_aligned = np.abs(np.sin(np.deg2rad(self.tensor[:, 4]))) <= np.sin(
            np.deg2rad(clip_angle_threshold)
        )
        axis_boxes = BoxMode.convert(
            self.tensor[near_axis_aligned], BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS
        )
        if len(axis_boxes):
            clipped = Boxes(axis_boxes)
            clipped.clip((height, width))
            xywh = BoxMode.convert(clipped.tensor, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            self.tensor[near_axis_aligned, :4] = BoxMode.convert(
                xywh, BoxMode.XYWH_ABS, BoxMode.XYWHA_ABS
            )[:, :4]

    def nonempty(self, threshold: float = 0.0) -> np.ndarray:
        return (self.tensor[:, 2] > threshold) & (self.tensor[:, 3] > threshold)

    def __getitem__(self, item) -> "RotatedBoxes":
        if isinstance(item, int):
            return RotatedBoxes(self.tensor[item].reshape(1, -1))
        selected = self.tensor[item]
        if selected.ndim != 2:
            raise AssertionError(
                f"Indexing on RotatedBoxes with {item} returned {selected.shape}"
            )
        return RotatedBoxes(selected)

    def __repr__(self) -> str:
        return "RotatedBoxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size, boundary_threshold: int = 0) -> np.ndarray:
        axis = Boxes(BoxMode.convert(self.tensor, BoxMode.XYWHA_ABS, BoxMode.XYXY_ABS))
        return axis.inside_box(box_size, boundary_threshold)

    def get_centers(self) -> np.ndarray:
        return self.tensor[:, :2]

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0] *= scale_x
        self.tensor[:, 1] *= scale_y
        self.tensor[:, 2] *= scale_x
        self.tensor[:, 3] *= scale_y

    @classmethod
    def cat(cls, boxes_list):
        if not boxes_list:
            return cls(np.empty((0, 5), dtype=np.float32))
        return cls(np.concatenate([box.tensor for box in boxes_list], axis=0))


def pairwise_iou(boxes1: RotatedBoxes, boxes2: RotatedBoxes) -> np.ndarray:
    return pairwise_iou_rotated(boxes1.tensor, boxes2.tensor)
