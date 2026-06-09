"""Box helper structures ported to NumPy."""

from __future__ import annotations

import math
from enum import IntEnum, unique
from typing import List, Tuple, Union

import numpy as np

from sam3_mlx._unsupported import raise_unsupported


_RawBoxType = Union[List[float], Tuple[float, ...], np.ndarray]


@unique
class BoxMode(IntEnum):
    """Enum of different ways to represent a box."""

    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3
    XYWHA_ABS = 4

    @staticmethod
    def convert(
        box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode"
    ) -> _RawBoxType:
        """Convert boxes between official SAM3/Detectron box formats."""
        if from_mode == to_mode:
            return box

        original_type = type(box)
        single_box = isinstance(box, (list, tuple))
        arr = np.asarray(box, dtype=np.float64)
        if single_box:
            if arr.size not in (4, 5):
                raise AssertionError("BoxMode.convert expects 4 or 5 values")
            arr = arr.reshape(1, -1)
        else:
            arr = arr.copy()

        if to_mode in (BoxMode.XYXY_REL, BoxMode.XYWH_REL) or from_mode in (
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ):
            raise AssertionError("Relative mode not yet supported!")

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            if arr.shape[-1] != 5:
                raise AssertionError("The last dimension must be 5 for XYWHA format")
            w = arr[:, 2].copy()
            h = arr[:, 3].copy()
            a = arr[:, 4]
            c = np.abs(np.cos(a * math.pi / 180.0))
            s = np.abs(np.sin(a * math.pi / 180.0))
            new_w = c * w + s * h
            new_h = c * h + s * w
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h
            arr = arr[:, :4]
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            arr = np.concatenate(
                [arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)], axis=1
            )
        elif to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
            arr[:, 2] += arr[:, 0]
            arr[:, 3] += arr[:, 1]
        elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
            arr[:, 2] -= arr[:, 0]
            arr[:, 3] -= arr[:, 1]
        else:
            raise_unsupported(
                "sam3_mlx.agent.helpers.boxes.BoxMode.convert",
                reason="port-gap",
                detail=(
                    f"Conversion from BoxMode {from_mode} to {to_mode} is "
                    "not supported yet."
                ),
                alternative="absolute XYXY/XYWH/XYWHA conversions",
            )

        if single_box:
            return original_type(arr.reshape(-1).tolist())
        return arr


class Boxes:
    """A NumPy-backed list of boxes in ``XYXY`` absolute coordinates."""

    def __init__(self, tensor):
        tensor = np.asarray(tensor, dtype=np.float32)
        if tensor.size == 0:
            tensor = tensor.reshape((-1, 4)).astype(np.float32)
        if tensor.ndim != 2 or tensor.shape[-1] != 4:
            raise AssertionError(tensor.shape)
        self.tensor = tensor

    def clone(self) -> "Boxes":
        return Boxes(self.tensor.copy())

    def to(self, device):
        if str(device) not in ("cpu", "None"):
            raise ValueError("NumPy Boxes only support CPU storage")
        return self.clone()

    def area(self) -> np.ndarray:
        box = self.tensor
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    def clip(self, box_size: Tuple[int, int]) -> None:
        if not np.isfinite(self.tensor).all():
            raise AssertionError("Box tensor contains infinite or NaN!")
        h, w = box_size
        self.tensor[:, 0::2] = np.clip(self.tensor[:, 0::2], 0, w)
        self.tensor[:, 1::2] = np.clip(self.tensor[:, 1::2], 0, h)

    def nonempty(self, threshold: float = 0.0) -> np.ndarray:
        widths = self.tensor[:, 2] - self.tensor[:, 0]
        heights = self.tensor[:, 3] - self.tensor[:, 1]
        return (widths > threshold) & (heights > threshold)

    def __getitem__(self, item) -> "Boxes":
        if isinstance(item, int):
            return Boxes(self.tensor[item].reshape(1, -1))
        selected = self.tensor[item]
        if selected.ndim != 2:
            raise AssertionError(
                f"Indexing on Boxes with {item} returned {selected.shape}"
            )
        return Boxes(selected)

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(
        self, box_size: Tuple[int, int], boundary_threshold: int = 0
    ) -> np.ndarray:
        height, width = box_size
        return (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )

    def get_centers(self) -> np.ndarray:
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        if not isinstance(boxes_list, (list, tuple)):
            raise AssertionError("boxes_list must be a list or tuple")
        if len(boxes_list) == 0:
            return cls(np.empty((0, 4), dtype=np.float32))
        if not all(isinstance(box, Boxes) for box in boxes_list):
            raise AssertionError("All entries must be Boxes")
        return cls(np.concatenate([box.tensor for box in boxes_list], axis=0))

    @property
    def device(self):
        return "cpu"

    def __iter__(self):
        yield from self.tensor


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> np.ndarray:
    boxes1_arr, boxes2_arr = boxes1.tensor, boxes2.tensor
    width_height = np.minimum(boxes1_arr[:, None, 2:], boxes2_arr[:, 2:]) - np.maximum(
        boxes1_arr[:, None, :2], boxes2_arr[:, :2]
    )
    width_height = np.clip(width_height, 0, None)
    return width_height.prod(axis=2)


def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> np.ndarray:
    area1 = boxes1.area()
    area2 = boxes2.area()
    inter = pairwise_intersection(boxes1, boxes2)
    denom = area1[:, None] + area2 - inter
    return np.where((inter > 0) & (denom > 0), inter / denom, 0.0)


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> np.ndarray:
    area2 = boxes2.area()
    inter = pairwise_intersection(boxes1, boxes2)
    return np.where((inter > 0) & (area2 > 0), inter / area2, 0.0)


def pairwise_point_box_distance(points, boxes: Boxes):
    points = np.asarray(points, dtype=np.float32)
    x = points[:, None, 0]
    y = points[:, None, 1]
    x0 = boxes.tensor[None, :, 0]
    y0 = boxes.tensor[None, :, 1]
    x1 = boxes.tensor[None, :, 2]
    y1 = boxes.tensor[None, :, 3]
    return np.stack([x - x0, y - y0, x1 - x, y1 - y], axis=2)


def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> np.ndarray:
    if len(boxes1) != len(boxes2):
        raise AssertionError(
            f"boxlists should have the same number of entries, got {len(boxes1)}, {len(boxes2)}"
        )
    area1 = boxes1.area()
    area2 = boxes2.area()
    lt = np.maximum(boxes1.tensor[:, :2], boxes2.tensor[:, :2])
    rb = np.minimum(boxes1.tensor[:, 2:], boxes2.tensor[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, 0] * wh[:, 1]
    denom = area1 + area2 - inter
    return np.where(denom > 0, inter / denom, 0.0)
