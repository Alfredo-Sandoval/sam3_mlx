"""Mask helper structures ported to NumPy/PIL."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from sam3_mlx.agent.helpers.boxes import Boxes


type Array = NDArray[np.generic]
type BoolArray = NDArray[np.bool_]
type FloatArray = NDArray[np.float64]
type Polygon = NDArray[np.generic]
type PolygonInstances = list[list[Polygon]]
type MaskIndex = int | slice | Sequence[int] | NDArray[np.integer] | BoolArray


def _array(value: object) -> Array:
    return cast(Array, np.asarray(value))


def _boxes_array(boxes: object) -> Array:
    return cast(Array, boxes.tensor) if isinstance(boxes, Boxes) else _array(boxes)


def _rounded_box(box: object) -> tuple[int, int, int, int]:
    numeric = np.asarray(box, dtype=np.float64)
    values = np.round(numeric).astype(np.int64).reshape(-1)
    if values.size != 4:
        raise ValueError("Boxes must contain exactly four coordinates.")
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


def polygon_area(x: Array, y: Array) -> float:
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygons_to_bitmask(
    polygons: Sequence[Polygon], height: int, width: int
) -> BoolArray:
    """Rasterize polygon coordinates into a bool mask."""
    if len(polygons) == 0:
        return np.zeros((height, width), dtype=bool)
    image = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        coords = np.asarray(polygon, dtype=float).reshape(-1, 2)
        if len(coords) >= 3:
            draw.polygon([tuple(point) for point in coords], outline=1, fill=1)
    return cast(BoolArray, np.asarray(image, dtype=bool))


def rasterize_polygons_within_box(
    polygons: list[Polygon], box: Array, mask_size: int
) -> BoolArray:
    """Rasterize polygons cropped/rescaled into a square mask."""
    w, h = box[2] - box[0], box[3] - box[1]
    polygons = copy.deepcopy(polygons)
    ratio_h = mask_size / max(h, 0.1)
    ratio_w = mask_size / max(w, 0.1)
    for polygon in polygons:
        polygon[0::2] = (polygon[0::2] - box[0]) * ratio_w
        polygon[1::2] = (polygon[1::2] - box[1]) * ratio_h
    return polygons_to_bitmask(polygons, mask_size, mask_size)


class BitMasks:
    """Store segmentation masks as a bool ``N,H,W`` NumPy array."""

    def __init__(self, tensor: object):
        tensor = cast(BoolArray, np.asarray(tensor, dtype=bool))
        if tensor.ndim != 3:
            raise AssertionError(tensor.shape)
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, *args: object, **kwargs: object) -> "BitMasks":
        del args, kwargs
        return BitMasks(self.tensor.copy())

    @property
    def device(self) -> str:
        return "cpu"

    def __getitem__(self, item: MaskIndex) -> "BitMasks":
        if isinstance(item, int):
            return BitMasks(self.tensor[item][None, :, :])
        selected = self.tensor[item]
        if selected.ndim != 3:
            raise AssertionError(
                f"Indexing on BitMasks with {item} returned {selected.shape}"
            )
        return BitMasks(selected)

    def __iter__(self) -> Iterator[BoolArray]:
        yield from self.tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self.tensor)})"

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def nonempty(self) -> BoolArray:
        return cast(
            BoolArray, self.tensor.reshape(self.tensor.shape[0], -1).any(axis=1)
        )

    @staticmethod
    def from_polygon_masks(
        polygon_masks: "PolygonMasks" | PolygonInstances,
        height: int,
        width: int,
    ) -> "BitMasks":
        if isinstance(polygon_masks, PolygonMasks):
            polygon_masks = polygon_masks.polygons
        masks = [
            polygons_to_bitmask(polygon, height, width) for polygon in polygon_masks
        ]
        if masks:
            return BitMasks(np.stack(masks, axis=0))
        return BitMasks(np.empty((0, height, width), dtype=bool))

    @staticmethod
    def from_roi_masks(
        roi_masks: "ROIMasks", boxes: object, height: int, width: int
    ) -> "BitMasks":
        return roi_masks.to_bitmasks(boxes, height, width)

    def crop_and_resize(self, boxes: object, mask_size: int) -> BoolArray:
        boxes_arr = _boxes_array(boxes)
        if len(boxes_arr) != len(self):
            raise AssertionError(f"{len(boxes_arr)} != {len(self)}")
        crops: list[BoolArray] = []
        for mask, box in zip(self.tensor, boxes_arr):
            x0, y0, x1, y1 = _rounded_box(box)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
            crop = mask[y0:y1, x0:x1].astype(np.uint8) * 255
            resized = Image.fromarray(crop).resize(
                (mask_size, mask_size), Image.Resampling.NEAREST
            )
            crops.append(np.asarray(resized, dtype=np.uint8) >= 128)
        if not crops:
            return np.empty((0, mask_size, mask_size), dtype=bool)
        return np.stack(crops, axis=0)

    def get_bounding_boxes(self) -> Boxes:
        boxes = np.zeros((self.tensor.shape[0], 4), dtype=np.float32)
        for idx, mask in enumerate(self.tensor):
            ys, xs = np.where(mask)
            if xs.size and ys.size:
                boxes[idx] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        return Boxes(boxes)

    @staticmethod
    def cat(bitmasks_list: Sequence[object]) -> "BitMasks":
        if not bitmasks_list:
            raise AssertionError("bitmasks_list must be non-empty")
        if not all(isinstance(bitmask, BitMasks) for bitmask in bitmasks_list):
            raise AssertionError("All entries must be BitMasks")
        typed_masks = cast(Sequence[BitMasks], bitmasks_list)
        return BitMasks(np.concatenate([bm.tensor for bm in typed_masks], axis=0))


class PolygonMasks:
    """Store segmentation masks as polygon coordinate arrays."""

    def __init__(self, polygons: object):
        if not isinstance(polygons, list):
            raise ValueError(
                "Cannot create PolygonMasks: expected a list of polygons per image. "
                f"Got {type(polygons)!r}."
            )

        polygon_instances = cast(list[object], polygons)

        def process_polygons(polygons_per_instance: object) -> list[Polygon]:
            if not isinstance(polygons_per_instance, list):
                raise ValueError(
                    "Cannot create polygons: expected a list of polygons per instance. "
                    f"Got {type(polygons_per_instance)!r}."
                )
            processed = [
                cast(Polygon, np.asarray(p, dtype=np.float64))
                for p in cast(list[object], polygons_per_instance)
            ]
            for polygon in processed:
                if len(polygon) % 2 != 0 or len(polygon) < 6:
                    raise ValueError(
                        f"Cannot create a polygon from {len(polygon)} coordinates."
                    )
            return processed

        self.polygons: PolygonInstances = [
            process_polygons(polygons_per_instance)
            for polygons_per_instance in polygon_instances
        ]

    def to(self, *args: object, **kwargs: object) -> "PolygonMasks":
        del args, kwargs
        return self

    @property
    def device(self) -> str:
        return "cpu"

    def get_bounding_boxes(self) -> Boxes:
        boxes = np.zeros((len(self.polygons), 4), dtype=np.float32)
        for idx, polygons_per_instance in enumerate(self.polygons):
            if not polygons_per_instance:
                continue
            coords = np.concatenate(
                [polygon.reshape(-1, 2) for polygon in polygons_per_instance], axis=0
            )
            boxes[idx] = [
                coords[:, 0].min(),
                coords[:, 1].min(),
                coords[:, 0].max(),
                coords[:, 1].max(),
            ]
        return Boxes(boxes)

    def nonempty(self) -> BoolArray:
        return cast(
            BoolArray,
            np.asarray([len(polygon) > 0 for polygon in self.polygons], dtype=bool),
        )

    def __getitem__(self, item: MaskIndex) -> "PolygonMasks":
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        elif isinstance(item, list):
            selected_polygons = [self.polygons[i] for i in item]
        else:
            item_arr = _array(item)
            if item_arr.dtype == bool:
                selected_polygons = [
                    polygon for keep, polygon in zip(item_arr, self.polygons) if keep
                ]
            else:
                selected_polygons = [self.polygons[int(i)] for i in item_arr.tolist()]
        return PolygonMasks(selected_polygons)

    def __iter__(self) -> Iterator[list[Polygon]]:
        return iter(self.polygons)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self.polygons)})"

    def __len__(self) -> int:
        return len(self.polygons)

    def crop_and_resize(self, boxes: object, mask_size: int) -> BoolArray:
        boxes_arr = _boxes_array(boxes)
        if len(boxes_arr) != len(self):
            raise AssertionError(f"{len(boxes_arr)} != {len(self)}")
        results = [
            rasterize_polygons_within_box(poly, _array(box), mask_size)
            for poly, box in zip(self.polygons, boxes_arr)
        ]
        if not results:
            return np.empty((0, mask_size, mask_size), dtype=bool)
        return np.stack(results, axis=0)

    def area(self) -> FloatArray:
        areas: list[float] = []
        for polygons_per_instance in self.polygons:
            area_per_instance = 0.0
            for polygon in polygons_per_instance:
                area_per_instance += polygon_area(polygon[0::2], polygon[1::2])
            areas.append(area_per_instance)
        return np.asarray(areas, dtype=np.float64)

    @staticmethod
    def cat(polymasks_list: Sequence[object]) -> "PolygonMasks":
        if not polymasks_list:
            raise AssertionError("polymasks_list must be non-empty")
        if not all(isinstance(polymask, PolygonMasks) for polymask in polymasks_list):
            raise AssertionError("All entries must be PolygonMasks")
        typed_masks = cast(Sequence[PolygonMasks], polymasks_list)
        return PolygonMasks(
            list(itertools.chain.from_iterable(pm.polygons for pm in typed_masks))
        )


class ROIMasks:
    """Represent masks by smaller masks defined in ROI boxes."""

    def __init__(self, tensor: object):
        tensor = _array(tensor)
        if tensor.ndim != 3:
            raise ValueError("ROIMasks must take masks with 3 dimensions.")
        self.tensor = tensor

    def to(self, device: object) -> "ROIMasks":
        if str(device) not in ("cpu", "None"):
            raise ValueError("NumPy ROIMasks only support CPU storage")
        return ROIMasks(self.tensor.copy())

    @property
    def device(self) -> str:
        return "cpu"

    def __len__(self) -> int:
        return int(self.tensor.shape[0])

    def __getitem__(self, item: MaskIndex) -> "ROIMasks":
        selected = self.tensor[item]
        if selected.ndim != 3:
            raise ValueError(
                f"Indexing on ROIMasks with {item} returns {selected.shape}!"
            )
        return ROIMasks(selected)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_instances={len(self.tensor)})"

    def to_bitmasks(
        self,
        boxes: object,
        height: int,
        width: int,
        threshold: float = 0.5,
    ) -> BitMasks:
        boxes_arr = _boxes_array(boxes)
        masks = cast(
            BoolArray, np.zeros((len(self), int(height), int(width)), dtype=bool)
        )
        for idx, (roi_mask, box) in enumerate(zip(self.tensor, boxes_arr)):
            x0, y0, x1, y1 = _rounded_box(box)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(max(x1, x0 + 1), int(width)), min(max(y1, y0 + 1), int(height))
            resized = Image.fromarray(
                (roi_mask > threshold).astype(np.uint8) * 255
            ).resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
            masks[idx, y0:y1, x0:x1] = np.asarray(resized, dtype=np.uint8) >= 128
        return BitMasks(masks)
