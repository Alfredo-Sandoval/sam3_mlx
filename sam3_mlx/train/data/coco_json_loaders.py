# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Image-safe COCO/SAM3 JSON loader surfaces for the MLX data port."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import mlx.core as mx
import numpy as np

from sam3_mlx.mlx_runtime import evaluate_boundary
from sam3_mlx.rle import CocoRle, ann_to_rle
from sam3_mlx.train._unsupported import raise_unsupported

MLX_COCO_JSON_BASE_COMMIT = "8896002f3b5fe333c962ddc7590fe018b6132156"


class CocoImage(TypedDict):
    id: int
    file_name: str
    width: int
    height: int
    queried_category: NotRequired[int]
    text_input: NotRequired[str]


class EvalImage(TypedDict):
    id: int
    file_name: str
    queried_category: int
    text_input: str


class CocoAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    area: NotRequired[float]
    segmentation: NotRequired[object]
    iscrowd: NotRequired[int]


class GroupedImage(TypedDict):
    image: CocoImage
    annotations: list[CocoAnnotation]


class ImageMetadata(TypedDict):
    id: int
    file_name: str
    original_img_id: int
    coco_img_id: int
    blurring_mask: NotRequired[object]


class AnnotationRecord(TypedDict):
    image_id: int
    bbox: mx.array
    area: float
    segmentation: CocoRle | None
    object_id: int
    is_crowd: bool
    id: int
    frame_index: NotRequired[int]
    source: NotRequired[str]


class QueryRecord(TypedDict):
    id: int
    original_cat_id: int
    object_ids_output: list[int]
    query_text: str
    query_processing_order: int
    image_id: int
    input_box: list[list[float]] | None
    input_box_label: list[int] | None
    input_points: list[list[float]] | None
    is_exhaustive: bool
    is_pixel_exhaustive: NotRequired[bool | None]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array.")
    return cast(list[object], value)


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a sequence.")
    return cast(Sequence[object], value)


def _int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{context} must be an integer.")
    return value


def _str(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string.")
    return value


def _float_list(value: object, context: str) -> list[float]:
    entries = _list(value, context)
    if not all(isinstance(entry, (int, float)) for entry in entries):
        raise TypeError(f"{context} values must be numeric.")
    return [float(cast(int | float, entry)) for entry in entries]


def _parse_image(value: object) -> CocoImage:
    item = _mapping(value, "COCO image")
    return {
        "id": _int(item.get("id"), "COCO image id"),
        "file_name": _str(item.get("file_name"), "COCO image file_name"),
        "width": _int(item.get("width"), "COCO image width"),
        "height": _int(item.get("height"), "COCO image height"),
    }


def _parse_annotation(value: object) -> CocoAnnotation:
    item = _mapping(value, "COCO annotation")
    annotation: CocoAnnotation = {
        "id": _int(item.get("id"), "COCO annotation id"),
        "image_id": _int(item.get("image_id"), "COCO annotation image_id"),
        "category_id": _int(item.get("category_id"), "COCO annotation category_id"),
        "bbox": _float_list(item.get("bbox"), "COCO annotation bbox"),
    }
    if "segmentation" in item:
        annotation["segmentation"] = item["segmentation"]
    if "iscrowd" in item:
        annotation["iscrowd"] = _int(item["iscrowd"], "COCO annotation iscrowd")
    return annotation


def _parse_eval_image(value: object) -> EvalImage:
    item = _mapping(value, "SAM3 eval image")
    return {
        "id": _int(item.get("id"), "SAM3 eval image id"),
        "file_name": _str(item.get("file_name"), "SAM3 eval image file_name"),
        "queried_category": _int(
            item.get("queried_category"), "SAM3 eval queried_category"
        ),
        "text_input": _str(item.get("text_input"), "SAM3 eval text_input"),
    }


def convert_boxlist_to_normalized_tensor(
    box_list: Sequence[Sequence[float]], image_width: int, image_height: int
) -> mx.array:
    """Convert COCO-style ``xywh`` boxes to normalized MLX ``xywh`` arrays."""

    boxes_np = np.asarray(box_list, dtype=np.float32).reshape(-1, 4)
    if boxes_np.size == 0:
        return mx.zeros((0, 4), dtype=mx.float32)
    boxes = mx.array(boxes_np, dtype=mx.float32)
    scale = mx.array(
        [image_width, image_height, image_width, image_height],
        dtype=mx.float32,
    )
    return mx.clip(boxes / scale, 0.0, 1.0)


def load_coco_and_group_by_image(
    json_path: str,
) -> tuple[list[GroupedImage], dict[int, str]]:
    """Load a COCO JSON file and group annotations deterministically by image."""

    with Path(json_path).open("r", encoding="utf-8") as handle:
        coco = _mapping(json.load(handle), "COCO root")

    images = {
        image["id"]: image
        for image in (
            _parse_image(value)
            for value in _list(coco.get("images", []), "COCO images")
        )
    }
    anns_by_image: defaultdict[int, list[CocoAnnotation]] = defaultdict(list)
    for value in _list(coco.get("annotations", []), "COCO annotations"):
        annotation = _parse_annotation(value)
        anns_by_image[annotation["image_id"]].append(annotation)

    grouped: list[GroupedImage] = []
    for image_id in sorted(images.keys()):
        grouped.append(
            {
                "image": images[image_id],
                "annotations": anns_by_image.get(image_id, []),
            }
        )

    cat_id_to_name: dict[int, str] = {}
    for value in _list(coco.get("categories", []), "COCO categories"):
        category = _mapping(value, "COCO category")
        cat_id_to_name[_int(category.get("id"), "COCO category id")] = _str(
            category.get("name"), "COCO category name"
        )
    return grouped, cat_id_to_name


class COCO_FROM_JSON:
    """COCO image API matching the official category-chunk query contract."""

    def __init__(
        self,
        annotation_file: str,
        prompts: str | Sequence[Mapping[str, object]] | None = None,
        include_negatives: bool = True,
        category_chunk_size: int | None = None,
        include_segmentation: bool = False,
    ) -> None:
        self._raw_data, self._cat_idx_to_text = load_coco_and_group_by_image(
            annotation_file
        )
        self._sorted_cat_ids = sorted(self._cat_idx_to_text.keys())
        self.include_negatives = include_negatives
        self.category_chunk_size = (
            category_chunk_size
            if category_chunk_size is not None
            else len(self._sorted_cat_ids)
        )
        if self.category_chunk_size <= 0:
            raise ValueError("category_chunk_size must be positive.")
        self.category_chunks = [
            self._sorted_cat_ids[i : i + self.category_chunk_size]
            for i in range(0, len(self._sorted_cat_ids), self.category_chunk_size)
        ]
        self.include_segmentation = include_segmentation
        self.prompts: dict[int, str] | None = None
        if prompts is not None:
            raw_prompts: object = (
                ast.literal_eval(prompts) if isinstance(prompts, str) else prompts
            )
            prompt_entries = _sequence(raw_prompts, "prompts")
            self.prompts = {}
            for value in prompt_entries:
                entry = _mapping(value, "prompt")
                prompt_id = _int(entry.get("id"), "prompt id")
                self.prompts[prompt_id] = _str(entry.get("name"), "prompt name")
            if len(self.prompts) != len(self._sorted_cat_ids):
                raise AssertionError(
                    "Number of prompts must match number of categories."
                )

    def getDatapointIds(self) -> list[int]:
        return list(range(len(self._raw_data) * len(self.category_chunks)))

    def loadQueriesAndAnnotationsFromDatapoint(
        self, idx: int
    ) -> tuple[list[QueryRecord], list[AnnotationRecord]]:
        img_idx = idx // len(self.category_chunks)
        chunk_idx = idx % len(self.category_chunks)
        cat_chunk = self.category_chunks[chunk_idx]

        queries: list[QueryRecord] = []
        annotations: list[AnnotationRecord] = []

        raw_annotations = self._raw_data[img_idx]["annotations"]
        image_info = self._raw_data[img_idx]["image"]
        width, height = image_info["width"], image_info["height"]

        cat_id_to_anns: defaultdict[int, list[CocoAnnotation]] = defaultdict(list)
        for raw_annotation in raw_annotations:
            cat_id_to_anns[raw_annotation["category_id"]].append(raw_annotation)

        for cat_id in cat_chunk:
            anns = cat_id_to_anns[cat_id]
            if len(anns) == 0 and not self.include_negatives:
                continue

            cur_ann_ids: list[int] = []
            for annotation_raw in anns:
                normalized_boxes = convert_boxlist_to_normalized_tensor(
                    [annotation_raw["bbox"]], width, height
                )
                bbox = normalized_boxes[0]
                area = bbox[2] * bbox[3]
                evaluate_boundary(area)
                annotation_id = len(annotations)
                annotation: AnnotationRecord = {
                    "id": annotation_id,
                    "object_id": annotation_id,
                    "image_id": 0,
                    "is_crowd": bool(annotation_raw.get("iscrowd", 0)),
                    "area": float(np.asarray(area)),
                    "bbox": bbox,
                    "segmentation": None,
                }

                if (
                    self.include_segmentation
                    and "segmentation" in annotation_raw
                    and annotation_raw["segmentation"] not in (None, [])
                ):
                    annotation["segmentation"] = ann_to_rle(
                        annotation_raw["segmentation"],
                        im_info={"height": height, "width": width},
                    )

                annotations.append(annotation)
                cur_ann_ids.append(annotation["id"])

            query: QueryRecord = {
                "id": len(queries),
                "original_cat_id": cat_id,
                "query_text": (
                    self._cat_idx_to_text[cat_id]
                    if self.prompts is None
                    else self.prompts[cat_id]
                ),
                "object_ids_output": cur_ann_ids,
                "query_processing_order": 0,
                "image_id": 0,
                "input_box": None,
                "input_box_label": None,
                "input_points": None,
                "is_exhaustive": True,
            }
            queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx: int) -> list[ImageMetadata]:
        img_idx = idx // len(self.category_chunks)
        img_data = self._raw_data[img_idx]["image"]
        return [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]


class SAM3_EVAL_API_FROM_JSON_NP:
    """SAM3 image noun-phrase eval API with no target annotations."""

    def __init__(self, annotation_file: str) -> None:
        with Path(annotation_file).open("r", encoding="utf-8") as handle:
            data = _mapping(json.load(handle), "SAM3 eval root")
        self._image_data = [
            _parse_eval_image(value)
            for value in _list(data.get("images"), "SAM3 eval images")
        ]

    def getDatapointIds(self) -> list[int]:
        return list(range(len(self._image_data)))

    def loadQueriesAndAnnotationsFromDatapoint(
        self, idx: int
    ) -> tuple[list[QueryRecord], list[AnnotationRecord]]:
        cur_img_data = self._image_data[idx]
        query: QueryRecord = {
            "id": 0,
            "original_cat_id": int(cur_img_data["queried_category"]),
            "object_ids_output": [],
            "query_text": cur_img_data["text_input"],
            "query_processing_order": 0,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }
        return [query], []

    def loadImagesFromDatapoint(self, idx: int) -> list[ImageMetadata]:
        img_data = self._image_data[idx]
        return [
            {
                "id": 0,
                "file_name": img_data["file_name"],
                "original_img_id": img_data["id"],
                "coco_img_id": img_data["id"],
            }
        ]


class SAM3_VEVAL_API_FROM_JSON_NP:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_unsupported("SAM3_VEVAL_API_FROM_JSON_NP")


__all__ = [
    "COCO_FROM_JSON",
    "MLX_COCO_JSON_BASE_COMMIT",
    "SAM3_EVAL_API_FROM_JSON_NP",
    "SAM3_VEVAL_API_FROM_JSON_NP",
    "ann_to_rle",
    "convert_boxlist_to_normalized_tensor",
    "load_coco_and_group_by_image",
]
