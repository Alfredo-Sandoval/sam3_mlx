# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Image-safe COCO/SAM3 JSON loader surfaces for the MLX data port."""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from sam3_mlx.train._unsupported import raise_unsupported

MLX_COCO_JSON_BASE_COMMIT = "8896002f3b5fe333c962ddc7590fe018b6132156"


def convert_boxlist_to_normalized_tensor(box_list, image_width, image_height):
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


def load_coco_and_group_by_image(json_path: str) -> Tuple[List[Dict], Dict[int, str]]:
    """Load a COCO JSON file and group annotations deterministically by image."""

    with Path(json_path).open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = {image["id"]: image for image in coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for annotation in coco.get("annotations", []):
        anns_by_image[annotation["image_id"]].append(annotation)

    grouped = []
    for image_id in sorted(images.keys()):
        grouped.append(
            {
                "image": images[image_id],
                "annotations": anns_by_image.get(image_id, []),
            }
        )

    cat_id_to_name = {
        category["id"]: category["name"] for category in coco.get("categories", [])
    }
    return grouped, cat_id_to_name


def _encode_binary_mask_to_uncompressed_rle(mask: np.ndarray) -> Dict:
    height, width = mask.shape
    flat = np.asarray(mask, dtype=np.uint8).T.reshape(-1)
    counts = []
    last_value = 0
    run_length = 0
    for value in flat:
        value = int(value)
        if value == last_value:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            last_value = value
    counts.append(run_length)
    return {"size": [height, width], "counts": counts}


def _polygons_to_mask(
    polygons: List[List[float]], height: int, width: int
) -> np.ndarray:
    mask = PILImage.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        draw.polygon([tuple(point) for point in points], outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def ann_to_rle(segm, im_info: Dict) -> Dict:
    """Convert COCO polygons or RLE annotations to a local COCO RLE dict."""

    height, width = int(im_info["height"]), int(im_info["width"])
    if isinstance(segm, list):
        mask = _polygons_to_mask(segm, height=height, width=width)
        return _encode_binary_mask_to_uncompressed_rle(mask)

    if not isinstance(segm, dict) or "counts" not in segm:
        raise TypeError("COCO segmentation must be polygons or an RLE dict.")

    counts = segm["counts"]
    if isinstance(counts, list):
        return {
            "size": [height, width],
            "counts": [int(count) for count in counts],
        }
    return segm


class COCO_FROM_JSON:
    """COCO image API matching the official category-chunk query contract."""

    def __init__(
        self,
        annotation_file,
        prompts=None,
        include_negatives=True,
        category_chunk_size=None,
        include_segmentation=False,
    ):
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
        self.prompts = None
        if prompts is not None:
            parsed_prompts = (
                ast.literal_eval(prompts) if isinstance(prompts, str) else prompts
            )
            self.prompts = {int(entry["id"]): entry["name"] for entry in parsed_prompts}
            if len(self.prompts) != len(self._sorted_cat_ids):
                raise AssertionError(
                    "Number of prompts must match number of categories."
                )

    def getDatapointIds(self):
        return list(range(len(self._raw_data) * len(self.category_chunks)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        img_idx = idx // len(self.category_chunks)
        chunk_idx = idx % len(self.category_chunks)
        cat_chunk = self.category_chunks[chunk_idx]

        queries = []
        annotations = []
        query_template = {
            "id": None,
            "original_cat_id": None,
            "object_ids_output": None,
            "query_text": None,
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }
        annot_template = {
            "image_id": 0,
            "bbox": None,
            "area": None,
            "segmentation": None,
            "object_id": None,
            "is_crowd": None,
            "id": None,
        }

        raw_annotations = self._raw_data[img_idx]["annotations"]
        image_info = self._raw_data[img_idx]["image"]
        width, height = image_info["width"], image_info["height"]

        cat_id_to_anns = defaultdict(list)
        for annotation in raw_annotations:
            cat_id_to_anns[annotation["category_id"]].append(annotation)

        for cat_id in cat_chunk:
            anns = cat_id_to_anns[cat_id]
            if len(anns) == 0 and not self.include_negatives:
                continue

            cur_ann_ids = []
            for annotation_raw in anns:
                annotation = annot_template.copy()
                annotation["id"] = len(annotations)
                annotation["object_id"] = annotation["id"]
                annotation["is_crowd"] = annotation_raw.get("iscrowd", 0)

                normalized_boxes = convert_boxlist_to_normalized_tensor(
                    [annotation_raw["bbox"]], width, height
                )
                bbox = normalized_boxes[0]
                area = bbox[2] * bbox[3]
                mx.eval(area)
                annotation["area"] = float(np.asarray(area))
                annotation["bbox"] = bbox

                if (
                    self.include_segmentation
                    and "segmentation" in annotation_raw
                    and annotation_raw["segmentation"] not in (None, [])
                ):
                    annotation["segmentation"] = ann_to_rle(
                        annotation_raw["segmentation"], im_info=image_info
                    )

                annotations.append(annotation)
                cur_ann_ids.append(annotation["id"])

            query = query_template.copy()
            query["id"] = len(queries)
            query["original_cat_id"] = cat_id
            query["query_text"] = (
                self._cat_idx_to_text[cat_id]
                if self.prompts is None
                else self.prompts[cat_id]
            )
            query["object_ids_output"] = cur_ann_ids
            queries.append(query)

        return queries, annotations

    def loadImagesFromDatapoint(self, idx):
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

    def __init__(self, annotation_file):
        with Path(annotation_file).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        self._image_data = data["images"]

    def getDatapointIds(self):
        return list(range(len(self._image_data)))

    def loadQueriesAndAnnotationsFromDatapoint(self, idx):
        cur_img_data = self._image_data[idx]
        query = {
            "id": 0,
            "original_cat_id": int(cur_img_data["queried_category"]),
            "object_ids_output": [],
            "query_text": cur_img_data["text_input"],
            "query_processing_order": 0,
            "ptr_x_query_id": None,
            "ptr_y_query_id": None,
            "image_id": 0,
            "input_box": None,
            "input_box_label": None,
            "input_points": None,
            "is_exhaustive": True,
        }
        return [query], []

    def loadImagesFromDatapoint(self, idx):
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
    def __init__(self, *args, **kwargs):
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
