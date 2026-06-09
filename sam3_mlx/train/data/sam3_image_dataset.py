# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Dataset dataclasses for official-shaped SAM3 image inputs.

Ported from ``third_party/facebook-sam3/sam3/train/data/sam3_image_dataset.py``.
The active surface is the image-only COCO/SAM3 JSON path backed by PIL and MLX.
Video frame loading, sharded annotations, zstd caching, and Torch worker
behavior stay explicit unsupported boundaries.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import mlx.core as mx
from PIL import Image as PILImage

from sam3_mlx.model.box_ops import box_xywh_to_xyxy
from sam3_mlx.train._unsupported import raise_unsupported
from sam3_mlx.train.data.coco_json_loaders import COCO_FROM_JSON

MLX_IMAGE_DATASET_BASE_COMMIT = "13ec0366cb85f7a025a9a36af94fa9eb9599b9d9"


@dataclass
class InferenceMetadata:
    """Metadata required for postprocessing."""

    coco_image_id: int
    original_image_id: int
    original_category_id: int
    original_size: Tuple[int, int]
    object_id: int
    frame_index: int
    is_conditioning_only: Optional[bool] = False


@dataclass
class FindQuery:
    query_text: str
    image_id: int
    object_ids_output: List[int]
    is_exhaustive: bool
    query_processing_order: int = 0
    input_bbox: Optional[mx.array] = None
    input_bbox_label: Optional[mx.array] = None
    input_points: Optional[mx.array] = None
    semantic_target: Optional[mx.array] = None
    is_pixel_exhaustive: Optional[bool] = None


@dataclass
class FindQueryLoaded(FindQuery):
    inference_metadata: Optional[InferenceMetadata] = None


@dataclass
class Object:
    bbox: mx.array
    area: Union[float, mx.array]
    object_id: Optional[int] = -1
    frame_index: Optional[int] = -1
    segment: Optional[Union[mx.array, dict]] = None
    is_crowd: bool = False
    source: Optional[str] = None


@dataclass
class Image:
    data: Union[mx.array, PILImage.Image]
    objects: List[Object]
    size: Tuple[int, int]
    blurring_mask: Optional[Dict[str, Any]] = None


@dataclass
class Datapoint:
    """Refers to an image/video and all its annotations."""

    find_queries: List[FindQueryLoaded]
    images: List[Image]
    raw_images: Optional[List[PILImage.Image]] = None


def _as_float_array(value) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    return mx.array(value, dtype=mx.float32)


def _denormalize_xywh_to_xyxy(boxes, height: int, width: int) -> mx.array:
    bbox = box_xywh_to_xyxy(_as_float_array(boxes)).reshape(-1, 4)
    scale = mx.array([width, height, width, height], dtype=mx.float32)
    return mx.clip(bbox * scale, a_min=0.0, a_max=scale)


class CustomCocoDetectionAPI:
    """Pure-Python image dataset surface for official-shaped COCO APIs."""

    def __init__(
        self,
        root: str,
        annFile: str,
        load_segmentation: bool,
        fix_fname: bool = False,
        training: bool = True,
        blurring_masks_path: Optional[str] = None,
        use_caching: bool = True,
        zstd_dict_path=None,
        filter_query=None,
        coco_json_loader: Callable = COCO_FROM_JSON,
        limit_ids: int = None,
        is_sharded_annotation_dir: bool = False,
    ) -> None:
        if use_caching is not True:
            raise_unsupported("CustomCocoDetectionAPI use_caching=False")
        if zstd_dict_path is not None:
            raise_unsupported("CustomCocoDetectionAPI zstd_dict_path")
        if is_sharded_annotation_dir:
            raise_unsupported("CustomCocoDetectionAPI sharded annotations")
        self.root = Path(root)
        self.annFile = Path(annFile)
        self.curr_epoch = 0
        self.load_segmentation = load_segmentation
        self.fix_fname = fix_fname
        self.filter_query = filter_query
        self.coco = None
        self.coco_json_loader = coco_json_loader
        self.limit_ids = limit_ids
        self.training = training
        self.blurring_masks_path = (
            Path(blurring_masks_path) if blurring_masks_path is not None else None
        )
        self.set_sharded_annotation_file(0)

    def _load_images(
        self, datapoint_id: int, img_ids_to_load: Optional[Set[int]] = None
    ) -> Tuple[List[Tuple[int, PILImage.Image]], List[Dict[str, Any]]]:
        all_images = []
        all_img_metadata = []
        for current_meta in self.coco.loadImagesFromDatapoint(datapoint_id):
            img_id = current_meta["id"]
            if img_ids_to_load is not None and img_id not in img_ids_to_load:
                continue
            current_meta = dict(current_meta)
            if self.fix_fname:
                current_meta["file_name"] = Path(current_meta["file_name"]).name

            rel_path = current_meta["file_name"]
            if rel_path.endswith(".mp4") or ".mp4@" in rel_path:
                raise_unsupported("CustomCocoDetectionAPI video frame loading")

            if self.blurring_masks_path is not None:
                mask_name = Path(rel_path).name.replace(".jpg", "-mask.json")
                mask_path = self.blurring_masks_path / mask_name
                if mask_path.exists():
                    with mask_path.open("r", encoding="utf-8") as handle:
                        current_meta["blurring_mask"] = json.load(handle)

            path = self.root / rel_path
            if not path.is_file():
                raise FileNotFoundError(
                    f"File not found: {path} from dataset: {self.annFile}"
                )
            with PILImage.open(path) as image:
                all_images.append((img_id, image.convert("RGB")))
            all_img_metadata.append(current_meta)
        return all_images, all_img_metadata

    def set_curr_epoch(self, epoch: int):
        self.curr_epoch = epoch

    def set_epoch(self, epoch: int):
        self.curr_epoch = epoch

    def set_sharded_annotation_file(self, data_epoch: int):
        del data_epoch
        if self.coco is not None:
            return
        if not self.annFile.is_file():
            raise FileNotFoundError(
                f"please provide valid annotation file. Missing: {self.annFile}"
            )
        loader_kwargs = {}
        if self.load_segmentation:
            loader_kwargs["include_segmentation"] = True
        self.coco = self.coco_json_loader(str(self.annFile), **loader_kwargs)
        ids_list = list(sorted(self.coco.getDatapointIds()))
        if self.limit_ids is not None:
            local_random = random.Random(len(ids_list))
            local_random.shuffle(ids_list)
            ids_list = ids_list[: self.limit_ids]
        self.ids = ids_list

    def __getitem__(self, index: int) -> Datapoint:
        return self._load_datapoint(index)

    def _load_datapoint(self, index: int) -> Datapoint:
        datapoint_id = self.ids[index]
        pil_images, img_metadata = self._load_images(datapoint_id)
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(
            datapoint_id
        )
        return self.load_queries(pil_images, annotations, queries, img_metadata)

    def load_queries(self, pil_images, annotations, queries, img_metadata):
        images: List[Image] = []
        id2index_img = {}
        id2index_obj = {}
        id2imsize = {}
        if len(pil_images) != len(img_metadata):
            raise AssertionError("pil_images and img_metadata length mismatch.")

        for index, (image_id, pil_image) in enumerate(pil_images):
            width, height = pil_image.size
            blurring_mask = img_metadata[index].get("blurring_mask")
            images.append(
                Image(
                    data=pil_image,
                    objects=[],
                    size=(height, width),
                    blurring_mask=blurring_mask,
                )
            )
            id2index_img[image_id] = index
            id2imsize[image_id] = (height, width)

        for annotation in annotations:
            image_id = id2index_img[annotation["image_id"]]
            height, width = id2imsize[annotation["image_id"]]
            bbox = _denormalize_xywh_to_xyxy(annotation["bbox"], height, width)
            segment = None
            if self.load_segmentation and "segmentation" in annotation:
                segment = annotation["segmentation"]
            images[image_id].objects.append(
                Object(
                    bbox=bbox[0],
                    area=annotation["area"],
                    object_id=annotation.get("object_id", -1),
                    frame_index=annotation.get("frame_index", -1),
                    segment=segment,
                    is_crowd=annotation.get("is_crowd", False),
                    source=annotation.get("source", ""),
                )
            )
            id2index_obj[annotation["id"]] = len(images[image_id].objects) - 1

        stage2num_queries = Counter()
        for query in queries:
            stage2num_queries[query["query_processing_order"]] += 1
        if stage2num_queries:
            num_queries_per_stage = stage2num_queries.most_common(1)[0][1]
            for stage, num_queries in stage2num_queries.items():
                if num_queries != num_queries_per_stage:
                    raise AssertionError(
                        f"Number of queries in stage {stage} is {num_queries}, "
                        f"expected {num_queries_per_stage}"
                    )

        find_queries = []
        for query in queries:
            height, width = id2imsize[query["image_id"]]
            if query.get("input_box") is not None and len(query["input_box"]) > 0:
                bbox = _denormalize_xywh_to_xyxy(query["input_box"], height, width)
                if query.get("input_box_label") is not None:
                    bbox_label = mx.array(
                        query["input_box_label"], dtype=mx.int64
                    ).reshape(-1)
                    if len(bbox_label) != len(bbox):
                        raise AssertionError("input_box_label length mismatch.")
                else:
                    bbox_label = mx.ones((len(bbox),), dtype=mx.int64)
            else:
                bbox = None
                bbox_label = None

            if query.get("input_points") is not None:
                points = mx.array(query["input_points"], dtype=mx.float32).reshape(
                    1, -1, 3
                )
                scale = mx.array([width, height, 1.0], dtype=mx.float32)
                points = mx.clip(points * scale, a_min=0.0, a_max=scale)
            else:
                points = None

            img_meta = img_metadata[id2index_img[query["image_id"]]]
            try:
                original_image_id = int(img_meta["original_img_id"])
            except (KeyError, TypeError, ValueError):
                original_image_id = -1
            try:
                coco_image_id = int(img_meta.get("coco_img_id", query["id"]))
            except (TypeError, ValueError):
                coco_image_id = -1
            try:
                original_category_id = int(query["original_cat_id"])
            except (KeyError, TypeError, ValueError):
                original_category_id = -1

            if query["object_ids_output"]:
                first_obj_id = query["object_ids_output"][0]
                obj_idx = id2index_obj[first_obj_id]
                image_idx = id2index_img[query["image_id"]]
                object_id = images[image_idx].objects[obj_idx].object_id
                frame_index = images[image_idx].objects[obj_idx].frame_index
            else:
                object_id = -1
                frame_index = -1

            find_queries.append(
                FindQueryLoaded(
                    query_text=query["query_text"] or "",
                    image_id=id2index_img[query["image_id"]],
                    input_bbox=bbox,
                    input_bbox_label=bbox_label,
                    input_points=points,
                    object_ids_output=[
                        id2index_obj[obj_id] for obj_id in query["object_ids_output"]
                    ],
                    is_exhaustive=query["is_exhaustive"],
                    is_pixel_exhaustive=query.get(
                        "is_pixel_exhaustive",
                        query["is_exhaustive"] if query["is_exhaustive"] else None,
                    ),
                    query_processing_order=query["query_processing_order"],
                    inference_metadata=InferenceMetadata(
                        coco_image_id=-1 if self.training else coco_image_id,
                        original_image_id=-1 if self.training else original_image_id,
                        frame_index=frame_index,
                        original_category_id=original_category_id,
                        original_size=(height, width),
                        object_id=object_id,
                    ),
                )
            )

        return Datapoint(
            find_queries=find_queries,
            images=images,
            raw_images=[image for _, image in pil_images],
        )

    def __len__(self) -> int:
        return len(self.ids)


class Sam3ImageDataset(CustomCocoDetectionAPI):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        max_ann_per_img: int,
        multiplier: int,
        training: bool,
        load_segmentation: bool = False,
        max_train_queries: int = 81,
        max_val_queries: int = 300,
        fix_fname: bool = False,
        is_sharded_annotation_dir: bool = False,
        blurring_masks_path: Optional[str] = None,
        use_caching: bool = True,
        zstd_dict_path=None,
        filter_query=None,
        coco_json_loader: Callable = COCO_FROM_JSON,
        limit_ids: int = None,
    ):
        super().__init__(
            img_folder,
            ann_file,
            fix_fname=fix_fname,
            load_segmentation=load_segmentation,
            training=training,
            blurring_masks_path=blurring_masks_path,
            use_caching=use_caching,
            zstd_dict_path=zstd_dict_path,
            filter_query=filter_query,
            coco_json_loader=coco_json_loader,
            limit_ids=limit_ids,
            is_sharded_annotation_dir=is_sharded_annotation_dir,
        )
        self._transforms = [] if transforms is None else transforms
        self.training = training
        self.max_ann_per_img = max_ann_per_img
        self.max_train_queries = max_train_queries
        self.max_val_queries = max_val_queries
        self.repeat_factors = [float(multiplier) for _ in self.ids]

    def __getitem__(self, idx):
        datapoint = super().__getitem__(idx)
        if self.filter_query is not None:
            datapoint = self.filter_query(datapoint)
        for query in datapoint.find_queries:
            if len(query.object_ids_output) > self.max_ann_per_img:
                raise ValueError(f"Too many outputs ({len(query.object_ids_output)})")
        max_queries = self.max_train_queries if self.training else self.max_val_queries
        if len(datapoint.find_queries) > max_queries:
            raise ValueError(f"Too many find queries ({len(datapoint.find_queries)})")
        if len(datapoint.find_queries) == 0:
            raise ValueError("No find queries")
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint


__all__ = [
    "CustomCocoDetectionAPI",
    "Datapoint",
    "FindQuery",
    "FindQueryLoaded",
    "Image",
    "InferenceMetadata",
    "MLX_IMAGE_DATASET_BASE_COMMIT",
    "Object",
    "Sam3ImageDataset",
]
