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
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import mlx.core as mx
import numpy as np
from PIL import Image as PILImage

from sam3_mlx.model.box_ops import box_xywh_to_xyxy
from sam3_mlx.rle import CocoRle
from sam3_mlx.train._unsupported import raise_unsupported
from sam3_mlx.train.data.coco_json_loaders import (
    COCO_FROM_JSON,
    AnnotationRecord,
    ImageMetadata,
    QueryRecord,
)
from sam3_mlx.train.transforms._array_contracts import (
    ArrayInput,
    mx_array,
    mx_ops,
)

MLX_IMAGE_DATASET_BASE_COMMIT = "13ec0366cb85f7a025a9a36af94fa9eb9599b9d9"


@dataclass
class InferenceMetadata:
    """Metadata required for postprocessing."""

    coco_image_id: int
    original_image_id: int
    original_category_id: int
    original_size: tuple[int, int]
    object_id: int
    frame_index: int
    is_conditioning_only: bool | None = False


@dataclass
class FindQuery:
    query_text: str
    image_id: int
    object_ids_output: list[int]
    is_exhaustive: bool
    query_processing_order: int = 0
    input_bbox: mx.array | None = None
    input_bbox_label: mx.array | None = None
    input_points: mx.array | None = None
    semantic_target: mx.array | CocoRle | None = None
    is_pixel_exhaustive: bool | None = None


@dataclass
class FindQueryLoaded(FindQuery):
    inference_metadata: InferenceMetadata | None = None


@dataclass
class Object:
    bbox: mx.array
    area: float | mx.array
    object_id: int | None = -1
    frame_index: int | None = -1
    segment: mx.array | CocoRle | dict[str, object] | None = None
    is_crowd: bool = False
    source: str | None = None


@dataclass
class Image:
    data: mx.array | PILImage.Image
    objects: list[Object]
    size: tuple[int, int]
    blurring_mask: object | None = None


@dataclass
class Datapoint:
    """Refers to an image/video and all its annotations."""

    find_queries: list[FindQueryLoaded]
    images: list[Image]
    raw_images: list[PILImage.Image] | None = None


class _CocoApi(Protocol):
    def getDatapointIds(self) -> list[int]: ...

    def loadImagesFromDatapoint(self, idx: int) -> list[ImageMetadata]: ...

    def loadQueriesAndAnnotationsFromDatapoint(
        self, idx: int
    ) -> tuple[list[QueryRecord], list[AnnotationRecord]]: ...


class _CocoLoader(Protocol):
    def __call__(
        self,
        annotation_file: str,
        prompts: str | Sequence[Mapping[str, object]] | None = None,
        include_negatives: bool = True,
        category_chunk_size: int | None = None,
        include_segmentation: bool = False,
    ) -> _CocoApi: ...


class _DatapointTransform(Protocol):
    def __call__(self, datapoint: Datapoint, **kwargs: int) -> Datapoint: ...


def _as_float_array(
    value: ArrayInput | Sequence[Sequence[float]],
) -> mx.array:
    if isinstance(value, mx.array):
        return mx_ops(value).astype(mx.float32)
    if isinstance(value, np.ndarray):
        return mx_array(value, dtype=mx.float32)
    return mx_array(np.asarray(value, dtype=np.float32), dtype=mx.float32)


def _denormalize_xywh_to_xyxy(
    boxes: ArrayInput | Sequence[Sequence[float]], height: int, width: int
) -> mx.array:
    bbox = mx_ops(box_xywh_to_xyxy(_as_float_array(boxes))).reshape(-1, 4)
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
        blurring_masks_path: str | None = None,
        use_caching: bool = True,
        zstd_dict_path: str | None = None,
        filter_query: _DatapointTransform | None = None,
        coco_json_loader: _CocoLoader = COCO_FROM_JSON,
        limit_ids: int | None = None,
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
        self.coco: _CocoApi | None = None
        self.coco_json_loader = coco_json_loader
        self.limit_ids = limit_ids
        self.training = training
        self.blurring_masks_path = (
            Path(blurring_masks_path) if blurring_masks_path is not None else None
        )
        self.set_sharded_annotation_file(0)

    def _load_images(
        self, datapoint_id: int, img_ids_to_load: Set[int] | None = None
    ) -> tuple[list[tuple[int, PILImage.Image]], list[ImageMetadata]]:
        if self.coco is None:
            raise RuntimeError("COCO loader must be initialized before loading images.")
        all_images: list[tuple[int, PILImage.Image]] = []
        all_img_metadata: list[ImageMetadata] = []
        for loaded_meta in self.coco.loadImagesFromDatapoint(datapoint_id):
            current_meta = loaded_meta.copy()
            img_id = current_meta["id"]
            if img_ids_to_load is not None and img_id not in img_ids_to_load:
                continue
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

    def set_curr_epoch(self, epoch: int) -> None:
        self.curr_epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.curr_epoch = epoch

    def set_sharded_annotation_file(self, data_epoch: int) -> None:
        del data_epoch
        if self.coco is not None:
            return
        if not self.annFile.is_file():
            raise FileNotFoundError(
                f"please provide valid annotation file. Missing: {self.annFile}"
            )
        self.coco = self.coco_json_loader(
            str(self.annFile), include_segmentation=self.load_segmentation
        )
        ids_list = list(sorted(self.coco.getDatapointIds()))
        if self.limit_ids is not None:
            local_random = random.Random(len(ids_list))
            local_random.shuffle(ids_list)
            ids_list = ids_list[: self.limit_ids]
        self.ids = ids_list

    def __getitem__(self, index: int) -> Datapoint:
        return self._load_datapoint(index)

    def _load_datapoint(self, index: int) -> Datapoint:
        if self.coco is None:
            raise RuntimeError("COCO loader must be initialized before dataset access.")
        datapoint_id = self.ids[index]
        pil_images, img_metadata = self._load_images(datapoint_id)
        queries, annotations = self.coco.loadQueriesAndAnnotationsFromDatapoint(
            datapoint_id
        )
        return self.load_queries(pil_images, annotations, queries, img_metadata)

    def load_queries(
        self,
        pil_images: Sequence[tuple[int, PILImage.Image]],
        annotations: Sequence[AnnotationRecord],
        queries: Sequence[QueryRecord],
        img_metadata: Sequence[ImageMetadata],
    ) -> Datapoint:
        images: list[Image] = []
        id2index_img: dict[int, int] = {}
        id2index_obj: dict[int, int] = {}
        id2imsize: dict[int, tuple[int, int]] = {}
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

        stage2num_queries: Counter[int] = Counter()
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

        find_queries: list[FindQueryLoaded] = []
        for query in queries:
            height, width = id2imsize[query["image_id"]]
            input_box = query["input_box"]
            input_box_label = query["input_box_label"]
            if input_box:
                bbox = _denormalize_xywh_to_xyxy(input_box, height, width)
                if input_box_label is not None:
                    bbox_label = mx_ops(
                        mx_array(input_box_label, dtype=mx.int64)
                    ).reshape(-1)
                    if len(bbox_label) != len(bbox):
                        raise AssertionError("input_box_label length mismatch.")
                else:
                    bbox_label = mx.ones((len(bbox),), dtype=mx.int64)
            else:
                bbox = None
                bbox_label = None

            input_points = query["input_points"]
            if input_points is not None:
                points = mx_ops(
                    mx_array(
                        np.asarray(input_points, dtype=np.float32), dtype=mx.float32
                    )
                ).reshape(1, -1, 3)
                scale = mx.array([width, height, 1.0], dtype=mx.float32)
                points = mx.clip(points * scale, a_min=0.0, a_max=scale)
            else:
                points = None

            img_meta = img_metadata[id2index_img[query["image_id"]]]
            original_image_id = img_meta["original_img_id"]
            coco_image_id = img_meta["coco_img_id"]
            original_category_id = query["original_cat_id"]

            if query["object_ids_output"]:
                first_obj_id = query["object_ids_output"][0]
                obj_idx = id2index_obj[first_obj_id]
                image_idx = id2index_img[query["image_id"]]
                object_id = images[image_idx].objects[obj_idx].object_id
                frame_index = images[image_idx].objects[obj_idx].frame_index
                object_id = -1 if object_id is None else object_id
                frame_index = -1 if frame_index is None else frame_index
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
        img_folder: str,
        ann_file: str,
        transforms: Sequence[_DatapointTransform] | None,
        max_ann_per_img: int,
        multiplier: int,
        training: bool,
        load_segmentation: bool = False,
        max_train_queries: int = 81,
        max_val_queries: int = 300,
        fix_fname: bool = False,
        is_sharded_annotation_dir: bool = False,
        blurring_masks_path: str | None = None,
        use_caching: bool = True,
        zstd_dict_path: str | None = None,
        filter_query: _DatapointTransform | None = None,
        coco_json_loader: _CocoLoader = COCO_FROM_JSON,
        limit_ids: int | None = None,
    ) -> None:
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
        self._transforms = list(transforms or ())
        self.training = training
        self.max_ann_per_img = max_ann_per_img
        self.max_train_queries = max_train_queries
        self.max_val_queries = max_val_queries
        self.repeat_factors = [float(multiplier) for _ in self.ids]

    def __getitem__(self, idx: int) -> Datapoint:
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
