# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""Query-filter transforms for official-shaped SAM3 image datapoints."""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Union

import mlx.core as mx
import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.train.data.sam3_image_dataset import Datapoint, FindQuery, Object


MLX_FILTER_QUERY_BASE_COMMIT = "dc33741d86020f34c73f9534deabff1007cdd886"


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, mx.array):
        mx.eval(value)
    return np.asarray(value)


def _scalar(value):
    array = _to_numpy(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    raise ValueError(f"Expected scalar value, got shape {array.shape}.")


class FilterDataPointQueries:
    find_ids_to_filter: set = None
    get_ids_to_filter: set = None
    obj_ids_to_filter: set = None

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        raise NotImplementedError

    def _do_filter_query(self, query: Union[FindQuery], query_id: int):
        del query
        if self.find_ids_to_filter is None:
            raise AssertionError("identify_queries_to_filter must run first.")
        return query_id in self.find_ids_to_filter


class FilterQueryWithText(FilterDataPointQueries):
    def __init__(
        self, exclude_find_keys: List[str] = None, exclude_get_keys: List[str] = None
    ):
        self.find_filter_keys = exclude_find_keys if exclude_find_keys else []
        self.get_filter_keys = exclude_get_keys if exclude_get_keys else []

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        del_find_ids = []
        for index, query in enumerate(datapoint.find_queries):
            if query.query_text in self.find_filter_keys:
                del_find_ids.append(index)
        self.find_ids_to_filter = set(del_find_ids)


class KeepMaxNumFindQueries(FilterDataPointQueries):
    def __init__(
        self, max_num_find_queries: int, retain_positive_queries: bool = False
    ):
        self.max_num_find_queries = max_num_find_queries
        self.retain_positive_queries = retain_positive_queries

    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        num_find_queries = len(datapoint.find_queries)
        if num_find_queries <= self.max_num_find_queries:
            self.find_ids_to_filter = set()
            return

        if not self.retain_positive_queries:
            num_to_filter = max(0, num_find_queries - self.max_num_find_queries)
            query_ids_to_filter = random.sample(
                range(num_find_queries), k=num_to_filter
            )
        else:
            pos_find_ids, neg_find_ids = [], []
            for index, query in enumerate(datapoint.find_queries):
                if len(query.object_ids_output) == 0:
                    neg_find_ids.append(index)
                else:
                    pos_find_ids.append(index)

            if len(pos_find_ids) >= self.max_num_find_queries:
                num_to_filter = len(pos_find_ids) - self.max_num_find_queries
                query_ids_to_filter = random.sample(pos_find_ids, k=num_to_filter)
                query_ids_to_filter.extend(neg_find_ids)
            else:
                num_to_filter = num_find_queries - self.max_num_find_queries
                query_ids_to_filter = random.sample(neg_find_ids, k=num_to_filter)

        if len(query_ids_to_filter) != num_find_queries - self.max_num_find_queries:
            raise AssertionError("Unexpected number of queries selected for filtering.")
        self.find_ids_to_filter = set(query_ids_to_filter)


class KeepMaxNumFindQueriesVideo(FilterDataPointQueries):
    def __init__(
        self,
        video_mosaic_max_num_find_queries_per_frame: int,
        retain_positive_queries: bool = False,
    ):
        del video_mosaic_max_num_find_queries_per_frame, retain_positive_queries
        raise_unsupported(
            "sam3_mlx.train.transforms.filter_query_transforms.KeepMaxNumFindQueriesVideo",
            reason="training-loop",
            detail=(
                "KeepMaxNumFindQueriesVideo is a video/mosaic transform; this "
                "MLX port is image-only."
            ),
        )


class KeepSemanticFindQueriesOnly(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = {
            index
            for index, query in enumerate(datapoint.find_queries)
            if query.input_bbox is not None
        }


class KeepUnaryFindQueriesOnly(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = set()


class FilterZeroBoxQueries(FilterDataPointQueries):
    @staticmethod
    def _is_zero_area_object(obj: Object):
        bbox = _to_numpy(obj.bbox)
        height = bbox[..., 3] - bbox[..., 1]
        width = bbox[..., 2] - bbox[..., 0]
        return bool(np.any((height == 0) | (width == 0)))

    def identify_queries_to_filter(self, datapoint):
        exclude_objects = set()
        for image_id, image in enumerate(datapoint.images):
            exclude_objects.update(
                (image_id, obj_id)
                for obj_id, obj in enumerate(image.objects)
                if self._is_zero_area_object(obj)
            )
        self.obj_ids_to_filter = exclude_objects
        self.find_ids_to_filter = {
            index
            for index, query in enumerate(datapoint.find_queries)
            if any(
                (query.image_id, object_id) in exclude_objects
                for object_id in query.object_ids_output
            )
        }


class FilterFindQueriesWithTooManyOut(FilterDataPointQueries):
    def __init__(self, max_num_objects: int):
        self.max_num_objects = max_num_objects

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = {
            index
            for index, query in enumerate(datapoint.find_queries)
            if len(query.object_ids_output) > self.max_num_objects
        }


class FilterEmptyTargets(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        for img_id, image in enumerate(datapoint.images):
            for obj_id, obj in enumerate(image.objects):
                if _scalar(obj.area) < 1e-6:
                    self.obj_ids_to_filter.add((img_id, obj_id))
        self.find_ids_to_filter = set()


class FilterNonExhaustiveFindQueries(FilterDataPointQueries):
    def __init__(self, exhaustivity_type: str):
        if exhaustivity_type not in ["pixel", "instance"]:
            raise AssertionError("exhaustivity_type must be 'pixel' or 'instance'.")
        self.exhaustivity_type = exhaustivity_type

    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        del_find_ids = []
        for index, query in enumerate(datapoint.find_queries):
            if self.exhaustivity_type == "instance":
                if not query.is_exhaustive:
                    del_find_ids.append(index)
            elif (
                query.is_pixel_exhaustive is not None and not query.is_pixel_exhaustive
            ):
                del_find_ids.append(index)
        self.find_ids_to_filter = set(del_find_ids)


class FilterInvalidGeometricQueries(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint):
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = {
            index
            for index, query in enumerate(datapoint.find_queries)
            if (
                query.input_bbox is not None
                and query.query_text == "geometric"
                and len(query.object_ids_output) == 0
            )
        }


class FlexibleFilterFindGetQueries:
    def __init__(
        self, query_filter: FilterDataPointQueries, enabled: bool = True
    ) -> None:
        self.query_filter = query_filter
        self.enabled = enabled

    def __call__(self, datapoint, **kwargs):
        del kwargs
        if not self.enabled:
            return datapoint

        self.query_filter.identify_queries_to_filter(datapoint=datapoint)
        for index, query in enumerate(datapoint.find_queries):
            if self.query_filter._do_filter_query(query, index):
                datapoint.find_queries[index] = None

        new_find_queries = [
            query for query in datapoint.find_queries if query is not None
        ]
        start_with_zero_check = len(new_find_queries) == 0 or any(
            query.query_processing_order == 0 for query in new_find_queries
        )
        if not start_with_zero_check:
            raise AssertionError(
                "Find queries must start at query_processing_order = 0."
            )

        datapoint.find_queries = new_find_queries
        if len(datapoint.find_queries) == 0:
            raise ValueError(
                "No find queries left in datapoint after filtering with "
                f"{self.query_filter}."
            )

        all_stages = sorted(
            {query.query_processing_order for query in datapoint.find_queries}
        )
        stage_map = {
            old_stage: new_stage for new_stage, old_stage in enumerate(all_stages)
        }
        for query in datapoint.find_queries:
            query.query_processing_order = stage_map[query.query_processing_order]

        for img_id in range(len(datapoint.images)):
            all_object_ids_to_keep = {
                obj_id
                for query in datapoint.find_queries
                for obj_id in query.object_ids_output
                if query.image_id == img_id
            }
            unused_ids = (
                set(range(len(datapoint.images[img_id].objects)))
                - all_object_ids_to_keep
            )
            obj_ids_to_filter = self.query_filter.obj_ids_to_filter or set()
            for tgt_img_id, tgt_obj_id in obj_ids_to_filter:
                if tgt_img_id == img_id:
                    unused_ids.add(tgt_obj_id)

            if unused_ids:
                old_objects = datapoint.images[img_id].objects
                object_old_to_new_map = {}
                new_objects = []
                for old_id, obj in enumerate(old_objects):
                    if old_id not in unused_ids:
                        object_old_to_new_map[old_id] = len(new_objects)
                        new_objects.append(obj)
                datapoint.images[img_id].objects = new_objects

                for query in datapoint.find_queries:
                    if query.image_id != img_id:
                        continue
                    old_ids = query.object_ids_output
                    query.object_ids_output = [
                        object_old_to_new_map[old_id]
                        for old_id in old_ids
                        if old_id not in unused_ids
                    ]

        images_to_keep = {query.image_id for query in datapoint.find_queries}
        old_img_to_new_img = {}
        new_images = []
        for img_id, image in enumerate(datapoint.images):
            if img_id in images_to_keep:
                old_img_to_new_img[img_id] = len(new_images)
                new_images.append(image)
        datapoint.images = new_images
        for query in datapoint.find_queries:
            query.image_id = old_img_to_new_img[query.image_id]

        return datapoint


class AddPrefixSuffixToFindText:
    def __init__(
        self,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        condition_on_text: bool = False,
        condition_text_list: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.condition_on_text = condition_on_text
        if self.condition_on_text:
            if condition_text_list is None:
                raise AssertionError(
                    "condition_text_list is required when condition_on_text=True."
                )
            self.condition_text_set = {
                text.lower().strip() for text in condition_text_list
            }
        self.enabled = enabled
        if self.enabled:
            logging.info(
                "AddPrefixSuffixToFindText: prefix=%s, suffix=%s, "
                "condition_on_text=%s, condition_text_list=%s",
                prefix,
                suffix,
                condition_on_text,
                condition_text_list,
            )

    def __call__(self, datapoint, **kwargs):
        del kwargs
        if not self.enabled:
            return datapoint
        for query in datapoint.find_queries:
            if query.query_text == "geometric":
                continue
            if (
                self.condition_on_text
                and query.query_text.lower().strip() not in self.condition_text_set
            ):
                continue
            if self.prefix is not None:
                query.query_text = self.prefix + query.query_text
            if self.suffix is not None:
                query.query_text = query.query_text + self.suffix
        return datapoint


class FilterCrowds(FilterDataPointQueries):
    def identify_queries_to_filter(self, datapoint: Datapoint) -> None:
        self.obj_ids_to_filter = set()
        self.find_ids_to_filter = set()
        for img_id, image in enumerate(datapoint.images):
            for obj_id, obj in enumerate(image.objects):
                if obj.is_crowd:
                    self.obj_ids_to_filter.add((img_id, obj_id))


class TextQueryToVisual:
    def __init__(self, probability, keep_text_queries=False) -> None:
        self.probability = probability
        if not 0 <= probability <= 1:
            raise AssertionError("probability must be between 0 and 1.")
        self.keep_text_queries = keep_text_queries

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        for query in datapoint.find_queries:
            if query.input_bbox is not None or query.input_points is not None:
                continue
            if len(query.object_ids_output) == 0:
                continue
            if query.query_processing_order > 0:
                continue
            if random.random() > self.probability:
                continue

            selected_vq_id = random.choice(query.object_ids_output)
            img_id = query.image_id
            query.input_bbox = datapoint.images[img_id].objects[selected_vq_id].bbox
            query.input_bbox_label = mx.ones((1,), dtype=mx.bool_)
            if not self.keep_text_queries:
                query.query_text = "visual"
        return datapoint


class RemoveInputBoxes:
    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        for query in datapoint.find_queries:
            if query.input_bbox is None:
                continue
            if query.query_text == "geometric":
                print("Warning: removing input box from geometric find query")
            query.input_bbox = None
            query.input_bbox_label = None
        return datapoint


class OverwriteTextQuery:
    def __init__(self, target_text, probability=1.0) -> None:
        self.probability = probability
        self.target_text = target_text
        if not 0 <= probability <= 1:
            raise AssertionError("probability must be between 0 and 1.")

    def __call__(self, datapoint: Datapoint, **kwargs):
        del kwargs
        for query in datapoint.find_queries:
            if random.random() <= self.probability:
                query.query_text = self.target_text
        return datapoint


__all__ = [
    "AddPrefixSuffixToFindText",
    "FilterCrowds",
    "FilterDataPointQueries",
    "FilterEmptyTargets",
    "FilterFindQueriesWithTooManyOut",
    "FilterInvalidGeometricQueries",
    "FilterNonExhaustiveFindQueries",
    "FilterQueryWithText",
    "FilterZeroBoxQueries",
    "FlexibleFilterFindGetQueries",
    "KeepMaxNumFindQueries",
    "KeepMaxNumFindQueriesVideo",
    "KeepSemanticFindQueriesOnly",
    "KeepUnaryFindQueriesOnly",
    "MLX_FILTER_QUERY_BASE_COMMIT",
    "OverwriteTextQuery",
    "RemoveInputBoxes",
    "TextQueryToVisual",
]
