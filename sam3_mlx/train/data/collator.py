"""MLX collator for official-shaped SAM3 image datapoints.

Ported from ``third_party/facebook-sam3/sam3/train/data/collator.py`` with
Torch tensor operations translated to explicit MLX arrays.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, List

import mlx.core as mx
import numpy as np

from sam3_mlx.model.data_misc import (
    BatchedDatapoint,
    BatchedFindTarget,
    BatchedInferenceMetadata,
    FindStage,
    NestedTensor,
    convert_my_tensors as _convert_my_tensors,
)
from sam3_mlx.train.data.sam3_image_dataset import Datapoint

MLX_COLLATOR_BASE_COMMIT = "13ec0366cb85f7a025a9a36af94fa9eb9599b9d9"

__all__ = [
    "BatchedDatapoint",
    "FindStage",
    "NestedTensor",
    "MLX_COLLATOR_BASE_COMMIT",
    "collate_fn_api",
    "collate_fn_api_with_chunking",
    "convert_my_tensors",
    "packed_to_padded_naive",
    "pad_tensor_list_to_longest",
]


def _as_array(value: Any, dtype=None) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(dtype) if dtype is not None else value
    return mx.array(value, dtype=dtype)


def _numel(value: mx.array) -> int:
    total = 1
    for dim in value.shape:
        total *= dim
    return total


def _to_numpy(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _ensure_image_array(value: Any) -> mx.array:
    if not isinstance(value, mx.array):
        raise TypeError(
            "collate_fn_api expects image data to be MLX arrays. Apply "
            "sam3_mlx.train.transforms.basic_for_api.ToTensorAPI before collation."
        )
    return value


def convert_my_tensors(obj):
    """Official collator API shim over the shared MLX dataclass converter."""

    return _convert_my_tensors(obj)


def packed_to_padded_naive(boxes_packed, num_boxes, fill_value=0):
    """Convert packed rows to the official padded row representation."""

    boxes_packed = _as_array(boxes_packed)
    num_boxes_list = [
        int(x) for x in _to_numpy(_as_array(num_boxes)).reshape(-1).tolist()
    ]
    batch_size = len(num_boxes_list)
    tail_shape = tuple(boxes_packed.shape[1:])
    max_num = max(num_boxes_list) if num_boxes_list else 0

    if batch_size == 0:
        return mx.zeros((0, 0, *tail_shape), dtype=boxes_packed.dtype)

    rows = []
    prev_idx = 0
    for count in num_boxes_list:
        next_idx = prev_idx + count
        row = boxes_packed[prev_idx:next_idx]
        prev_idx = next_idx
        pad_count = max_num - count
        if pad_count:
            pad = mx.full(
                (pad_count, *tail_shape), fill_value, dtype=boxes_packed.dtype
            )
            row = mx.concat([row, pad], axis=0)
        rows.append(row)
    return mx.stack(rows, axis=0)


def pad_tensor_list_to_longest(
    tensors: List[mx.array], dim=0, pad_val=0
) -> List[mx.array]:
    """Pad MLX arrays in-place along ``dim`` to the longest sequence."""

    if not tensors:
        return tensors
    arrays = [_as_array(tensor) for tensor in tensors]
    normalized_dim = dim % arrays[0].ndim
    pad_len = max(tensor.shape[normalized_dim] for tensor in arrays)
    for index, tensor in enumerate(arrays):
        pad_count = pad_len - tensor.shape[normalized_dim]
        if pad_count == 0:
            tensors[index] = tensor
            continue
        pad_shape = list(tensor.shape)
        pad_shape[normalized_dim] = pad_count
        pad = mx.full(tuple(pad_shape), pad_val, dtype=tensor.dtype)
        tensors[index] = mx.concat([tensor, pad], axis=normalized_dim)
    return tensors


def collate_fn_api_with_chunking(
    batch,
    num_chunks,
    dict_key,
    with_seg_masks=False,
    input_points_embedding_dim=257,
    repeats: int = 0,
    load_image_in_fp16: bool = False,
):
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    if not batch:
        raise ValueError("collate_fn_api_with_chunking requires a non-empty batch")

    batch_chunks = [batch[i::num_chunks] for i in range(min(num_chunks, len(batch)))]
    return [
        collate_fn_api(
            chunk,
            dict_key,
            with_seg_masks,
            input_points_embedding_dim,
            repeats,
            load_image_in_fp16,
        )
        for chunk in batch_chunks
    ]


def collate_fn_api(
    batch: List[Datapoint],
    dict_key,
    with_seg_masks=False,
    input_points_embedding_dim=257,
    repeats: int = 0,
    load_image_in_fp16: bool = False,
):
    if not batch:
        raise ValueError("collate_fn_api requires a non-empty batch")
    if not any(data.find_queries for data in batch):
        raise ValueError("collate_fn_api requires at least one find query")

    img_batch = []
    text_batch = []
    raw_images = None

    num_stages = (
        max(q.query_processing_order for data in batch for q in data.find_queries) + 1
    )

    stages = [
        FindStage(
            img_ids=[],
            text_ids=[],
            input_boxes=[],
            input_boxes_label=[],
            input_boxes_mask=[],
            input_points=[],
            input_points_mask=[],
            object_ids=[],
        )
        for _ in range(num_stages)
    ]
    find_targets = [
        BatchedFindTarget(
            num_boxes=[],
            boxes=[],
            boxes_padded=[],
            is_exhaustive=[],
            segments=[],
            semantic_segments=[],
            is_valid_segment=[],
            repeated_boxes=[],
            object_ids=[],
            object_ids_padded=[],
        )
        for _ in range(num_stages)
    ]
    find_metadatas = [
        BatchedInferenceMetadata(
            coco_image_id=[],
            original_size=[],
            object_id=[],
            frame_index=[],
            original_image_id=[],
            original_category_id=[],
            is_conditioning_only=[],
        )
        for _ in range(num_stages)
    ]

    offset_img_id = 0
    offset_query_id = [0 for _ in range(num_stages)]
    for data in batch:
        img_batch.extend([_ensure_image_array(img.data) for img in data.images])

        if data.raw_images is not None:
            if raw_images is None:
                raw_images = []
            raw_images.extend(data.raw_images)

        for q in data.find_queries:
            stage_id = q.query_processing_order
            offset_query_id[stage_id] += 1

        for q in data.find_queries:
            if q.image_id < 0 or q.image_id >= len(data.images):
                raise IndexError(
                    f"query image_id={q.image_id} is out of range for "
                    f"{len(data.images)} image(s)."
                )
            stage_id = q.query_processing_order
            stages[stage_id].img_ids.append(q.image_id + offset_img_id)
            if q.query_text not in text_batch:
                text_batch.append(q.query_text)
            stages[stage_id].text_ids.append(text_batch.index(q.query_text))

            if q.inference_metadata is None:
                raise ValueError(
                    "inference_metadata must be provided when FindQueryLoaded is created."
                )
            for field in fields(q.inference_metadata):
                getattr(find_metadatas[stage_id], field.name).append(
                    getattr(q.inference_metadata, field.name)
                )

            if q.input_bbox is not None:
                input_bbox = _as_array(q.input_bbox)
                if _numel(input_bbox) % 4 != 0:
                    raise ValueError(
                        "input_bbox must contain a multiple of four values"
                    )
                if q.input_bbox_label is None:
                    raise ValueError(
                        "input_bbox_label must be provided with input_bbox"
                    )
                nb_boxes = _numel(input_bbox) // 4
                input_bbox_label = _as_array(q.input_bbox_label)
                if _numel(input_bbox_label) != nb_boxes:
                    raise ValueError("input_bbox_label length must match input_bbox")
                stages[stage_id].input_boxes.append(input_bbox.reshape(nb_boxes, 4))
                stages[stage_id].input_boxes_label.append(
                    input_bbox_label.reshape(nb_boxes)
                )
                stages[stage_id].input_boxes_mask.append(
                    mx.zeros((nb_boxes,), dtype=mx.bool_)
                )
            else:
                stages[stage_id].input_boxes.append(mx.zeros((0, 4), dtype=mx.float32))
                stages[stage_id].input_boxes_label.append(
                    mx.zeros((0,), dtype=mx.int64)
                )
                stages[stage_id].input_boxes_mask.append(mx.ones((0,), dtype=mx.bool_))

            if q.input_points is not None:
                input_points = _as_array(q.input_points)
                if input_points.ndim == 3 and input_points.shape[0] == 1:
                    input_points = input_points[0]
                if input_points.ndim != 2 or input_points.shape[-1] != 3:
                    raise ValueError(
                        "input_points must be raw point prompts with shape (N, 3)"
                    )
                stages[stage_id].input_points.append(input_points)
                stages[stage_id].input_points_mask.append(
                    mx.zeros((input_points.shape[0],), dtype=mx.bool_)
                )
            else:
                stages[stage_id].input_points.append(mx.zeros((0, 3), dtype=mx.float32))
                stages[stage_id].input_points_mask.append(
                    mx.zeros((0,), dtype=mx.bool_)
                )

            current_out_boxes = []
            current_out_object_ids = []
            stages[stage_id].object_ids.append(q.object_ids_output)
            for object_id in q.object_ids_output:
                if object_id < 0 or object_id >= len(data.images[q.image_id].objects):
                    raise IndexError(
                        f"object_id={object_id} is out of range for query image "
                        f"{q.image_id} with "
                        f"{len(data.images[q.image_id].objects)} object(s)."
                    )
                current_out_boxes.append(
                    _as_array(data.images[q.image_id].objects[object_id].bbox)
                )
                current_out_object_ids.append(object_id)
            find_targets[stage_id].boxes.extend(current_out_boxes)
            find_targets[stage_id].object_ids.extend(current_out_object_ids)
            if repeats > 0:
                for _ in range(repeats):
                    find_targets[stage_id].repeated_boxes.extend(current_out_boxes)
            find_targets[stage_id].num_boxes.append(len(current_out_boxes))
            find_targets[stage_id].is_exhaustive.append(q.is_exhaustive)

            if with_seg_masks:
                current_seg_mask = []
                current_is_valid_segment = []
                for object_id in q.object_ids_output:
                    seg_mask = data.images[q.image_id].objects[object_id].segment
                    if seg_mask is not None:
                        current_seg_mask.append(_as_array(seg_mask))
                        current_is_valid_segment.append(1)
                    else:
                        dummy_mask = mx.zeros(
                            data.images[q.image_id].data.shape[-2:], dtype=mx.bool_
                        )
                        current_seg_mask.append(dummy_mask)
                        current_is_valid_segment.append(0)
                find_targets[stage_id].segments.extend(current_seg_mask)
                find_targets[stage_id].is_valid_segment.extend(current_is_valid_segment)
            else:
                find_targets[stage_id].segments = None
                find_targets[stage_id].is_valid_segment = None

            if q.semantic_target is not None:
                find_targets[stage_id].semantic_segments.append(
                    _as_array(q.semantic_target)
                )

        offset_img_id += len(data.images)

    for i in range(len(stages)):
        stages[i].input_points = pad_tensor_list_to_longest(
            stages[i].input_points, dim=0, pad_val=0
        )
        stages[i].input_points_mask = pad_tensor_list_to_longest(
            stages[i].input_points_mask, dim=0, pad_val=1
        )

    for i in range(len(stages)):
        stages[i].input_boxes = pad_tensor_list_to_longest(
            stages[i].input_boxes, dim=0, pad_val=0
        )
        stages[i].input_boxes_label = pad_tensor_list_to_longest(
            stages[i].input_boxes_label, dim=0, pad_val=0
        )
        stages[i].input_boxes_mask = pad_tensor_list_to_longest(
            stages[i].input_boxes_mask, dim=0, pad_val=1
        )

    for i in range(len(stages)):
        stages[i] = convert_my_tensors(stages[i])
        find_targets[i] = convert_my_tensors(find_targets[i])
        find_metadatas[i] = convert_my_tensors(find_metadatas[i])
        find_targets[i].boxes = find_targets[i].boxes.reshape(-1, 4)
        find_targets[i].repeated_boxes = find_targets[i].repeated_boxes.reshape(-1, 4)
        find_targets[i].boxes_padded = packed_to_padded_naive(
            find_targets[i].boxes, find_targets[i].num_boxes
        )
        find_targets[i].object_ids_padded = packed_to_padded_naive(
            find_targets[i].object_ids, find_targets[i].num_boxes, fill_value=-1
        )

    for img in img_batch[1:]:
        if img.shape != img_batch[0].shape:
            raise ValueError("All images must have the same size")
    image_batch = mx.stack(img_batch, axis=0)
    if load_image_in_fp16:
        image_batch = image_batch.astype(mx.float16)

    return {
        dict_key: BatchedDatapoint(
            img_batch=image_batch,
            find_text_batch=text_batch,
            find_inputs=stages,
            find_targets=find_targets,
            find_metadatas=find_metadatas,
            raw_images=raw_images,
        )
    }
