from __future__ import annotations

from collections import defaultdict
from enum import Enum
import math
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.perflib.masks_ops import mask_iou
from sam3_mlx.model.box_ops import fast_diag_box_iou
from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.model.sam3_tracker_utils import fill_holes_in_mask_scores, mask_to_box
from sam3_mlx.model.sam3_video_base import (
    LazyAssociateDetTrkResult,
    Sam3VideoBase,
    _associate_det_trk_compilable,
    realize_adt_result,
)
from sam3_mlx.model.multiplex_utils import raise_unsupported_multiplex_runtime


SAM3_COLLECTIVE_OP_TIMEOUT_SEC = 180


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


def _array_to_numpy(value: Any, *, dtype=None) -> np.ndarray:
    return to_numpy(value, dtype=dtype, copy=False)


def _score_to_float(value: Any) -> float:
    array = _array_to_numpy(value, dtype=np.float32)
    return float(array.reshape(-1)[0])


def _is_floating_array(value: Any) -> bool:
    if _is_mlx_array(value):
        return value.dtype in {mx.float16, mx.float32, mx.bfloat16}
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        dtype = np.asarray(value).dtype
    return np.issubdtype(dtype, np.floating)


def _copy_metadata_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, defaultdict):
        copied = defaultdict(value.default_factory)
        for key, item in value.items():
            copied[key] = _copy_metadata_value(item)
        return copied
    if isinstance(value, dict):
        return {key: _copy_metadata_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_metadata_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_metadata_value(item) for item in value)
    if isinstance(value, set):
        return set(value)
    return value


def _torch_bool_argsort_desc_np(values: np.ndarray) -> np.ndarray:
    """Replicate PyTorch CPU bool ``argsort(descending=True)`` tie ordering."""
    bool_values = np.asarray(values, dtype=bool).reshape(-1)
    order = list(range(bool_values.size))

    def comp(left: int, right: int) -> bool:
        return bool(bool_values[left] and not bool_values[right])

    def swap(left: int, right: int) -> None:
        order[left], order[right] = order[right], order[left]

    def sort3(first: int, second: int, third: int) -> bool:
        if not comp(order[second], order[first]):
            if not comp(order[third], order[second]):
                return False
            swap(second, third)
            if comp(order[second], order[first]):
                swap(first, second)
            return True
        if comp(order[third], order[second]):
            swap(first, third)
            return True
        swap(first, second)
        if comp(order[third], order[second]):
            swap(second, third)
        return True

    def sort4(first: int, second: int, third: int, fourth: int) -> None:
        sort3(first, second, third)
        if comp(order[fourth], order[third]):
            swap(third, fourth)
            if comp(order[third], order[second]):
                swap(second, third)
                if comp(order[second], order[first]):
                    swap(first, second)

    def sort5(first: int, second: int, third: int, fourth: int, fifth: int) -> None:
        sort4(first, second, third, fourth)
        if comp(order[fifth], order[fourth]):
            swap(fourth, fifth)
            if comp(order[fourth], order[third]):
                swap(third, fourth)
                if comp(order[third], order[second]):
                    swap(second, third)
                    if comp(order[second], order[first]):
                        swap(first, second)

    def insertion_sort(first: int, last: int) -> None:
        for index in range(first + 1, last):
            previous = index - 1
            if comp(order[index], order[previous]):
                item = order[index]
                read = previous
                write = index
                while True:
                    order[write] = order[read]
                    write = read
                    if write == first:
                        break
                    read -= 1
                    if not comp(item, order[read]):
                        break
                order[write] = item

    def insertion_sort_unguarded(first: int, last: int) -> None:
        for index in range(first + 1, last):
            previous = index - 1
            if comp(order[index], order[previous]):
                item = order[index]
                read = previous
                write = index
                while True:
                    order[write] = order[read]
                    write = read
                    read -= 1
                    if not comp(item, order[read]):
                        break
                order[write] = item

    def insertion_sort_incomplete(first: int, last: int) -> bool:
        length = last - first
        if length <= 1:
            return True
        if length == 2:
            if comp(order[last - 1], order[first]):
                swap(first, last - 1)
            return True
        if length == 3:
            sort3(first, first + 1, last - 1)
            return True
        if length == 4:
            sort4(first, first + 1, first + 2, last - 1)
            return True
        if length == 5:
            sort5(first, first + 1, first + 2, first + 3, last - 1)
            return True

        previous = first + 2
        sort3(first, first + 1, previous)
        move_limit = 8
        moves = 0
        for index in range(previous + 1, last):
            if comp(order[index], order[previous]):
                item = order[index]
                read = previous
                write = index
                while True:
                    order[write] = order[read]
                    write = read
                    if write == first:
                        break
                    read -= 1
                    if not comp(item, order[read]):
                        break
                order[write] = item
                moves += 1
                if moves == move_limit:
                    return index + 1 == last
            previous = index
        return True

    def partition_equals_on_right(first: int, last: int) -> tuple[int, bool]:
        begin = first
        pivot = order[first]
        while True:
            first += 1
            if not comp(order[first], pivot):
                break
        if begin == first - 1:
            while first < last:
                last -= 1
                if comp(order[last], pivot):
                    break
        else:
            while True:
                last -= 1
                if comp(order[last], pivot):
                    break
        already_partitioned = first >= last
        while first < last:
            swap(first, last)
            while True:
                first += 1
                if not comp(order[first], pivot):
                    break
            while True:
                last -= 1
                if comp(order[last], pivot):
                    break
        pivot_pos = first - 1
        if begin != pivot_pos:
            order[begin] = order[pivot_pos]
        order[pivot_pos] = pivot
        return pivot_pos, already_partitioned

    def partition_equals_on_left(first: int, last: int) -> int:
        begin = first
        pivot = order[first]
        if comp(pivot, order[last - 1]):
            while True:
                first += 1
                if comp(pivot, order[first]):
                    break
        else:
            while True:
                first += 1
                if not (first < last and not comp(pivot, order[first])):
                    break
        if first < last:
            while True:
                last -= 1
                if not comp(pivot, order[last]):
                    break
        while first < last:
            swap(first, last)
            while True:
                first += 1
                if comp(pivot, order[first]):
                    break
            while True:
                last -= 1
                if not comp(pivot, order[last]):
                    break
        pivot_pos = first - 1
        if begin != pivot_pos:
            order[begin] = order[pivot_pos]
        order[pivot_pos] = pivot
        return first

    def introsort(first: int, last: int, depth: int, leftmost: bool = True) -> None:
        insertion_limit = 24
        ninther_threshold = 128
        while True:
            length = last - first
            if length <= 1:
                return
            if length == 2:
                if comp(order[last - 1], order[first]):
                    swap(first, last - 1)
                return
            if length == 3:
                sort3(first, first + 1, last - 1)
                return
            if length == 4:
                sort4(first, first + 1, first + 2, last - 1)
                return
            if length == 5:
                sort5(first, first + 1, first + 2, first + 3, last - 1)
                return
            if length < insertion_limit:
                if leftmost:
                    insertion_sort(first, last)
                else:
                    insertion_sort_unguarded(first, last)
                return
            if depth == 0:
                order[first:last] = sorted(
                    order[first:last],
                    key=lambda index: (not bool_values[index], index),
                )
                return
            depth -= 1
            half = length // 2
            if length > ninther_threshold:
                sort3(first, first + half, last - 1)
                sort3(first + 1, first + half - 1, last - 2)
                sort3(first + 2, first + half + 1, last - 3)
                sort3(first + half - 1, first + half, first + half + 1)
                swap(first, first + half)
            else:
                sort3(first + half, first, last - 1)

            if not leftmost and not comp(order[first - 1], order[first]):
                first = partition_equals_on_left(first, last)
                continue

            pivot_pos, already_partitioned = partition_equals_on_right(first, last)
            if already_partitioned:
                first_sorted = insertion_sort_incomplete(first, pivot_pos)
                if insertion_sort_incomplete(pivot_pos + 1, last):
                    if first_sorted:
                        return
                    last = pivot_pos
                    continue
                if first_sorted:
                    first = pivot_pos + 1
                    continue

            introsort(first, pivot_pos, depth, leftmost)
            leftmost = False
            first = pivot_pos + 1

    size = len(order)
    if size > 1:
        introsort(0, size, 2 * (size.bit_length() - 1), True)
    return np.asarray(order, dtype=np.int64)


class MaskletConfirmationStatus(Enum):
    UNCONFIRMED = 1
    CONFIRMED = 2


class Sam3MultiplexTrackerPredictor(nn.Module):
    def __init__(
        self,
        config_file: Any,
        checkpoint_file: Any = None,
        hydra_overrides: Any = None,
        per_obj_inference: bool = False,
        fill_hole_area: int = 0,
        use_fa3: bool = False,
        use_rope_real: bool = False,
        keep_first_cond_frame: bool = False,
        is_multiplex: bool = False,
        is_multiplex_dynamic: bool = False,
        use_memory_selection: bool = False,
    ):
        del config_file, checkpoint_file, hydra_overrides, per_obj_inference
        del fill_hole_area, use_fa3, use_rope_real, keep_first_cond_frame
        del is_multiplex, is_multiplex_dynamic, use_memory_selection
        super().__init__()
        raise_unsupported_multiplex_runtime("Sam3MultiplexTrackerPredictor")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise_unsupported_multiplex_runtime("Sam3MultiplexTrackerPredictor.forward")

    def add_output_per_object(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise_unsupported_multiplex_runtime(
            "Sam3MultiplexTrackerPredictor.add_output_per_object"
        )


class Sam3MultiplexBase(Sam3VideoBase):
    def __init__(
        self,
        tracker: Any,
        detector: Any,
        ckpt_path: Any = None,
        sam3_ckpt_path: Any = None,
        score_threshold_detection: float = 0.5,
        image_only_det_thresh: float = 0.5,
        det_nms_thresh: float = 0.0,
        det_nms_use_iom: bool = False,
        assoc_iou_thresh: float = 0.5,
        trk_assoc_iou_thresh: float = 0.5,
        new_det_thresh: float = 0.5,
        hotstart_delay: int = 0,
        hotstart_unmatch_thresh: int = 3,
        hotstart_dup_thresh: int = 3,
        suppress_unmatched_only_within_hotstart: bool = True,
        init_trk_keep_alive: int = 0,
        max_trk_keep_alive: int = 8,
        min_trk_keep_alive: int = -4,
        suppress_overlapping_based_on_recent_occlusion_threshold: float = 0.0,
        allow_unoccluded_to_suppress: bool = False,
        decrease_trk_keep_alive_for_empty_masklets: bool = False,
        o2o_matching_masklets_enable: bool = False,
        suppress_det_close_to_boundary: bool = False,
        fill_hole_area: int = 16,
        sprinkle_removal_area: int = 16,
        max_num_objects: int = 128,
        max_num_kboxes: int = 20,
        recondition_every_nth_frame: int = -1,
        use_iom_recondition: bool = False,
        iom_thresh_recondition: float = 0.8,
        iou_thresh_recondition: float = 0.8,
        is_multiplex: bool = False,
        masklet_confirmation_enable: bool = False,
        masklet_confirmation_consecutive_det_thresh: int = 3,
        reconstruction_bbox_iou_thresh: float = 0.0,
        reconstruction_bbox_det_score: float = 0.5,
        reapply_no_object_pointer: bool = False,
        running_in_prod: bool = False,
        use_batched_grounding: bool = False,
        batched_grounding_batch_size: int = 1,
        **kwargs: Any,
    ):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected Sam3MultiplexBase keyword argument(s): {names}"
            )
        if ckpt_path is not None or sam3_ckpt_path is not None:
            raise_unsupported_multiplex_runtime("Sam3MultiplexBase checkpoint loading")

        super().__init__(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=score_threshold_detection,
            det_nms_thresh=det_nms_thresh,
            assoc_iou_thresh=assoc_iou_thresh,
            trk_assoc_iou_thresh=trk_assoc_iou_thresh,
            new_det_thresh=new_det_thresh,
            hotstart_delay=hotstart_delay,
            hotstart_unmatch_thresh=hotstart_unmatch_thresh,
            hotstart_dup_thresh=hotstart_dup_thresh,
            suppress_unmatched_only_within_hotstart=(
                suppress_unmatched_only_within_hotstart
            ),
            init_trk_keep_alive=init_trk_keep_alive,
            max_trk_keep_alive=max_trk_keep_alive,
            min_trk_keep_alive=min_trk_keep_alive,
            suppress_overlapping_based_on_recent_occlusion_threshold=(
                suppress_overlapping_based_on_recent_occlusion_threshold
            ),
            decrease_trk_keep_alive_for_empty_masklets=(
                decrease_trk_keep_alive_for_empty_masklets
            ),
            o2o_matching_masklets_enable=o2o_matching_masklets_enable,
            suppress_det_close_to_boundary=suppress_det_close_to_boundary,
            fill_hole_area=fill_hole_area,
            max_num_objects=max_num_objects,
            recondition_every_nth_frame=recondition_every_nth_frame,
            masklet_confirmation_enable=masklet_confirmation_enable,
            masklet_confirmation_consecutive_det_thresh=(
                masklet_confirmation_consecutive_det_thresh
            ),
            reconstruction_bbox_iou_thresh=reconstruction_bbox_iou_thresh,
            reconstruction_bbox_det_score=reconstruction_bbox_det_score,
        )
        self.image_only_det_thresh = image_only_det_thresh
        self.det_nms_use_iom = det_nms_use_iom
        self.is_multiplex = is_multiplex
        self.running_in_prod = running_in_prod
        if hasattr(self.detector, "running_in_prod"):
            self.detector.running_in_prod = running_in_prod

        tracker_is_multiplex = getattr(self.tracker, "is_multiplex", is_multiplex)
        detector_is_multiplex = getattr(self.detector, "is_multiplex", is_multiplex)
        assert self.is_multiplex == tracker_is_multiplex == detector_is_multiplex, (
            "is_multiplex must be the same for all models: "
            f"{self.is_multiplex=}, {tracker_is_multiplex=}, {detector_is_multiplex=}"
        )

        self.allow_unoccluded_to_suppress = allow_unoccluded_to_suppress
        self.sprinkle_removal_area = sprinkle_removal_area
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self._dist_pg_cpu = None
        self._profiler = None
        self._frame_count = 0
        self._profile_save_dir = os.getenv("PROFILE_SAVE_DIR", "/tmp/profiling")
        self._profiling_enabled = os.getenv("ENABLE_PROFILING", "0").lower() == "1"

        if max_num_objects > 0:
            multiplex_divisor = (
                self.tracker.multiplex_controller.allowed_bucket_capacity
                if self.is_multiplex
                else 1
            )
            self.num_obj_for_compile = math.ceil(
                max_num_objects / (self.world_size * multiplex_divisor)
            )
            self.max_num_objects = max_num_objects
        else:
            self.max_num_objects = 10000
            self.num_obj_for_compile = 16
        self.max_num_kboxes = max_num_kboxes
        self.use_iom_recondition = use_iom_recondition
        self.iom_thresh_recondition = iom_thresh_recondition
        self.iou_thresh_recondition = iou_thresh_recondition
        self.reapply_no_object_pointer = reapply_no_object_pointer
        self.use_batched_grounding = use_batched_grounding
        self.batched_grounding_batch_size = batched_grounding_batch_size

        if self.is_multiplex:
            assert not self.tracker.multiplex_controller.training, (
                "This model class should only be used for eval."
            )
            self.bucket_capacity = (
                self.tracker.multiplex_controller.allowed_bucket_capacity
            )

    def _initialize_metadata(self) -> dict[str, Any]:
        """Initialize the SAM3 masklet metadata structure."""
        score_key = (
            "obj_id_to_sam2_score_frame_wise"
            if self.is_multiplex
            else "obj_id_to_tracker_score_frame_wise"
        )
        tracker_metadata: dict[str, Any] = {
            "obj_ids_per_gpu": [
                np.array([], dtype=np.int64) for _ in range(self.world_size)
            ],
            "obj_ids_all_gpu": np.array([], dtype=np.int64),
            "num_obj_per_gpu": np.zeros(self.world_size, dtype=np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            score_key: defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        if self.is_multiplex:
            tracker_metadata["gpu_metadata"] = {"N_obj": 0}
            tracker_metadata["num_buc_per_gpu"] = np.zeros(
                self.world_size,
                dtype=np.int64,
            )

        if self.is_multiplex or self.rank == 0:
            rank0_metadata: dict[str, Any] = {
                "obj_first_frame_idx": {},
                "unmatched_frame_inds": defaultdict(list),
                "trk_keep_alive": defaultdict(int),
                "overlap_pair_to_frame_inds": defaultdict(list),
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(set),
            }
            if self.masklet_confirmation_enable:
                rank0_metadata["masklet_confirmation"] = {
                    "status": np.array([], dtype=np.int64),
                    "consecutive_det_num": np.array([], dtype=np.int64),
                }
            tracker_metadata["rank0_metadata"] = rank0_metadata
        return tracker_metadata

    def _active_tracker_object_count(self, tracker_metadata: dict[str, Any]) -> int:
        if not tracker_metadata:
            return 0
        return int(np.asarray(tracker_metadata.get("obj_ids_all_gpu", [])).size)

    def _create_planning_metadata(
        self,
        tracker_metadata_prev: dict[str, Any],
    ) -> dict[str, Any]:
        """Create the planner metadata shell from prior frame state."""
        score_key = "obj_id_to_tracker_score_frame_wise"
        if score_key not in tracker_metadata_prev:
            score_key = "obj_id_to_sam2_score_frame_wise"

        metadata: dict[str, Any] = {
            "obj_ids_per_gpu": _copy_metadata_value(
                tracker_metadata_prev["obj_ids_per_gpu"]
            ),
            "obj_ids_all_gpu": None,
            "num_obj_per_gpu": _copy_metadata_value(
                tracker_metadata_prev["num_obj_per_gpu"]
            ),
            "obj_id_to_score": _copy_metadata_value(
                tracker_metadata_prev["obj_id_to_score"]
            ),
            score_key: _copy_metadata_value(tracker_metadata_prev[score_key]),
            "obj_id_to_last_occluded": {},
            "max_obj_id": _copy_metadata_value(tracker_metadata_prev["max_obj_id"]),
        }
        if "rank0_metadata" in tracker_metadata_prev:
            metadata["rank0_metadata"] = _copy_metadata_value(
                tracker_metadata_prev["rank0_metadata"]
            )
        if self.is_multiplex:
            metadata["num_buc_per_gpu"] = _copy_metadata_value(
                tracker_metadata_prev["num_buc_per_gpu"]
            )
            metadata["gpu_metadata"] = tracker_metadata_prev["gpu_metadata"]
        elif "gpu_metadata" in tracker_metadata_prev:
            metadata["gpu_metadata"] = tracker_metadata_prev["gpu_metadata"]
        return metadata

    def _post_execution_phase_hook(
        self,
        tracker_states_local: list[Any],
        tracker_metadata_new: dict[str, Any] | None,
    ) -> None:
        """Update multiplex bucket counts after a local execution phase."""
        if self.is_multiplex and tracker_metadata_new is not None:
            tracker_metadata_new["num_buc_per_gpu"][self.rank] = (
                self._count_buckets_in_states(tracker_states_local)
            )

    def _count_buckets_in_states(self, tracker_states_local: list[Any]) -> int:
        """Count dynamic multiplex buckets across local tracker states."""
        if not self.is_multiplex:
            return 0
        total_buckets = 0
        for state in tracker_states_local:
            if "multiplex_state" in state:
                total_buckets += int(state["multiplex_state"].num_buckets)
        return total_buckets

    def _tracker_state_obj_ids_array(
        self,
        tracker_states_local: list[Any],
    ) -> np.ndarray:
        obj_ids: list[int] = []
        for state in tracker_states_local:
            if not isinstance(state, dict):
                continue
            multiplex_state = state.get("multiplex_state")
            state_obj_ids = state.get("obj_ids")
            if state_obj_ids is None and multiplex_state is not None:
                state_obj_ids = getattr(multiplex_state, "object_ids", [])
            if state_obj_ids is None:
                state_obj_ids = []
            if _is_mlx_array(state_obj_ids) or isinstance(state_obj_ids, np.ndarray):
                obj_ids.extend(
                    int(obj_id)
                    for obj_id in _array_to_numpy(
                        state_obj_ids,
                        dtype=np.int64,
                    )
                    .reshape(-1)
                    .tolist()
                )
            else:
                obj_ids.extend(int(obj_id) for obj_id in state_obj_ids)
        return np.asarray(obj_ids, dtype=np.int64)

    def _align_hotstart_gpu_metadata_to_object_ids(
        self,
        tracker_metadata: dict[str, Any],
        gpu_metadata: dict[str, Any],
        *,
        obj_ids_prev: np.ndarray,
        obj_ids_updated: np.ndarray,
        frame_idx: int,
    ) -> dict[str, Any]:
        obj_ids_prev = np.asarray(obj_ids_prev, dtype=np.int64).reshape(-1)
        obj_ids_updated = np.asarray(obj_ids_updated, dtype=np.int64).reshape(-1)
        if obj_ids_updated.size == 0:
            return self._empty_hotstart_gpu_metadata()

        required_keys = (
            "obj_first_frame",
            "consecutive_unmatch_count",
            "trk_keep_alive",
            "removed_mask",
            "overlap_pair_counts",
            "last_occluded_tensor",
        )
        if any(key not in gpu_metadata for key in required_keys):
            rank0_metadata = _copy_metadata_value(tracker_metadata)
            rank0_metadata["obj_ids_all_gpu"] = obj_ids_updated
            return self._hotstart_gpu_metadata_from_rank0(
                rank0_metadata,
                frame_idx=frame_idx,
                num_objects=int(obj_ids_updated.size),
            )

        num_prev_metadata = int(gpu_metadata.get("N_obj", 0))
        if num_prev_metadata != obj_ids_prev.size:
            if num_prev_metadata == obj_ids_updated.size:
                current_metadata = {
                    **tracker_metadata,
                    "obj_ids_all_gpu": obj_ids_updated,
                }
                return self._ensure_hotstart_gpu_metadata(
                    current_metadata,
                    gpu_metadata,
                    frame_idx=frame_idx,
                    num_objects=int(obj_ids_updated.size),
                )
            raise ValueError(
                "hotstart gpu metadata N_obj must match previous tracker object ids; "
                f"got {num_prev_metadata} and {obj_ids_prev.size}."
            )

        previous_metadata = {**tracker_metadata, "obj_ids_all_gpu": obj_ids_prev}
        gpu_metadata = self._ensure_hotstart_gpu_metadata(
            previous_metadata,
            gpu_metadata,
            frame_idx=frame_idx,
            num_objects=int(obj_ids_prev.size),
        )
        old_index_by_obj_id = {
            int(obj_id): idx for idx, obj_id in enumerate(obj_ids_prev.tolist())
        }
        missing_obj_ids = [
            int(obj_id)
            for obj_id in obj_ids_updated.tolist()
            if int(obj_id) not in old_index_by_obj_id
        ]
        if missing_obj_ids:
            raise ValueError(
                "hotstart gpu metadata previous object ids must contain the current "
                f"tracker object ids; missing {missing_obj_ids}."
            )

        keep_indices = mx.array(
            [old_index_by_obj_id[int(obj_id)] for obj_id in obj_ids_updated],
            dtype=mx.int64,
        )
        aligned: dict[str, Any] = {"N_obj": int(obj_ids_updated.size)}
        for key in (
            "obj_first_frame",
            "consecutive_unmatch_count",
            "trk_keep_alive",
            "removed_mask",
            "last_occluded_tensor",
        ):
            aligned[key] = mx.take(gpu_metadata[key], keep_indices, axis=0)
        aligned["overlap_pair_counts"] = mx.take(
            mx.take(
                gpu_metadata["overlap_pair_counts"],
                keep_indices,
                axis=0,
            ),
            keep_indices,
            axis=1,
        )
        return aligned

    def _sync_execution_phase_metadata(
        self,
        tracker_states_local: list[Any],
        tracker_metadata_new: dict[str, Any],
        *,
        frame_idx: int,
    ) -> None:
        obj_ids_prev_value = tracker_metadata_new.get("obj_ids_all_gpu")
        obj_ids_prev = (
            np.asarray(obj_ids_prev_value, dtype=np.int64).reshape(-1)
            if obj_ids_prev_value is not None
            else np.array([], dtype=np.int64)
        )
        obj_ids_local = self._tracker_state_obj_ids_array(tracker_states_local)
        tracker_metadata_new["obj_ids_per_gpu"][self.rank] = obj_ids_local
        tracker_metadata_new["num_obj_per_gpu"][self.rank] = int(obj_ids_local.size)
        tracker_metadata_new["obj_ids_all_gpu"] = np.concatenate(
            [
                np.asarray(obj_ids, dtype=np.int64).reshape(-1)
                for obj_ids in tracker_metadata_new["obj_ids_per_gpu"]
            ]
        )

        if not self.is_multiplex:
            return

        tracker_metadata_new["num_buc_per_gpu"][self.rank] = (
            self._count_execution_phase_buckets(tracker_states_local)
        )
        tracker_metadata_new["gpu_metadata"] = (
            self._align_hotstart_gpu_metadata_to_object_ids(
                tracker_metadata_new,
                tracker_metadata_new.get("gpu_metadata", {"N_obj": 0}),
                obj_ids_prev=obj_ids_prev,
                obj_ids_updated=tracker_metadata_new["obj_ids_all_gpu"],
                frame_idx=frame_idx,
            )
        )

    def _count_execution_phase_buckets(
        self,
        tracker_states_local: list[Any],
    ) -> int:
        num_unpacked_objects = 0
        num_packed_buckets = 0
        for state in tracker_states_local:
            if not isinstance(state, dict):
                continue
            multiplex_state = state.get("multiplex_state")
            state_obj_ids = state.get("obj_ids")
            if state_obj_ids is None and multiplex_state is not None:
                state_obj_ids = getattr(multiplex_state, "object_ids", [])
            if state_obj_ids is None:
                state_obj_ids = []
            if _is_mlx_array(state_obj_ids) or isinstance(state_obj_ids, np.ndarray):
                obj_count = int(
                    _array_to_numpy(state_obj_ids, dtype=np.int64).reshape(-1).size
                )
            else:
                obj_count = len(state_obj_ids)
            if multiplex_state is not None:
                num_packed_buckets += int(multiplex_state.num_buckets)
            else:
                num_unpacked_objects += obj_count
        num_unpacked_buckets = (
            math.ceil(num_unpacked_objects / self.bucket_capacity)
            if num_unpacked_objects > 0
            else 0
        )
        return num_packed_buckets + num_unpacked_buckets

    def update_masklet_confirmation_status(
        self,
        rank0_metadata: dict[str, Any],
        obj_ids_all_gpu_prev: np.ndarray,
        obj_ids_all_gpu_updated: np.ndarray,
        det_to_matched_trk_obj_ids: dict[int, Any],
        new_det_obj_ids: np.ndarray,
    ) -> dict[str, Any]:
        confirmation_data = rank0_metadata["masklet_confirmation"]
        status_prev = np.asarray(confirmation_data["status"], dtype=np.int64)
        consecutive_det_num_prev = np.asarray(
            confirmation_data["consecutive_det_num"],
            dtype=np.int64,
        )
        obj_ids_prev = np.asarray(obj_ids_all_gpu_prev, dtype=np.int64).reshape(-1)
        obj_ids_updated = np.asarray(obj_ids_all_gpu_updated, dtype=np.int64).reshape(
            -1
        )

        if status_prev.shape[0] != obj_ids_prev.shape[0]:
            raise ValueError(
                "masklet_confirmation status length must match previous obj ids, "
                f"got {status_prev.shape[0]} and {obj_ids_prev.shape[0]}."
            )
        if consecutive_det_num_prev.shape[0] != obj_ids_prev.shape[0]:
            raise ValueError(
                "masklet_confirmation consecutive_det_num length must match previous "
                f"obj ids, got {consecutive_det_num_prev.shape[0]} and "
                f"{obj_ids_prev.shape[0]}."
            )

        status = np.full(
            obj_ids_updated.shape,
            MaskletConfirmationStatus.UNCONFIRMED.value,
            dtype=np.int64,
        )
        consecutive_det_num = np.zeros(obj_ids_updated.shape, dtype=np.int64)

        if obj_ids_prev.size > 0 and obj_ids_updated.size > 0:
            obj_id_to_new_idx = {
                int(obj_id): idx for idx, obj_id in enumerate(obj_ids_updated)
            }
            for old_idx, obj_id in enumerate(obj_ids_prev):
                new_idx = obj_id_to_new_idx.get(int(obj_id))
                if new_idx is not None:
                    status[new_idx] = status_prev[old_idx]
                    consecutive_det_num[new_idx] = consecutive_det_num_prev[old_idx]

        matched_obj_ids = {
            int(obj_id)
            for obj_id in np.asarray(new_det_obj_ids, dtype=np.int64).reshape(-1)
        }
        for matched_trk_ids in det_to_matched_trk_obj_ids.values():
            matched_obj_ids.update(
                int(obj_id)
                for obj_id in np.asarray(matched_trk_ids, dtype=np.int64).reshape(-1)
            )

        for idx, obj_id in enumerate(obj_ids_updated):
            if int(obj_id) in matched_obj_ids:
                consecutive_det_num[idx] += 1
            else:
                consecutive_det_num[idx] = 0
            if (
                consecutive_det_num[idx]
                >= self.masklet_confirmation_consecutive_det_thresh
            ):
                status[idx] = MaskletConfirmationStatus.CONFIRMED.value

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return rank0_metadata

    def _hotstart_gpu_metadata_from_rank0(
        self,
        tracker_metadata_prev: dict[str, Any],
        *,
        frame_idx: int,
        num_objects: int,
    ) -> dict[str, Any]:
        obj_ids = np.asarray(
            tracker_metadata_prev.get("obj_ids_all_gpu", []),
            dtype=np.int64,
        ).reshape(-1)
        if obj_ids.size != num_objects:
            raise ValueError(
                "hotstart gpu metadata must align with tracker object ids; got "
                f"{num_objects} objects and {obj_ids.size} object ids."
            )

        rank0_metadata = tracker_metadata_prev.get("rank0_metadata", {})
        obj_first_frame_idx = rank0_metadata.get("obj_first_frame_idx", {})
        unmatched_frame_inds = rank0_metadata.get("unmatched_frame_inds", {})
        trk_keep_alive = rank0_metadata.get("trk_keep_alive", {})
        removed_obj_ids = {
            int(obj_id) for obj_id in rank0_metadata.get("removed_obj_ids", set())
        }
        overlap_pair_to_frame_inds = rank0_metadata.get(
            "overlap_pair_to_frame_inds",
            {},
        )
        obj_id_to_last_occluded = tracker_metadata_prev.get(
            "obj_id_to_last_occluded", {}
        )
        obj_id_to_idx = {
            int(obj_id): idx for idx, obj_id in enumerate(obj_ids.tolist())
        }

        overlap_pair_counts = np.zeros((num_objects, num_objects), dtype=np.int64)
        for key, frame_indices in overlap_pair_to_frame_inds.items():
            if len(key) != 2:
                continue
            first_obj_id, obj_id = (int(key[0]), int(key[1]))
            first_idx = obj_id_to_idx.get(first_obj_id)
            obj_idx = obj_id_to_idx.get(obj_id)
            if first_idx is not None and obj_idx is not None:
                overlap_pair_counts[first_idx, obj_idx] = len(frame_indices)

        return {
            "N_obj": int(num_objects),
            "obj_first_frame": mx.array(
                [
                    int(obj_first_frame_idx.get(int(obj_id), frame_idx))
                    for obj_id in obj_ids.tolist()
                ],
                dtype=mx.int64,
            ),
            "consecutive_unmatch_count": mx.array(
                [
                    len(unmatched_frame_inds.get(int(obj_id), []))
                    for obj_id in obj_ids.tolist()
                ],
                dtype=mx.int64,
            ),
            "trk_keep_alive": mx.array(
                [
                    int(trk_keep_alive.get(int(obj_id), 0))
                    for obj_id in obj_ids.tolist()
                ],
                dtype=mx.int64,
            ),
            "removed_mask": mx.array(
                [int(obj_id) in removed_obj_ids for obj_id in obj_ids.tolist()],
                dtype=mx.bool_,
            ),
            "overlap_pair_counts": mx.array(overlap_pair_counts, dtype=mx.int64),
            "last_occluded_tensor": mx.array(
                [
                    int(obj_id_to_last_occluded.get(int(obj_id), -1))
                    for obj_id in obj_ids.tolist()
                ],
                dtype=mx.int64,
            ),
        }

    def _ensure_hotstart_gpu_metadata(
        self,
        tracker_metadata_prev: dict[str, Any],
        gpu_metadata_prev: dict[str, Any],
        *,
        frame_idx: int,
        num_objects: int,
    ) -> dict[str, Any]:
        required_shapes = {
            "obj_first_frame": (num_objects,),
            "consecutive_unmatch_count": (num_objects,),
            "trk_keep_alive": (num_objects,),
            "removed_mask": (num_objects,),
            "overlap_pair_counts": (num_objects, num_objects),
            "last_occluded_tensor": (num_objects,),
        }
        if any(key not in gpu_metadata_prev for key in required_shapes):
            return self._hotstart_gpu_metadata_from_rank0(
                tracker_metadata_prev,
                frame_idx=frame_idx,
                num_objects=num_objects,
            )

        metadata: dict[str, Any] = {"N_obj": int(gpu_metadata_prev.get("N_obj", 0))}
        if metadata["N_obj"] != num_objects:
            raise ValueError(
                "hotstart gpu metadata N_obj must match association objects; got "
                f"{metadata['N_obj']} and {num_objects}."
            )
        dtype_by_key = {
            "obj_first_frame": mx.int64,
            "consecutive_unmatch_count": mx.int64,
            "trk_keep_alive": mx.int64,
            "removed_mask": mx.bool_,
            "overlap_pair_counts": mx.int64,
            "last_occluded_tensor": mx.int64,
        }
        for key, expected_shape in required_shapes.items():
            value = gpu_metadata_prev[key]
            value_mx = value if _is_mlx_array(value) else mx.array(value)
            value_mx = value_mx.astype(dtype_by_key[key])
            if value_mx.shape != expected_shape:
                raise ValueError(
                    f"hotstart gpu metadata {key} must have shape "
                    f"{expected_shape}, got {value_mx.shape}."
                )
            metadata[key] = value_mx
        return metadata

    def _process_hotstart_gpu(
        self,
        frame_idx: int,
        reverse: bool,
        adt_result: Any,
        tracker_metadata_prev: dict[str, Any],
        gpu_metadata_prev: dict[str, Any],
    ) -> tuple[Any, Any, dict[str, Any]]:
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._process_hotstart_gpu(distributed)"
            )
        if not isinstance(adt_result, LazyAssociateDetTrkResult):
            empty_mask = mx.zeros((0,), dtype=mx.bool_)
            return empty_mask, empty_mask, {"N_obj": 0}

        trk_is_unmatched = (
            adt_result.trk_is_unmatched
            if _is_mlx_array(adt_result.trk_is_unmatched)
            else mx.array(adt_result.trk_is_unmatched)
        ).astype(mx.bool_)
        trk_is_nonempty = (
            adt_result.trk_is_nonempty
            if _is_mlx_array(adt_result.trk_is_nonempty)
            else mx.array(adt_result.trk_is_nonempty)
        ).astype(mx.bool_)
        im_mask = (
            adt_result.im_mask
            if _is_mlx_array(adt_result.im_mask)
            else mx.array(adt_result.im_mask)
        ).astype(mx.bool_)
        num_objects = int(trk_is_unmatched.shape[0])
        if trk_is_nonempty.shape != trk_is_unmatched.shape:
            raise ValueError(
                "hotstart association track metadata must have matching shapes; "
                f"got {trk_is_unmatched.shape} and {trk_is_nonempty.shape}."
            )
        if im_mask.ndim != 2 or int(im_mask.shape[1]) != num_objects:
            raise ValueError(
                "hotstart association im_mask must have shape (N_det, N_obj); "
                f"got {im_mask.shape} for {num_objects} objects."
            )
        gpu_metadata = self._ensure_hotstart_gpu_metadata(
            tracker_metadata_prev,
            gpu_metadata_prev,
            frame_idx=frame_idx,
            num_objects=num_objects,
        )

        obj_first_frame = gpu_metadata["obj_first_frame"]
        consecutive_unmatch_count = gpu_metadata["consecutive_unmatch_count"]
        trk_keep_alive = gpu_metadata["trk_keep_alive"]
        removed_mask = gpu_metadata["removed_mask"]
        overlap_pair_counts = gpu_metadata["overlap_pair_counts"]
        last_occluded = gpu_metadata["last_occluded_tensor"]

        trk_is_matched = mx.any(im_mask, axis=0)
        trk_keep_alive = mx.where(
            trk_is_matched,
            trk_keep_alive + 1,
            trk_keep_alive - 1,
        )
        trk_keep_alive = mx.clip(
            trk_keep_alive,
            int(self.min_trk_keep_alive),
            int(self.max_trk_keep_alive),
        )
        if self.decrease_trk_keep_alive_for_empty_masklets:
            trk_keep_alive = mx.where(
                ~trk_is_nonempty,
                mx.clip(
                    trk_keep_alive - 1,
                    int(self.min_trk_keep_alive),
                    int(self.max_trk_keep_alive),
                ),
                trk_keep_alive,
            )

        consecutive_unmatch_count = mx.where(
            trk_is_unmatched,
            consecutive_unmatch_count + 1,
            consecutive_unmatch_count,
        )

        if im_mask.shape[0] == 0 or num_objects == 0:
            overlap_increment = mx.zeros((num_objects, num_objects), dtype=mx.int64)
        else:
            tracks_per_det = mx.sum(im_mask.astype(mx.int64), axis=1)
            multi_match_tracks = im_mask & (tracks_per_det > 1)[:, None]
            multi_match_float = multi_match_tracks.astype(mx.float32)
            overlap_increment = mx.matmul(
                mx.swapaxes(multi_match_float, 0, 1),
                multi_match_float,
            ).astype(mx.int64)
            overlap_increment = mx.triu(overlap_increment, k=1)
        overlap_pair_counts = overlap_pair_counts + overlap_increment

        hotstart_diff = (
            int(frame_idx) - int(self.hotstart_delay)
            if not reverse
            else int(frame_idx) + int(self.hotstart_delay)
        )
        is_within_hotstart = (
            obj_first_frame > hotstart_diff
            if not reverse
            else obj_first_frame < hotstart_diff
        )
        remove_by_unmatch = (
            is_within_hotstart
            & (consecutive_unmatch_count >= int(self.hotstart_unmatch_thresh))
            & ~removed_mask
        )

        if self.suppress_unmatched_only_within_hotstart:
            suppress_by_unmatch = mx.zeros((num_objects,), dtype=mx.bool_)
        else:
            suppress_by_unmatch = (
                (trk_keep_alive <= 0) & ~removed_mask & ~remove_by_unmatch
            )

        if num_objects == 0:
            remove_by_overlap = mx.zeros((0,), dtype=mx.bool_)
        else:
            first_frames_i = obj_first_frame[:, None]
            first_frames_j = obj_first_frame[None, :]
            is_earlier_matrix = (
                first_frames_i < first_frames_j
                if not reverse
                else first_frames_i > first_frames_j
            )
            overlap_with_earlier = mx.where(
                is_earlier_matrix,
                overlap_pair_counts,
                mx.zeros_like(overlap_pair_counts),
            )
            max_overlap_with_earlier = mx.max(overlap_with_earlier, axis=0)
            remove_by_overlap = (
                is_within_hotstart
                & (max_overlap_with_earlier >= int(self.hotstart_dup_thresh))
                & ~removed_mask
            )

        to_remove = remove_by_unmatch | remove_by_overlap
        to_suppress = suppress_by_unmatch
        removed_mask = removed_mask | to_remove

        return (
            to_remove,
            to_suppress,
            {
                "N_obj": num_objects,
                "obj_first_frame": obj_first_frame,
                "consecutive_unmatch_count": consecutive_unmatch_count,
                "trk_keep_alive": trk_keep_alive,
                "removed_mask": removed_mask,
                "overlap_pair_counts": overlap_pair_counts,
                "last_occluded_tensor": last_occluded,
            },
        )

    def _compact_hotstart_gpu_metadata(
        self,
        gpu_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        num_objects = int(gpu_metadata.get("N_obj", 0))
        if num_objects == 0:
            return self._empty_hotstart_gpu_metadata()
        removed_mask = (
            gpu_metadata["removed_mask"]
            if _is_mlx_array(gpu_metadata["removed_mask"])
            else mx.array(gpu_metadata["removed_mask"])
        ).astype(mx.bool_)
        keep_indices_np = np.nonzero(
            ~_array_to_numpy(removed_mask, dtype=bool).reshape(-1)
        )[0].astype(np.int64)
        keep_indices = mx.array(keep_indices_np, dtype=mx.int64)
        compacted: dict[str, Any] = {"N_obj": int(keep_indices_np.size)}
        for key in (
            "obj_first_frame",
            "consecutive_unmatch_count",
            "trk_keep_alive",
            "removed_mask",
            "last_occluded_tensor",
        ):
            compacted[key] = mx.take(gpu_metadata[key], keep_indices, axis=0)
        overlap_pair_counts = mx.take(
            mx.take(gpu_metadata["overlap_pair_counts"], keep_indices, axis=0),
            keep_indices,
            axis=1,
        )
        compacted["overlap_pair_counts"] = overlap_pair_counts
        return compacted

    def _empty_hotstart_gpu_metadata(self) -> dict[str, Any]:
        return {
            "N_obj": 0,
            "obj_first_frame": mx.zeros((0,), dtype=mx.int64),
            "consecutive_unmatch_count": mx.zeros((0,), dtype=mx.int64),
            "trk_keep_alive": mx.zeros((0,), dtype=mx.int64),
            "removed_mask": mx.zeros((0,), dtype=mx.bool_),
            "overlap_pair_counts": mx.zeros((0, 0), dtype=mx.int64),
            "last_occluded_tensor": mx.zeros((0,), dtype=mx.int64),
        }

    def _extend_hotstart_gpu_metadata_for_new_objects(
        self,
        gpu_metadata: dict[str, Any],
        *,
        frame_idx: int,
        num_new_objects: int,
    ) -> dict[str, Any]:
        num_new = int(num_new_objects)
        if num_new <= 0:
            return gpu_metadata
        old_num = int(gpu_metadata.get("N_obj", 0))
        if old_num == 0 and "obj_first_frame" not in gpu_metadata:
            gpu_metadata = self._empty_hotstart_gpu_metadata()
        gpu_metadata["obj_first_frame"] = mx.concatenate(
            [
                gpu_metadata["obj_first_frame"],
                mx.full((num_new,), int(frame_idx), dtype=mx.int64),
            ],
            axis=0,
        )
        gpu_metadata["consecutive_unmatch_count"] = mx.concatenate(
            [
                gpu_metadata["consecutive_unmatch_count"],
                mx.zeros((num_new,), dtype=mx.int64),
            ],
            axis=0,
        )
        gpu_metadata["trk_keep_alive"] = mx.concatenate(
            [
                gpu_metadata["trk_keep_alive"],
                mx.full(
                    (num_new,),
                    int(self.init_trk_keep_alive),
                    dtype=mx.int64,
                ),
            ],
            axis=0,
        )
        gpu_metadata["removed_mask"] = mx.concatenate(
            [
                gpu_metadata["removed_mask"],
                mx.zeros((num_new,), dtype=mx.bool_),
            ],
            axis=0,
        )
        gpu_metadata["last_occluded_tensor"] = mx.concatenate(
            [
                gpu_metadata["last_occluded_tensor"],
                mx.full((num_new,), -1, dtype=mx.int64),
            ],
            axis=0,
        )
        new_num = old_num + num_new
        old_overlap = gpu_metadata["overlap_pair_counts"]
        top = mx.concatenate(
            [
                old_overlap,
                mx.zeros((old_num, num_new), dtype=mx.int64),
            ],
            axis=1,
        )
        bottom = mx.zeros((num_new, new_num), dtype=mx.int64)
        gpu_metadata["overlap_pair_counts"] = mx.concatenate([top, bottom], axis=0)
        gpu_metadata["N_obj"] = new_num
        return gpu_metadata

    def _process_hotstart(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: dict[int, Any],
        new_det_obj_ids: Any,
        empty_trk_obj_ids: Any,
        unmatched_trk_obj_ids: Any,
        rank0_metadata: dict[str, Any],
        tracker_metadata: dict[str, Any],
    ) -> tuple[set[int], dict[str, Any]]:
        del num_frames, tracker_metadata
        frame_idx_int = int(frame_idx)
        matched_trks_by_det = {
            int(det_idx): _array_to_numpy(obj_ids, dtype=np.int64).reshape(-1)
            for det_idx, obj_ids in det_to_matched_trk_obj_ids.items()
        }
        new_det_obj_ids_np = _array_to_numpy(new_det_obj_ids, dtype=np.int64).reshape(
            -1
        )
        empty_trk_obj_ids_np = _array_to_numpy(
            empty_trk_obj_ids,
            dtype=np.int64,
        ).reshape(-1)
        unmatched_trk_obj_ids_np = _array_to_numpy(
            unmatched_trk_obj_ids,
            dtype=np.int64,
        ).reshape(-1)

        obj_first_frame_idx = rank0_metadata.setdefault("obj_first_frame_idx", {})
        unmatched_frame_inds = rank0_metadata.setdefault(
            "unmatched_frame_inds",
            defaultdict(list),
        )
        trk_keep_alive = rank0_metadata.setdefault("trk_keep_alive", defaultdict(int))
        overlap_pair_to_frame_inds = rank0_metadata.setdefault(
            "overlap_pair_to_frame_inds",
            defaultdict(list),
        )
        removed_obj_ids = rank0_metadata.setdefault("removed_obj_ids", set())
        suppressed_obj_ids_by_frame = rank0_metadata.setdefault(
            "suppressed_obj_ids",
            defaultdict(set),
        )
        suppressed_obj_ids = suppressed_obj_ids_by_frame.setdefault(
            frame_idx_int, set()
        )

        obj_ids_newly_removed: set[int] = set()
        hotstart_diff = (
            frame_idx_int - self.hotstart_delay
            if not reverse
            else frame_idx_int + self.hotstart_delay
        )

        for obj_id in new_det_obj_ids_np:
            obj_id_int = int(obj_id)
            obj_first_frame_idx.setdefault(obj_id_int, frame_idx_int)
            if obj_id_int in trk_keep_alive:
                raise AssertionError(
                    f"New detection object {obj_id_int} already has hotstart state."
                )
            trk_keep_alive[obj_id_int] = int(self.init_trk_keep_alive)

        matched_trks = {
            int(obj_id)
            for matched_trks_per_det in matched_trks_by_det.values()
            for obj_id in matched_trks_per_det
        }
        for obj_id in matched_trks:
            trk_keep_alive[obj_id] = min(
                int(self.max_trk_keep_alive),
                int(trk_keep_alive[obj_id]) + 1,
            )

        for obj_id in unmatched_trk_obj_ids_np:
            obj_id_int = int(obj_id)
            unmatched_frame_inds.setdefault(obj_id_int, []).append(frame_idx_int)
            trk_keep_alive[obj_id_int] = max(
                int(self.min_trk_keep_alive),
                int(trk_keep_alive[obj_id_int]) - 1,
            )

        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids_np:
                obj_id_int = int(obj_id)
                trk_keep_alive[obj_id_int] = max(
                    int(self.min_trk_keep_alive),
                    int(trk_keep_alive[obj_id_int]) - 1,
                )

        for obj_id, frame_indices in unmatched_frame_inds.items():
            obj_id_int = int(obj_id)
            if obj_id_int in removed_obj_ids or obj_id_int in obj_ids_newly_removed:
                continue
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (
                    obj_first_frame_idx[obj_id_int] > hotstart_diff and not reverse
                ) or (obj_first_frame_idx[obj_id_int] < hotstart_diff and reverse)
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id_int)
            if (
                trk_keep_alive[obj_id_int] <= 0
                and not self.suppress_unmatched_only_within_hotstart
                and obj_id_int not in removed_obj_ids
                and obj_id_int not in obj_ids_newly_removed
            ):
                suppressed_obj_ids.add(obj_id_int)

        for matched_trk_obj_ids in matched_trks_by_det.values():
            if len(matched_trk_obj_ids) < 2:
                continue
            matched_ids = [int(obj_id) for obj_id in matched_trk_obj_ids]
            first_appear_obj_id = (
                min(matched_ids, key=lambda obj_id: obj_first_frame_idx[obj_id])
                if not reverse
                else max(matched_ids, key=lambda obj_id: obj_first_frame_idx[obj_id])
            )
            for obj_id in matched_ids:
                if obj_id != first_appear_obj_id:
                    overlap_pair_to_frame_inds.setdefault(
                        (first_appear_obj_id, obj_id),
                        [],
                    ).append(frame_idx_int)

        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            obj_id_int = int(obj_id)
            if obj_id_int in removed_obj_ids or obj_id_int in obj_ids_newly_removed:
                continue
            is_within_hotstart = (
                obj_first_frame_idx[obj_id_int] > hotstart_diff and not reverse
            ) or (obj_first_frame_idx[obj_id_int] < hotstart_diff and reverse)
            if is_within_hotstart and len(frame_indices) >= self.hotstart_dup_thresh:
                obj_ids_newly_removed.add(obj_id_int)

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, rank0_metadata

    def build_outputs(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: dict[str, Any],
        tracker_low_res_masks_global: Any,
        tracker_obj_scores_global: Any,
        tracker_metadata_prev: dict[str, Any],
        sam2_update_plan: dict[str, Any] | None = None,
        orig_vid_height: int = 0,
        orig_vid_width: int = 0,
        reconditioned_obj_ids: set[int] | None = None,
        det_to_matched_trk_obj_ids: dict[int, Any] | None = None,
        tracker_update_plan: dict[str, Any] | None = None,
    ) -> dict[int, Any]:
        del frame_idx, num_frames, reverse, tracker_obj_scores_global
        del det_to_matched_trk_obj_ids
        update_plan = (
            sam2_update_plan if sam2_update_plan is not None else tracker_update_plan
        )
        if update_plan is None:
            raise TypeError("build_outputs requires sam2_update_plan.")
        if orig_vid_height <= 0 or orig_vid_width <= 0:
            raise ValueError("orig_vid_height and orig_vid_width must be positive.")

        tracker_masks = (
            tracker_low_res_masks_global
            if _is_mlx_array(tracker_low_res_masks_global)
            else mx.array(tracker_low_res_masks_global)
        )
        if tracker_masks.ndim == 4 and tracker_masks.shape[1] == 1:
            tracker_masks = tracker_masks[:, 0, :, :]
        if tracker_masks.ndim != 3:
            raise ValueError(
                "tracker_low_res_masks_global must have shape (N, H, W) or "
                f"(N, 1, H, W), got {tracker_masks.shape}."
            )

        det_masks = det_out["mask"]
        det_masks_mx = det_masks if _is_mlx_array(det_masks) else mx.array(det_masks)
        if det_masks_mx.ndim == 4 and det_masks_mx.shape[0] == 1:
            det_masks_mx = det_masks_mx[0]
        if det_masks_mx.ndim != 3:
            raise ValueError(
                "det_out['mask'] must have shape (N, H, W) or (1, N, H, W), "
                f"got {det_masks_mx.shape}."
            )

        obj_id_to_mask: dict[int, Any] = {}

        existing_obj_ids_all = np.asarray(
            tracker_metadata_prev["obj_ids_all_gpu"],
            dtype=np.int64,
        ).reshape(-1)
        per_gpu_parts = [
            np.asarray(obj_ids, dtype=np.int64).reshape(-1)
            for obj_ids in tracker_metadata_prev["obj_ids_per_gpu"]
        ]
        existing_obj_ids_per_gpu = (
            np.concatenate(per_gpu_parts)
            if per_gpu_parts
            else np.array([], dtype=np.int64)
        )
        use_per_gpu_ids = (
            existing_obj_ids_per_gpu.size != existing_obj_ids_all.size
            or not np.array_equal(existing_obj_ids_per_gpu, existing_obj_ids_all)
        )
        existing_obj_ids = (
            existing_obj_ids_per_gpu if use_per_gpu_ids else existing_obj_ids_all
        )
        existing_video_masks = interpolate(
            tracker_masks.astype(mx.float32)[:, None, :, :],
            size=(int(orig_vid_height), int(orig_vid_width)),
            mode="bilinear",
            align_corners=False,
        )
        num_masks = int(existing_video_masks.shape[0])
        num_ids = int(existing_obj_ids.size)
        if num_masks < num_ids:
            pad = mx.zeros(
                (
                    num_ids - num_masks,
                    1,
                    int(orig_vid_height),
                    int(orig_vid_width),
                ),
                dtype=existing_video_masks.dtype,
            )
            existing_video_masks = mx.concat([existing_video_masks, pad], axis=0)
        elif num_masks > num_ids:
            existing_video_masks = existing_video_masks[:num_ids]
        existing_binary = existing_video_masks > 0
        for obj_id, mask in zip(existing_obj_ids.tolist(), existing_binary):
            obj_id_to_mask[int(obj_id)] = mask

        new_det_fa_inds = np.asarray(
            update_plan["new_det_fa_inds"],
            dtype=np.int64,
        ).reshape(-1)
        new_det_obj_ids = np.asarray(
            update_plan["new_det_obj_ids"],
            dtype=np.int64,
        ).reshape(-1)
        if new_det_fa_inds.shape[0] != new_det_obj_ids.shape[0]:
            raise ValueError(
                "new_det_obj_ids must have the same length as new_det_fa_inds, "
                f"got {new_det_obj_ids.shape[0]} and {new_det_fa_inds.shape[0]}."
            )
        if new_det_fa_inds.size > 0:
            if np.any(new_det_fa_inds < 0) or np.any(
                new_det_fa_inds >= det_masks_mx.shape[0]
            ):
                raise ValueError(
                    "new_det_fa_inds contains an index outside det_out['mask']; "
                    f"valid range is [0, {det_masks_mx.shape[0]})."
                )

            selected = mx.take(
                det_masks_mx,
                mx.array(new_det_fa_inds, dtype=mx.int64),
                axis=0,
            )
            new_det_low_res_masks = selected.astype(mx.float32)[:, None, :, :]
            new_det_low_res_masks = fill_holes_in_mask_scores(
                new_det_low_res_masks,
                fill_hole_area=self.fill_hole_area,
                sprinkle_removal_area=self.sprinkle_removal_area,
                fill_holes=True,
                remove_sprinkles=True,
            )
            new_video_masks = interpolate(
                new_det_low_res_masks,
                size=(int(orig_vid_height), int(orig_vid_width)),
                mode="bilinear",
                align_corners=False,
            )
            new_binary = new_video_masks > 0
            for obj_id, mask in zip(new_det_obj_ids.tolist(), new_binary):
                obj_id_to_mask[int(obj_id)] = mask

        if reconditioned_obj_ids:
            trk_id_to_max_iou_high_conf_det = update_plan.get(
                "trk_id_to_max_iou_high_conf_det",
                {},
            )
            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(int(obj_id))
                if det_idx is None:
                    continue
                det_idx_int = int(det_idx)
                if det_idx_int < 0 or det_idx_int >= det_masks_mx.shape[0]:
                    raise ValueError(
                        "trk_id_to_max_iou_high_conf_det contains an index outside "
                        f"det_out['mask']; valid range is [0, {det_masks_mx.shape[0]})."
                    )
                det_mask = det_masks_mx[det_idx_int].astype(mx.float32)
                det_video_mask = interpolate(
                    det_mask[None, None, :, :],
                    size=(int(orig_vid_height), int(orig_vid_width)),
                    mode="bilinear",
                    align_corners=False,
                )
                obj_id_to_mask[int(obj_id)] = (det_video_mask > 0)[0]
        return obj_id_to_mask

    def _normalize_image_only_detection_output(
        self,
        sam3_image_out: dict[str, Any],
    ) -> tuple[Any, Any, Any | None]:
        pred_logits = sam3_image_out["pred_logits"]
        if pred_logits.ndim == 3 and pred_logits.shape[0] == 1:
            pred_logits = pred_logits[0]
        if pred_logits.ndim == 2 and pred_logits.shape[-1] == 1:
            pred_logits = pred_logits[:, 0]
        if pred_logits.ndim != 1:
            raise ValueError(
                "SAM3 image-only pred_logits must have shape (1, N, 1), "
                f"(N, 1), or (N,), got {sam3_image_out['pred_logits'].shape}."
            )
        pred_scores = mx.sigmoid(pred_logits)

        pred_masks = sam3_image_out["pred_masks"]
        if (
            pred_masks.ndim == 5
            and pred_masks.shape[0] == 1
            and pred_masks.shape[2] == 1
        ):
            pred_masks = pred_masks[0, :, 0]
        elif pred_masks.ndim == 4 and pred_masks.shape[0] == 1:
            pred_masks = pred_masks[0]
        elif pred_masks.ndim != 3:
            raise ValueError(
                "SAM3 image-only pred_masks must have shape (1, N, H, W), "
                f"(1, N, 1, H, W), or (N, H, W), got {pred_masks.shape}."
            )
        if pred_masks.ndim != 3:
            raise ValueError(
                "SAM3 image-only pred_masks must normalize to shape (N, H, W), "
                f"got {pred_masks.shape}."
            )
        if pred_masks.shape[0] != pred_scores.shape[0]:
            raise ValueError(
                "SAM3 image-only pred_logits and pred_masks disagree on detection "
                f"count: {pred_scores.shape[0]} vs {pred_masks.shape[0]}."
            )
        pred_boxes = sam3_image_out.get("pred_boxes_xyxy")
        if pred_boxes is not None:
            pred_boxes = (
                pred_boxes if _is_mlx_array(pred_boxes) else mx.array(pred_boxes)
            )
            if pred_boxes.ndim == 3 and pred_boxes.shape[0] == 1:
                pred_boxes = pred_boxes[0]
            if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
                raise ValueError(
                    "SAM3 image-only pred_boxes_xyxy must have shape (1, N, 4) "
                    f"or (N, 4), got {sam3_image_out['pred_boxes_xyxy'].shape}."
                )
            if pred_boxes.shape[0] != pred_scores.shape[0]:
                raise ValueError(
                    "SAM3 image-only pred_logits and pred_boxes_xyxy disagree "
                    f"on detection count: {pred_scores.shape[0]} vs {pred_boxes.shape[0]}."
                )
        return pred_scores, pred_masks, pred_boxes

    def _normalize_video_detection_outputs(
        self,
        det_out: dict[str, Any],
        det_keep: Any,
    ) -> tuple[dict[str, Any], Any]:
        det_masks = det_out["mask"]
        det_masks = det_masks if _is_mlx_array(det_masks) else mx.array(det_masks)
        if det_masks.ndim == 4 and det_masks.shape[0] == 1:
            det_masks = det_masks[0]
        if det_masks.ndim != 3:
            raise ValueError(
                "video detector masks must have shape (N, H, W) or (1, N, H, W), "
                f"got {det_masks.shape}."
            )

        det_scores = det_out["scores"]
        det_scores = det_scores if _is_mlx_array(det_scores) else mx.array(det_scores)
        if det_scores.ndim == 2 and det_scores.shape[0] == 1:
            det_scores = det_scores[0]
        if det_scores.ndim != 1:
            raise ValueError(
                "video detector scores must have shape (N,) or (1, N), "
                f"got {det_scores.shape}."
            )
        if det_scores.shape[0] != det_masks.shape[0]:
            raise ValueError(
                "video detector scores and masks disagree on detection count: "
                f"{det_scores.shape[0]} vs {det_masks.shape[0]}."
            )

        det_keep_mx = det_keep if _is_mlx_array(det_keep) else mx.array(det_keep)
        if det_keep_mx.ndim == 2 and det_keep_mx.shape[0] == 1:
            det_keep_mx = det_keep_mx[0]
        if det_keep_mx.ndim != 1:
            raise ValueError(
                "video detector keep mask must have shape (N,) or (1, N), "
                f"got {det_keep_mx.shape}."
            )
        if det_keep_mx.shape[0] != det_masks.shape[0]:
            raise ValueError(
                "video detector keep mask and masks disagree on detection count: "
                f"{det_keep_mx.shape[0]} vs {det_masks.shape[0]}."
            )

        normalized = dict(det_out)
        normalized["mask"] = det_masks
        normalized["scores"] = det_scores.astype(mx.float32)
        bbox = normalized.get("bbox")
        if bbox is not None:
            bbox_mx = bbox if _is_mlx_array(bbox) else mx.array(bbox)
            if bbox_mx.ndim == 3 and bbox_mx.shape[0] == 1:
                bbox_mx = bbox_mx[0]
            normalized["bbox"] = bbox_mx
        return normalized, det_keep_mx.astype(mx.bool_)

    def _image_only_detection_keep_indices(
        self,
        pred_scores: Any,
        pred_masks: Any,
        pred_boxes: Any | None = None,
        *,
        apply_boundary_suppression: bool = False,
    ) -> tuple[np.ndarray, int]:
        del pred_masks
        keep = pred_scores >= self.image_only_det_thresh
        detector_keep = pred_scores > self.score_threshold_detection
        if apply_boundary_suppression:
            if pred_boxes is None:
                raise ValueError(
                    "suppress_det_close_to_boundary=True requires pred_boxes_xyxy."
                )
            boundary_keep = self._suppress_detections_close_to_boundary(pred_boxes)
            keep = keep & boundary_keep
            detector_keep = detector_keep & boundary_keep
        # Match the official no-tracklet image-only startup path: image prompt
        # outputs are thresholded, not reduced by the video object limit, and
        # assigned object IDs after the detector keep-mask true-first partition.
        keep_np = to_numpy(keep, dtype=bool, copy=False).reshape(-1)
        detector_keep_np = to_numpy(
            detector_keep,
            dtype=bool,
            copy=False,
        ).reshape(-1)
        sorted_indices = _torch_bool_argsort_desc_np(detector_keep_np)
        keep_indices = sorted_indices[keep_np[sorted_indices]].astype(np.int64)
        return keep_indices, 0

    def _materialize_initial_detection_frame(
        self,
        *,
        frame_idx: int,
        pred_scores: Any,
        pred_masks: Any,
        tracker_metadata_prev: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
        pred_boxes: Any | None = None,
        apply_boundary_suppression: bool = False,
    ) -> tuple[dict[int, Any], dict[int, float], dict[str, Any], dict[str, int], Any]:
        keep_indices, num_dropped = self._image_only_detection_keep_indices(
            pred_scores,
            pred_masks,
            pred_boxes,
            apply_boundary_suppression=apply_boundary_suppression,
        )

        prev_max_obj_id = int(tracker_metadata_prev.get("max_obj_id", -1))
        new_obj_ids = np.arange(
            prev_max_obj_id + 1,
            prev_max_obj_id + 1 + keep_indices.size,
            dtype=np.int64,
        )
        tracker_metadata_new = self._initialize_metadata()
        tracker_metadata_new["max_obj_id"] = (
            int(new_obj_ids[-1]) if new_obj_ids.size > 0 else prev_max_obj_id
        )
        tracker_metadata_new["obj_ids_all_gpu"] = new_obj_ids
        tracker_metadata_new["obj_ids_per_gpu"][0] = new_obj_ids
        tracker_metadata_new["num_obj_per_gpu"][0] = int(new_obj_ids.size)
        if self.is_multiplex:
            tracker_metadata_new["gpu_metadata"] = {"N_obj": int(new_obj_ids.size)}
            tracker_metadata_new["num_buc_per_gpu"][0] = (
                math.ceil(new_obj_ids.size / self.bucket_capacity)
                if new_obj_ids.size > 0
                else 0
            )

        obj_id_to_mask: dict[int, Any] = {}
        obj_id_to_score: dict[int, float] = {}
        if keep_indices.size > 0:
            selected = mx.array(keep_indices, dtype=mx.int64)
            selected_scores = mx.take(pred_scores, selected, axis=0)
            selected_masks = mx.take(pred_masks, selected, axis=0)
            selected_masks = fill_holes_in_mask_scores(
                selected_masks.astype(mx.float32)[:, None, :, :],
                fill_hole_area=self.fill_hole_area,
                sprinkle_removal_area=self.sprinkle_removal_area,
                fill_holes=True,
                remove_sprinkles=True,
            )[:, 0, :, :]
            if selected_masks.shape[-2:] != (orig_vid_height, orig_vid_width):
                selected_masks = interpolate(
                    selected_masks[:, None, :, :],
                    size=(orig_vid_height, orig_vid_width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0, :, :]
            selected_masks = selected_masks > 0
            scores_np = to_numpy(
                selected_scores,
                dtype=np.float32,
                copy=False,
            ).reshape(-1)
            for local_idx, obj_id in enumerate(new_obj_ids.tolist()):
                obj_id_int = int(obj_id)
                obj_id_to_mask[obj_id_int] = selected_masks[local_idx][None, :, :]
                obj_id_to_score[obj_id_int] = float(scores_np[local_idx])
                tracker_metadata_new["rank0_metadata"]["obj_first_frame_idx"][
                    obj_id_int
                ] = frame_idx

        tracker_metadata_new["obj_id_to_score"] = obj_id_to_score
        tracker_metadata_new["obj_id_to_sam2_score_frame_wise"][frame_idx].update(
            obj_id_to_score
        )
        if self.masklet_confirmation_enable:
            confirmation = tracker_metadata_new["rank0_metadata"][
                "masklet_confirmation"
            ]
            confirmation["status"] = np.full(
                new_obj_ids.shape,
                MaskletConfirmationStatus.CONFIRMED.value,
                dtype=np.int64,
            )
            confirmation["consecutive_det_num"] = np.full(
                new_obj_ids.shape,
                self.masklet_confirmation_consecutive_det_thresh,
                dtype=np.int64,
            )

        frame_stats = {
            "num_obj_tracked": int(new_obj_ids.size),
            "num_obj_dropped": num_dropped,
        }
        tracker_obj_scores_global = mx.zeros((0,), dtype=mx.float32)
        return (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        )

    def _run_image_only_detection_frame(
        self,
        *,
        frame_idx: int,
        input_batch: Any,
        geometric_prompt: Any,
        tracker_metadata_prev: dict[str, Any],
        feature_cache: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
    ) -> tuple[
        dict[int, Any], dict[int, float], list[Any], dict[str, Any], dict[str, int], Any
    ]:
        if hasattr(self.detector, "forward_video_grounding_multigpu"):
            det_out, det_keep = self.run_backbone_and_detection(
                frame_idx=frame_idx,
                num_frames=len(input_batch.find_inputs),
                reverse=False,
                input_batch=input_batch,
                geometric_prompt=geometric_prompt,
                feature_cache=feature_cache,
                use_batched_grounding=self.use_batched_grounding,
                batched_grounding_batch_size=self.batched_grounding_batch_size,
            )
            det_out, det_keep = self._normalize_video_detection_outputs(
                det_out,
                det_keep,
            )
            del det_keep
            pred_scores = det_out["scores"]
            pred_masks = det_out["mask"]
            pred_boxes = det_out.get("bbox")
        else:
            text_batch_key = tuple(input_batch.find_text_batch)
            if (
                "text" not in feature_cache
                or text_batch_key not in feature_cache["text"]
            ):
                text_outputs = self.detector.backbone.forward_text(
                    input_batch.find_text_batch,
                    device=self.device,
                )
                feature_cache["text"] = {text_batch_key: text_outputs}
            else:
                text_outputs = feature_cache["text"][text_batch_key]

            backbone_out = {
                "img_batch_all_stages": input_batch.img_batch,
                **text_outputs,
            }
            find_target = (
                input_batch.find_targets[frame_idx]
                if input_batch.find_targets is not None
                else None
            )
            if self.use_batched_grounding and hasattr(
                self.detector,
                "forward_video_grounding_batched_multigpu",
            ):
                sam3_image_out, _ = (
                    self.detector.forward_video_grounding_batched_multigpu(
                        backbone_out=backbone_out,
                        find_inputs=input_batch.find_inputs,
                        geometric_prompt=geometric_prompt,
                        frame_idx=frame_idx,
                        num_frames=len(input_batch.find_inputs),
                        grounding_cache=feature_cache.setdefault("grounding_cache", {}),
                        batch_size=self.batched_grounding_batch_size,
                    )
                )
            else:
                sam3_image_out, _ = self.detector.forward_video_grounding(
                    backbone_out=backbone_out,
                    find_input=input_batch.find_inputs[frame_idx],
                    find_target=find_target,
                    geometric_prompt=geometric_prompt,
                )
            pred_scores, pred_masks, pred_boxes = (
                self._normalize_image_only_detection_output(sam3_image_out)
            )
        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        ) = self._materialize_initial_detection_frame(
            frame_idx=frame_idx,
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            pred_boxes=pred_boxes,
            tracker_metadata_prev=tracker_metadata_prev,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            apply_boundary_suppression=self.suppress_det_close_to_boundary,
        )
        return (
            obj_id_to_mask,
            obj_id_to_score,
            [],
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        )

    def _seed_tracker_states_from_frame_masks(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        obj_id_to_mask: dict[int, Any],
        feature_cache: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
    ) -> list[Any]:
        if not obj_id_to_mask:
            return []
        obj_ids = sorted(obj_id_to_mask)
        masks = mx.stack(
            [
                (
                    obj_id_to_mask[obj_id]
                    if _is_mlx_array(obj_id_to_mask[obj_id])
                    else mx.array(obj_id_to_mask[obj_id])
                )[0].astype(mx.float32)
                for obj_id in obj_ids
            ],
            axis=0,
        )
        return self._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=num_frames,
            new_obj_ids=np.asarray(obj_ids, dtype=np.int64),
            new_obj_masks=masks,
            tracker_states_local=[],
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            feature_cache=feature_cache,
        )

    def _seed_tracker_states_from_detection_logits(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: Any,
        pred_masks: Any,
        keep_indices: np.ndarray,
        feature_cache: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
    ) -> list[Any]:
        new_obj_ids_np = np.asarray(new_obj_ids, dtype=np.int64).reshape(-1)
        if new_obj_ids_np.size == 0:
            return []
        if new_obj_ids_np.size != keep_indices.size:
            raise ValueError(
                "new_obj_ids must align with kept detector masks; got "
                f"{new_obj_ids_np.size} ids and {keep_indices.size} masks."
            )
        pred_masks_mx = (
            pred_masks if _is_mlx_array(pred_masks) else mx.array(pred_masks)
        )
        selected_masks = mx.take(
            pred_masks_mx.astype(mx.float32),
            mx.array(keep_indices, dtype=mx.int64),
            axis=0,
        )
        return self._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=num_frames,
            new_obj_ids=new_obj_ids_np,
            new_obj_masks=selected_masks,
            tracker_states_local=[],
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            feature_cache=feature_cache,
        )

    @staticmethod
    def _ensure_empty_packed_current_output(
        sam2_state: Any,
        frame_idx: int,
    ) -> None:
        if not isinstance(sam2_state, dict):
            return
        multiplex_state = sam2_state.get("multiplex_state")
        if multiplex_state is None:
            return
        if int(getattr(multiplex_state, "total_valid_entries", -1)) != 0:
            return
        output_dict = sam2_state.setdefault("output_dict", {})
        cond_outputs = output_dict.setdefault("cond_frame_outputs", {})
        non_cond_outputs = output_dict.setdefault("non_cond_frame_outputs", {})
        if frame_idx in cond_outputs or frame_idx in non_cond_outputs:
            return
        cond_outputs[frame_idx] = {"conditioning_objects": set()}

    def _run_detector_startup_frame(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: Any,
        geometric_prompt: Any,
        tracker_metadata_prev: dict[str, Any],
        feature_cache: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
    ) -> tuple[
        dict[int, Any], dict[int, float], list[Any], dict[str, Any], dict[str, int], Any
    ]:
        if not hasattr(self.detector, "backbone"):
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._det_track_one_frame(video-tracker-runtime)"
            )
        detector_method = (
            "forward_video_grounding_batched_multigpu"
            if self.use_batched_grounding
            else "forward_video_grounding_multigpu"
        )
        if not hasattr(self.detector, detector_method):
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._det_track_one_frame(video-tracker-runtime)"
            )
        if not hasattr(self.tracker, "init_state") or not (
            hasattr(self.tracker, "add_new_masks")
            or hasattr(self.tracker, "add_new_mask")
        ):
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._det_track_one_frame(video-tracker-runtime)"
            )

        det_out, det_keep = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            use_batched_grounding=self.use_batched_grounding,
            batched_grounding_batch_size=self.batched_grounding_batch_size,
        )
        pred_scores = det_out["scores"]
        pred_masks = det_out["mask"]
        if pred_scores.ndim == 2 and pred_scores.shape[0] == 1:
            pred_scores = pred_scores[0]
        if pred_masks.ndim == 4 and pred_masks.shape[0] == 1:
            pred_masks = pred_masks[0]
        pred_scores = mx.where(
            det_keep.reshape(-1),
            pred_scores.reshape(-1),
            mx.array(-1.0, dtype=pred_scores.dtype),
        )
        keep_indices, _num_dropped = self._image_only_detection_keep_indices(
            pred_scores,
            pred_masks,
        )
        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        ) = self._materialize_initial_detection_frame(
            frame_idx=frame_idx,
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            tracker_metadata_prev=tracker_metadata_prev,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
        )
        tracker_states_local_new = self._seed_tracker_states_from_detection_logits(
            frame_idx=frame_idx,
            num_frames=num_frames,
            new_obj_ids=tracker_metadata_new["obj_ids_all_gpu"],
            pred_masks=pred_masks,
            keep_indices=keep_indices,
            feature_cache=feature_cache,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
        )
        return (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        )

    def _convert_low_res_mask_to_video_res(
        self,
        low_res_mask: Any,
        *,
        orig_vid_height: int,
        orig_vid_width: int,
    ) -> Any:
        low_res_mask_mx = (
            low_res_mask if _is_mlx_array(low_res_mask) else mx.array(low_res_mask)
        )
        if low_res_mask_mx.ndim != 2:
            raise ValueError(
                f"low_res_mask must have shape (H, W), got {low_res_mask_mx.shape}."
            )
        mask_4d = low_res_mask_mx[None, None, :, :].astype(mx.float32)
        video_res_mask = interpolate(
            mask_4d,
            size=(int(orig_vid_height), int(orig_vid_width)),
            mode="bilinear",
            align_corners=False,
        )
        return video_res_mask[0] > 0.0

    def _propogate_tracker_one_frame_local_gpu(
        self,
        tracker_states_local: list[dict[str, Any]],
        frame_idx: int,
        reverse: bool,
        run_mem_encoder: bool = True,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        obj_ids_local: list[Any] = []
        low_res_masks_local: list[Any] = []
        sam2_scores_local: list[Any] = []
        seen_obj_ids: set[Any] = set()

        for sam2_state in tracker_states_local:
            if len(sam2_state.get("obj_ids", [])) == 0:
                continue
            frame_results = self.tracker.propagate_in_video(
                sam2_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,
                reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                propagate_preflight=False,
            )
            for result in frame_results:
                (
                    result_frame_idx,
                    state_obj_ids,
                    state_low_res_masks,
                    _state_video_res_masks,
                    state_sam2_scores,
                ) = result
                if int(result_frame_idx) != int(frame_idx):
                    raise ValueError(
                        "Tracker propagation returned frame_idx="
                        f"{result_frame_idx}, expected {frame_idx}."
                    )

                if _is_mlx_array(state_obj_ids):
                    state_obj_ids_list = (
                        _array_to_numpy(
                            state_obj_ids,
                            dtype=np.int64,
                        )
                        .reshape(-1)
                        .tolist()
                    )
                elif isinstance(state_obj_ids, np.ndarray):
                    state_obj_ids_list = state_obj_ids.reshape(-1).tolist()
                else:
                    state_obj_ids_list = list(state_obj_ids)

                state_low_res_masks_mx = (
                    state_low_res_masks
                    if _is_mlx_array(state_low_res_masks)
                    else mx.array(state_low_res_masks)
                )
                state_sam2_scores_mx = (
                    state_sam2_scores
                    if _is_mlx_array(state_sam2_scores)
                    else mx.array(state_sam2_scores)
                )
                if state_low_res_masks_mx.shape[0] != len(state_obj_ids_list):
                    raise ValueError(
                        "Tracker low-res mask batch must match obj_ids; got "
                        f"{state_low_res_masks_mx.shape[0]} masks for "
                        f"{len(state_obj_ids_list)} object ids."
                    )
                if state_sam2_scores_mx.shape[0] != len(state_obj_ids_list):
                    raise ValueError(
                        "Tracker score batch must match obj_ids; got "
                        f"{state_sam2_scores_mx.shape[0]} scores for "
                        f"{len(state_obj_ids_list)} object ids."
                    )

                for obj_idx, obj_id in enumerate(state_obj_ids_list):
                    if obj_id in seen_obj_ids:
                        raise ValueError(
                            f"Duplicate obj_id {obj_id} in local SAM2 states."
                        )
                    seen_obj_ids.add(obj_id)
                    low_res_mask = state_low_res_masks_mx[obj_idx]
                    if low_res_mask.ndim == 3 and low_res_mask.shape[0] == 1:
                        low_res_mask = low_res_mask[0]
                    if low_res_mask.ndim != 2:
                        raise ValueError(
                            "Tracker low-res mask entries must have shape (H, W) "
                            f"or (1, H, W), got {low_res_mask.shape}."
                        )
                    obj_ids_local.append(obj_id)
                    low_res_masks_local.append(low_res_mask.astype(mx.float32))
                    sam2_score = state_sam2_scores_mx[obj_idx]
                    if int(np.prod(sam2_score.shape)) == 1:
                        sam2_score = sam2_score.reshape(())
                    sam2_scores_local.append(sam2_score)

        return obj_ids_local, low_res_masks_local, sam2_scores_local

    def _ensure_tracker_metadata_for_local_states(
        self,
        tracker_metadata: dict[str, Any],
        obj_ids_local: list[Any],
    ) -> None:
        if not tracker_metadata:
            tracker_metadata.update(self._initialize_metadata())

        obj_ids_all = np.asarray(
            tracker_metadata.get("obj_ids_all_gpu", []),
            dtype=np.int64,
        ).reshape(-1)
        if obj_ids_all.size == 0 and obj_ids_local:
            obj_ids_all = np.asarray(obj_ids_local, dtype=np.int64)
            tracker_metadata["obj_ids_all_gpu"] = obj_ids_all
            tracker_metadata["obj_ids_per_gpu"][self.rank] = obj_ids_all
            tracker_metadata["num_obj_per_gpu"][self.rank] = int(obj_ids_all.size)
            tracker_metadata["max_obj_id"] = int(obj_ids_all.max())
            if self.is_multiplex:
                tracker_metadata["gpu_metadata"] = {"N_obj": int(obj_ids_all.size)}
                tracker_metadata["num_buc_per_gpu"][self.rank] = (
                    math.ceil(obj_ids_all.size / self.bucket_capacity)
                    if obj_ids_all.size > 0
                    else 0
                )
            rank0_metadata = tracker_metadata.setdefault(
                "rank0_metadata",
                self._initialize_metadata()["rank0_metadata"],
            )
            first_frame = rank0_metadata.setdefault("obj_first_frame_idx", {})
            for obj_id in obj_ids_all.tolist():
                first_frame.setdefault(int(obj_id), 0)

        expected_ids = obj_ids_all.tolist()
        if expected_ids != [int(obj_id) for obj_id in obj_ids_local]:
            raise ValueError(
                "Local tracker propagation obj_ids do not match tracker metadata "
                f"order: {obj_ids_local} != {expected_ids}."
            )

        tracker_metadata.setdefault("obj_id_to_score", {})
        tracker_metadata.setdefault("obj_id_to_last_occluded", {})
        tracker_metadata.setdefault(
            "obj_id_to_sam2_score_frame_wise",
            defaultdict(dict),
        )
        rank0_metadata = tracker_metadata.setdefault(
            "rank0_metadata",
            self._initialize_metadata()["rank0_metadata"],
        )
        rank0_metadata.setdefault("removed_obj_ids", set())
        rank0_metadata.setdefault("suppressed_obj_ids", defaultdict(set))

    def _run_local_tracker_states_only_frame(
        self,
        *,
        frame_idx: int,
        reverse: bool,
        tracker_states_local: list[Any],
        tracker_metadata_prev: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
    ) -> tuple[
        dict[int, Any], dict[int, float], list[Any], dict[str, Any], dict[str, int], Any
    ]:
        obj_ids_local, low_res_masks_local, sam2_scores_local = (
            self._propogate_tracker_one_frame_local_gpu(
                tracker_states_local,
                frame_idx=frame_idx,
                reverse=reverse,
                run_mem_encoder=True,
            )
        )
        self._ensure_tracker_metadata_for_local_states(
            tracker_metadata_prev,
            obj_ids_local,
        )

        frame_scores = tracker_metadata_prev[
            "obj_id_to_sam2_score_frame_wise"
        ].setdefault(frame_idx, {})
        obj_id_to_mask: dict[int, Any] = {}
        obj_id_to_score: dict[int, float] = {}
        score_map = tracker_metadata_prev.get("obj_id_to_score", {})
        sam2_scores_global = []
        for obj_idx, obj_id in enumerate(obj_ids_local):
            obj_id_int = int(obj_id)
            sam2_score_prob = mx.sigmoid(
                (
                    sam2_scores_local[obj_idx]
                    if _is_mlx_array(sam2_scores_local[obj_idx])
                    else mx.array(sam2_scores_local[obj_idx])
                ).astype(mx.float32)
            )
            frame_scores[obj_id_int] = sam2_score_prob
            sam2_scores_global.append(sam2_score_prob)
            obj_id_to_mask[obj_id_int] = self._convert_low_res_mask_to_video_res(
                low_res_masks_local[obj_idx],
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
            )
            obj_id_to_score[obj_id_int] = float(
                score_map.get(obj_id_int, _score_to_float(sam2_score_prob))
            )

        if sam2_scores_global:
            tracker_obj_scores_global = mx.stack(sam2_scores_global, axis=0)
        else:
            tracker_obj_scores_global = mx.zeros((0,), dtype=mx.float32)

        frame_stats = {
            "num_obj_tracked": int(len(obj_ids_local)),
            "num_obj_dropped": 0,
        }
        return (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local,
            tracker_metadata_prev,
            frame_stats,
            tracker_obj_scores_global,
        )

    def _recondition_masklets(
        self,
        frame_idx: int,
        det_out: dict[str, Any],
        trk_id_to_max_iou_high_conf_det: dict[int, int],
        tracker_states_local: list[Any],
        tracker_metadata: dict[str, Any],
        tracker_obj_scores_global: Any,
        tracker_low_res_masks_global: Any,
    ) -> tuple[list[Any], set[int], Any]:
        reconditioned_obj_ids: set[int] = set()
        if not trk_id_to_max_iou_high_conf_det:
            return (
                tracker_states_local,
                reconditioned_obj_ids,
                tracker_low_res_masks_global,
            )

        input_mask_size = getattr(self.tracker, "input_mask_size", None)
        if input_mask_size is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._recondition_masklets(input_mask_size)"
            )
        add_new_masks = getattr(self.tracker, "add_new_masks", None)
        add_new_mask = getattr(self.tracker, "add_new_mask", None)
        if add_new_masks is None and add_new_mask is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._recondition_masklets(add_mask)"
            )

        obj_ids_all = np.asarray(
            tracker_metadata["obj_ids_all_gpu"],
            dtype=np.int64,
        ).reshape(-1)
        tracker_scores = (
            tracker_obj_scores_global
            if _is_mlx_array(tracker_obj_scores_global)
            else mx.array(tracker_obj_scores_global)
        ).astype(mx.float32)
        if tracker_scores.shape[0] != obj_ids_all.size:
            raise ValueError(
                "tracker_obj_scores_global must align with obj_ids_all_gpu for "
                f"reconditioning; got {tracker_scores.shape[0]} scores for "
                f"{obj_ids_all.size} object ids."
            )

        valid_obj_ids: list[int] = []
        valid_obj_indices: list[int] = []
        valid_det_indices: list[int] = []
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            matches = np.flatnonzero(obj_ids_all == int(trk_obj_id))
            if matches.size == 0:
                raise ValueError(
                    "trk_id_to_max_iou_high_conf_det references unknown object id "
                    f"{trk_obj_id}."
                )
            obj_idx = int(matches[0])
            if _score_to_float(mx.sigmoid(tracker_scores[obj_idx])) <= 0.8:
                continue
            valid_obj_ids.append(int(trk_obj_id))
            valid_obj_indices.append(obj_idx)
            valid_det_indices.append(int(det_idx))

        if not valid_obj_ids:
            return (
                tracker_states_local,
                reconditioned_obj_ids,
                tracker_low_res_masks_global,
            )

        det_masks = (
            det_out["mask"]
            if _is_mlx_array(det_out["mask"])
            else mx.array(det_out["mask"])
        )
        if det_masks.ndim != 3:
            raise ValueError(
                "reconditioning detector masks must have shape (N, H, W), "
                f"got {det_masks.shape}."
            )
        det_count = int(det_masks.shape[0])
        for det_idx in valid_det_indices:
            if det_idx < 0 or det_idx >= det_count:
                raise ValueError(
                    "trk_id_to_max_iou_high_conf_det contains an index outside "
                    f"det_out['mask']; valid range is [0, {det_count})."
                )

        det_idx_mx = mx.array(valid_det_indices, dtype=mx.int64)
        new_masks = mx.take(det_masks.astype(mx.float32), det_idx_mx, axis=0)
        current_masks = (
            tracker_low_res_masks_global
            if _is_mlx_array(tracker_low_res_masks_global)
            else mx.array(tracker_low_res_masks_global)
        ).astype(mx.float32)
        obj_idx_mx = mx.array(valid_obj_indices, dtype=mx.int64)
        old_masks = mx.take(current_masks, obj_idx_mx, axis=0)
        if new_masks.shape[-2:] != old_masks.shape[-2:]:
            raise ValueError(
                "reconditioning detector masks and tracker masks must share "
                f"low-res shape, got {new_masks.shape[-2:]} and {old_masks.shape[-2:]}."
            )

        binary_agreement = (new_masks > 0) == (old_masks > 0)
        updated_masks = mx.where(binary_agreement, old_masks, new_masks)
        updated_masks = fill_holes_in_mask_scores(
            updated_masks[:, None, :, :],
            fill_hole_area=self.fill_hole_area,
            sprinkle_removal_area=self.sprinkle_removal_area,
            fill_holes=True,
            remove_sprinkles=True,
        )[:, 0, :, :]

        current_mask_list = [
            current_masks[idx] for idx in range(current_masks.shape[0])
        ]
        for local_idx, obj_idx in enumerate(valid_obj_indices):
            current_mask_list[obj_idx] = updated_masks[local_idx]
        tracker_low_res_masks_global = mx.stack(current_mask_list, axis=0)

        recondition_masks = (
            interpolate(
                new_masks[:, None, :, :],
                size=(int(input_mask_size), int(input_mask_size)),
                mode="bilinear",
                align_corners=False,
            )[:, 0, :, :]
            > 0
        )

        state_to_recondition_info: dict[int, list[tuple[int, Any]]] = {}
        for local_idx, trk_obj_id in enumerate(valid_obj_ids):
            for state_idx, inference_state in enumerate(tracker_states_local):
                if trk_obj_id in inference_state.get("obj_ids", []):
                    state_to_recondition_info.setdefault(state_idx, []).append(
                        (trk_obj_id, recondition_masks[local_idx])
                    )
                    break

        for state_idx, recondition_list in state_to_recondition_info.items():
            inference_state = tracker_states_local[state_idx]
            obj_ids_to_recondition = [item[0] for item in recondition_list]
            masks_to_recondition = mx.stack([item[1] for item in recondition_list])
            if add_new_masks is not None:
                add_new_masks(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_ids=obj_ids_to_recondition,
                    masks=masks_to_recondition,
                    reconditioning=True,
                )
            else:
                for obj_idx, obj_id in enumerate(obj_ids_to_recondition):
                    add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=masks_to_recondition[obj_idx],
                    )
            reconditioned_obj_ids.update(obj_ids_to_recondition)
            propagate_preflight = getattr(
                self.tracker,
                "propagate_in_video_preflight",
                None,
            )
            if propagate_preflight is not None:
                propagate_preflight(inference_state, run_mem_encoder=True)

        return tracker_states_local, reconditioned_obj_ids, tracker_low_res_masks_global

    def _should_recondition_from_bbox_iou(
        self,
        *,
        det_out: dict[str, Any],
        trk_id_to_max_iou_high_conf_det: dict[int, int],
        tracker_metadata_prev: dict[str, Any],
        tracker_low_res_masks_global: Any,
    ) -> bool:
        if self.reconstruction_bbox_iou_thresh <= 0:
            return False
        if not trk_id_to_max_iou_high_conf_det:
            return False
        if "bbox" not in det_out:
            raise ValueError(
                "bbox reconditioning requires det_out['bbox'] when "
                "reconstruction_bbox_iou_thresh > 0."
            )

        obj_ids_all = np.asarray(
            tracker_metadata_prev["obj_ids_all_gpu"],
            dtype=np.int64,
        ).reshape(-1)
        det_indices: list[int] = []
        tracker_indices: list[int] = []
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            matches = np.flatnonzero(obj_ids_all == int(trk_obj_id))
            if matches.size == 0:
                continue
            tracker_indices.append(int(matches[0]))
            det_indices.append(int(det_idx))
        if not det_indices:
            return False

        det_boxes = (
            det_out["bbox"]
            if _is_mlx_array(det_out["bbox"])
            else mx.array(det_out["bbox"])
        )
        det_scores = (
            det_out["scores"]
            if _is_mlx_array(det_out["scores"])
            else mx.array(det_out["scores"])
        ).astype(mx.float32)
        if det_boxes.ndim != 2 or det_boxes.shape[-1] != 4:
            raise ValueError(
                f"det_out['bbox'] must have shape (N, 4), got {det_boxes.shape}."
            )
        det_count = int(det_boxes.shape[0])
        for det_idx in det_indices:
            if det_idx < 0 or det_idx >= det_count:
                raise ValueError(
                    "trk_id_to_max_iou_high_conf_det contains an index outside "
                    f"det_out['bbox']; valid range is [0, {det_count})."
                )

        tracker_masks = (
            tracker_low_res_masks_global
            if _is_mlx_array(tracker_low_res_masks_global)
            else mx.array(tracker_low_res_masks_global)
        )
        if tracker_masks.ndim != 3:
            raise ValueError(
                "tracker_low_res_masks_global must have shape (N, H, W), "
                f"got {tracker_masks.shape}."
            )
        tracker_count = int(tracker_masks.shape[0])
        for tracker_idx in tracker_indices:
            if tracker_idx < 0 or tracker_idx >= tracker_count:
                raise ValueError(
                    "tracker metadata references a mask index outside "
                    f"tracker_low_res_masks_global; valid range is [0, {tracker_count})."
                )

        det_idx_mx = mx.array(det_indices, dtype=mx.int64)
        tracker_idx_mx = mx.array(tracker_indices, dtype=mx.int64)
        det_boxes_bbox_iou = mx.take(det_boxes.astype(mx.float32), det_idx_mx, axis=0)
        det_scores_bbox_iou = mx.take(det_scores, det_idx_mx, axis=0)
        sam2_mask = mx.take(tracker_masks.astype(mx.float32), tracker_idx_mx, axis=0)
        sam2_box_pixels = mask_to_box((sam2_mask > 0)[:, None, :, :]).reshape(-1, 4)
        mask_height, mask_width = sam2_mask.shape[-2:]
        sam2_box_normalized = sam2_box_pixels.astype(mx.float32) / mx.array(
            [mask_width, mask_height, mask_width, mask_height],
            dtype=mx.float32,
        )
        iou = fast_diag_box_iou(det_boxes_bbox_iou, sam2_box_normalized)
        iou_np = _array_to_numpy(iou, dtype=np.float32).reshape(-1)
        score_np = _array_to_numpy(det_scores_bbox_iou, dtype=np.float32).reshape(-1)
        recondition = (iou_np < float(self.reconstruction_bbox_iou_thresh)) & (
            score_np >= float(self.reconstruction_bbox_det_score)
        )
        return bool(np.any(recondition))

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: dict[str, Any],
        det_keep: Any,
        tracker_low_res_masks_global: Any,
        tracker_obj_scores_global: Any,
        tracker_metadata_prev: dict[str, Any],
        tracker_states_local: list[Any],
        is_image_only: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.rank != 0 or self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase.run_tracker_update_planning_phase(distributed)"
            )

        tracker_metadata_new = self._create_planning_metadata(tracker_metadata_prev)
        rank0_metadata_new = _copy_metadata_value(
            tracker_metadata_prev["rank0_metadata"]
        )

        det_masks = det_out["mask"]
        det_scores = det_out["scores"].astype(mx.float32)
        tracker_obj_ids_prev = np.asarray(
            tracker_metadata_prev["obj_ids_all_gpu"],
            dtype=np.int64,
        ).reshape(-1)
        adt_lazy_result = self._associate_det_trk(
            det_masks=det_masks,
            det_scores=det_scores,
            det_keep=det_keep,
            trk_masks=tracker_low_res_masks_global,
            trk_obj_ids=tracker_obj_ids_prev,
            default_det_thresh=self.image_only_det_thresh if is_image_only else None,
        )
        hotstart_to_remove = None
        hotstart_to_suppress = None
        hotstart_gpu_metadata_new = None
        hotstart_planning_enabled = not hasattr(self, "_warm_up_complete") or bool(
            self._warm_up_complete
        )
        if self.is_multiplex and hotstart_planning_enabled:
            (
                hotstart_to_remove,
                hotstart_to_suppress,
                hotstart_gpu_metadata_new,
            ) = self._process_hotstart_gpu(
                frame_idx=frame_idx,
                reverse=reverse,
                adt_result=adt_lazy_result,
                tracker_metadata_prev=tracker_metadata_prev,
                gpu_metadata_prev=tracker_metadata_prev.get("gpu_metadata", {}),
            )
            tracker_metadata_new["gpu_metadata"] = hotstart_gpu_metadata_new
        adt_result = realize_adt_result(
            adt_lazy_result,
            tracker_metadata_prev,
            det_masks,
        )
        new_det_obj_ids, new_det_gpu_ids, num_obj_dropped_due_to_limit = (
            adt_result.get_new_det_gpu_ids(
                tracker_metadata_prev,
                is_image_only,
                det_scores,
                self,
            )
        )
        if hotstart_planning_enabled:
            obj_ids_newly_removed, rank0_metadata_new = self._process_hotstart(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_to_matched_trk_obj_ids=adt_result.det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
                empty_trk_obj_ids=adt_result.empty_trk_obj_ids,
                unmatched_trk_obj_ids=adt_result.unmatched_trk_obj_ids,
                rank0_metadata=rank0_metadata_new,
                tracker_metadata=tracker_metadata_prev,
            )
        else:
            obj_ids_newly_removed = set()
        tracker_metadata_new["rank0_metadata"] = rank0_metadata_new
        if hotstart_to_remove is not None:
            to_remove_np = _array_to_numpy(hotstart_to_remove, dtype=bool).reshape(-1)
            removed_by_gpu = {
                int(obj_id)
                for obj_id, remove in zip(tracker_obj_ids_prev.tolist(), to_remove_np)
                if remove
            }
            if removed_by_gpu != {int(obj_id) for obj_id in obj_ids_newly_removed}:
                raise ValueError(
                    "hotstart gpu metadata removal diverged from object-id "
                    f"metadata; got {removed_by_gpu} and {obj_ids_newly_removed}."
                )
        if hotstart_to_suppress is not None:
            to_suppress_np = _array_to_numpy(hotstart_to_suppress, dtype=bool).reshape(
                -1
            )
            suppressed_by_gpu = {
                int(obj_id)
                for obj_id, suppress in zip(
                    tracker_obj_ids_prev.tolist(),
                    to_suppress_np,
                )
                if suppress
            }
            suppressed_by_rank0 = set(
                rank0_metadata_new.get("suppressed_obj_ids", {}).get(frame_idx, set())
            )
            if suppressed_by_gpu != {int(obj_id) for obj_id in suppressed_by_rank0}:
                raise ValueError(
                    "hotstart gpu metadata suppression diverged from object-id "
                    f"metadata; got {suppressed_by_gpu} and {suppressed_by_rank0}."
                )
        reconditioned_obj_ids: set[int] = set()
        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(adt_result.trk_id_to_max_iou_high_conf_det) > 0
        )
        should_recondition_iou = self._should_recondition_from_bbox_iou(
            det_out=det_out,
            trk_id_to_max_iou_high_conf_det=(
                adt_result.trk_id_to_max_iou_high_conf_det
            ),
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
        )
        if should_recondition_periodic or should_recondition_iou:
            (
                tracker_states_local,
                reconditioned_obj_ids,
                tracker_low_res_masks_global,
            ) = self._recondition_masklets(
                frame_idx,
                det_out,
                adt_result.trk_id_to_max_iou_high_conf_det,
                tracker_states_local,
                tracker_metadata_prev,
                tracker_obj_scores_global,
                tracker_low_res_masks_global,
            )

        batch_size = int(tracker_low_res_masks_global.shape[0])
        if (
            batch_size > 0
            and hotstart_planning_enabled
            and self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0
        ):
            obj_ids_newly_removed_set = {
                int(obj_id) for obj_id in obj_ids_newly_removed
            }
            to_remove_mask = mx.array(
                [
                    int(obj_id) in obj_ids_newly_removed_set
                    for obj_id in tracker_obj_ids_prev.tolist()
                ],
                dtype=mx.bool_,
            )
            tracker_low_res_masks_global = (
                self._suppress_overlapping_based_on_recent_occlusion(
                    frame_idx,
                    tracker_low_res_masks_global,
                    tracker_metadata_prev,
                    tracker_metadata_new,
                    to_remove_mask=to_remove_mask,
                    reverse=reverse,
                )
            )

        for rank in range(self.world_size):
            updated_obj_ids_this_gpu = np.asarray(
                tracker_metadata_prev["obj_ids_per_gpu"][rank],
                dtype=np.int64,
            ).reshape(-1)
            new_det_obj_ids_this_gpu = np.asarray(
                new_det_obj_ids[new_det_gpu_ids == rank],
                dtype=np.int64,
            ).reshape(-1)
            if new_det_obj_ids_this_gpu.size > 0:
                updated_obj_ids_this_gpu = np.concatenate(
                    [updated_obj_ids_this_gpu, new_det_obj_ids_this_gpu]
                )
            if obj_ids_newly_removed:
                keep = ~np.isin(
                    updated_obj_ids_this_gpu,
                    list(obj_ids_newly_removed),
                )
                updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[keep]
            tracker_metadata_new["obj_ids_per_gpu"][rank] = updated_obj_ids_this_gpu
            tracker_metadata_new["num_obj_per_gpu"][rank] = int(
                updated_obj_ids_this_gpu.size
            )

        tracker_metadata_new["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata_new["obj_ids_per_gpu"]
        )
        if self.is_multiplex and hotstart_planning_enabled:
            num_current = int(tracker_metadata_new["obj_ids_all_gpu"].size)
            if hotstart_gpu_metadata_new is None:
                hotstart_gpu_metadata_new = self._hotstart_gpu_metadata_from_rank0(
                    tracker_metadata_new,
                    frame_idx=frame_idx,
                    num_objects=num_current,
                )
            hotstart_gpu_metadata_new = self._compact_hotstart_gpu_metadata(
                hotstart_gpu_metadata_new
            )
            hotstart_gpu_metadata_new = (
                self._extend_hotstart_gpu_metadata_for_new_objects(
                    hotstart_gpu_metadata_new,
                    frame_idx=frame_idx,
                    num_new_objects=int(new_det_obj_ids.size),
                )
            )
            if int(hotstart_gpu_metadata_new["N_obj"]) != num_current:
                raise ValueError(
                    "hotstart gpu metadata size must match updated object ids; got "
                    f"{hotstart_gpu_metadata_new['N_obj']} and {num_current}."
                )
            tracker_metadata_new["gpu_metadata"] = hotstart_gpu_metadata_new
            tracker_metadata_new["num_buc_per_gpu"][self.rank] = (
                math.ceil(num_current / self.bucket_capacity) if num_current > 0 else 0
            )

        det_scores_np = _array_to_numpy(det_scores, dtype=np.float32).reshape(-1)
        score_key = (
            "obj_id_to_sam2_score_frame_wise"
            if self.is_multiplex
            else "obj_id_to_tracker_score_frame_wise"
        )
        tracker_metadata_new.setdefault(score_key, defaultdict(dict))
        frame_scores = tracker_metadata_new[score_key].setdefault(frame_idx, {})
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(
                {
                    int(obj_id): float(det_scores_np[int(det_idx)])
                    for obj_id, det_idx in zip(
                        new_det_obj_ids.tolist(),
                        adt_result.new_det_fa_inds.tolist(),
                    )
                }
            )
            frame_scores.update(
                {
                    int(obj_id): mx.array(
                        float(det_scores_np[int(det_idx)]),
                        dtype=mx.float32,
                    )
                    for obj_id, det_idx in zip(
                        new_det_obj_ids.tolist(),
                        adt_result.new_det_fa_inds.tolist(),
                    )
                }
            )
            tracker_metadata_new["max_obj_id"] = max(
                int(tracker_metadata_new["max_obj_id"]),
                int(np.max(new_det_obj_ids)),
            )

        for obj_id in obj_ids_newly_removed:
            obj_id_int = int(obj_id)
            tracker_metadata_new["obj_id_to_score"][obj_id_int] = -1.0e4
            frame_scores[obj_id_int] = mx.array(-1.0e4, dtype=mx.float32)
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id_int, None)

        if self.masklet_confirmation_enable:
            tracker_metadata_new["rank0_metadata"] = (
                self.update_masklet_confirmation_status(
                    rank0_metadata=tracker_metadata_new["rank0_metadata"],
                    obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids_all_gpu"],
                    obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids_all_gpu"],
                    det_to_matched_trk_obj_ids=(adt_result.det_to_matched_trk_obj_ids),
                    new_det_obj_ids=new_det_obj_ids,
                )
            )

        sam2_update_plan = {
            "new_det_fa_inds": adt_result.new_det_fa_inds,
            "new_det_obj_ids": new_det_obj_ids,
            "new_det_gpu_ids": new_det_gpu_ids,
            "unmatched_trk_obj_ids": adt_result.unmatched_trk_obj_ids,
            "det_to_matched_trk_obj_ids": adt_result.det_to_matched_trk_obj_ids,
            "obj_ids_newly_removed": obj_ids_newly_removed,
            "num_obj_dropped_due_to_limit": int(num_obj_dropped_due_to_limit),
            "trk_id_to_max_iou_high_conf_det": (
                adt_result.trk_id_to_max_iou_high_conf_det
            ),
            "reconditioned_obj_ids": reconditioned_obj_ids,
            "tracker_low_res_masks_global": tracker_low_res_masks_global,
        }
        return sam2_update_plan, tracker_metadata_new

    def _tracker_add_new_objects(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: Any,
        new_obj_masks: Any,
        tracker_states_local: list[Any],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: dict[str, Any],
    ) -> list[Any]:
        new_obj_ids_np = _array_to_numpy(new_obj_ids, dtype=np.int64).reshape(-1)
        if new_obj_ids_np.size == 0:
            return tracker_states_local

        new_obj_masks_mx = (
            new_obj_masks if _is_mlx_array(new_obj_masks) else mx.array(new_obj_masks)
        ).astype(mx.float32)
        if new_obj_masks_mx.ndim != 3:
            raise ValueError(
                "new_obj_masks must have shape (N, H, W), "
                f"got {new_obj_masks_mx.shape}."
            )
        if new_obj_masks_mx.shape[0] != new_obj_ids_np.size:
            raise ValueError(
                "new_obj_masks batch must match new_obj_ids, "
                f"got {new_obj_masks_mx.shape[0]} masks for {new_obj_ids_np.size} ids."
            )

        prev_sam2_state = tracker_states_local[0] if tracker_states_local else None

        def _init_sam2_state(*, copy_backbone_out: bool) -> dict[str, Any]:
            init_state = getattr(self.tracker, "init_state", None)
            if init_state is None:
                raise_unsupported_multiplex_runtime(
                    "Sam3MultiplexBase._tracker_add_new_objects(init_state)"
                )
            new_sam2_state = init_state(
                cached_features=feature_cache,
                video_height=int(orig_vid_height),
                video_width=int(orig_vid_width),
                num_frames=int(num_frames),
            )
            if not isinstance(new_sam2_state, dict):
                raise TypeError("tracker.init_state must return a SAM2 state dict.")
            if copy_backbone_out:
                new_sam2_state["backbone_out"] = (
                    prev_sam2_state.get("backbone_out", None)
                    if isinstance(prev_sam2_state, dict)
                    else None
                )
            return new_sam2_state

        if getattr(self.tracker, "is_multiplex_dynamic", False):
            best_state = None
            best_available_slots = math.inf
            num_new_objects = int(new_obj_ids_np.size)
            for state in tracker_states_local:
                if not isinstance(state, dict):
                    continue
                multiplex_state = state.get("multiplex_state")
                available_slots = getattr(multiplex_state, "available_slots", None)
                if (
                    available_slots is not None
                    and available_slots >= num_new_objects
                    and available_slots < best_available_slots
                ):
                    best_state = state
                    best_available_slots = available_slots

            if best_state is not None:
                sam2_state = best_state
            else:
                sam2_state = _init_sam2_state(copy_backbone_out=True)
                tracker_states_local.append(sam2_state)
        elif tracker_states_local and getattr(self.tracker, "per_obj_inference", False):
            sam2_state = tracker_states_local[0]
        else:
            sam2_state = _init_sam2_state(copy_backbone_out=bool(prev_sam2_state))
            if getattr(self.tracker, "per_obj_inference", False):
                tracker_states_local = [sam2_state]
            else:
                tracker_states_local.append(sam2_state)
        sam2_state.setdefault("obj_ids", [])
        if isinstance(sam2_state, dict):
            sam2_state["cached_features"] = feature_cache
            self._ensure_empty_packed_current_output(sam2_state, frame_idx)

        input_mask_size = getattr(self.tracker, "input_mask_size", None)
        target_size = (
            (int(input_mask_size), int(input_mask_size))
            if input_mask_size is not None
            else (int(orig_vid_height), int(orig_vid_width))
        )
        if new_obj_masks_mx.shape[-2:] != target_size:
            new_obj_masks_mx = interpolate(
                new_obj_masks_mx[:, None, :, :],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )[:, 0, :, :]
        new_obj_masks_mx = new_obj_masks_mx > 0

        add_new_masks = getattr(self.tracker, "add_new_masks", None)
        add_new_mask = getattr(self.tracker, "add_new_mask", None)
        obj_ids = [int(obj_id) for obj_id in new_obj_ids_np.tolist()]
        if add_new_masks is not None:
            add_new_masks(
                inference_state=sam2_state,
                frame_idx=frame_idx,
                obj_ids=obj_ids,
                masks=new_obj_masks_mx,
                add_mask_to_memory=True,
            )
        elif add_new_mask is not None:
            for obj_idx, obj_id in enumerate(obj_ids):
                add_new_mask(
                    inference_state=sam2_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=new_obj_masks_mx[obj_idx],
                    add_mask_to_memory=True,
                )
        else:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._tracker_add_new_objects(add_mask)"
            )

        propagate_preflight = getattr(
            self.tracker, "propagate_in_video_preflight", None
        )
        if propagate_preflight is not None:
            propagate_preflight(sam2_state, run_mem_encoder=True)
        return tracker_states_local

    def _tracker_remove_objects(
        self,
        tracker_states_local: list[Any],
        obj_ids: Any,
    ) -> None:
        if isinstance(obj_ids, set):
            obj_ids_np = np.asarray(sorted(obj_ids), dtype=np.int64).reshape(-1)
        else:
            obj_ids_np = _array_to_numpy(obj_ids, dtype=np.int64).reshape(-1)
        if obj_ids_np.size == 0:
            return
        remove_objects = getattr(self.tracker, "remove_objects", None)
        remove_object = getattr(self.tracker, "remove_object", None)
        kept_states: list[Any] = []
        for sam2_state in tracker_states_local:
            if remove_objects is not None:
                new_obj_ids, _ = remove_objects(
                    sam2_state,
                    obj_ids_np.tolist(),
                    strict=False,
                    need_output=False,
                )
            elif remove_object is not None:
                new_obj_ids = sam2_state.get("obj_ids", [])
                for obj_id in obj_ids_np.tolist():
                    new_obj_ids, _ = remove_object(
                        sam2_state,
                        int(obj_id),
                        strict=False,
                        need_output=False,
                    )
            else:
                new_obj_ids = self._packed_state_remove_objects(
                    sam2_state,
                    obj_ids_np,
                )
                if new_obj_ids is None:
                    raise_unsupported_multiplex_runtime(
                        "Sam3MultiplexBase._tracker_remove_objects(remove_object)"
                    )
            if len(new_obj_ids) > 0:
                kept_states.append(sam2_state)
        tracker_states_local[:] = kept_states

    def _packed_state_remove_objects(
        self,
        sam2_state: Any,
        obj_ids_np: np.ndarray,
    ) -> list[int] | None:
        if not isinstance(sam2_state, dict):
            return None
        multiplex_state = sam2_state.get("multiplex_state")
        if multiplex_state is None or not hasattr(multiplex_state, "remove_objects"):
            return None

        if getattr(multiplex_state, "assignments", None) is None:
            sam2_state["obj_ids"] = []
            return []

        state_obj_ids = sam2_state.get(
            "obj_ids",
            getattr(multiplex_state, "object_ids", None),
        )
        if state_obj_ids is None:
            raise ValueError(
                "Packed tracker state removal requires obj_ids or "
                "multiplex_state.object_ids."
            )

        active_obj_ids = _array_to_numpy(state_obj_ids, dtype=np.int64).reshape(-1)
        expected_entries = int(getattr(multiplex_state, "total_valid_entries"))
        if active_obj_ids.size != expected_entries:
            raise ValueError(
                "Packed tracker state obj_ids must map one-to-one with "
                f"multiplex entries; got {active_obj_ids.size} obj_ids for "
                f"{expected_entries} entries."
            )

        remove_set = {int(obj_id) for obj_id in obj_ids_np.tolist()}
        indices_to_remove = [
            idx
            for idx, obj_id in enumerate(active_obj_ids.tolist())
            if int(obj_id) in remove_set
        ]
        if not indices_to_remove:
            return [int(obj_id) for obj_id in active_obj_ids.tolist()]

        multiplex_state.remove_objects(indices_to_remove, strict=False)
        if getattr(multiplex_state, "object_ids", None) is not None:
            new_obj_ids = [int(obj_id) for obj_id in multiplex_state.object_ids]
        else:
            remove_indices = set(indices_to_remove)
            new_obj_ids = [
                int(obj_id)
                for idx, obj_id in enumerate(active_obj_ids.tolist())
                if idx not in remove_indices
            ]
        sam2_state["obj_ids"] = new_obj_ids
        return new_obj_ids

    def _tracker_update_memories(
        self,
        sam2_inference_states: list[Any],
        frame_idx: int,
        tracker_metadata: dict[str, Any],
        low_res_masks: Any,
    ) -> None:
        if len(sam2_inference_states) == 0:
            return

        run_memory_encoder = getattr(self.tracker, "_run_memory_encoder", None)
        if run_memory_encoder is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._tracker_update_memories(_run_memory_encoder)"
            )
        add_output_per_object = getattr(self.tracker, "add_output_per_object", None)
        if add_output_per_object is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._tracker_update_memories(add_output_per_object)"
            )

        maskmem_backbone = getattr(self.tracker, "maskmem_backbone", None)
        mask_downsampler = getattr(maskmem_backbone, "mask_downsampler", None)
        interpol_size = getattr(mask_downsampler, "interpol_size", None)
        if interpol_size is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._tracker_update_memories(interpol_size)"
            )
        high_res_h, high_res_w = (int(interpol_size[0]), int(interpol_size[1]))

        low_res_masks_mx = (
            low_res_masks if _is_mlx_array(low_res_masks) else mx.array(low_res_masks)
        ).astype(mx.float32)
        if low_res_masks_mx.ndim != 3:
            raise ValueError(
                "low_res_masks must have shape (N, H, W), "
                f"got {low_res_masks_mx.shape}."
            )

        expected_num_objects = sum(
            len(state.get("obj_ids", []))
            for state in sam2_inference_states
            if isinstance(state, dict)
        )
        if low_res_masks_mx.shape[0] != expected_num_objects:
            raise ValueError(
                "low_res_masks batch must match local SAM2 object count; got "
                f"{low_res_masks_mx.shape[0]} masks for {expected_num_objects} objects."
            )

        high_res_masks = interpolate(
            low_res_masks_mx[:, None, :, :],
            size=(high_res_h, high_res_w),
            mode="bilinear",
            align_corners=False,
        )
        suppress_masks = getattr(
            self.tracker, "_suppress_object_pw_area_shrinkage", None
        )
        if suppress_masks is not None:
            high_res_masks = suppress_masks(high_res_masks)

        mask_has_object = mx.sum(high_res_masks > 0, axis=(-1, -2)) > 0
        object_score_logits = mx.where(
            mask_has_object,
            mx.full(mask_has_object.shape, 10.0, dtype=mx.float32),
            mx.full(mask_has_object.shape, -10.0, dtype=mx.float32),
        )

        dynamic_multiplex = bool(
            self.is_multiplex and getattr(self.tracker, "is_multiplex_dynamic", False)
        )
        object_idx_assignment: dict[int, list[int]] = {}
        if dynamic_multiplex:
            all_object_ids: list[int] = []
            object_id_to_state_idx: dict[int, int] = {}
            for state_idx, sam2_state in enumerate(sam2_inference_states):
                obj_ids = [int(obj_id) for obj_id in sam2_state.get("obj_ids", [])]
                all_object_ids.extend(obj_ids)
                for obj_id in obj_ids:
                    object_id_to_state_idx[obj_id] = state_idx
                object_idx_assignment[state_idx] = []
            sorted_indices = sorted(
                range(len(all_object_ids)),
                key=lambda idx: all_object_ids[idx],
            )
            for global_idx, local_idx in enumerate(sorted_indices):
                obj_id = all_object_ids[local_idx]
                object_idx_assignment[object_id_to_state_idx[obj_id]].append(global_idx)

        start_idx_state = int(
            np.sum(
                np.asarray(
                    tracker_metadata.get("num_obj_per_gpu", [0]),
                    dtype=np.int64,
                )[: self.rank]
            )
        )
        for state_idx, sam2_state in enumerate(sam2_inference_states):
            num_obj_per_state = len(sam2_state.get("obj_ids", []))
            if num_obj_per_state == 0:
                continue
            if dynamic_multiplex:
                local_idx = mx.array(object_idx_assignment[state_idx], dtype=mx.int64)
                local_high_res_masks = mx.take(high_res_masks, local_idx, axis=0)
                local_object_score_logits = mx.take(
                    object_score_logits,
                    local_idx,
                    axis=0,
                )
            else:
                end_idx_state = start_idx_state + num_obj_per_state
                local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
                local_object_score_logits = object_score_logits[
                    start_idx_state:end_idx_state
                ]

            encoded_mem = tuple(
                run_memory_encoder(
                    sam2_state,
                    frame_idx,
                    num_obj_per_state,
                    local_high_res_masks,
                    local_object_score_logits,
                    is_mask_from_pts=False,
                )
            )
            if self.is_multiplex:
                if len(encoded_mem) != 4:
                    raise ValueError(
                        "Multiplex tracker memory encoder must return "
                        "(maskmem_features, maskmem_pos_enc, image_features, "
                        "image_pos_enc)."
                    )
                (
                    local_maskmem_features,
                    local_maskmem_pos_enc,
                    local_image_features,
                    local_image_pos_enc,
                ) = encoded_mem
            else:
                if len(encoded_mem) != 2:
                    raise ValueError(
                        "Tracker memory encoder must return "
                        "(maskmem_features, maskmem_pos_enc)."
                    )
                local_maskmem_features, local_maskmem_pos_enc = encoded_mem
                local_image_features = local_image_pos_enc = None

            output_dict = sam2_state.get("output_dict")
            if output_dict is None:
                raise ValueError(
                    "SAM2 state must contain output_dict for memory update."
                )
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                storage_outputs = output_dict.get(storage_key, {})
                if frame_idx not in storage_outputs:
                    continue
                current_out = storage_outputs[frame_idx]
                current_out["maskmem_features"] = local_maskmem_features
                current_out["maskmem_pos_enc"] = [pos for pos in local_maskmem_pos_enc]
                if self.is_multiplex:
                    current_out["image_features"] = local_image_features
                    current_out["image_pos_enc"] = local_image_pos_enc
                    if self.reapply_no_object_pointer:
                        self._reapply_no_object_pointer_for_suppressed(
                            sam2_state,
                            current_out,
                            local_object_score_logits,
                        )
                elif self.reapply_no_object_pointer:
                    raise_unsupported_multiplex_runtime(
                        "Sam3MultiplexBase._tracker_update_memories("
                        "reapply_no_object_pointer_non_multiplex)"
                    )
                add_output_per_object(
                    inference_state=sam2_state,
                    frame_idx=frame_idx,
                    current_out=current_out,
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _reapply_no_object_pointer_for_suppressed(
        self,
        sam2_state: dict[str, Any],
        current_out: dict[str, Any],
        local_object_score_logits: Any,
    ) -> None:
        no_obj_ptr_linear = getattr(self.tracker, "no_obj_ptr_linear", None)
        if no_obj_ptr_linear is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._tracker_update_memories(no_obj_ptr_linear)"
            )
        if "obj_ptr" not in current_out:
            raise ValueError(
                "current_out must contain obj_ptr when reapply_no_object_pointer=True."
            )
        if "object_score_logits" not in current_out:
            raise ValueError(
                "current_out must contain object_score_logits when "
                "reapply_no_object_pointer=True."
            )
        multiplex_state = sam2_state.get("multiplex_state")
        if multiplex_state is None:
            raise ValueError(
                "SAM2 state must contain multiplex_state when "
                "reapply_no_object_pointer=True."
            )
        if not hasattr(self.tracker, "object_score_logit_threshold"):
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._tracker_update_memories("
                "object_score_logit_threshold)"
            )

        previous_logits = (
            current_out["object_score_logits"]
            if _is_mlx_array(current_out["object_score_logits"])
            else mx.array(current_out["object_score_logits"])
        ).astype(mx.float32)
        new_logits = (
            local_object_score_logits
            if _is_mlx_array(local_object_score_logits)
            else mx.array(local_object_score_logits)
        ).astype(mx.float32)
        if previous_logits.ndim == 1:
            previous_logits = previous_logits[:, None]
        if new_logits.ndim == 1:
            new_logits = new_logits[:, None]
        if previous_logits.shape != new_logits.shape:
            raise ValueError(
                "current_out object_score_logits must match local object score "
                f"logits shape; got {previous_logits.shape} and {new_logits.shape}."
            )
        if previous_logits.ndim != 2 or previous_logits.shape[1] != 1:
            raise ValueError(
                "object score logits must have shape (N, 1), "
                f"got {previous_logits.shape}."
            )

        threshold = float(self.tracker.object_score_logit_threshold)
        newly_suppressed = (previous_logits > threshold) & (new_logits < 0)
        any_suppressed = _array_to_numpy(
            mx.any(newly_suppressed),
            dtype=bool,
        ).reshape(-1)[0]
        if not bool(any_suppressed):
            return
        existing_pointers = multiplex_state.demux(current_out["obj_ptr"])
        if newly_suppressed.shape[0] != existing_pointers.shape[0]:
            raise ValueError(
                "object_score_logits must align with demuxed obj_ptr entries; got "
                f"{newly_suppressed.shape[0]} logits for "
                f"{existing_pointers.shape[0]} pointers."
            )
        replacement_pointers = no_obj_ptr_linear(existing_pointers)
        if replacement_pointers.shape != existing_pointers.shape:
            raise ValueError(
                "tracker.no_obj_ptr_linear must preserve obj_ptr shape; got "
                f"{replacement_pointers.shape} for {existing_pointers.shape}."
            )
        suppress_mask = newly_suppressed.astype(existing_pointers.dtype)
        suppress_mask = suppress_mask.reshape(
            (suppress_mask.shape[0],) + (1,) * (existing_pointers.ndim - 1)
        )
        new_pointers = (
            suppress_mask * replacement_pointers
            + (1.0 - suppress_mask) * existing_pointers
        )
        current_out["obj_ptr"] = multiplex_state.mux(new_pointers)

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: Any,
        last_occluded: Any,
        obj_ids: Any,
        frame_idx: int | None = None,
        reverse: bool = False,
    ) -> Any:
        del frame_idx
        masks = (
            binary_low_res_masks
            if _is_mlx_array(binary_low_res_masks)
            else mx.array(binary_low_res_masks)
        ).astype(mx.bool_)
        if masks.ndim != 3:
            raise ValueError(
                f"binary_low_res_masks must have shape (N, H, W), got {masks.shape}."
            )
        obj_ids_np = np.asarray(obj_ids, dtype=np.int64).reshape(-1)
        if masks.shape[0] != obj_ids_np.size:
            raise ValueError(
                "binary_low_res_masks and obj_ids must have the same length; "
                f"got {masks.shape[0]} masks for {obj_ids_np.size} object ids."
            )
        if obj_ids_np.size <= 1:
            return mx.zeros((obj_ids_np.size,), dtype=mx.bool_)

        last_occ = (
            last_occluded if _is_mlx_array(last_occluded) else mx.array(last_occluded)
        ).astype(mx.int64)
        if last_occ.shape[0] != obj_ids_np.size:
            raise ValueError(
                "last_occluded must align with obj_ids; got "
                f"{last_occ.shape[0]} values for {obj_ids_np.size} object ids."
            )

        iou = mask_iou(masks, masks)
        idx = mx.arange(obj_ids_np.size, dtype=mx.int64)
        upper_tri = idx[:, None] < idx[None, :]
        overlapping_pairs = (
            iou >= float(self.suppress_overlapping_based_on_recent_occlusion_threshold)
        ) & upper_tri

        last_i = last_occ[:, None]
        last_j = last_occ[None, :]
        if reverse:
            i_more_recent = last_i < last_j
            j_more_recent = last_j < last_i
        else:
            i_more_recent = last_i > last_j
            j_more_recent = last_j > last_i

        if self.allow_unoccluded_to_suppress:
            suppress_i_mask = overlapping_pairs & i_more_recent
            suppress_j_mask = overlapping_pairs & j_more_recent
        else:
            suppress_i_mask = overlapping_pairs & i_more_recent & (last_j > -1)
            suppress_j_mask = overlapping_pairs & j_more_recent & (last_i > -1)
        return mx.any(suppress_i_mask, axis=1) | mx.any(suppress_j_mask, axis=0)

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: Any,
        tracker_metadata_prev: dict[str, Any],
        tracker_metadata_new: dict[str, Any],
        to_remove_mask: Any | None = None,
        reverse: bool = False,
    ) -> Any:
        if self.suppress_overlapping_based_on_recent_occlusion_threshold <= 0.0:
            return tracker_low_res_masks_global

        masks = (
            tracker_low_res_masks_global
            if _is_mlx_array(tracker_low_res_masks_global)
            else mx.array(tracker_low_res_masks_global)
        ).astype(mx.float32)
        if masks.ndim != 3:
            raise ValueError(
                "tracker_low_res_masks_global must have shape (N, H, W), "
                f"got {masks.shape}."
            )
        obj_ids = np.asarray(
            tracker_metadata_prev["obj_ids_all_gpu"],
            dtype=np.int64,
        ).reshape(-1)
        if masks.shape[0] != obj_ids.size:
            raise ValueError(
                "Mask/metadata count mismatch in _suppress_overlapping: "
                f"batch_size={masks.shape[0]}, num_ids={obj_ids.size}, "
                f"frame_idx={frame_idx}."
            )
        if obj_ids.size == 0:
            return masks

        gpu_metadata = tracker_metadata_new.setdefault("gpu_metadata", {})
        last_occluded_tensor = gpu_metadata.get("last_occluded_tensor")
        if last_occluded_tensor is None:
            last_occluded_by_id = tracker_metadata_prev.get(
                "obj_id_to_last_occluded",
                tracker_metadata_new.get("obj_id_to_last_occluded", {}),
            )
            last_occluded_np = np.array(
                [
                    int(last_occluded_by_id.get(int(obj_id), -1))
                    for obj_id in obj_ids.tolist()
                ],
                dtype=np.int64,
            )
            last_occluded = mx.array(last_occluded_np, dtype=mx.int64)
        else:
            last_occluded = (
                last_occluded_tensor
                if _is_mlx_array(last_occluded_tensor)
                else mx.array(last_occluded_tensor)
            ).astype(mx.int64)
        if last_occluded.shape[0] != obj_ids.size:
            raise ValueError(
                "last_occluded_tensor must align with obj_ids; got "
                f"{last_occluded.shape[0]} values for {obj_ids.size} object ids."
            )

        if to_remove_mask is None:
            remove_mask = mx.zeros((obj_ids.size,), dtype=mx.bool_)
        elif isinstance(to_remove_mask, set):
            raise TypeError(
                "to_remove_mask must be a boolean vector aligned with obj_ids, "
                "not an object-id set."
            )
        else:
            remove_mask = (
                to_remove_mask
                if _is_mlx_array(to_remove_mask)
                else mx.array(to_remove_mask)
            ).astype(mx.bool_)
            if remove_mask.shape[0] != obj_ids.size:
                raise ValueError(
                    "to_remove_mask must align with obj_ids; got "
                    f"{remove_mask.shape[0]} values for {obj_ids.size} object ids."
                )
        always_occluded = mx.full(last_occluded.shape, 100000, dtype=mx.int64)
        last_occluded_prev = mx.where(remove_mask, always_occluded, last_occluded)

        binary_masks = masks > 0
        to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
            binary_masks,
            last_occluded_prev,
            obj_ids,
            frame_idx=frame_idx,
            reverse=reverse,
        )
        is_obj_occluded = ~mx.any(
            binary_masks.reshape(binary_masks.shape[0], -1),
            axis=1,
        )
        is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
        last_occluded_new = mx.where(
            is_obj_occluded_or_suppressed,
            mx.full(last_occluded_prev.shape, int(frame_idx), dtype=mx.int64),
            last_occluded_prev,
        )
        gpu_metadata["last_occluded_tensor"] = last_occluded_new
        last_occluded_np = _array_to_numpy(
            last_occluded_new,
            dtype=np.int64,
        ).reshape(-1)
        tracker_metadata_new["obj_id_to_last_occluded"] = {
            int(obj_id): int(last_occluded_np[idx])
            for idx, obj_id in enumerate(obj_ids.tolist())
            if int(last_occluded_np[idx]) >= 0
        }

        suppress_shape = (to_suppress.shape[0],) + (1,) * (masks.ndim - 1)
        return mx.where(
            to_suppress.reshape(suppress_shape),
            mx.full(masks.shape, -10.0, dtype=masks.dtype),
            masks,
        )

    def _can_update_tracker_memories(self) -> bool:
        if getattr(self.tracker, "_run_memory_encoder", None) is None:
            return False
        if getattr(self.tracker, "add_output_per_object", None) is None:
            return False
        maskmem_backbone = getattr(self.tracker, "maskmem_backbone", None)
        mask_downsampler = getattr(maskmem_backbone, "mask_downsampler", None)
        return getattr(mask_downsampler, "interpol_size", None) is not None

    def run_tracker_update_execution_phase(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: dict[str, Any],
        tracker_states_local: list[Any],
        tracker_update_plan: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: dict[str, Any],
        tracker_metadata_new: dict[str, Any] | None = None,
    ) -> list[Any]:
        new_det_fa_inds = np.asarray(
            tracker_update_plan["new_det_fa_inds"],
            dtype=np.int64,
        ).reshape(-1)
        new_det_obj_ids = np.asarray(
            tracker_update_plan["new_det_obj_ids"],
            dtype=np.int64,
        ).reshape(-1)
        new_det_gpu_ids = np.asarray(
            tracker_update_plan["new_det_gpu_ids"],
            dtype=np.int64,
        ).reshape(-1)
        is_on_this_rank = new_det_gpu_ids == self.rank
        new_det_fa_inds_local = new_det_fa_inds[is_on_this_rank]
        new_det_obj_ids_local = new_det_obj_ids[is_on_this_rank]
        tracker_low_res_masks_global = tracker_update_plan.get(
            "tracker_low_res_masks_global"
        )
        obj_ids_newly_removed = tracker_update_plan.get("obj_ids_newly_removed", set())
        if (
            tracker_low_res_masks_global is not None
            and tracker_states_local
            and self._can_update_tracker_memories()
        ):
            memory_masks = (
                tracker_low_res_masks_global
                if _is_mlx_array(tracker_low_res_masks_global)
                else mx.array(tracker_low_res_masks_global)
            )
            if memory_masks.shape[0] > 0:
                memory_metadata = tracker_metadata_new
                if memory_metadata is None:
                    num_local_objects = sum(
                        len(state.get("obj_ids", []))
                        for state in tracker_states_local
                        if isinstance(state, dict)
                    )
                    memory_metadata = {
                        "num_obj_per_gpu": np.array(
                            [num_local_objects],
                            dtype=np.int64,
                        )
                    }
                if (
                    tracker_metadata_new is not None
                    and self.suppress_overlapping_based_on_recent_occlusion_threshold
                    > 0.0
                ):
                    memory_obj_ids = np.array(
                        [
                            int(obj_id)
                            for state in tracker_states_local
                            if isinstance(state, dict)
                            for obj_id in state.get("obj_ids", [])
                        ],
                        dtype=np.int64,
                    )
                    memory_metadata_prev = {
                        **memory_metadata,
                        "obj_ids_all_gpu": memory_obj_ids,
                        "obj_id_to_last_occluded": tracker_metadata_new.get(
                            "obj_id_to_last_occluded",
                            {},
                        ),
                    }
                    remove_set = {
                        int(obj_id)
                        for obj_id in (
                            sorted(obj_ids_newly_removed)
                            if isinstance(obj_ids_newly_removed, set)
                            else np.asarray(obj_ids_newly_removed).reshape(-1).tolist()
                        )
                    }
                    to_remove_mask = mx.array(
                        [
                            int(obj_id) in remove_set
                            for obj_id in memory_obj_ids.tolist()
                        ],
                        dtype=mx.bool_,
                    )
                    memory_masks = self._suppress_overlapping_based_on_recent_occlusion(
                        frame_idx,
                        memory_masks,
                        memory_metadata_prev,
                        tracker_metadata_new,
                        to_remove_mask=to_remove_mask,
                        reverse=reverse,
                    )
                self._tracker_update_memories(
                    tracker_states_local,
                    frame_idx,
                    tracker_metadata=memory_metadata,
                    low_res_masks=memory_masks,
                )
        if new_det_fa_inds_local.size > 0:
            det_masks = det_out["mask"]
            det_masks_mx = (
                det_masks if _is_mlx_array(det_masks) else mx.array(det_masks)
            )
            new_det_masks = mx.take(
                det_masks_mx,
                mx.array(new_det_fa_inds_local, dtype=mx.int64),
                axis=0,
            )
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                feature_cache=feature_cache,
            )

        if obj_ids_newly_removed:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)

        if tracker_metadata_new is not None:
            self._sync_execution_phase_metadata(
                tracker_states_local,
                tracker_metadata_new,
                frame_idx=frame_idx,
            )
        return tracker_states_local

    def _can_run_local_detector_update(
        self,
        input_batch: Any,
    ) -> bool:
        if input_batch is None or not hasattr(self.detector, "backbone"):
            return False
        detector_method = (
            "forward_video_grounding_batched_multigpu"
            if self.use_batched_grounding
            else "forward_video_grounding_multigpu"
        )
        return hasattr(self.detector, detector_method)

    def _run_local_tracker_detector_frame(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: Any,
        geometric_prompt: Any,
        tracker_states_local: list[Any],
        tracker_metadata_prev: dict[str, Any],
        feature_cache: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
    ) -> tuple[
        dict[int, Any], dict[int, float], list[Any], dict[str, Any], dict[str, int], Any
    ]:
        det_out, det_keep = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            use_batched_grounding=self.use_batched_grounding,
            batched_grounding_batch_size=self.batched_grounding_batch_size,
        )
        det_out, det_keep = self._normalize_video_detection_outputs(det_out, det_keep)

        obj_ids_local, low_res_masks_local, sam2_scores_local = (
            self._propogate_tracker_one_frame_local_gpu(
                tracker_states_local,
                frame_idx=frame_idx,
                reverse=reverse,
                run_mem_encoder=False,
            )
        )
        self._ensure_tracker_metadata_for_local_states(
            tracker_metadata_prev,
            obj_ids_local,
        )

        if low_res_masks_local:
            tracker_low_res_masks_global = mx.stack(low_res_masks_local, axis=0)
        else:
            tracker_low_res_masks_global = mx.zeros(
                (0, *det_out["mask"].shape[-2:]),
                dtype=mx.float32,
            )
        if sam2_scores_local:
            tracker_obj_scores_global = mx.stack(
                [
                    (
                        score
                        if _is_mlx_array(score)
                        else mx.array(score, dtype=mx.float32)
                    ).reshape(())
                    for score in sam2_scores_local
                ],
                axis=0,
            )
        else:
            tracker_obj_scores_global = mx.zeros((0,), dtype=mx.float32)

        tracker_update_plan, tracker_metadata_new = (
            self.run_tracker_update_planning_phase(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                det_keep=det_keep,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_states_local=tracker_states_local,
                is_image_only=is_image_only,
            )
        )
        tracker_low_res_masks_global = tracker_update_plan.get(
            "tracker_low_res_masks_global",
            tracker_low_res_masks_global,
        )
        tracker_states_local_new = self.run_tracker_update_execution_phase(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            det_out=det_out,
            tracker_states_local=tracker_states_local,
            tracker_update_plan=tracker_update_plan,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            feature_cache=feature_cache,
            tracker_metadata_new=tracker_metadata_new,
        )
        obj_id_to_mask = self.build_outputs(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_obj_scores_global=tracker_obj_scores_global,
            tracker_metadata_prev=tracker_metadata_prev,
            sam2_update_plan=tracker_update_plan,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            reconditioned_obj_ids=tracker_update_plan.get(
                "reconditioned_obj_ids",
                set(),
            ),
            det_to_matched_trk_obj_ids=tracker_update_plan.get(
                "det_to_matched_trk_obj_ids",
                {},
            ),
        )

        score_key = (
            "obj_id_to_sam2_score_frame_wise"
            if self.is_multiplex
            else "obj_id_to_tracker_score_frame_wise"
        )
        frame_scores = tracker_metadata_new[score_key].setdefault(frame_idx, {})
        tracker_obj_scores_prob = mx.sigmoid(
            tracker_obj_scores_global.astype(mx.float32)
        )
        for obj_idx, obj_id in enumerate(obj_ids_local):
            frame_scores[int(obj_id)] = tracker_obj_scores_prob[obj_idx]

        frame_stats = {
            "num_obj_tracked": int(np.sum(tracker_metadata_new["num_obj_per_gpu"])),
            "num_obj_dropped": int(tracker_update_plan["num_obj_dropped_due_to_limit"]),
        }
        return (
            obj_id_to_mask,
            tracker_metadata_new["obj_id_to_score"],
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_prob,
        )

    def _cache_tracker_backbone_features(
        self,
        *,
        frame_idx: int,
        input_batch: Any,
        sam3_image_out: dict[str, Any],
        feature_cache: dict[str, Any],
    ) -> None:
        def _check_cache_keys(
            keys: tuple[str, ...],
            *,
            label: str,
        ) -> bool:
            present = [key in sam3_image_out for key in keys]
            if not any(present):
                return False
            if not all(present):
                missing = [
                    key for key, is_present in zip(keys, present) if not is_present
                ]
                raise ValueError(
                    f"{label} backbone feature cache is incomplete; missing "
                    f"{', '.join(missing)}."
                )
            return True

        sam2_keys = (
            "sam2_backbone_fpn_0",
            "sam2_backbone_fpn_1",
            "sam2_backbone_fpn_2",
            "sam2_backbone_pos_enc",
        )
        if not _check_cache_keys(sam2_keys, label="SAM2"):
            return
        sam_mask_decoder = getattr(self.tracker, "sam_mask_decoder", None)
        if sam_mask_decoder is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase.run_backbone_and_detection(sam2-backbone-cache)"
            )

        backbone_cache: dict[str, Any] = {}
        interactive_keys = (
            "interactive_backbone_fpn_0",
            "interactive_backbone_fpn_1",
            "interactive_backbone_fpn_2",
            "interactive_backbone_pos_enc",
        )
        has_interactive_cache = self.is_multiplex and _check_cache_keys(
            interactive_keys,
            label="Interactive",
        )
        if has_interactive_cache:
            interactive_sam_mask_decoder = getattr(
                self.tracker,
                "interactive_sam_mask_decoder",
                None,
            )
            if interactive_sam_mask_decoder is None:
                raise_unsupported_multiplex_runtime(
                    "Sam3MultiplexBase.run_backbone_and_detection("
                    "interactive-backbone-cache)"
                )
            interactive_backbone_fpn = [
                interactive_sam_mask_decoder.conv_s0(
                    sam3_image_out["interactive_backbone_fpn_0"]
                ),
                interactive_sam_mask_decoder.conv_s1(
                    sam3_image_out["interactive_backbone_fpn_1"]
                ),
                sam3_image_out["interactive_backbone_fpn_2"],
            ]
            backbone_cache["interactive"] = {
                "vision_features": interactive_backbone_fpn[-1],
                "vision_mask": None,
                "vision_pos_enc": sam3_image_out["interactive_backbone_pos_enc"],
                "backbone_fpn": interactive_backbone_fpn,
            }

        sam2_backbone_fpn = [
            sam_mask_decoder.conv_s0(sam3_image_out["sam2_backbone_fpn_0"]),
            sam_mask_decoder.conv_s1(sam3_image_out["sam2_backbone_fpn_1"]),
            sam3_image_out["sam2_backbone_fpn_2"],
        ]
        sam2_backbone_out = {
            "vision_features": sam2_backbone_fpn[-1],
            "vision_mask": None,
            "vision_pos_enc": sam3_image_out["sam2_backbone_pos_enc"],
            "backbone_fpn": sam2_backbone_fpn,
        }
        if has_interactive_cache:
            backbone_cache["sam2_backbone_out"] = sam2_backbone_out
        else:
            backbone_cache = sam2_backbone_out
        feature_cache[frame_idx] = (
            input_batch.img_batch[frame_idx : frame_idx + 1],
            backbone_cache,
        )

    def run_backbone_and_detection(
        self,
        frame_idx: int,
        num_frames: int,
        input_batch: Any,
        geometric_prompt: Any,
        feature_cache: dict[str, Any],
        reverse: bool,
        use_batched_grounding: bool = False,
        batched_grounding_batch_size: int = 16,
    ) -> tuple[dict[str, Any], Any]:
        text_batch_key = tuple(input_batch.find_text_batch)
        if "text" not in feature_cache or text_batch_key not in feature_cache["text"]:
            text_outputs = self.detector.backbone.forward_text(
                input_batch.find_text_batch,
                device=self.device,
            )
            feature_cache["text"] = {text_batch_key: text_outputs}
        else:
            text_outputs = feature_cache["text"][text_batch_key]

        tracking_bounds = feature_cache.get("tracking_bounds", {})
        max_frame_num_to_track = tracking_bounds.get("max_frame_num_to_track")
        start_frame_idx = tracking_bounds.get("propagate_in_video_start_frame_idx")
        backbone_out = {
            "img_batch_all_stages": input_batch.img_batch,
            **text_outputs,
        }

        if use_batched_grounding:
            sam3_image_out, _ = self.detector.forward_video_grounding_batched_multigpu(
                backbone_out=backbone_out,
                find_inputs=input_batch.find_inputs,
                geometric_prompt=geometric_prompt,
                frame_idx=frame_idx,
                num_frames=num_frames,
                grounding_cache=feature_cache.setdefault("grounding_cache", {}),
                track_in_reverse=reverse,
                return_sam2_backbone_feats=True,
                run_nms=self.det_nms_thresh > 0.0,
                nms_prob_thresh=self.score_threshold_detection,
                nms_iou_thresh=self.det_nms_thresh,
                nms_use_iom=self.det_nms_use_iom,
                max_frame_num_to_track=max_frame_num_to_track,
                propagate_in_video_start_frame_idx=start_frame_idx,
                feature_cache=feature_cache,
                batch_size=batched_grounding_batch_size,
            )
        else:
            sam3_image_out, _ = self.detector.forward_video_grounding_multigpu(
                backbone_out=backbone_out,
                find_inputs=input_batch.find_inputs,
                geometric_prompt=geometric_prompt,
                frame_idx=frame_idx,
                num_frames=num_frames,
                multigpu_buffer=feature_cache.setdefault("multigpu_buffer", {}),
                track_in_reverse=reverse,
                return_sam2_backbone_feats=True,
                run_nms=self.det_nms_thresh > 0.0,
                nms_prob_thresh=self.score_threshold_detection,
                nms_iou_thresh=self.det_nms_thresh,
                nms_use_iom=self.det_nms_use_iom,
                max_frame_num_to_track=max_frame_num_to_track,
                propagate_in_video_start_frame_idx=start_frame_idx,
                feature_cache=feature_cache,
            )

        pred_logits = sam3_image_out["pred_logits"]
        pred_probs = mx.sigmoid(pred_logits.squeeze(-1))
        pos_pred_mask = pred_probs > self.score_threshold_detection
        if self.suppress_det_close_to_boundary:
            keep = self._suppress_detections_close_to_boundary(
                sam3_image_out["pred_boxes_xyxy"]
            )
            pos_pred_mask = pos_pred_mask & keep
        det_out = {
            "bbox": sam3_image_out["pred_boxes_xyxy"],
            "mask": sam3_image_out["pred_masks"],
            "scores": pred_probs,
        }
        self._cache_tracker_backbone_features(
            frame_idx=frame_idx,
            input_batch=input_batch,
            sam3_image_out=sam3_image_out,
            feature_cache=feature_cache,
        )
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
        return det_out, pos_pred_mask

    def _associate_det_trk(
        self,
        det_masks: Any,
        det_scores: Any,
        det_keep: Any,
        trk_masks: Any,
        trk_obj_ids: np.ndarray,
        default_det_thresh: float | None = None,
    ) -> LazyAssociateDetTrkResult:
        if not _is_floating_array(det_masks):
            raise TypeError("det_masks must be floating-point mask logits.")
        if not _is_floating_array(trk_masks):
            raise TypeError("trk_masks must be floating-point mask logits.")
        trk_obj_ids = np.asarray(trk_obj_ids)
        if trk_masks.shape[0] != len(trk_obj_ids):
            raise ValueError(
                "trk_masks and trk_obj_ids must have the same length, "
                f"got {trk_masks.shape[0]} and {len(trk_obj_ids)}."
            )

        det_masks_mx = det_masks if _is_mlx_array(det_masks) else mx.array(det_masks)
        trk_masks_mx = trk_masks if _is_mlx_array(trk_masks) else mx.array(trk_masks)
        det_scores_mx = (
            det_scores if _is_mlx_array(det_scores) else mx.array(det_scores)
        )
        det_keep_mx = det_keep if _is_mlx_array(det_keep) else mx.array(det_keep)

        if det_masks_mx.ndim != 3:
            raise ValueError(
                f"det_masks must have shape (N, H, W), got {det_masks_mx.shape}."
            )
        if trk_masks_mx.ndim != 3:
            raise ValueError(
                f"trk_masks must have shape (M, H, W), got {trk_masks_mx.shape}."
            )
        if det_scores_mx.shape[0] != det_masks_mx.shape[0]:
            raise ValueError(
                "det_scores length must match det_masks, "
                f"got {det_scores_mx.shape[0]} and {det_masks_mx.shape[0]}."
            )
        if det_keep_mx.shape[0] != det_masks_mx.shape[0]:
            raise ValueError(
                "det_keep length must match det_masks, "
                f"got {det_keep_mx.shape[0]} and {det_masks_mx.shape[0]}."
            )

        if det_masks_mx.shape[-2:] != trk_masks_mx.shape[-2:]:
            if np.prod(det_masks_mx.shape[-2:]) < np.prod(trk_masks_mx.shape[-2:]):
                trk_masks_mx = interpolate(
                    trk_masks_mx[:, None, :, :],
                    size=det_masks_mx.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0, :, :]
            else:
                det_masks_mx = interpolate(
                    det_masks_mx[:, None, :, :],
                    size=trk_masks_mx.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0, :, :]

        trk_count = trk_masks_mx.shape[0]
        if trk_count < self.max_num_objects:
            padding_size = self.max_num_objects - trk_count
            padding = mx.zeros(
                (padding_size, *trk_masks_mx.shape[1:]),
                dtype=trk_masks_mx.dtype,
            )
            trk_masks_for_assoc = mx.concat([trk_masks_mx, padding], axis=0)
        else:
            trk_masks_for_assoc = trk_masks_mx

        result = _associate_det_trk_compilable(
            det_masks_mx,
            det_scores_mx,
            det_keep_mx,
            trk_masks_for_assoc,
            self.new_det_thresh if default_det_thresh is None else default_det_thresh,
            self.trk_assoc_iou_thresh,
            self.assoc_iou_thresh,
            0.8,
            self.use_iom_recondition,
            self.o2o_matching_masklets_enable,
            self.iom_thresh_recondition,
            self.iou_thresh_recondition,
        )
        (
            trk_is_unmatched,
            trk_is_nonempty,
            is_new_det,
            det_to_max_iou_trk_idx,
            det_is_high_conf,
            det_is_high_iou,
            det_keep_out,
            im_mask,
        ) = result
        return LazyAssociateDetTrkResult(
            trk_is_unmatched[:trk_count],
            trk_is_nonempty[:trk_count],
            is_new_det,
            det_to_max_iou_trk_idx,
            det_is_high_conf,
            det_is_high_iou,
            det_keep_out,
            im_mask[:, :trk_count],
        )

    def _det_track_one_frame(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: Any,
        geometric_prompt: Any,
        tracker_states_local: list[Any],
        tracker_metadata_prev: dict[str, Any],
        feature_cache: dict[str, Any],
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
    ) -> Any:
        if self.rank != 0 or self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._det_track_one_frame(distributed)"
            )
        if tracker_states_local:
            if self._can_run_local_detector_update(input_batch):
                return self._run_local_tracker_detector_frame(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    input_batch=input_batch,
                    geometric_prompt=geometric_prompt,
                    tracker_states_local=tracker_states_local,
                    tracker_metadata_prev=tracker_metadata_prev,
                    feature_cache=feature_cache,
                    orig_vid_height=orig_vid_height,
                    orig_vid_width=orig_vid_width,
                    is_image_only=is_image_only,
                )
            return self._run_local_tracker_states_only_frame(
                frame_idx=frame_idx,
                reverse=reverse,
                tracker_states_local=tracker_states_local,
                tracker_metadata_prev=tracker_metadata_prev,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
            )
        if not is_image_only:
            if self._active_tracker_object_count(tracker_metadata_prev) > 0:
                raise_unsupported_multiplex_runtime(
                    "Sam3MultiplexBase._det_track_one_frame(existing-tracklets)"
                )
            if tracker_metadata_prev == {}:
                tracker_metadata_prev.update(self._initialize_metadata())
            return self._run_detector_startup_frame(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                input_batch=input_batch,
                geometric_prompt=geometric_prompt,
                tracker_metadata_prev=tracker_metadata_prev,
                feature_cache=feature_cache,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
            )
        if self._active_tracker_object_count(tracker_metadata_prev) > 0:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexBase._det_track_one_frame(existing-tracklets)"
            )
        if tracker_metadata_prev == {}:
            tracker_metadata_prev.update(self._initialize_metadata())
        return self._run_image_only_detection_frame(
            frame_idx=frame_idx,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            tracker_metadata_prev=tracker_metadata_prev,
            feature_cache=feature_cache,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
        )


class Sam3MultiplexPredictorWrapper(Sam3MultiplexTrackerPredictor):
    """
    Lightweight wrapper for an already-constructed MLX model.

    The official wrapper also enters a Torch autocast context. The MLX shell
    keeps only the delegation behavior that is safe without Torch.
    """

    def __init__(
        self,
        model: Any,
        per_obj_inference: bool = False,
        fill_hole_area: int = 0,
        is_multiplex: bool = True,
        is_multiplex_dynamic: bool = True,
    ):
        nn.Module.__init__(self)
        self.model = model
        self.per_obj_inference = per_obj_inference
        self.fill_hole_area = fill_hole_area
        self.is_multiplex = is_multiplex
        self.is_multiplex_dynamic = is_multiplex_dynamic

    def __getattr__(self, name: str) -> Any:
        try:
            model = self["model"]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if name == "model":
            return model
        return getattr(model, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise_unsupported_multiplex_runtime("Sam3MultiplexPredictorWrapper.forward")

    def add_output_per_object(self, *args: Any, **kwargs: Any) -> Any:
        if self.per_obj_inference:
            return None
        if hasattr(self.model, "_add_output_per_object"):
            return self.model._add_output_per_object(*args, **kwargs)
        raise_unsupported_multiplex_runtime(
            "Sam3MultiplexPredictorWrapper.add_output_per_object"
        )
