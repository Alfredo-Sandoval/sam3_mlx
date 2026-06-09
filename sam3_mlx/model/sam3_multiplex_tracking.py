from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.data_misc import (
    BatchedDatapoint,
    BatchedPointer,
    FindStage,
    interpolate,
)
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.io_utils import load_resource_as_video_frames
from sam3_mlx.model.multiplex_utils import (
    MultiplexState,
    raise_unsupported_multiplex_runtime,
)
from sam3_mlx.model.sam3_multiplex_base import (
    MaskletConfirmationStatus,
    Sam3MultiplexBase,
)
from sam3_mlx.model.sam3_tracker_base import NO_OBJ_SCORE
from sam3_mlx.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3_mlx.model.box_ops import box_xywh_to_cxcywh, box_xyxy_to_xywh
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.perflib.masks_ops import masks_to_boxes


DUMMY_OUTPUT = "DUMMY_OUTPUT"


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


def _mlx_to(data: Any, *args: Any, **kwargs: Any) -> Any:
    device = kwargs.pop("device", None)
    dtype = kwargs.pop("dtype", None)
    if len(args) > 1:
        raise TypeError("recursive_to accepts at most one positional target.")
    if args:
        arg = args[0]
        if arg is None or isinstance(arg, str):
            device = arg
        else:
            dtype = arg
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"Unsupported recursive_to kwargs for MLX array: {names}.")
    if device not in (None, "mlx"):
        raise_unsupported(
            f"sam3_mlx.model.sam3_multiplex_tracking.recursive_to(device={device!r})",
            reason="unsupported-device",
            detail="recursive_to only supports the explicit MLX device.",
            alternative="device='mlx' or device=None",
        )
    return data.astype(dtype) if dtype is not None else data


def _array_to_numpy(value: Any, *, dtype=None) -> np.ndarray:
    return to_numpy(value, dtype=dtype, copy=False)


HOTSTART_GPU_METADATA_KEYS = (
    "obj_first_frame",
    "consecutive_unmatch_count",
    "trk_keep_alive",
    "removed_mask",
    "overlap_pair_counts",
    "last_occluded_tensor",
)


def _copy_first_axis_slice(value: Any, start: int, stop: int) -> Any:
    sliced = value[start:stop]
    if _is_mlx_array(sliced):
        return mx.array(sliced)
    if isinstance(sliced, np.ndarray):
        return sliced.copy()
    clone = getattr(sliced, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(sliced, "copy", None)
    return copy() if callable(copy) else sliced


def _demux_and_slice_first_axis(value: Any, multiplex_state: Any, obj_idx: int) -> Any:
    if value is None:
        return None
    if multiplex_state is not None:
        try:
            value = multiplex_state.demux(value)
        except (AssertionError, IndexError, ValueError):
            return None
    return _copy_first_axis_slice(value, obj_idx, obj_idx + 1)


def _score_to_float(value: Any) -> float:
    array = _array_to_numpy(value, dtype=np.float32)
    return float(array.reshape(()))


def _mask_to_mlx_bool(mask: Any) -> Any:
    mask = mask if _is_mlx_array(mask) else mx.array(mask)
    if mask.dtype != mx.bool_:
        raise TypeError("Sam3MultiplexTracking postprocess expects boolean masks.")
    if mask.ndim == 2:
        mask = mask[None, ...]
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(
            "Each obj_id_to_mask entry must have shape (1, H, W) or (H, W), "
            f"got {mask.shape}."
        )
    return mask


def _empty_postprocessed_output(
    height: int,
    width: int,
    frame_stats: Any,
    *,
    include_prod_outputs: bool,
) -> dict[str, Any]:
    output = {
        "out_obj_ids": np.zeros(0, dtype=np.int64),
        "out_probs": np.zeros(0, dtype=np.float32),
        "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
        "out_binary_masks": np.zeros((0, height, width), dtype=bool),
        "frame_stats": frame_stats,
    }
    if include_prod_outputs:
        output["out_centers"] = np.zeros((0, 2), dtype=np.float32)
    return output


def _hidden_obj_ids(
    removed_obj_ids: Any,
    suppressed_obj_ids: Any,
    unconfirmed_obj_ids: Any,
) -> set[Any]:
    hidden = set()
    if suppressed_obj_ids is not None:
        hidden.update(suppressed_obj_ids)
    if removed_obj_ids is not None:
        hidden.update(removed_obj_ids)
    if unconfirmed_obj_ids is not None:
        hidden.update(unconfirmed_obj_ids)
    return hidden


def _mask_centers(binary_masks: Any) -> np.ndarray:
    binary_masks = (
        binary_masks if _is_mlx_array(binary_masks) else mx.array(binary_masks)
    )
    if binary_masks.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    height, width = binary_masks.shape[-2:]
    masks_f32 = binary_masks.astype(mx.float32)
    y = mx.arange(height, dtype=mx.float32).reshape(1, height, 1)
    x = mx.arange(width, dtype=mx.float32).reshape(1, 1, width)
    mass = mx.maximum(mx.sum(masks_f32, axis=(1, 2)), mx.array(1e-6, dtype=mx.float32))
    center_x = mx.sum(masks_f32 * x, axis=(1, 2)) / mass / width
    center_y = mx.sum(masks_f32 * y, axis=(1, 2)) / mass / height
    return _array_to_numpy(mx.stack([center_x, center_y], axis=1), dtype=np.float32)


def _point_count(points: Any) -> int:
    if _is_mlx_array(points):
        shape = tuple(points.shape)
        if not shape:
            return 0
        if len(shape) >= 2 and shape[-1] in (2, 3):
            return int(shape[-2])
        return int(shape[0])
    return len(points)


def recursive_to(data: Any, *args: Any, **kwargs: Any) -> Any:
    if _is_mlx_array(data):
        return _mlx_to(data, *args, **kwargs)
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, Mapping):
        ret = type(data)()
        for key, value in data.items():
            ret[key] = recursive_to(value, *args, **kwargs)
        return ret
    if isinstance(data, tuple):
        return tuple(recursive_to(value, *args, **kwargs) for value in data)
    if isinstance(data, Sequence) and not isinstance(data, str):
        return type(data)(recursive_to(value, *args, **kwargs) for value in data)
    if is_dataclass(data):
        ret_cls = type(data)
        ret_fields = {
            field.name: recursive_to(getattr(data, field.name), *args, **kwargs)
            for field in fields(data)
        }
        return ret_cls(**ret_fields)
    return data


class Sam3MultiplexTracking(Sam3MultiplexBase):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        image_size: int = 1008,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        compile_model: bool = False,
        postprocess_batch_size: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        if postprocess_batch_size < 1:
            raise ValueError("postprocess_batch_size must be >= 1.")
        self.compile_model = compile_model
        if hasattr(self.detector, "compile_model"):
            self.detector.compile_model = compile_model
        self.postprocess_batch_size = postprocess_batch_size
        self._compiled_for_propagation = False

    def _construct_initial_input_batch(self, inference_state: dict[str, Any], images):
        num_frames = len(images)
        image_tensors = getattr(images, "images", None)
        if image_tensors is None:
            raise ValueError("Loaded video frames must expose normalized MLX images.")

        find_text_batch = ["<text placeholder>", "visual", "geometric"]
        dummy_ptrs = BatchedPointer(
            stage_ids=mx.array([], dtype=mx.int64),
            query_ids=mx.array([], dtype=mx.int64),
            object_ids=mx.array([], dtype=mx.int64),
            ptr_mask=mx.array([], dtype=mx.bool_),
            ptr_types=mx.array([], dtype=mx.int64),
        )
        stages = [
            FindStage(
                img_ids=mx.array([stage_id], dtype=mx.int64),
                img_ids_np=np.array([stage_id], dtype=np.int64),
                text_ids=mx.array([self.TEXT_ID_FOR_TEXT], dtype=mx.int64),
                input_boxes=mx.zeros((258, 1), dtype=mx.float32),
                input_boxes_before_embed=mx.zeros((0, 1, 4), dtype=mx.float32),
                input_boxes_mask=mx.zeros((1, 0), dtype=mx.bool_),
                input_boxes_label=mx.zeros((0, 1), dtype=mx.int64),
                input_points=mx.zeros((1, 0, 257), dtype=mx.float32),
                input_points_before_embed=mx.zeros((1, 0, 3), dtype=mx.float32),
                input_points_mask=mx.zeros((1, 0), dtype=mx.bool_),
                ptrs=dummy_ptrs,
                ptrs_seg=dummy_ptrs,
                object_ids=[],
            )
            for stage_id in range(num_frames)
        ]

        input_batch = BatchedDatapoint(
            img_batch=image_tensors,
            find_text_batch=find_text_batch,
            find_inputs=stages,
            find_targets=[None] * num_frames,
            find_metadatas=[None] * num_frames,
            get_queries=None,
        )
        inference_state["input_batch"] = input_batch
        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=mx.zeros((0, 1, 4), dtype=mx.float32),
            box_mask=mx.zeros((1, 0), dtype=mx.bool_),
            box_labels=mx.zeros((0, 1), dtype=mx.int64),
            point_embeddings=mx.zeros((0, 1, 2), dtype=mx.float32),
            point_mask=mx.zeros((1, 0), dtype=mx.bool_),
            point_labels=mx.zeros((0, 1), dtype=mx.int64),
        )
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["text_prompt"] = None
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    @staticmethod
    def _clear_find_input_geometry(find_input: FindStage) -> None:
        find_input.input_boxes_before_embed = mx.zeros((0, 1, 4), dtype=mx.float32)
        find_input.input_boxes_mask = mx.zeros((1, 0), dtype=mx.bool_)
        find_input.input_boxes_label = mx.zeros((0, 1), dtype=mx.int64)
        find_input.input_points_before_embed = mx.zeros((1, 0, 3), dtype=mx.float32)
        find_input.input_points_mask = mx.zeros((1, 0), dtype=mx.bool_)

    @staticmethod
    def _set_find_input_box_geometry(
        find_input: FindStage,
        boxes_cxcywh: Any,
        box_labels: Any,
    ) -> None:
        num_boxes = int(boxes_cxcywh.shape[0])
        find_input.input_boxes_before_embed = boxes_cxcywh.reshape(num_boxes, 1, 4)
        find_input.input_boxes_mask = mx.zeros((1, num_boxes), dtype=mx.bool_)
        find_input.input_boxes_label = box_labels.reshape(num_boxes, 1)

    def _get_visual_prompt(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        boxes_cxcywh: Any,
        box_labels: Any,
    ):
        del frame_idx
        boxes_cxcywh = mx.array(boxes_cxcywh, dtype=mx.float32)
        box_labels = mx.array(box_labels, dtype=mx.int64)
        batch_size = 1
        geometric_prompt = Prompt(
            box_embeddings=mx.zeros((0, batch_size, 4), dtype=mx.float32),
            box_mask=mx.zeros((batch_size, 0), dtype=mx.bool_),
            point_embeddings=None,
            point_mask=None,
        )
        geometric_prompt.append_boxes(
            boxes=boxes_cxcywh.reshape(-1, batch_size, 4),
            labels=box_labels.reshape(-1, batch_size),
        )
        return boxes_cxcywh, box_labels, geometric_prompt

    def _cache_frame_outputs(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_id_to_mask: dict[Any, Any],
        suppressed_obj_ids=None,
        removed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ) -> None:
        if "cached_frame_outputs" not in inference_state:
            inference_state["cached_frame_outputs"] = {}
        filtered_obj_id_to_mask = obj_id_to_mask.copy()
        objects_to_exclude = set()
        if suppressed_obj_ids is not None:
            objects_to_exclude.update(suppressed_obj_ids)
        if removed_obj_ids is not None:
            objects_to_exclude.update(removed_obj_ids)
        if unconfirmed_obj_ids is not None:
            objects_to_exclude.update(unconfirmed_obj_ids)
        for obj_id in objects_to_exclude:
            filtered_obj_id_to_mask.pop(obj_id, None)
        inference_state["cached_frame_outputs"][frame_idx] = filtered_obj_id_to_mask

    def _build_sam2_output(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        refined_obj_id_to_mask: dict[Any, Any] | None = None,
    ) -> dict[Any, Any]:
        cached_frame_outputs = inference_state.get("cached_frame_outputs", {})
        if frame_idx not in cached_frame_outputs:
            return {}

        obj_id_to_mask = cached_frame_outputs[frame_idx].copy()
        if refined_obj_id_to_mask is not None:
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                if refined_mask is None:
                    raise ValueError(
                        f"Refined mask data must be provided for obj_id {obj_id}."
                    )
                obj_id_to_mask[obj_id] = refined_mask
        return obj_id_to_mask

    def _postprocess_output(
        self,
        inference_state: dict[str, Any],
        out: dict[str, Any],
        removed_obj_ids=None,
        suppressed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ) -> dict[str, Any]:
        obj_id_to_mask = out["obj_id_to_mask"]
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        height = int(inference_state["orig_height"])
        width = int(inference_state["orig_width"])
        frame_stats = out.get("frame_stats", None)
        if len(curr_obj_ids) == 0:
            return _empty_postprocessed_output(
                height,
                width,
                frame_stats,
                include_prod_outputs=self.running_in_prod,
            )

        masks = mx.concat(
            [_mask_to_mlx_bool(obj_id_to_mask[obj_id]) for obj_id in curr_obj_ids],
            axis=0,
        )
        hidden = _hidden_obj_ids(
            removed_obj_ids,
            suppressed_obj_ids,
            unconfirmed_obj_ids,
        )
        # Python-metadata boundary: object ids, hidden ids, and public output
        # ordering are Python-owned, so materialize keep indices once here.
        has_area = _array_to_numpy(mx.any(masks, axis=(1, 2)), dtype=bool)
        keep = np.array(
            [
                bool(has_area[index]) and obj_id not in hidden
                for index, obj_id in enumerate(curr_obj_ids)
            ],
            dtype=bool,
        )
        if not keep.any():
            return _empty_postprocessed_output(
                height,
                width,
                frame_stats,
                include_prod_outputs=self.running_in_prod,
            )

        keep_indices = np.flatnonzero(keep).astype(np.int64)
        kept_obj_ids = [curr_obj_ids[index] for index in keep_indices.tolist()]
        kept_masks = mx.take(masks, mx.array(keep_indices, dtype=mx.int64), axis=0)

        kept_probs = np.array(
            [
                _score_to_float(out["obj_id_to_score"][obj_id])
                for obj_id in kept_obj_ids
            ],
            dtype=np.float32,
        )
        sam2_score_by_obj_id = out.get("obj_id_to_sam2_score", {})
        kept_sam2_probs = mx.array(
            [
                _score_to_float(sam2_score_by_obj_id.get(obj_id, 0.0))
                for obj_id in kept_obj_ids
            ],
            dtype=mx.float32,
        )

        out_boxes_xyxy = masks_to_boxes(kept_masks, kept_obj_ids)
        out_boxes_xywh = box_xyxy_to_xywh(out_boxes_xyxy) / mx.array(
            [width, height, width, height],
            dtype=mx.float32,
        )

        if kept_masks.shape[0] > 1 and hasattr(
            self.tracker,
            "_apply_object_wise_non_overlapping_constraints",
        ):
            constrained = self.tracker._apply_object_wise_non_overlapping_constraints(
                kept_masks[:, None, :, :],
                kept_sam2_probs[:, None],
                background_value=0,
            )
            kept_masks = (constrained[:, 0, :, :] > 0).astype(mx.bool_)

        out_centers = _mask_centers(kept_masks) if self.running_in_prod else None
        out_binary_masks = _array_to_numpy(kept_masks, dtype=bool)
        outputs = {
            "out_obj_ids": np.array(kept_obj_ids, dtype=np.int64),
            "out_probs": kept_probs,
            "out_boxes_xywh": _array_to_numpy(out_boxes_xywh, dtype=np.float32),
            "out_binary_masks": out_binary_masks,
            "frame_stats": frame_stats,
        }
        if self.running_in_prod:
            outputs["out_centers"] = out_centers
        return outputs

    def _postprocess_output_batched(
        self,
        height: int,
        width: int,
        batched_outs,
    ) -> list[dict[str, Any]]:
        inference_state = {
            "orig_height": height,
            "orig_width": width,
        }
        return [
            self._postprocess_output(
                inference_state,
                out,
                removed_obj_ids=removed_obj_ids,
                suppressed_obj_ids=suppressed_obj_ids,
                unconfirmed_obj_ids=unconfirmed_obj_ids,
            )
            for out, removed_obj_ids, suppressed_obj_ids, unconfirmed_obj_ids in batched_outs
        ]

    def _compile_model(self) -> None:
        """Mark the model as compiled for propagation; ``mx.compile`` is not used in the MLX runtime."""
        self._compiled_for_propagation = True

    def _new_generator_state(self) -> dict[str, Any]:
        return {
            "hotstart_buffer": [],
            "hotstart_removed_obj_ids": set(),
            "unconfirmed_obj_ids_per_frame": {},
            "postprocess_yield_list": [],
        }

    def _unconfirmed_ids_for_yield_frame(
        self,
        *,
        yield_frame_idx: int,
        num_frames: int,
        reverse: bool,
        unconfirmed_status_delay: int,
        unconfirmed_obj_ids_per_frame: dict[int, Any],
    ) -> Any:
        unconfirmed_status_frame_idx = (
            yield_frame_idx + unconfirmed_status_delay
            if not reverse
            else yield_frame_idx - unconfirmed_status_delay
        )
        unconfirmed_status_frame_idx = max(
            0,
            min(unconfirmed_status_frame_idx, num_frames - 1),
        )
        return unconfirmed_obj_ids_per_frame.get(unconfirmed_status_frame_idx, None)

    def _postprocess_propagation_batch(
        self,
        inference_state: dict[str, Any],
        batch_to_process,
        *,
        reverse: bool,
        unconfirmed_status_delay: int,
        unconfirmed_obj_ids_per_frame: dict[int, Any],
    ):
        if self.rank != 0:
            return [
                (yield_frame_idx, DUMMY_OUTPUT)
                for yield_frame_idx, _, _ in batch_to_process
            ]

        height = int(inference_state["orig_height"])
        width = int(inference_state["orig_width"])
        num_frames = int(inference_state["num_frames"])
        batched_outs = []
        frame_indices = []
        for yield_frame_idx, yield_out, removed_obj_ids_snapshot in batch_to_process:
            suppressed_obj_ids = yield_out.get("suppressed_obj_ids", None)
            unconfirmed_obj_ids = self._unconfirmed_ids_for_yield_frame(
                yield_frame_idx=yield_frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                unconfirmed_status_delay=unconfirmed_status_delay,
                unconfirmed_obj_ids_per_frame=unconfirmed_obj_ids_per_frame,
            )
            batched_outs.append(
                (
                    yield_out,
                    removed_obj_ids_snapshot,
                    suppressed_obj_ids,
                    unconfirmed_obj_ids,
                )
            )
            frame_indices.append(yield_frame_idx)
            self._cache_frame_outputs(
                inference_state,
                yield_frame_idx,
                yield_out["obj_id_to_mask"],
                suppressed_obj_ids=suppressed_obj_ids,
                removed_obj_ids=removed_obj_ids_snapshot,
                unconfirmed_obj_ids=unconfirmed_obj_ids,
            )

        if self.postprocess_batch_size > 1:
            postprocessed_outs = self._postprocess_output_batched(
                height,
                width,
                batched_outs,
            )
        else:
            postprocessed_outs = [
                self._postprocess_output(
                    inference_state,
                    yield_out,
                    removed_obj_ids_snapshot,
                    suppressed_obj_ids,
                    unconfirmed_obj_ids,
                )
                for (
                    yield_out,
                    removed_obj_ids_snapshot,
                    suppressed_obj_ids,
                    unconfirmed_obj_ids,
                ) in batched_outs
            ]
        return list(zip(frame_indices, postprocessed_outs))

    def _record_bucket_utilization(self, inference_state: dict[str, Any]) -> None:
        if not self.is_multiplex:
            return
        total_valid_objects = 0
        total_num_buckets = 0
        for state in inference_state.get("sam2_inference_states", []):
            obj_ids = state.get("obj_ids", []) if isinstance(state, Mapping) else []
            multiplex_state = (
                state.get("multiplex_state") if isinstance(state, Mapping) else None
            )
            if multiplex_state is None:
                continue
            total_valid_entries = getattr(multiplex_state, "total_valid_entries", None)
            if total_valid_entries is not None:
                assert len(obj_ids) == total_valid_entries
            total_valid_objects += len(obj_ids)
            total_num_buckets += int(getattr(multiplex_state, "num_buckets", 0))
        if total_num_buckets == 0:
            return
        inference_state["bucket_utilization_stats"] = {
            "total_valid_objects": total_valid_objects,
            "total_num_buckets": total_num_buckets,
            "bucket_utilization_rate": (
                total_valid_objects / (total_num_buckets * self.bucket_capacity)
            )
            * 100,
            "subscription_rate": (total_valid_objects / total_num_buckets) * 100,
        }

    def _run_single_frame_inference(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        reverse: bool,
        is_instance_processing: bool = False,
    ) -> dict[str, Any]:
        del is_instance_processing
        input_batch = inference_state["input_batch"]
        tracker_states_local = inference_state["sam2_inference_states"]
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            feature_cache=inference_state["feature_cache"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            is_image_only=inference_state["is_image_only"],
        )
        inference_state["sam2_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        if self.rank == 0:
            self._cache_frame_outputs(inference_state, frame_idx, obj_id_to_mask)

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,
            "obj_id_to_sam2_score": tracker_metadata_new[
                "obj_id_to_sam2_score_frame_wise"
            ][frame_idx],
        }
        if self.rank == 0:
            rank0_metadata = tracker_metadata_new["rank0_metadata"]
            out["removed_obj_ids"] = rank0_metadata["removed_obj_ids"]
            out["suppressed_obj_ids"] = rank0_metadata["suppressed_obj_ids"][frame_idx]
            out["frame_stats"] = frame_stats
            if self.masklet_confirmation_enable:
                status = rank0_metadata["masklet_confirmation"]["status"]
                is_unconfirmed = status == MaskletConfirmationStatus.UNCONFIRMED.value
                out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids_all_gpu"][
                    is_unconfirmed
                ].tolist()
            else:
                out["unconfirmed_obj_ids"] = []
        return out

    def _propagate_in_video_impl(
        self,
        inference_state: dict[str, Any],
        *,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        is_instance_processing: bool = False,
        generator_state: dict[str, Any] | None = None,
        flush_hotstart_at_end: bool = True,
        persist_generator_state: bool = False,
    ):
        self._compile_model()
        processing_order, end_frame_idx = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse=reverse,
        )
        feature_cache = inference_state.setdefault("feature_cache", {})
        feature_cache["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }

        if generator_state is None:
            generator_state = self._new_generator_state()
        hotstart_buffer = generator_state["hotstart_buffer"]
        hotstart_removed_obj_ids = generator_state["hotstart_removed_obj_ids"]
        unconfirmed_obj_ids_per_frame = generator_state["unconfirmed_obj_ids_per_frame"]
        postprocess_yield_list = generator_state.get("postprocess_yield_list", [])
        unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1

        for frame_idx in processing_order:
            out = self._run_single_frame_inference(
                inference_state,
                frame_idx,
                reverse,
                is_instance_processing=is_instance_processing,
            )
            if self.rank == 0:
                unconfirmed_obj_ids = out.get("unconfirmed_obj_ids", None)
                if unconfirmed_obj_ids is not None:
                    unconfirmed_obj_ids_per_frame[frame_idx] = unconfirmed_obj_ids

            if self.hotstart_delay > 0:
                hotstart_buffer.append((frame_idx, out))
                if self.rank == 0:
                    hotstart_removed_obj_ids.update(out.get("removed_obj_ids", set()))

                if frame_idx == end_frame_idx and flush_hotstart_at_end:
                    yield_list = hotstart_buffer
                    hotstart_buffer = []
                elif len(hotstart_buffer) >= self.hotstart_delay:
                    yield_list = hotstart_buffer[:1]
                    hotstart_buffer = hotstart_buffer[1:]
                else:
                    yield_list = []
            else:
                yield_list = [(frame_idx, out)]

            for yield_frame_idx, yield_out in yield_list:
                postprocess_yield_list.append(
                    (yield_frame_idx, yield_out, set(hotstart_removed_obj_ids))
                )

            while len(postprocess_yield_list) >= self.postprocess_batch_size:
                batch_to_process = postprocess_yield_list[: self.postprocess_batch_size]
                postprocess_yield_list = postprocess_yield_list[
                    self.postprocess_batch_size :
                ]
                yield from self._postprocess_propagation_batch(
                    inference_state,
                    batch_to_process,
                    reverse=reverse,
                    unconfirmed_status_delay=unconfirmed_status_delay,
                    unconfirmed_obj_ids_per_frame=unconfirmed_obj_ids_per_frame,
                )

        if flush_hotstart_at_end and hotstart_buffer:
            for yield_frame_idx, yield_out in hotstart_buffer:
                postprocess_yield_list.append(
                    (yield_frame_idx, yield_out, set(hotstart_removed_obj_ids))
                )
            hotstart_buffer = []

        if postprocess_yield_list:
            yield from self._postprocess_propagation_batch(
                inference_state,
                postprocess_yield_list,
                reverse=reverse,
                unconfirmed_status_delay=unconfirmed_status_delay,
                unconfirmed_obj_ids_per_frame=unconfirmed_obj_ids_per_frame,
            )
            postprocess_yield_list = []

        if persist_generator_state:
            generator_state["hotstart_buffer"] = hotstart_buffer
            generator_state["hotstart_removed_obj_ids"] = hotstart_removed_obj_ids
            generator_state["unconfirmed_obj_ids_per_frame"] = (
                unconfirmed_obj_ids_per_frame
            )
            generator_state["postprocess_yield_list"] = postprocess_yield_list

        self._record_bucket_utilization(inference_state)

    def init_state(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            if "resource_path" in kwargs:
                raise TypeError(
                    "resource_path passed both positionally and by keyword."
                )
            kwargs["resource_path"] = args[0]
            if len(args) > 1:
                raise TypeError("init_state accepts at most one positional argument.")
        resource_path = kwargs.pop("resource_path")
        offload_video_to_cpu = kwargs.pop("offload_video_to_cpu", False)
        offload_state_to_cpu = kwargs.pop("offload_state_to_cpu", False)
        async_loading_frames = kwargs.pop("async_loading_frames", False)
        use_torchcodec = kwargs.pop("use_torchcodec", False)
        use_cv2 = kwargs.pop("use_cv2", False)
        input_is_mp4 = kwargs.pop("input_is_mp4", False)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected init_state keyword argument(s): {names}")
        del input_is_mp4
        if offload_state_to_cpu:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.init_state(offload_state_to_cpu=True)"
            )
        if use_torchcodec:
            video_loader_type = "torchcodec"
        elif use_cv2:
            video_loader_type = "cv2"
        else:
            video_loader_type = "cv2"
        images = load_resource_as_video_frames(
            resource_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=self.image_mean,
            img_std=self.image_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
        inference_state: dict[str, Any] = {
            "image_size": self.image_size,
            "num_frames": len(images),
            "device": "mlx",
            "orig_height": images.orig_height,
            "orig_width": images.orig_width,
            "constants": {},
        }
        self._construct_initial_input_batch(inference_state, images)
        inference_state["sam2_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["is_image_only"] = len(images) == 1
        return inference_state

    def reset_state(self, inference_state: Any) -> Any:
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for frame_idx in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[frame_idx].text_ids = mx.array(
                [self.TEXT_ID_FOR_TEXT],
                dtype=mx.int64,
            )
            self._clear_find_input_geometry(
                inference_state["input_batch"].find_inputs[frame_idx]
            )
            inference_state["previous_stages_out"][frame_idx] = None
            inference_state["per_frame_raw_point_input"][frame_idx] = None
            inference_state["per_frame_raw_box_input"][frame_idx] = None
            inference_state["per_frame_visual_prompt"][frame_idx] = None
            inference_state["per_frame_geometric_prompt"][frame_idx] = None
            inference_state["per_frame_cur_step"][frame_idx] = 0
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        inference_state["sam2_inference_states"].clear()
        inference_state["tracker_metadata"].clear()
        inference_state["feature_cache"].clear()
        inference_state["cached_frame_outputs"] = {}
        return None

    def add_fake_objects_to_inference_state(
        self,
        inference_state: dict[str, Any],
        num_objects: int,
        frame_idx: int,
    ) -> dict[str, Any]:
        """Synthesize warm-up objects by calling the tracker-add path directly."""
        if self.rank != 0 or self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.add_fake_objects_to_inference_state(distributed)"
            )
        num_objects = int(num_objects)
        if num_objects < 0:
            raise ValueError("num_objects must be non-negative.")
        frame_idx = int(frame_idx)
        num_frames = int(inference_state["num_frames"])
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )

        mask_downsampler = getattr(
            getattr(self.tracker, "maskmem_backbone", None),
            "mask_downsampler",
            None,
        )
        interpol_size = getattr(mask_downsampler, "interpol_size", None)
        input_mask_size = getattr(self.tracker, "input_mask_size", None)
        if interpol_size is not None:
            high_res_h, high_res_w = (int(interpol_size[0]), int(interpol_size[1]))
        elif input_mask_size is not None:
            high_res_h = high_res_w = int(input_mask_size)
        else:
            high_res_h = int(inference_state["orig_height"])
            high_res_w = int(inference_state["orig_width"])

        new_obj_ids = np.arange(num_objects, dtype=np.int64)
        new_obj_masks = mx.ones(
            (num_objects, high_res_h, high_res_w),
            dtype=mx.float32,
        )
        inference_state["sam2_inference_states"] = self._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=num_frames,
            new_obj_ids=new_obj_ids,
            new_obj_masks=new_obj_masks,
            tracker_states_local=inference_state.setdefault(
                "sam2_inference_states", []
            ),
            orig_vid_height=int(inference_state["orig_height"]),
            orig_vid_width=int(inference_state["orig_width"]),
            feature_cache=inference_state.setdefault("feature_cache", {}),
        )

        obj_id_to_mask: dict[int, Any] = {}
        if num_objects > 0:
            video_res_masks = (
                interpolate(
                    new_obj_masks[:, None, :, :],
                    size=(
                        int(inference_state["orig_height"]),
                        int(inference_state["orig_width"]),
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                > 0
            )
            for obj_idx, obj_id in enumerate(new_obj_ids.tolist()):
                obj_id_to_mask[int(obj_id)] = video_res_masks[obj_idx]
        for cache_frame_idx in range(num_frames):
            self._cache_frame_outputs(
                inference_state,
                cache_frame_idx,
                obj_id_to_mask,
            )

        tracker_metadata = self._initialize_metadata()
        tracker_metadata["obj_ids_per_gpu"][0] = new_obj_ids.copy()
        tracker_metadata["obj_ids_all_gpu"] = new_obj_ids.copy()
        tracker_metadata["num_obj_per_gpu"][0] = num_objects
        tracker_metadata["max_obj_id"] = num_objects
        tracker_metadata["obj_id_to_score"] = {
            int(obj_id): 1.0 for obj_id in new_obj_ids.tolist()
        }
        rank0_metadata = tracker_metadata["rank0_metadata"]
        rank0_metadata["obj_first_frame_idx"] = {
            int(obj_id): frame_idx for obj_id in new_obj_ids.tolist()
        }
        rank0_metadata["trk_keep_alive"] = defaultdict(
            int,
            {
                int(obj_id): int(self.init_trk_keep_alive)
                for obj_id in new_obj_ids.tolist()
            },
        )
        tracker_metadata["obj_id_to_last_occluded"] = {
            int(obj_id): -1 for obj_id in new_obj_ids.tolist()
        }
        if self.masklet_confirmation_enable:
            rank0_metadata["masklet_confirmation"] = {
                "status": np.zeros(num_objects, dtype=np.int64),
                "consecutive_det_num": np.zeros(num_objects, dtype=np.int64),
            }
        tracker_metadata["gpu_metadata"] = self._hotstart_gpu_metadata_from_rank0(
            tracker_metadata,
            frame_idx=frame_idx,
            num_objects=num_objects,
        )
        if self.is_multiplex:
            tracker_metadata["num_buc_per_gpu"][0] = self._count_buckets_in_states(
                inference_state["sam2_inference_states"]
            )
        inference_state["tracker_metadata"] = tracker_metadata
        return inference_state

    def _hotstart_gpu_metadata_has_tensors(
        self, gpu_metadata: Mapping[str, Any]
    ) -> bool:
        return all(key in gpu_metadata for key in HOTSTART_GPU_METADATA_KEYS)

    def _rebuild_hotstart_gpu_metadata_from_rank0(
        self,
        tracker_metadata: dict[str, Any],
        *,
        frame_idx: int | None,
    ) -> dict[str, Any]:
        num_objects = int(
            np.asarray(
                tracker_metadata.get("obj_ids_all_gpu", []),
                dtype=np.int64,
            ).size
        )
        if num_objects == 0:
            return self._empty_hotstart_gpu_metadata()
        return self._hotstart_gpu_metadata_from_rank0(
            tracker_metadata,
            frame_idx=0 if frame_idx is None else int(frame_idx),
            num_objects=num_objects,
        )

    def _sync_hotstart_gpu_metadata_after_removal(
        self,
        tracker_metadata: dict[str, Any],
        *,
        previous_obj_ids: Any,
        keep_mask: Any,
        frame_idx: int | None,
    ) -> None:
        if not self.is_multiplex or "gpu_metadata" not in tracker_metadata:
            return
        previous_obj_ids_np = np.asarray(previous_obj_ids, dtype=np.int64).reshape(-1)
        keep_mask_np = np.asarray(keep_mask, dtype=bool).reshape(-1)
        if keep_mask_np.shape != previous_obj_ids_np.shape:
            raise ValueError(
                "hotstart gpu metadata removal mask must match previous object ids; "
                f"got {keep_mask_np.shape} and {previous_obj_ids_np.shape}."
            )
        gpu_metadata_prev = tracker_metadata.get("gpu_metadata", {})
        if self._hotstart_gpu_metadata_has_tensors(gpu_metadata_prev):
            previous_tracker_metadata = dict(tracker_metadata)
            previous_tracker_metadata["obj_ids_all_gpu"] = previous_obj_ids_np
            gpu_metadata = self._ensure_hotstart_gpu_metadata(
                previous_tracker_metadata,
                gpu_metadata_prev,
                frame_idx=0 if frame_idx is None else int(frame_idx),
                num_objects=int(previous_obj_ids_np.size),
            )
            gpu_metadata["removed_mask"] = gpu_metadata["removed_mask"] | mx.array(
                ~keep_mask_np, dtype=mx.bool_
            )
            tracker_metadata["gpu_metadata"] = self._compact_hotstart_gpu_metadata(
                gpu_metadata
            )
        else:
            tracker_metadata["gpu_metadata"] = (
                self._rebuild_hotstart_gpu_metadata_from_rank0(
                    tracker_metadata,
                    frame_idx=frame_idx,
                )
            )

    def _sync_hotstart_gpu_metadata_after_addition(
        self,
        tracker_metadata: dict[str, Any],
        *,
        frame_idx: int,
        num_new_objects: int,
    ) -> None:
        if (
            not self.is_multiplex
            or num_new_objects <= 0
            or "gpu_metadata" not in tracker_metadata
        ):
            return
        obj_ids_all = np.asarray(
            tracker_metadata.get("obj_ids_all_gpu", []),
            dtype=np.int64,
        ).reshape(-1)
        current_num = int(obj_ids_all.size)
        previous_num = current_num - int(num_new_objects)
        if previous_num < 0:
            raise ValueError(
                "hotstart gpu metadata addition count exceeds current object ids; "
                f"got {num_new_objects} new objects for {current_num} ids."
            )
        gpu_metadata_prev = tracker_metadata.get("gpu_metadata", {})
        if self._hotstart_gpu_metadata_has_tensors(gpu_metadata_prev):
            previous_tracker_metadata = dict(tracker_metadata)
            previous_tracker_metadata["obj_ids_all_gpu"] = obj_ids_all[:previous_num]
            gpu_metadata = self._ensure_hotstart_gpu_metadata(
                previous_tracker_metadata,
                gpu_metadata_prev,
                frame_idx=int(frame_idx),
                num_objects=previous_num,
            )
            tracker_metadata["gpu_metadata"] = (
                self._extend_hotstart_gpu_metadata_for_new_objects(
                    gpu_metadata,
                    frame_idx=int(frame_idx),
                    num_new_objects=int(num_new_objects),
                )
            )
        else:
            previous_tracker_metadata = dict(tracker_metadata)
            previous_tracker_metadata["obj_ids_all_gpu"] = obj_ids_all[:previous_num]
            gpu_metadata = (
                self._empty_hotstart_gpu_metadata()
                if previous_num == 0
                else self._hotstart_gpu_metadata_from_rank0(
                    previous_tracker_metadata,
                    frame_idx=int(frame_idx),
                    num_objects=previous_num,
                )
            )
            tracker_metadata["gpu_metadata"] = (
                self._extend_hotstart_gpu_metadata_for_new_objects(
                    gpu_metadata,
                    frame_idx=int(frame_idx),
                    num_new_objects=int(num_new_objects),
                )
            )

    def _fetch_removed_object_frame(
        self,
        inference_state: dict[str, Any],
        frame_idx: int | None,
    ) -> tuple[int | None, dict[str, Any] | None]:
        if frame_idx is None:
            return frame_idx, None
        if self.rank != 0:
            return frame_idx, DUMMY_OUTPUT
        if frame_idx in inference_state.get("cached_frame_outputs", {}):
            cached_frame_outputs = inference_state.get("cached_frame_outputs", {})
            tracker_metadata = inference_state.get("tracker_metadata", {})
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})
            suppressed_by_frame = rank0_metadata.get("suppressed_obj_ids", {})
            obj_id_to_sam2_score = tracker_metadata.get(
                "obj_id_to_sam2_score_frame_wise",
                {},
            ).get(frame_idx, {})
            obj_id_to_score = tracker_metadata.get("obj_id_to_score", {})
            frame_scores = {
                obj_id: obj_id_to_sam2_score.get(
                    obj_id,
                    obj_id_to_score.get(obj_id, 0.0),
                )
                for obj_id in cached_frame_outputs[frame_idx]
            }
            return (
                frame_idx,
                self._postprocess_output(
                    inference_state,
                    {
                        "obj_id_to_mask": cached_frame_outputs[frame_idx],
                        "obj_id_to_score": {
                            obj_id: obj_id_to_score.get(obj_id, frame_scores[obj_id])
                            for obj_id in cached_frame_outputs[frame_idx]
                        },
                        "obj_id_to_sam2_score": frame_scores,
                    },
                    suppressed_obj_ids=suppressed_by_frame.get(frame_idx, set()),
                ),
            )
        return (
            frame_idx,
            _empty_postprocessed_output(
                int(inference_state["orig_height"]),
                int(inference_state["orig_width"]),
                frame_stats=None,
                include_prod_outputs=self.running_in_prod,
            ),
        )

    def _validate_removable_sam2_states(
        self,
        inference_state: dict[str, Any],
    ) -> None:
        for sam2_state in inference_state.get("sam2_inference_states", []):
            if not isinstance(sam2_state, Mapping):
                raise_unsupported_multiplex_runtime(
                    "Sam3MultiplexTracking.remove_object(existing-tracker-states)"
                )
            multiplex_state = sam2_state.get("multiplex_state")
            has_packed_ids = (
                multiplex_state is not None
                and getattr(multiplex_state, "object_ids", None) is not None
            )
            if "obj_ids" not in sam2_state and not has_packed_ids:
                raise_unsupported_multiplex_runtime(
                    "Sam3MultiplexTracking.remove_object(existing-tracker-states)"
                )

    def _get_gpu_id_by_obj_id(
        self,
        inference_state: dict[str, Any],
        obj_id: Any,
    ) -> int | None:
        obj_id_int = int(obj_id)
        obj_ids_per_gpu = inference_state.get("tracker_metadata", {}).get(
            "obj_ids_per_gpu",
            [],
        )
        for rank, obj_ids in enumerate(obj_ids_per_gpu):
            if obj_id_int in np.asarray(obj_ids, dtype=np.int64).reshape(-1).tolist():
                return rank
        return None

    def _remove_object_from_sam2_states(
        self,
        inference_state: dict[str, Any],
        obj_id: Any,
        frame_idx: int | None,
        strict: bool,
    ) -> tuple[int | None, dict[str, Any] | None]:
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.remove_object(existing-tracker-states-distributed)"
            )
        self._validate_removable_sam2_states(inference_state)
        if frame_idx is not None:
            frame_idx = int(frame_idx)
            num_frames = int(inference_state["num_frames"])
            if not 0 <= frame_idx < num_frames:
                raise ValueError(
                    f"frame_idx={frame_idx} is out of range for {num_frames} frames."
                )
        try:
            obj_id_int = int(obj_id)
        except (TypeError, ValueError) as exc:
            raise TypeError("obj_id must be integer-compatible.") from exc

        tracker_metadata = inference_state.setdefault("tracker_metadata", {})
        if not tracker_metadata:
            tracker_metadata.update(self._initialize_metadata())
        rank0_metadata = tracker_metadata.setdefault(
            "rank0_metadata",
            self._initialize_metadata()["rank0_metadata"],
        )

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id_int)
        found_in_cache = any(
            isinstance(frame_outputs, Mapping) and obj_id_int in frame_outputs
            for frame_outputs in inference_state.get(
                "cached_frame_outputs", {}
            ).values()
        )
        if obj_rank is None and not found_in_cache:
            if strict:
                raise ValueError(
                    f"Object id {obj_id_int} does not exist in the tracking state."
                )
            return self._fetch_removed_object_frame(inference_state, frame_idx)

        previous_obj_ids_all = np.asarray(
            tracker_metadata.get("obj_ids_all_gpu", []),
            dtype=np.int64,
        ).reshape(-1)
        keep_mask_all = previous_obj_ids_all != obj_id_int
        if obj_rank is not None:
            tracker_states_local = inference_state["sam2_inference_states"]
            self._tracker_remove_objects(tracker_states_local, [obj_id_int])

            obj_ids_per_gpu = tracker_metadata["obj_ids_per_gpu"]
            kept_ids = np.asarray(obj_ids_per_gpu[obj_rank], dtype=np.int64)
            kept_ids = kept_ids[kept_ids != obj_id_int]
            obj_ids_per_gpu[obj_rank] = kept_ids
            tracker_metadata["num_obj_per_gpu"][obj_rank] = int(kept_ids.size)
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(obj_ids_per_gpu)
            if "num_buc_per_gpu" in tracker_metadata:
                remaining_bucket_count = self._count_buckets_in_states(
                    tracker_states_local
                )
                tracker_metadata["num_buc_per_gpu"][obj_rank] = (
                    remaining_bucket_count
                    if remaining_bucket_count > 0
                    else int(np.ceil(kept_ids.size / max(int(self.bucket_capacity), 1)))
                    if kept_ids.size > 0
                    else 0
                )

        for frame_outputs in inference_state.get("cached_frame_outputs", {}).values():
            if isinstance(frame_outputs, dict):
                frame_outputs.pop(obj_id_int, None)

        rank0_metadata.setdefault("removed_obj_ids", set()).add(obj_id_int)
        for suppressed_obj_ids in rank0_metadata.setdefault(
            "suppressed_obj_ids",
            defaultdict(set),
        ).values():
            suppressed_obj_ids.discard(obj_id_int)
        rank0_metadata.get("obj_first_frame_idx", {}).pop(obj_id_int, None)
        rank0_metadata.get("trk_keep_alive", {}).pop(obj_id_int, None)
        rank0_metadata.get("unmatched_frame_inds", {}).pop(obj_id_int, None)
        if obj_rank is not None:
            self._sync_hotstart_gpu_metadata_after_removal(
                tracker_metadata,
                previous_obj_ids=previous_obj_ids_all,
                keep_mask=keep_mask_all,
                frame_idx=frame_idx,
            )
        if "generator_state" in inference_state:
            inference_state["generator_state"].setdefault(
                "hotstart_removed_obj_ids",
                set(),
            ).add(obj_id_int)

        tracker_metadata.get("obj_id_to_score", {}).pop(obj_id_int, None)
        tracker_metadata.get("obj_id_to_last_occluded", {}).pop(obj_id_int, None)
        for score_key in (
            "obj_id_to_sam2_score_frame_wise",
            "obj_id_to_tracker_score_frame_wise",
        ):
            for frame_scores in tracker_metadata.get(score_key, {}).values():
                frame_scores.pop(obj_id_int, None)

        return self._fetch_removed_object_frame(inference_state, frame_idx)

    def remove_object(
        self,
        inference_state: dict[str, Any],
        obj_id: Any,
        frame_idx: int | None = 0,
        strict: bool = False,
        is_user_action: bool = False,
    ) -> tuple[int | None, dict[str, Any] | None]:
        """Remove an object from cached/image-only state or local SAM2 states."""
        del is_user_action
        if inference_state.get("sam2_inference_states"):
            return self._remove_object_from_sam2_states(
                inference_state,
                obj_id=obj_id,
                frame_idx=frame_idx,
                strict=strict,
            )
        num_frames = int(inference_state["num_frames"])
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )
        try:
            obj_id_int = int(obj_id)
        except (TypeError, ValueError) as exc:
            raise TypeError("obj_id must be integer-compatible.") from exc

        tracker_metadata = inference_state.setdefault("tracker_metadata", {})
        if not tracker_metadata:
            tracker_metadata.update(self._initialize_metadata())

        cached_frame_outputs = inference_state.setdefault("cached_frame_outputs", {})
        active_ids = np.asarray(
            tracker_metadata.get("obj_ids_all_gpu", []),
            dtype=np.int64,
        ).reshape(-1)
        found_in_active = bool(np.any(active_ids == obj_id_int))
        found_in_cache = any(
            isinstance(frame_outputs, Mapping) and obj_id_int in frame_outputs
            for frame_outputs in cached_frame_outputs.values()
        )
        if not found_in_active and not found_in_cache:
            if strict:
                raise ValueError(
                    f"Object id {obj_id_int} does not exist in the tracking state."
                )
            return (
                frame_idx,
                _empty_postprocessed_output(
                    int(inference_state["orig_height"]),
                    int(inference_state["orig_width"]),
                    frame_stats=None,
                    include_prod_outputs=self.running_in_prod,
                ),
            )

        rank0_metadata = tracker_metadata.setdefault(
            "rank0_metadata",
            self._initialize_metadata()["rank0_metadata"],
        )
        rank0_metadata.setdefault("removed_obj_ids", set()).add(obj_id_int)
        if "generator_state" in inference_state:
            inference_state["generator_state"].setdefault(
                "hotstart_removed_obj_ids",
                set(),
            ).add(obj_id_int)

        for frame_outputs in cached_frame_outputs.values():
            if isinstance(frame_outputs, dict):
                frame_outputs.pop(obj_id_int, None)

        keep_mask = active_ids != obj_id_int
        kept_ids = active_ids[keep_mask]
        tracker_metadata["obj_ids_all_gpu"] = kept_ids
        tracker_metadata["obj_ids_per_gpu"][0] = kept_ids
        tracker_metadata["num_obj_per_gpu"][0] = int(kept_ids.size)
        if self.is_multiplex:
            tracker_metadata["num_buc_per_gpu"][0] = (
                int(np.ceil(kept_ids.size / self.bucket_capacity))
                if kept_ids.size > 0
                else 0
            )
        tracker_metadata.get("obj_id_to_score", {}).pop(obj_id_int, None)
        tracker_metadata.get("obj_id_to_last_occluded", {}).pop(obj_id_int, None)
        for score_key in (
            "obj_id_to_sam2_score_frame_wise",
            "obj_id_to_tracker_score_frame_wise",
        ):
            for frame_scores in tracker_metadata.get(score_key, {}).values():
                frame_scores.pop(obj_id_int, None)

        rank0_metadata.get("obj_first_frame_idx", {}).pop(obj_id_int, None)
        rank0_metadata.get("trk_keep_alive", {}).pop(obj_id_int, None)
        rank0_metadata.get("unmatched_frame_inds", {}).pop(obj_id_int, None)
        if (
            self.masklet_confirmation_enable
            and "masklet_confirmation" in rank0_metadata
        ):
            confirmation = rank0_metadata["masklet_confirmation"]
            for key in ("status", "consecutive_det_num"):
                values = confirmation.get(key)
                if values is not None and len(values) == len(active_ids):
                    confirmation[key] = np.asarray(values)[keep_mask]
        self._sync_hotstart_gpu_metadata_after_removal(
            tracker_metadata,
            previous_obj_ids=active_ids,
            keep_mask=keep_mask,
            frame_idx=frame_idx,
        )

        remaining_masks = cached_frame_outputs.get(frame_idx, {})
        framewise_scores = {}
        for score_key in (
            "obj_id_to_sam2_score_frame_wise",
            "obj_id_to_tracker_score_frame_wise",
        ):
            framewise_scores.update(
                tracker_metadata.get(score_key, {}).get(frame_idx, {})
            )
        score_map = tracker_metadata.get("obj_id_to_score", {})
        out_scores = {
            remaining_obj_id: framewise_scores.get(
                remaining_obj_id,
                score_map.get(remaining_obj_id, 0.0),
            )
            for remaining_obj_id in remaining_masks
        }
        return (
            frame_idx,
            self._postprocess_output(
                inference_state,
                {
                    "obj_id_to_mask": remaining_masks,
                    "obj_id_to_score": out_scores,
                    "obj_id_to_sam2_score": framewise_scores,
                    "frame_stats": {
                        "num_obj_removed": 1,
                        "num_obj_remaining": len(remaining_masks),
                    },
                },
            ),
        )

    def _get_processing_order(
        self,
        inference_state: dict[str, Any],
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ):
        num_frames = inference_state["num_frames"]
        previous_stages_out = inference_state["previous_stages_out"]
        if all(out is None for out in previous_stages_out) and start_frame_idx is None:
            raise RuntimeError(
                "No prompts are received on any frames. Please add prompt on at "
                "least one frame before propagation."
            )
        if start_frame_idx is None:
            start_frame_idx = min(
                frame_idx
                for frame_idx, out in enumerate(previous_stages_out)
                if out is not None
            )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            processing_order = range(start_frame_idx - 1, end_frame_idx - 1, -1)
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track,
                num_frames - 1,
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order, end_frame_idx

    def propagate_in_video(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            if "inference_state" in kwargs:
                raise TypeError(
                    "inference_state passed both positionally and by keyword."
                )
            kwargs["inference_state"] = args[0]
            if len(args) > 1:
                raise TypeError(
                    "propagate_in_video accepts at most one positional argument."
                )
        inference_state = kwargs.pop("inference_state")
        start_frame_idx = kwargs.pop("start_frame_idx", None)
        max_frame_num_to_track = kwargs.pop("max_frame_num_to_track", None)
        reverse = kwargs.pop("reverse", False)
        kwargs.pop("output_prob_thresh", 0.5)
        kwargs.pop("compute_stability_score", False)
        is_instance_processing = kwargs.pop("is_instance_processing", False)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected propagate_in_video keyword argument(s): {names}"
            )

        yield from self._propagate_in_video_impl(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            is_instance_processing=is_instance_processing,
        )

    def _init_backbone_out(self, inference_state: dict[str, Any]) -> dict[str, Any]:
        input_batch = inference_state["input_batch"]
        text_outputs = self.detector.backbone.forward_text(
            input_batch.find_text_batch,
            device=self.device,
        )
        inference_state.setdefault("feature_cache", {})["text"] = {
            tuple(input_batch.find_text_batch): text_outputs
        }
        return {
            "img_batch_all_stages": input_batch.img_batch,
            **text_outputs,
        }

    def add_prompt(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        text_str: str | None = None,
        clear_old_points: bool = True,
        points: Any = None,
        point_labels: Any = None,
        boxes_xywh: Any = None,
        box_labels: Any = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
    ) -> Any:
        del output_prob_thresh
        frame_idx = int(frame_idx)
        num_frames = int(inference_state["num_frames"])
        if text_str is None and points is None and boxes_xywh is None:
            raise ValueError("add_prompt requires text_str, boxes_xywh, or points.")
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )
        if points is not None or point_labels is not None or not clear_old_points:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTracking.add_prompt(point-prompts)"
            )
        if not clear_old_boxes:
            raise ValueError("clear_old_boxes must be True.")
        if (boxes_xywh is None) != (box_labels is None):
            raise ValueError("boxes_xywh and box_labels must be provided together.")
        if text_str is not None and not isinstance(text_str, str):
            raise TypeError("text_str must be a string when provided.")

        self.reset_state(inference_state)

        if text_str is not None:
            inference_state["text_prompt"] = text_str
            inference_state["input_batch"].find_text_batch[0] = text_str
            for find_input in inference_state["input_batch"].find_inputs:
                find_input.text_ids = mx.full(
                    find_input.text_ids.shape,
                    self.TEXT_ID_FOR_TEXT,
                    dtype=mx.int64,
                )

        if boxes_xywh is not None:
            boxes_xywh_mx = mx.array(boxes_xywh, dtype=mx.float32)
            box_labels_mx = mx.array(box_labels, dtype=mx.int64)
            if boxes_xywh_mx.ndim != 2 or boxes_xywh_mx.shape[-1] != 4:
                raise ValueError(
                    f"boxes_xywh must have shape (N, 4), got {boxes_xywh_mx.shape}."
                )
            if boxes_xywh_mx.shape[0] == 0:
                raise ValueError("boxes_xywh must contain at least one box.")
            if (
                box_labels_mx.ndim != 1
                or box_labels_mx.shape[0] != boxes_xywh_mx.shape[0]
            ):
                raise ValueError(
                    "box_labels must have shape (N,) and match boxes_xywh; "
                    f"got {box_labels_mx.shape} for {boxes_xywh_mx.shape[0]} boxes."
                )
            boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh_mx)
            boxes_in_range = mx.all(
                (boxes_xywh_mx >= 0.0)
                & (boxes_xywh_mx <= 1.0)
                & (boxes_cxcywh >= 0.0)
                & (boxes_cxcywh <= 1.0)
            )
            if not bool(_array_to_numpy(boxes_in_range, dtype=bool).reshape(())):
                raise ValueError("boxes_xywh must be normalized to [0, 1].")
            inference_state["per_frame_raw_box_input"][frame_idx] = (
                boxes_cxcywh,
                box_labels_mx,
            )
            self._set_find_input_box_geometry(
                inference_state["input_batch"].find_inputs[frame_idx],
                boxes_cxcywh,
                box_labels_mx,
            )
            _, _, geometric_prompt = self._get_visual_prompt(
                inference_state,
                frame_idx,
                boxes_cxcywh,
                box_labels_mx,
            )
            inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        inference_state["backbone_out"] = self._init_backbone_out(inference_state)
        out = self._run_single_frame_inference(
            inference_state,
            frame_idx,
            reverse=False,
        )
        return frame_idx, self._postprocess_output(inference_state, out)

    def forward(self, input: Any, is_inference: bool = False) -> Any:
        del is_inference
        if input.raw_images is None:
            raise ValueError("Sam3MultiplexTracking.forward requires input.raw_images.")
        if not input.find_metadatas or input.find_metadatas[0] is None:
            raise ValueError(
                "Sam3MultiplexTracking.forward requires find_metadatas[0]."
            )

        metadata = input.find_metadatas[0]
        prompt_ids = _array_to_numpy(
            metadata.original_category_id,
            dtype=np.int64,
        ).reshape(-1)
        prompt_list = list(input.find_text_batch)
        if len(prompt_ids) < len(prompt_list):
            raise ValueError(
                "original_category_id must contain at least one id per text prompt."
            )

        original_rank = self.rank
        original_world_size = self.world_size
        detector_has_rank = hasattr(self.detector, "rank")
        detector_has_world_size = hasattr(self.detector, "world_size")
        original_detector_rank = getattr(self.detector, "rank", None)
        original_detector_world_size = getattr(self.detector, "world_size", None)

        self.rank = 0
        self.world_size = 1
        if detector_has_rank:
            self.detector.rank = 0
        if detector_has_world_size:
            self.detector.world_size = 1

        try:
            tracking_res = defaultdict(dict)
            scores_labels = {}
            inference_state = self.init_state(resource_path=input.raw_images)
            for prompt_id, prompt in zip(prompt_ids, prompt_list):
                _, prompt_out = self.add_prompt(
                    inference_state,
                    frame_idx=0,
                    text_str=prompt,
                )
                start_obj_id = max(scores_labels.keys(), default=-1) + 1
                obj_ids_this_prompt = set()
                if inference_state["is_image_only"]:
                    prompt_outputs = [(0, prompt_out)]
                else:
                    prompt_outputs = self.propagate_in_video(
                        inference_state,
                        start_frame_idx=0,
                        max_frame_num_to_track=inference_state["num_frames"],
                        reverse=False,
                    )
                for frame_idx, out in prompt_outputs:
                    for obj_id, mask in zip(
                        out["out_obj_ids"],
                        out["out_binary_masks"],
                        strict=True,
                    ):
                        output_obj_id = int(obj_id) + start_obj_id
                        # evaluator-export-boundary: prep_for_evaluator builds COCO-style
                        # NumPy/RLE payloads, so the postprocessed masks remain on host.
                        tracking_res[int(frame_idx)][output_obj_id] = np.asarray(
                            mask,
                            dtype=bool,
                        )[None, :, :]
                        obj_ids_this_prompt.add(output_obj_id)

                obj_id_to_score = inference_state["tracker_metadata"]["obj_id_to_score"]
                for obj_id, score in obj_id_to_score.items():
                    output_obj_id = int(obj_id) + start_obj_id
                    if output_obj_id in obj_ids_this_prompt:
                        scores_labels[output_obj_id] = (score, int(prompt_id))
                self.reset_state(inference_state)

            video_id = int(
                _array_to_numpy(metadata.original_image_id, dtype=np.int64).reshape(-1)[
                    0
                ]
            )
            preds = self.prep_for_evaluator(
                input.raw_images,
                tracking_res,
                scores_labels,
            )
            return {video_id: preds}
        finally:
            self.rank = original_rank
            self.world_size = original_world_size
            if detector_has_rank:
                self.detector.rank = original_detector_rank
            if detector_has_world_size:
                self.detector.world_size = original_detector_world_size


class Sam3MultiplexTrackingProd(Sam3MultiplexTracking):
    def init_state(
        self,
        resource_path: Any,
        offload_video_to_cpu: bool = False,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
        input_is_mp4: bool = False,
    ) -> Any:
        inference_state = super().init_state(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
            input_is_mp4=input_is_mp4,
        )
        inference_state["generator_state"] = self._new_generator_state()
        return inference_state

    def reset_state(self, inference_state: dict[str, Any]) -> None:
        super().reset_state(inference_state)
        inference_state["generator_state"] = self._new_generator_state()

    def propagate_in_video(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            if "inference_state" in kwargs:
                raise TypeError(
                    "inference_state passed both positionally and by keyword."
                )
            kwargs["inference_state"] = args[0]
            if len(args) > 1:
                raise TypeError(
                    "propagate_in_video accepts at most one positional argument."
                )
        inference_state = kwargs.pop("inference_state")
        start_frame_idx = kwargs.pop("start_frame_idx", None)
        max_frame_num_to_track = kwargs.pop("max_frame_num_to_track", None)
        reverse = kwargs.pop("reverse", False)
        kwargs.pop("output_prob_thresh", 0.5)
        kwargs.pop("compute_stability_score", False)
        is_instance_processing = kwargs.pop("is_instance_processing", False)
        is_last_batch = kwargs.pop("is_last_batch", True)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected propagate_in_video keyword argument(s): {names}"
            )

        generator_state = inference_state.setdefault(
            "generator_state",
            self._new_generator_state(),
        )
        yield from self._propagate_in_video_impl(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            is_instance_processing=is_instance_processing,
            generator_state=generator_state,
            flush_hotstart_at_end=is_last_batch,
            persist_generator_state=True,
        )


class Sam3MultiplexTrackingWithInteractivity(Sam3MultiplexTracking):
    def __init__(
        self,
        use_prev_mem_frame: bool = False,
        use_stateless_refinement: bool = False,
        refinement_detector_cond_frame_removal_window: int = 30 * 4,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.use_prev_mem_frame = use_prev_mem_frame
        self.use_stateless_refinement = use_stateless_refinement
        self.refinement_detector_cond_frame_removal_window = (
            refinement_detector_cond_frame_removal_window
        )

    def init_state(self, *args: Any, **kwargs: Any) -> Any:
        inference_state = super().init_state(*args, **kwargs)
        inference_state["action_history"] = []
        if bool(getattr(self.tracker, "per_obj_inference", False)):
            inference_state["sam2_inference_states"] = [
                self._init_new_sam2_state(inference_state)
            ]
        return inference_state

    def reset_state(self, inference_state: Any) -> None:
        super().reset_state(inference_state)
        inference_state.setdefault("action_history", []).clear()
        if bool(getattr(self.tracker, "per_obj_inference", False)):
            inference_state["sam2_inference_states"] = [
                self._init_new_sam2_state(inference_state)
            ]

    def _init_new_sam2_state(self, inference_state: dict[str, Any]) -> dict[str, Any]:
        init_state = getattr(self.tracker, "init_state", None)
        if init_state is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity._init_new_sam2_state"
            )
        sam2_state = init_state(
            cached_features=inference_state.get("feature_cache", {}),
            video_height=int(inference_state["orig_height"]),
            video_width=int(inference_state["orig_width"]),
            num_frames=int(inference_state["num_frames"]),
        )
        if not isinstance(sam2_state, dict):
            raise TypeError("tracker.init_state must return a SAM2 state dict.")
        sam2_state.setdefault("obj_ids", [])
        return sam2_state

    def _create_singleton_multiplex_state(
        self,
        source_state: Mapping[str, Any],
        obj_id: int,
    ) -> MultiplexState:
        controller = getattr(self.tracker, "multiplex_controller", None)
        get_state = getattr(controller, "get_state", None)
        if get_state is not None:
            return get_state(
                num_valid_entries=1,
                device=source_state.get("device", "mlx"),
                dtype=mx.float32,
                random=False,
                object_ids=[obj_id],
            )

        source_multiplex_state = source_state.get("multiplex_state")
        multiplex_count = int(
            getattr(
                source_multiplex_state,
                "multiplex_count",
                getattr(controller, "multiplex_count", 1),
            )
        )
        allowed_bucket_capacity = int(
            getattr(
                source_multiplex_state,
                "allowed_bucket_capacity",
                getattr(controller, "allowed_bucket_capacity", 1),
            )
        )
        multiplex_count = max(multiplex_count, 1)
        allowed_bucket_capacity = max(min(allowed_bucket_capacity, multiplex_count), 1)
        return MultiplexState(
            [[0] + [-1] * (multiplex_count - 1)],
            device=source_state.get("device", "mlx"),
            dtype=mx.float32,
            allowed_bucket_capacity=allowed_bucket_capacity,
            object_ids=[obj_id],
        )

    def _extract_object_to_singleton_state(
        self,
        inference_state: dict[str, Any],
        obj_id: Any,
        obj_rank: int,
    ) -> None:
        if self.rank != obj_rank:
            return
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity."
                "_extract_object_to_singleton_state(distributed)"
            )

        obj_id_int = int(obj_id)
        tracker_states_local = inference_state.setdefault("sam2_inference_states", [])
        source_state: dict[str, Any] | None = None
        source_state_idx: int | None = None
        for idx, state in enumerate(tracker_states_local):
            if isinstance(state, dict) and obj_id_int in state.get("obj_ids", []):
                source_state = state
                source_state_idx = idx
                break
        if source_state is None or source_state_idx is None:
            raise ValueError(
                f"Object id {obj_id_int} is not present in local SAM2 states."
            )
        if len(source_state.get("obj_ids", [])) <= 1:
            return

        obj_idx_in_source = self._lookup_existing_obj_idx(source_state, obj_id_int)
        if obj_idx_in_source is None:
            raise ValueError(
                f"Object id {obj_id_int} is missing from source obj_id mapping."
            )
        multiplex_state = source_state.get("multiplex_state")
        singleton_consolidated_outputs: dict[str, dict[Any, dict[str, Any]]] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            source_outputs = source_state.get("output_dict", {}).get(storage_key, {})
            for output_frame_idx, source_frame_out in source_outputs.items():
                pred_masks = source_frame_out.get("pred_masks")
                object_score_logits = source_frame_out.get("object_score_logits")
                if pred_masks is None or object_score_logits is None:
                    raise ValueError(
                        "Packed singleton extraction requires pred_masks and "
                        f"object_score_logits for {storage_key} frame {output_frame_idx}."
                    )
                if pred_masks.shape[0] < obj_idx_in_source + 1:
                    continue
                singleton_frame_out = {
                    "pred_masks": _copy_first_axis_slice(
                        pred_masks,
                        obj_idx_in_source,
                        obj_idx_in_source + 1,
                    ),
                    "object_score_logits": _copy_first_axis_slice(
                        object_score_logits,
                        obj_idx_in_source,
                        obj_idx_in_source + 1,
                    ),
                    "image_features": source_frame_out.get("image_features"),
                    "image_pos_enc": source_frame_out.get("image_pos_enc"),
                    "local_obj_id_to_idx": {obj_id_int: 0},
                    "maskmem_features": _demux_and_slice_first_axis(
                        source_frame_out.get("maskmem_features"),
                        multiplex_state,
                        obj_idx_in_source,
                    ),
                }

                maskmem_pos_enc = source_frame_out.get("maskmem_pos_enc")
                if maskmem_pos_enc is None:
                    singleton_frame_out["maskmem_pos_enc"] = None
                else:
                    singleton_frame_out["maskmem_pos_enc"] = [
                        _demux_and_slice_first_axis(
                            level_enc, multiplex_state, obj_idx_in_source
                        )
                        if level_enc is not None
                        else None
                        for level_enc in maskmem_pos_enc
                    ]

                if "obj_ptr" in source_frame_out and getattr(
                    self.tracker, "use_obj_ptrs_in_encoder", False
                ):
                    singleton_frame_out["obj_ptr"] = _demux_and_slice_first_axis(
                        source_frame_out["obj_ptr"],
                        multiplex_state,
                        obj_idx_in_source,
                    )
                if "conditioning_objects" in source_frame_out:
                    singleton_frame_out["conditioning_objects"] = (
                        {0}
                        if obj_idx_in_source in source_frame_out["conditioning_objects"]
                        else set()
                    )
                singleton_consolidated_outputs[storage_key][output_frame_idx] = (
                    singleton_frame_out
                )

        extracted_point_inputs = (
            source_state.get("point_inputs_per_obj", {})
            .get(
                obj_idx_in_source,
                {},
            )
            .copy()
        )
        extracted_mask_inputs = (
            source_state.get("mask_inputs_per_obj", {})
            .get(
                obj_idx_in_source,
                {},
            )
            .copy()
        )

        extracted_obj_cond_outputs: dict[Any, Any] = {}
        extracted_obj_non_cond_outputs: dict[Any, Any] = {}
        obj_output_dict = source_state.get("output_dict_per_obj", {}).get(
            obj_idx_in_source,
        )
        if obj_output_dict is not None:
            cond_input_keys = (
                extracted_point_inputs.keys() | extracted_mask_inputs.keys()
            )
            extracted_obj_cond_outputs = {
                frame_key: frame_out
                for frame_key, frame_out in obj_output_dict.get(
                    "cond_frame_outputs",
                    {},
                ).items()
                if frame_key in cond_input_keys
            }
            extracted_obj_non_cond_outputs = obj_output_dict.get(
                "non_cond_frame_outputs",
                {},
            ).copy()

        extracted_temp_cond_outputs: dict[Any, Any] = {}
        extracted_temp_non_cond_outputs: dict[Any, Any] = {}
        temp_obj_output_dict = source_state.get("temp_output_dict_per_obj", {}).get(
            obj_idx_in_source,
        )
        if temp_obj_output_dict is not None:
            extracted_temp_cond_outputs = temp_obj_output_dict.get(
                "cond_frame_outputs",
                {},
            ).copy()
            extracted_temp_non_cond_outputs = temp_obj_output_dict.get(
                "non_cond_frame_outputs",
                {},
            ).copy()

        remove_object = getattr(self.tracker, "remove_object", None)
        if remove_object is None:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity."
                "_extract_object_to_singleton_state(remove_object)"
            )
        remaining_obj_ids, _ = remove_object(
            source_state,
            obj_id_int,
            strict=False,
            need_output=False,
        )

        new_sam2_state = self._init_new_sam2_state(inference_state)
        new_sam2_state["obj_id_to_idx"] = {obj_id_int: 0}
        new_sam2_state["obj_idx_to_id"] = {0: obj_id_int}
        new_sam2_state["obj_ids"] = [obj_id_int]
        new_sam2_state["point_inputs_per_obj"] = {0: extracted_point_inputs}
        new_sam2_state["mask_inputs_per_obj"] = {0: extracted_mask_inputs}
        new_sam2_state["output_dict_per_obj"] = {
            0: {
                "cond_frame_outputs": extracted_obj_cond_outputs,
                "non_cond_frame_outputs": extracted_obj_non_cond_outputs,
            }
        }
        new_sam2_state["temp_output_dict_per_obj"] = {
            0: {
                "cond_frame_outputs": extracted_temp_cond_outputs,
                "non_cond_frame_outputs": extracted_temp_non_cond_outputs,
            }
        }

        new_multiplex_state = self._create_singleton_multiplex_state(
            source_state,
            obj_id_int,
        )
        new_sam2_state["multiplex_state"] = new_multiplex_state
        if getattr(self.tracker, "use_obj_ptrs_in_encoder", False):
            for storage_outputs in singleton_consolidated_outputs.values():
                for frame_out in storage_outputs.values():
                    if frame_out.get("obj_ptr") is not None:
                        frame_out["obj_ptr"] = new_multiplex_state.mux(
                            frame_out["obj_ptr"]
                        )
        new_sam2_state["output_dict"] = singleton_consolidated_outputs

        for key in ("first_ann_frame_idx", "tracking_has_started"):
            if key in source_state:
                new_sam2_state[key] = source_state[key]
        new_sam2_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }

        tracker_states_local.append(new_sam2_state)
        if len(remaining_obj_ids) == 0:
            tracker_states_local.pop(source_state_idx)

    def add_action_history(
        self,
        inference_state: dict[str, Any],
        action_type: str,
        frame_idx: int | None = None,
        obj_ids: list[Any] | None = None,
    ) -> None:
        instance_actions = {"add", "remove", "refine"}
        propagation_actions = {
            "propagation_full",
            "propagation_partial",
            "propagation_fetch",
            "propagation_cancel",
        }
        if action_type not in instance_actions | propagation_actions:
            valid = sorted(instance_actions | propagation_actions)
            raise ValueError(
                f"Invalid action type: {action_type}, must be one of {valid}."
            )
        inference_state.setdefault("action_history", []).append(
            {
                "type": action_type,
                "frame_idx": frame_idx,
                "obj_ids": obj_ids,
            }
        )

    def add_prompt(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        text_str: str | None = None,
        clear_old_points: bool = True,
        points: Any = None,
        point_labels: Any = None,
        boxes_xywh: Any = None,
        box_labels: Any = None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: Any = None,
        rel_coordinates: bool = True,
    ) -> Any:
        if points is not None:
            del output_prob_thresh, clear_old_boxes, box_labels
            if text_str is not None or boxes_xywh is not None:
                raise ValueError(
                    "When points are provided, text_str and boxes_xywh must be None."
                )
            if obj_id is None:
                raise ValueError("obj_id must be provided when points are provided.")
            if point_labels is None:
                raise ValueError(
                    "point_labels must be provided when points are provided."
                )
            return self.add_sam2_new_points(
                inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=point_labels,
                clear_old_points=clear_old_points,
                rel_coordinates=rel_coordinates,
                use_prev_mem_frame=self.use_prev_mem_frame,
            )
        orig_use_batched_grounding = self.use_batched_grounding
        self.use_batched_grounding = False
        try:
            return super().add_prompt(
                inference_state,
                frame_idx=frame_idx,
                text_str=text_str,
                clear_old_points=clear_old_points,
                points=points,
                point_labels=point_labels,
                boxes_xywh=boxes_xywh,
                box_labels=box_labels,
                clear_old_boxes=clear_old_boxes,
                output_prob_thresh=output_prob_thresh,
            )
        finally:
            self.use_batched_grounding = orig_use_batched_grounding

    def add_sam2_new_points(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_id: Any,
        points: Any,
        labels: Any,
        clear_old_points: bool,
        rel_coordinates: bool = True,
        use_prev_mem_frame: bool = False,
    ) -> Any:
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity.add_sam2_new_points(distributed)"
            )
        frame_idx = int(frame_idx)
        num_frames = int(inference_state["num_frames"])
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )

        tracker_metadata = inference_state.setdefault("tracker_metadata", {})
        if not tracker_metadata:
            tracker_metadata.update(self._initialize_metadata())
        tracker_metadata.setdefault(
            "obj_id_to_sam2_score_frame_wise",
            defaultdict(dict),
        )
        rank0_metadata = tracker_metadata.setdefault(
            "rank0_metadata",
            self._initialize_metadata()["rank0_metadata"],
        )

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
        object_has_been_refined = self._has_object_been_refined(inference_state, obj_id)
        if (
            obj_rank is not None
            and self.use_stateless_refinement
            and not object_has_been_refined
        ):
            self.remove_object(
                inference_state,
                obj_id=obj_id,
                frame_idx=None,
                strict=False,
                is_user_action=False,
            )
            obj_rank = None
        elif obj_rank is not None and not object_has_been_refined:
            if self.rank == obj_rank and not bool(
                getattr(self.tracker, "per_obj_inference", False)
            ):
                tracker_states = self._get_sam2_inference_states_by_obj_ids(
                    inference_state,
                    [obj_id],
                )
                if len(tracker_states) != 1:
                    raise ValueError(
                        "Expected exactly one SAM2 state for obj_id "
                        f"{obj_id}, found {len(tracker_states)}."
                    )
                if len(tracker_states[0].get("obj_ids", [])) > 1:
                    self._extract_object_to_singleton_state(
                        inference_state,
                        obj_id,
                        obj_rank,
                    )
        if obj_rank is None:
            num_prev_obj = int(np.sum(tracker_metadata["num_obj_per_gpu"]))
            if num_prev_obj >= int(self.max_num_objects):
                return frame_idx, None
            obj_rank = 0
            tracker_states_local = inference_state.setdefault(
                "sam2_inference_states",
                [],
            )
            if bool(getattr(self.tracker, "per_obj_inference", False)):
                if not tracker_states_local:
                    tracker_states_local.append(
                        self._init_new_sam2_state(inference_state)
                    )
                sam2_state = tracker_states_local[0]
            else:
                sam2_state = self._init_new_sam2_state(inference_state)
                tracker_states_local.append(sam2_state)
            obj_ids_per_gpu = tracker_metadata["obj_ids_per_gpu"]
            obj_ids_per_gpu[obj_rank] = np.concatenate(
                [
                    np.asarray(obj_ids_per_gpu[obj_rank], dtype=np.int64),
                    np.array([int(obj_id)], dtype=np.int64),
                ]
            )
            tracker_metadata["num_obj_per_gpu"][obj_rank] = int(
                obj_ids_per_gpu[obj_rank].size
            )
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(obj_ids_per_gpu)
            tracker_metadata["max_obj_id"] = max(
                int(tracker_metadata.get("max_obj_id", -1)),
                int(obj_id),
            )
            if "num_buc_per_gpu" in tracker_metadata:
                tracker_metadata["num_buc_per_gpu"][obj_rank] = int(
                    np.ceil(
                        tracker_metadata["num_obj_per_gpu"][obj_rank]
                        / max(int(self.bucket_capacity), 1)
                    )
                )
            rank0_metadata.setdefault("obj_first_frame_idx", {})[int(obj_id)] = (
                frame_idx
            )
            self._sync_hotstart_gpu_metadata_after_addition(
                tracker_metadata,
                frame_idx=frame_idx,
                num_new_objects=1,
            )
            self.add_action_history(
                inference_state,
                "add",
                frame_idx=frame_idx,
                obj_ids=[obj_id],
            )
        else:
            tracker_states = self._get_sam2_inference_states_by_obj_ids(
                inference_state,
                [obj_id],
            )
            if len(tracker_states) != 1:
                raise ValueError(
                    "Expected exactly one SAM2 state for obj_id "
                    f"{obj_id}, found {len(tracker_states)}."
                )
            sam2_state = tracker_states[0]
            self.add_action_history(
                inference_state,
                "refine",
                frame_idx=frame_idx,
                obj_ids=[obj_id],
            )

        tracker_metadata["obj_id_to_score"][obj_id] = 1.0
        tracker_metadata["obj_id_to_sam2_score_frame_wise"][frame_idx][obj_id] = (
            mx.array(1.0, dtype=mx.float32)
        )
        rank0_metadata.setdefault("removed_obj_ids", set()).discard(obj_id)
        for suppressed_obj_ids in rank0_metadata.setdefault(
            "suppressed_obj_ids",
            defaultdict(set),
        ).values():
            suppressed_obj_ids.discard(obj_id)
        confirmation = rank0_metadata.get("masklet_confirmation")
        if confirmation is not None:
            obj_ids_all_gpu = np.asarray(
                tracker_metadata["obj_ids_all_gpu"],
                dtype=np.int64,
            )
            obj_indices = np.where(obj_ids_all_gpu == int(obj_id))[0]
            if obj_indices.size > 0:
                obj_idx = int(obj_indices[0])
                if obj_idx < len(confirmation["status"]):
                    confirmation["status"][obj_idx] = (
                        MaskletConfirmationStatus.CONFIRMED.value
                    )
                    confirmation["consecutive_det_num"][obj_idx] = (
                        self.masklet_confirmation_consecutive_det_thresh
                    )

        should_restore_original_mask = _point_count(points) == 0 and bool(
            inference_state.get("is_image_only", False)
        )
        mask_input = (
            self._get_mask_input(sam2_state, frame_idx, obj_id)
            if should_restore_original_mask
            else None
        )
        if mask_input is not None and 0 in tuple(mask_input.shape):
            mask_input = None
        add_new_mask = getattr(self.tracker, "add_new_mask", None)
        if mask_input is not None and add_new_mask is not None:
            self.tracker.clear_all_points_in_frame(
                sam2_state,
                frame_idx,
                obj_id,
                need_output=False,
            )
            frame_idx, obj_ids, _low_res_masks, video_res_masks = add_new_mask(
                sam2_state,
                frame_idx,
                obj_id,
                mask_input,
            )
        else:
            frame_idx, obj_ids, _low_res_masks, video_res_masks = (
                self.tracker.add_new_points(
                    inference_state=sam2_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=clear_old_points,
                    rel_coordinates=rel_coordinates,
                    use_prev_mem_frame=use_prev_mem_frame,
                )
            )
        if video_res_masks is not None and len(video_res_masks) > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks,
                fill_hole_area=self.fill_hole_area,
                sprinkle_removal_area=self.sprinkle_removal_area,
                fill_holes=True,
                remove_sprinkles=True,
            )
        self._sync_sam2_inputs_before_point_preflight(sam2_state)
        self.tracker.propagate_in_video_preflight(sam2_state, run_mem_encoder=True)
        if not inference_state.get("is_image_only", False):
            self.clear_detector_added_cond_frame_in_sam2(sam2_state, obj_id, frame_idx)

        if video_res_masks is None or len(video_res_masks) == 0:
            refined_obj_id_to_mask = None
        else:
            if obj_id not in obj_ids:
                raise ValueError(
                    f"Tracker add_new_points did not return obj_id {obj_id}."
                )
            new_mask_data = video_res_masks[list(obj_ids).index(obj_id)] > 0.0
            refined_obj_id_to_mask = {obj_id: new_mask_data}

        obj_id_to_mask = self._build_sam2_output(
            inference_state,
            frame_idx,
            refined_obj_id_to_mask,
        )
        suppressed_obj_ids = rank0_metadata.get("suppressed_obj_ids", {}).get(
            frame_idx,
            set(),
        )
        self._cache_frame_outputs(
            inference_state,
            frame_idx,
            obj_id_to_mask,
            suppressed_obj_ids=suppressed_obj_ids,
        )
        out_scores = {
            cached_obj_id: tracker_metadata["obj_id_to_score"].get(cached_obj_id, 0.0)
            for cached_obj_id in obj_id_to_mask
        }
        return frame_idx, self._postprocess_output(
            inference_state,
            {
                "obj_id_to_mask": obj_id_to_mask,
                "obj_id_to_score": out_scores,
                "obj_id_to_sam2_score": tracker_metadata[
                    "obj_id_to_sam2_score_frame_wise"
                ][frame_idx],
            },
            suppressed_obj_ids=suppressed_obj_ids,
        )

    def _sync_sam2_inputs_before_point_preflight(
        self, sam2_state: dict[str, Any]
    ) -> None:
        mask_inputs_per_obj = sam2_state.get("mask_inputs_per_obj")
        point_inputs_per_obj = sam2_state.get("point_inputs_per_obj")
        if not isinstance(mask_inputs_per_obj, Mapping) or not isinstance(
            point_inputs_per_obj,
            Mapping,
        ):
            return

        for obj_idx, mask_inputs_per_frame in list(mask_inputs_per_obj.items()):
            if not isinstance(mask_inputs_per_frame, dict):
                continue
            point_inputs_per_frame = point_inputs_per_obj.get(obj_idx, {})
            point_frame_indices = (
                set(point_inputs_per_frame)
                if isinstance(point_inputs_per_frame, Mapping)
                else set()
            )
            for frame_id in list(mask_inputs_per_frame):
                if frame_id not in point_frame_indices:
                    mask_inputs_per_frame.pop(frame_id, None)

        input_frames: set[int] = set()
        for point_inputs_per_frame in point_inputs_per_obj.values():
            if isinstance(point_inputs_per_frame, Mapping):
                input_frames.update(
                    int(frame_id) for frame_id in point_inputs_per_frame
                )
        for mask_inputs_per_frame in mask_inputs_per_obj.values():
            if isinstance(mask_inputs_per_frame, Mapping):
                input_frames.update(int(frame_id) for frame_id in mask_inputs_per_frame)

        temp_frames: set[int] = set()
        for temp_outputs in sam2_state.get("temp_output_dict_per_obj", {}).values():
            if not isinstance(temp_outputs, Mapping):
                continue
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                frame_outputs = temp_outputs.get(storage_key, {})
                if isinstance(frame_outputs, Mapping):
                    temp_frames.update(int(frame_id) for frame_id in frame_outputs)

        previous_input_frames = input_frames - temp_frames
        cond_outputs = sam2_state.get("output_dict", {}).get("cond_frame_outputs", {})
        cond_frame_outputs: set[int] = set()
        non_cond_frame_outputs: set[int] = set()
        for frame_id in previous_input_frames:
            if frame_id in cond_outputs:
                cond_frame_outputs.add(frame_id)
            else:
                non_cond_frame_outputs.add(frame_id)
        sam2_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": cond_frame_outputs,
            "non_cond_frame_outputs": non_cond_frame_outputs,
        }

    def _has_object_been_refined(
        self,
        inference_state: dict[str, Any],
        obj_id: Any,
    ) -> bool:
        for action in inference_state.get("action_history", []):
            if action["type"] in {"add", "refine"} and action.get("obj_ids"):
                if obj_id in action["obj_ids"]:
                    return True
        return False

    def parse_action_history_for_propagation(
        self,
        inference_state: dict[str, Any],
    ) -> tuple[str, list[Any] | None]:
        action_history = inference_state.setdefault("action_history", [])
        if (
            len(action_history) == 1
            and action_history[0]["type"] == "propagation_cancel"
        ):
            return "propagation_full", None
        if (
            len(action_history) >= 2
            and action_history[-1]["type"] == "propagation_cancel"
        ):
            action_before_cancellation = action_history[-2]
            if (
                action_before_cancellation["type"] == "propagation_fetch"
                and len(action_history) >= 3
            ):
                action_before_cancellation = action_history[-3]
            return (
                action_before_cancellation["type"],
                action_before_cancellation.get("obj_ids", None),
            )
        return self._parse_action_history_for_propagation(
            action_history,
            int(inference_state["num_frames"]),
        )

    def _parse_action_history_for_propagation(
        self,
        action_history: list[dict[str, Any]],
        num_frames: int,
    ) -> tuple[str, list[Any] | None]:
        if len(action_history) == 0:
            return "propagation_full", None

        last_action = action_history[-1]
        if "propagation" in last_action["type"]:
            if last_action["type"] == "propagation_fetch":
                return "propagation_fetch", None
            if last_action["type"] in {"propagation_partial", "propagation_full"}:
                has_previous_propagation = len(action_history) > 1 and action_history[
                    -2
                ]["type"] in {"propagation_partial", "propagation_full"}
                if has_previous_propagation or last_action["frame_idx"] in {
                    0,
                    num_frames - 1,
                }:
                    return "propagation_fetch", None
                return last_action["type"], last_action["obj_ids"]

        obj_ids: list[Any] = []
        for action in reversed(action_history):
            if "propagation" in action["type"]:
                break
            if action["type"] in {"add", "refine"}:
                obj_ids.extend(action["obj_ids"])
        if obj_ids:
            return "propagation_partial", sorted(set(obj_ids))
        return "propagation_fetch", None

    def fetch_and_process_single_frame_results(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
    ) -> tuple[int, dict[str, Any]]:
        cached_frame_outputs = inference_state.get("cached_frame_outputs", {})
        if frame_idx not in cached_frame_outputs:
            raise ValueError(
                f"No cached multiplex output is available for frame_idx={frame_idx}."
            )
        tracker_metadata = inference_state.get("tracker_metadata", {})
        rank0_metadata = tracker_metadata.get("rank0_metadata", {})
        suppressed_by_frame = rank0_metadata.get("suppressed_obj_ids", {})
        obj_id_to_sam2_score = tracker_metadata.get(
            "obj_id_to_sam2_score_frame_wise",
            {},
        ).get(frame_idx, {})
        obj_id_to_score = tracker_metadata.get("obj_id_to_score", {})
        frame_scores = {
            obj_id: obj_id_to_sam2_score.get(
                obj_id,
                obj_id_to_score.get(obj_id, 0.0),
            )
            for obj_id in cached_frame_outputs[frame_idx]
        }
        return (
            frame_idx,
            self._postprocess_output(
                inference_state,
                {
                    "obj_id_to_mask": cached_frame_outputs[frame_idx],
                    "obj_id_to_score": {
                        obj_id: obj_id_to_score.get(obj_id, frame_scores[obj_id])
                        for obj_id in cached_frame_outputs[frame_idx]
                    },
                    "obj_id_to_sam2_score": frame_scores,
                },
                suppressed_obj_ids=suppressed_by_frame.get(frame_idx, set()),
            ),
        )

    def cancel_propagation(self, inference_state: Any) -> Any:
        self.add_action_history(
            inference_state,
            action_type="propagation_cancel",
            obj_ids=None,
            frame_idx=None,
        )
        if "generator_state" in inference_state:
            inference_state["generator_state"] = self._new_generator_state()
        return None

    def remove_object(
        self,
        inference_state: Any,
        obj_id: Any,
        frame_idx: int | None,
        strict: bool = False,
        is_user_action: bool = False,
    ) -> Any:
        result = super().remove_object(
            inference_state,
            obj_id=obj_id,
            frame_idx=frame_idx,
            strict=strict,
            is_user_action=is_user_action,
        )
        if is_user_action:
            self.add_action_history(
                inference_state,
                action_type="remove",
                frame_idx=frame_idx,
                obj_ids=[int(obj_id)],
            )
        return result

    def _get_sam2_inference_states_by_obj_ids(
        self,
        inference_state: dict[str, Any],
        obj_ids: list[Any],
    ) -> list[Any]:
        requested_obj_ids = set(obj_ids)
        return [
            state
            for state in inference_state.get("sam2_inference_states", [])
            if requested_obj_ids & set(state.get("obj_ids", []))
        ]

    def _lookup_existing_obj_idx(
        self,
        inference_state: dict[str, Any],
        obj_id: Any,
    ) -> int | None:
        obj_id_to_idx = inference_state.get("obj_id_to_idx", {})
        if obj_id in obj_id_to_idx:
            return int(obj_id_to_idx[obj_id])
        if "obj_ids" in inference_state:
            obj_id_matches = np.flatnonzero(
                np.asarray(inference_state["obj_ids"]) == obj_id
            )
            if obj_id_matches.size:
                return int(obj_id_matches[0])
        return None

    def _get_mask_input(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_id: Any,
    ) -> Any:
        obj_idx = self._lookup_existing_obj_idx(inference_state, obj_id)
        if obj_idx is None:
            return None

        mask_inputs_per_frame = inference_state.get("mask_inputs_per_obj", {}).get(
            obj_idx,
            {},
        )
        if frame_idx not in mask_inputs_per_frame:
            return None

        mask_input = mask_inputs_per_frame[frame_idx]
        if _is_mlx_array(mask_input):
            if mask_input.ndim == 4 and mask_input.shape[:2] == (1, 1):
                return mask_input[0, 0]
            if mask_input.ndim == 3 and mask_input.shape[0] == 1:
                return mask_input[0]
            return mask_input

        mask_input_np = np.asarray(mask_input)
        if mask_input_np.ndim == 4 and mask_input_np.shape[:2] == (1, 1):
            return mask_input_np[0, 0]
        if mask_input_np.ndim == 3 and mask_input_np.shape[0] == 1:
            return mask_input_np[0]
        return mask_input

    def _convert_low_res_mask_to_video_res(
        self,
        low_res_mask: Any,
        inference_state: dict[str, Any],
    ) -> Any:
        if low_res_mask is None:
            return None

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
            size=(
                int(inference_state["orig_height"]),
                int(inference_state["orig_width"]),
            ),
            mode="bilinear",
            align_corners=False,
        )
        return video_res_mask[0] > 0.0

    def _gather_obj_id_to_mask_across_gpus(
        self,
        inference_state: dict[str, Any],
        obj_id_to_mask_local: dict[Any, Any],
    ) -> Any:
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity."
                "_gather_obj_id_to_mask_across_gpus(distributed)"
            )

        tracker_metadata = inference_state.get("tracker_metadata", {})
        obj_ids_per_gpu = tracker_metadata.get("obj_ids_per_gpu", [])
        if self.rank >= len(obj_ids_per_gpu):
            raise ValueError(
                f"rank={self.rank} is out of range for obj_ids_per_gpu "
                f"with {len(obj_ids_per_gpu)} entries."
            )

        h_mask = w_mask = int(self.tracker.low_res_mask_size)
        low_res_masks_local = []
        for obj_id in obj_ids_per_gpu[self.rank]:
            if obj_id in obj_id_to_mask_local:
                low_res_mask = obj_id_to_mask_local[obj_id]
                low_res_mask_mx = (
                    low_res_mask
                    if _is_mlx_array(low_res_mask)
                    else mx.array(low_res_mask)
                )
                if low_res_mask_mx.shape != (h_mask, w_mask):
                    raise ValueError(
                        "Each low-res mask must have shape "
                        f"({h_mask}, {w_mask}), got {low_res_mask_mx.shape} "
                        f"for obj_id {obj_id}."
                    )
                low_res_masks_local.append(low_res_mask_mx.astype(mx.float32))
            else:
                low_res_masks_local.append(
                    mx.full((h_mask, w_mask), NO_OBJ_SCORE, dtype=mx.float32)
                )

        if not low_res_masks_local:
            return mx.zeros((0, h_mask, w_mask), dtype=mx.float32)
        return mx.stack(low_res_masks_local, axis=0)

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
                        "Tracker partial propagation returned frame_idx="
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

    def clear_detector_added_cond_frame_in_sam2(
        self,
        sam2_state: dict[str, Any],
        obj_id: Any,
        refined_frame_idx: int,
    ) -> None:
        obj_idx = self._lookup_existing_obj_idx(sam2_state, obj_id)
        if obj_idx is None:
            return

        mask_inputs_per_obj = sam2_state.get("mask_inputs_per_obj", {})
        mask_inputs_per_frame = mask_inputs_per_obj.get(obj_idx, {})
        if not mask_inputs_per_frame:
            return

        point_inputs_per_obj = sam2_state.get("point_inputs_per_obj", {})
        point_frame_indices = set(point_inputs_per_obj.get(obj_idx, {}))
        window = int(self.refinement_detector_cond_frame_removal_window)
        frame_indices_to_clear = [
            int(frame_idx)
            for frame_idx in mask_inputs_per_frame
            if frame_idx not in point_frame_indices
            and abs(int(frame_idx) - int(refined_frame_idx)) <= window
        ]

        obj_ids_on_state = list(sam2_state.get("obj_id_to_idx", {}).keys())
        for frame_idx in frame_indices_to_clear:
            for obj_id_to_clear in obj_ids_on_state:
                self.tracker.clear_all_points_in_frame(
                    sam2_state,
                    frame_idx,
                    obj_id_to_clear,
                    need_output=False,
                )

    def propagate_in_video(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            if "inference_state" in kwargs:
                raise TypeError(
                    "inference_state passed both positionally and by keyword."
                )
            kwargs["inference_state"] = args[0]
            if len(args) > 1:
                raise TypeError(
                    "propagate_in_video accepts at most one positional argument."
                )
        inference_state = kwargs.pop("inference_state")
        start_frame_idx = kwargs.pop("start_frame_idx", None)
        max_frame_num_to_track = kwargs.pop("max_frame_num_to_track", None)
        reverse = kwargs.pop("reverse", False)
        kwargs.pop("output_prob_thresh", 0.5)
        kwargs.pop("compute_stability_score", False)
        is_instance_processing = kwargs.pop("is_instance_processing", False)
        is_last_batch = kwargs.pop("is_last_batch", True)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected propagate_in_video keyword argument(s): {names}"
            )

        propagation_type, obj_ids = self.parse_action_history_for_propagation(
            inference_state
        )
        self.add_action_history(
            inference_state,
            action_type=propagation_type,
            obj_ids=obj_ids,
            frame_idx=start_frame_idx,
        )

        if propagation_type == "propagation_full":
            yield from self._propagate_in_video_impl(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
                is_instance_processing=is_instance_processing,
                flush_hotstart_at_end=is_last_batch,
            )
            return

        processing_order, _ = self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )
        if propagation_type == "propagation_fetch":
            for frame_idx in processing_order:
                if self.rank == 0:
                    yield self.fetch_and_process_single_frame_results(
                        inference_state,
                        frame_idx,
                    )
                else:
                    yield frame_idx, DUMMY_OUTPUT
            return

        if propagation_type != "propagation_partial":
            raise ValueError(f"Unexpected propagation_type={propagation_type!r}.")
        if self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexTrackingWithInteractivity."
                "propagate_in_video(propagation_partial-distributed)"
            )
        if obj_ids is None:
            raise ValueError("propagation_partial requires obj_ids.")

        tracker_states_local = self._get_sam2_inference_states_by_obj_ids(
            inference_state,
            obj_ids,
        )
        for sam2_state in tracker_states_local:
            self.tracker.propagate_in_video_preflight(
                sam2_state,
                run_mem_encoder=True,
            )

        tracker_metadata = inference_state["tracker_metadata"]
        framewise_sam2_scores = tracker_metadata.setdefault(
            "obj_id_to_sam2_score_frame_wise",
            defaultdict(dict),
        )
        for frame_idx in processing_order:
            obj_ids_local, low_res_masks_local, sam2_scores_local = (
                self._propogate_tracker_one_frame_local_gpu(
                    tracker_states_local,
                    frame_idx=frame_idx,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
            )
            refined_obj_data = {
                obj_id: (sam2_scores_local[obj_idx], low_res_masks_local[obj_idx])
                for obj_idx, obj_id in enumerate(obj_ids_local)
                if obj_id in obj_ids
            }
            frame_scores = framewise_sam2_scores.setdefault(frame_idx, {})
            for obj_id, (refined_score, _) in refined_obj_data.items():
                frame_scores[obj_id] = refined_score

            if self.rank != 0:
                yield frame_idx, DUMMY_OUTPUT
                continue

            refined_obj_id_to_mask = {
                obj_id: self._convert_low_res_mask_to_video_res(
                    refined_mask_low_res,
                    inference_state,
                )
                for obj_id, (_, refined_mask_low_res) in refined_obj_data.items()
            }
            obj_id_to_mask = self._build_sam2_output(
                inference_state,
                frame_idx,
                refined_obj_id_to_mask,
            )
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})
            suppressed_obj_ids = rank0_metadata.get("suppressed_obj_ids", {}).get(
                frame_idx,
                set(),
            )
            self._cache_frame_outputs(
                inference_state,
                frame_idx,
                obj_id_to_mask,
                suppressed_obj_ids=suppressed_obj_ids,
            )
            obj_id_to_score = tracker_metadata.get("obj_id_to_score", {})
            yield (
                frame_idx,
                self._postprocess_output(
                    inference_state,
                    {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": {
                            obj_id: obj_id_to_score.get(
                                obj_id,
                                frame_scores.get(obj_id, 0.0),
                            )
                            for obj_id in obj_id_to_mask
                        },
                        "obj_id_to_sam2_score": frame_scores,
                    },
                    suppressed_obj_ids=suppressed_obj_ids,
                ),
            )
