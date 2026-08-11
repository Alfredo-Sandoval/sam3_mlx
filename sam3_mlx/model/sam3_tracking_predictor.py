# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX interactive tracker predictor.

Ported from the official SAM3 ``sam3_tracking_predictor`` at upstream commit
``2814fa619404a722d03e9a012e083e4f293a4e53``.

Tracker port status:

- Cached-feature point/box prompting and forward propagation are
  implemented on top of the ported ``Sam3TrackerBase.track_step`` slices.
- Video decoding, direct mask prompts, and object-batched forward propagation
  are implemented on top of cached or loaded MLX frames. CPU/state offload
  remains an explicit fail-fast boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from collections import OrderedDict
from pathlib import Path
from typing import (
    Literal,
    NoReturn,
    NotRequired,
    Required,
    TypeAlias,
    TypeVar,
    TypedDict,
    TypeGuard,
    Unpack,
    cast,
    overload,
)

import mlx.core as mx
import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.mlx_runtime import evaluate_boundary
from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.model.io_utils import load_resource_as_video_frames
from sam3_mlx.model.sam3_tracker_base import (
    BackboneFeatureOutput,
    NO_OBJ_SCORE,
    PointInputs,
    Sam3TrackerBase,
    TrackerOutputState,
    TrackerBaseOptions,
    TrackerStoredFrameOutput,
    concat_points,
)
from sam3_mlx.model.sam3_tracker_utils import fill_holes_in_mask_scores


ArrayInput: TypeAlias = (
    mx.array | np.ndarray | list[object] | tuple[object, ...] | bool | int | float
)
ObjectId: TypeAlias = object
StorageKey: TypeAlias = Literal["cond_frame_outputs", "non_cond_frame_outputs"]
ImageFrames: TypeAlias = mx.array | list[ArrayInput] | tuple[ArrayInput, ...]


class _TrackerBackboneWrapper(TypedDict):
    tracker_backbone_out: BackboneFeatureOutput


CachedBackboneOutput: TypeAlias = BackboneFeatureOutput | _TrackerBackboneWrapper
CachedFeature: TypeAlias = (
    tuple[mx.array, CachedBackboneOutput]
    | tuple[
        mx.array,
        CachedBackboneOutput,
        list[mx.array],
        list[mx.array],
        list[tuple[int, int]],
    ]
)


class _TemporaryFrameOutput(TypedDict):
    pred_masks: Required[mx.array | None]
    pred_masks_video_res: NotRequired[mx.array]
    obj_ptr: Required[mx.array]
    object_score_logits: Required[mx.array]
    iou_score: NotRequired[mx.array]
    eff_iou_score: NotRequired[mx.array]
    maskmem_features: Required[mx.array | None]
    maskmem_pos_enc: Required[list[mx.array] | None]


class _TemporaryOutputState(TypedDict):
    cond_frame_outputs: dict[int, _TemporaryFrameOutput]
    non_cond_frame_outputs: dict[int, _TemporaryFrameOutput]


class _VideoConsolidatedOutput(TypedDict):
    pred_masks_video_res: mx.array
    obj_ptr: mx.array
    object_score_logits: mx.array
    iou_score: NotRequired[mx.array]
    eff_iou_score: NotRequired[mx.array]
    maskmem_features: None
    maskmem_pos_enc: None


class _FrameTrackingInfo(TypedDict):
    reverse: bool


class _ConsolidatedFrameIndices(TypedDict):
    cond_frame_outputs: set[int]
    non_cond_frame_outputs: set[int]


class _ModelConstants(TypedDict, total=False):
    maskmem_pos_enc: list[mx.array]


class InferenceState(TypedDict):
    offload_video_to_cpu: bool
    offload_state_to_cpu: bool
    device: str
    storage_device: str
    video_height: int
    video_width: int
    num_frames: int
    point_inputs_per_obj: dict[int, dict[int, PointInputs]]
    mask_inputs_per_obj: dict[int, dict[int, mx.array]]
    images: ImageFrames | None
    cached_features: dict[int, CachedFeature]
    constants: _ModelConstants
    obj_id_to_idx: OrderedDict[ObjectId, int]
    obj_idx_to_id: OrderedDict[int, ObjectId]
    obj_ids: list[ObjectId]
    output_dict: TrackerOutputState
    first_ann_frame_idx: int | None
    output_dict_per_obj: dict[int, TrackerOutputState]
    temp_output_dict_per_obj: dict[int, _TemporaryOutputState]
    consolidated_frame_inds: _ConsolidatedFrameIndices
    tracking_has_started: bool
    frames_already_tracked: dict[int, _FrameTrackingInfo]


def _unsupported_tracking_predictor(
    method: str, *, detail: str | None = None
) -> NoReturn:
    raise_unsupported(
        f"sam3_mlx.model.sam3_tracking_predictor.Sam3TrackerPredictor.{method}",
        reason="video-tracker",
        detail=detail
        or (
            "This tracker predictor slice is outside the current MLX runtime increment."
        ),
        alternative=(
            "sam3_mlx.model.sam3_tracker_utils or "
            "sam3_mlx.model.sam3_video_inference.Sam3VideoInference"
        ),
    )


def _is_mlx_array(value: object) -> TypeGuard[mx.array]:
    return type(value).__module__.startswith("mlx.")


def _as_mlx_array(value: ArrayInput, *, dtype: mx.Dtype) -> mx.array:
    if _is_mlx_array(value):
        return value.astype(dtype)
    return mx.array(value, dtype=dtype)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, np.integer):
        normalized = value.item()
    else:
        raise ValueError(f"{name} must be a positive integer.")
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _eval_tree(*values: object) -> None:
    arrays: list[mx.array] = []

    def visit(value: object) -> None:
        if _is_mlx_array(value):
            arrays.append(value)
        elif isinstance(value, dict):
            for child in cast(dict[object, object], value).values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in cast(Sequence[object], value):
                visit(child)

    for value in values:
        visit(value)
    if arrays:
        evaluate_boundary(*arrays)


@overload
def _copy_output_slice(value: None, obj_slice: slice) -> None: ...


@overload
def _copy_output_slice(value: mx.array, obj_slice: slice) -> mx.array: ...


@overload
def _copy_output_slice(value: list[mx.array], obj_slice: slice) -> list[mx.array]: ...


def _copy_output_slice(
    value: mx.array | list[mx.array] | None,
    obj_slice: slice,
) -> mx.array | list[mx.array] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [x[obj_slice] for x in value]
    return value[obj_slice]


def _coerce_obj_id_list(obj_ids: object) -> list[ObjectId]:
    if isinstance(obj_ids, (str, bytes)):
        return [obj_ids]
    if _is_mlx_array(obj_ids):
        evaluate_boundary(obj_ids)
        return cast(list[ObjectId], np.asarray(obj_ids).reshape(-1).tolist())
    if isinstance(obj_ids, np.ndarray):
        return cast(list[ObjectId], obj_ids.reshape(-1).tolist())
    try:
        return list(cast(Iterable[object], obj_ids))
    except TypeError:
        return [obj_ids]


@overload
def _take_output_indices(value: None, obj_indices: list[int]) -> None: ...


@overload
def _take_output_indices(value: mx.array, obj_indices: list[int]) -> mx.array: ...


@overload
def _take_output_indices(
    value: list[mx.array], obj_indices: list[int]
) -> list[mx.array]: ...


def _take_output_indices(
    value: mx.array | list[mx.array] | None,
    obj_indices: list[int],
) -> mx.array | list[mx.array] | None:
    if value is None:
        return None
    index = mx.array(obj_indices, dtype=mx.int64)
    if isinstance(value, list):
        return [mx.take(x, index, axis=0) for x in value]
    return mx.take(value, index, axis=0)


def _has_stored_pred_masks(
    output: _TemporaryFrameOutput,
) -> TypeGuard[TrackerStoredFrameOutput]:
    return output["pred_masks"] is not None


_ValueT = TypeVar("_ValueT")


def _remap_integer_keys(
    container: dict[int, _ValueT],
    old_to_new: dict[int, int],
    retained_old_keys: list[int],
) -> None:
    new_values = {
        old_to_new[old_key]: container[old_key]
        for old_key in retained_old_keys
        if old_key in container
    }
    container.clear()
    container.update(new_values)


def _validate_cached_features(
    cached_features: object | None,
) -> dict[int, CachedFeature]:
    if cached_features is None:
        return {}
    if not isinstance(cached_features, dict):
        raise TypeError("cached_features must be a dict keyed by frame index.")
    return cast(dict[int, CachedFeature], cached_features)


def _broadcast_batch(value: mx.array, batch_size: int) -> mx.array:
    if value.shape[0] == batch_size:
        return value
    if value.shape[0] != 1:
        raise ValueError(
            f"Cannot broadcast batch dimension {value.shape[0]} to {batch_size}."
        )
    return mx.broadcast_to(value, (batch_size,) + tuple(value.shape[1:]))


def _broadcast_backbone_out(
    backbone_out: BackboneFeatureOutput,
    batch_size: int,
) -> BackboneFeatureOutput:
    expanded = backbone_out.copy()
    expanded["backbone_fpn"] = [
        _broadcast_batch(feature, batch_size)
        for feature in backbone_out["backbone_fpn"]
    ]
    expanded["vision_pos_enc"] = [
        _broadcast_batch(pos, batch_size) for pos in backbone_out["vision_pos_enc"]
    ]
    return expanded


def _broadcast_flat_features(
    features: list[mx.array], batch_size: int
) -> list[mx.array]:
    expanded: list[mx.array] = []
    for feature in features:
        if feature.shape[1] == batch_size:
            expanded.append(feature)
        elif feature.shape[1] == 1:
            expanded.append(
                mx.broadcast_to(
                    feature,
                    (feature.shape[0], batch_size) + tuple(feature.shape[2:]),
                )
            )
        else:
            raise ValueError(
                f"Cannot broadcast feature batch {feature.shape[1]} to {batch_size}."
            )
    return expanded


class Sam3TrackerPredictor(Sam3TrackerBase):
    """
    Official-name interactive tracker predictor for the MLX tracker slice.

    The supported runtime path requires precomputed per-frame visual features in
    ``inference_state["cached_features"]`` or pre-loaded MLX frames plus a
    tracker backbone. It intentionally does not fall back to Torch-only or video
    decoding.
    """

    def __init__(
        self,
        backbone: object | None = None,
        transformer: object | None = None,
        maskmem_backbone: object | None = None,
        clear_non_cond_mem_around_input: bool = False,
        clear_non_cond_mem_for_multi_obj: bool = False,
        fill_hole_area: int = 0,
        always_start_from_first_ann_frame: bool = False,
        max_point_num_in_prompt_enc: int = 16,
        non_overlap_masks_for_output: bool = True,
        **kwargs: Unpack[TrackerBaseOptions],
    ) -> None:
        if transformer is None or maskmem_backbone is None:
            from sam3_mlx.model_builder import (
                create_tracker_maskmem_backbone,
                create_tracker_transformer,
            )

            if transformer is None:
                transformer = create_tracker_transformer()
            if maskmem_backbone is None:
                maskmem_backbone = create_tracker_maskmem_backbone()

        if clear_non_cond_mem_for_multi_obj and not clear_non_cond_mem_around_input:
            _unsupported_tracking_predictor(
                "__init__(clear_non_cond_mem_for_multi_obj=True)",
                detail=(
                    "clear_non_cond_mem_for_multi_obj only has an effect when "
                    "clear_non_cond_mem_around_input=True."
                ),
            )
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            maskmem_backbone=maskmem_backbone,
            **kwargs,
        )
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output
        self.iter_use_prev_mask_pred = True
        self.add_all_frames_to_correct_as_cond = True

    def init_state(
        self,
        video_height: int | None = None,
        video_width: int | None = None,
        num_frames: int | None = None,
        video_path: str | Path | None = None,
        images: ImageFrames | None = None,
        cached_features: dict[int, CachedFeature] | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> InferenceState:
        """Initialize a cached-feature inference state."""
        if offload_video_to_cpu or offload_state_to_cpu:
            _unsupported_tracking_predictor(
                "init_state(offload)",
                detail="Torch-style video/state offload is not used by the MLX runtime.",
            )
        if async_loading_frames:
            _unsupported_tracking_predictor(
                "init_state(async_loading_frames)",
                detail="Async TorchCodec-style frame loading is not ported.",
            )
        if video_path is not None:
            loaded_frames = load_resource_as_video_frames(
                resource_path=video_path,
                image_size=self.image_size,
                offload_video_to_cpu=False,
                img_mean=(0.5, 0.5, 0.5),
                img_std=(0.5, 0.5, 0.5),
                async_loading_frames=False,
                video_loader_type="cv2",
            )
            frame_images = getattr(loaded_frames, "images", None)
            if frame_images is None:
                raise ValueError(
                    "Loaded video frames must expose normalized MLX images."
                )
            if images is not None:
                raise ValueError("Provide either video_path or images, not both.")
            images = frame_images
            if video_height is None:
                video_height = int(loaded_frames.orig_height)
            if video_width is None:
                video_width = int(loaded_frames.orig_width)
            if num_frames is None:
                num_frames = len(loaded_frames)
        if images is not None:
            if _is_mlx_array(images):
                if len(images.shape) != 4:
                    raise ValueError("images must have shape [N, C, H, W].")
                num_image_frames = int(images.shape[0])
                image_height = int(images.shape[-2])
                image_width = int(images.shape[-1])
            elif isinstance(images, (list, tuple)):
                if len(images) == 0:
                    raise ValueError("images must contain at least one frame.")
                num_image_frames = len(images)
                first_image = _as_mlx_array(images[0], dtype=mx.float32)
                if len(first_image.shape) not in (3, 4):
                    raise ValueError(
                        "image frames must have shape [C, H, W] or [1, C, H, W]."
                    )
                image_height = int(first_image.shape[-2])
                image_width = int(first_image.shape[-1])
            else:
                raise TypeError(
                    "images must be an MLX array or a list/tuple of frames."
                )
            if num_frames is None:
                num_frames = num_image_frames
            elif int(num_frames) != num_image_frames:
                raise ValueError(
                    "num_frames must match len(images). Got "
                    f"{num_frames} and {num_image_frames}."
                )
            if video_height is None:
                video_height = image_height
            if video_width is None:
                video_width = image_width

        if video_height is None or video_width is None or num_frames is None:
            raise ValueError(
                "video_height, video_width, and num_frames are required when "
                "video_path and images are not provided."
            )
        video_height = _positive_integer(video_height, name="video_height")
        video_width = _positive_integer(video_width, name="video_width")
        num_frames = _positive_integer(num_frames, name="num_frames")
        typed_cached_features = _validate_cached_features(cached_features)

        inference_state = InferenceState(
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            device=self.device,
            storage_device=self.device,
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            point_inputs_per_obj={},
            mask_inputs_per_obj={},
            images=images,
            cached_features=typed_cached_features,
            constants={},
            obj_id_to_idx=OrderedDict(),
            obj_idx_to_id=OrderedDict(),
            obj_ids=[],
            output_dict={
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            first_ann_frame_idx=None,
            output_dict_per_obj={},
            temp_output_dict_per_obj={},
            consolidated_frame_inds={
                "cond_frame_outputs": set(),
                "non_cond_frame_outputs": set(),
            },
            tracking_has_started=False,
            frames_already_tracked={},
        )
        self.clear_all_points_in_video(inference_state)
        return inference_state

    def _obj_id_to_idx(self, inference_state: InferenceState, obj_id: ObjectId) -> int:
        """Map a client object id to an MLX object slot."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id)
        if obj_idx is not None:
            return obj_idx

        if inference_state["tracking_has_started"]:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        obj_idx = len(inference_state["obj_id_to_idx"])
        inference_state["obj_id_to_idx"][obj_id] = obj_idx
        inference_state["obj_idx_to_id"][obj_idx] = obj_id
        inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
        inference_state["point_inputs_per_obj"][obj_idx] = {}
        inference_state["mask_inputs_per_obj"][obj_idx] = {}
        inference_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        inference_state["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        return obj_idx

    def _obj_idx_to_id(self, inference_state: InferenceState, obj_idx: int) -> ObjectId:
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state: InferenceState) -> int:
        """Get the number of active object ids in the session."""
        return len(inference_state["obj_idx_to_id"])

    def add_new_points_or_box(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        obj_id: ObjectId,
        points: ArrayInput | None = None,
        labels: ArrayInput | None = None,
        clear_old_points: bool = True,
        rel_coordinates: bool = True,
        use_prev_mem_frame: bool = False,
        normalize_coords: bool = True,
        box: ArrayInput | None = None,
    ) -> tuple[int, list[ObjectId], None, mx.array]:
        """Add point or box prompts for the single supported object."""
        del normalize_coords  # retained for upstream signature compatibility
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < inference_state["num_frames"]:
            raise ValueError(f"frame_idx {frame_idx} is outside the video frame range.")

        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = mx.zeros((0, 2), dtype=mx.float32)
        else:
            points = _as_mlx_array(points, dtype=mx.float32)
        if labels is None:
            labels = mx.zeros((0,), dtype=mx.int32)
        else:
            labels = _as_mlx_array(labels, dtype=mx.int32)

        if len(points.shape) == 2:
            points = points[None]
        if len(labels.shape) == 1:
            labels = labels[None]
        if tuple(points.shape[:2]) != tuple(labels.shape):
            raise ValueError(
                "points must have shape [B, P, 2] and labels must have shape [B, P]."
            )
        if points.shape[0] != 1:
            _unsupported_tracking_predictor(
                "add_new_points_or_box(batch_size)",
                detail="The MLX predictor increment supports one object/batch row.",
            )

        if rel_coordinates:
            points = points * self.image_size

        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since box prompt "
                    "must be provided before any point prompt (please use "
                    "clear_old_points=True instead)"
                )
            box = _as_mlx_array(box, dtype=mx.float32)
            if rel_coordinates:
                box = box * self.image_size
            box_coords = mx.reshape(box, (1, 2, 2))
            box_labels = mx.array([[2, 3]], dtype=mx.int32)
            points = mx.concat([box_coords, points], axis=1)
            labels = mx.concat([box_labels, labels], axis=1)

        if clear_old_points:
            point_inputs = None
        else:
            point_inputs = point_inputs_per_frame.get(frame_idx)
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_init_cond_frame
            else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        )
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        num_points = point_inputs["point_coords"].shape[1]
        if num_points > self.max_point_num_in_prompt_enc > 0:
            num_first = self.max_point_num_in_prompt_enc // 2
            num_last = self.max_point_num_in_prompt_enc - num_first
            point_inputs["point_coords"] = mx.concat(
                [
                    point_inputs["point_coords"][:, :num_first],
                    point_inputs["point_coords"][:, -num_last:],
                ],
                axis=1,
            )
            point_inputs["point_labels"] = mx.concat(
                [
                    point_inputs["point_labels"][:, :num_first],
                    point_inputs["point_labels"][:, -num_last:],
                ],
                axis=1,
            )

        prev_sam_mask_logits = None
        if self.iter_use_prev_mask_pred:
            prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            if prev_out is not None and prev_out["pred_masks"] is not None:
                prev_sam_mask_logits = mx.clip(prev_out["pred_masks"], -32.0, 32.0)

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            use_prev_mem_frame=use_prev_mem_frame,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, None, video_res_masks

    def add_new_mask(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        obj_id: ObjectId,
        mask: ArrayInput,
        add_mask_to_memory: bool = False,
    ) -> tuple[int, list[ObjectId], None, mx.array]:
        """Add a binary mask prompt for the single supported object."""
        del add_mask_to_memory  # official keeps this future-facing flag unused here
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < inference_state["num_frames"]:
            raise ValueError(f"frame_idx {frame_idx} is outside the video frame range.")

        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        mask = _as_mlx_array(mask, dtype=mx.float32)
        if len(mask.shape) != 2:
            raise ValueError("mask must have shape [H, W].")

        mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])
        mask_inputs_orig = mask[None, None]
        if (mask_h, mask_w) != (self.input_mask_size, self.input_mask_size):
            mask_inputs = interpolate(
                mask_inputs_orig,
                size=(self.input_mask_size, self.input_mask_size),
                align_corners=False,
                mode="bilinear",
            )
        else:
            mask_inputs = mask_inputs_orig

        video_h = inference_state["video_height"]
        video_w = inference_state["video_width"]
        if (mask_h, mask_w) != (video_h, video_w):
            mask_inputs_video_res = interpolate(
                mask_inputs_orig,
                size=(video_h, video_w),
                align_corners=False,
                mode="bilinear",
            )
        else:
            mask_inputs_video_res = mask_inputs_orig
        mask_inputs_video_res = mask_inputs_video_res > 0.5

        mask_inputs_per_frame[frame_idx] = mask_inputs_video_res
        point_inputs_per_frame.pop(frame_idx, None)

        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_init_cond_frame
            else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        )
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
        )
        current_out["pred_masks"] = None
        current_out["pred_masks_video_res"] = mx.where(
            mask_inputs_video_res,
            -NO_OBJ_SCORE,
            NO_OBJ_SCORE,
        ).astype(mx.float32)
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        for obj_idx2, obj_temp_output_dict2 in temp_output_dict_per_obj.items():
            if obj_idx2 == obj_idx:
                continue
            current_out2 = obj_temp_output_dict2[storage_key].get(frame_idx)
            if current_out2 is not None and "pred_masks_video_res" in current_out2:
                current_out2["pred_masks_video_res"] = mx.where(
                    mask_inputs_video_res,
                    NO_OBJ_SCORE,
                    current_out2["pred_masks_video_res"],
                )

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, None, video_res_masks

    def add_new_points(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        obj_id: ObjectId,
        points: ArrayInput | None = None,
        labels: ArrayInput | None = None,
        clear_old_points: bool = True,
        rel_coordinates: bool = True,
        use_prev_mem_frame: bool = False,
        normalize_coords: bool = True,
        box: ArrayInput | None = None,
    ) -> tuple[int, list[ObjectId], None, mx.array]:
        return self.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points,
            rel_coordinates=rel_coordinates,
            use_prev_mem_frame=use_prev_mem_frame,
            normalize_coords=normalize_coords,
            box=box,
        )

    def _get_orig_video_res_output(
        self, inference_state: InferenceState, any_res_masks: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Resize mask logits to the original video resolution."""
        video_h = inference_state["video_height"]
        video_w = inference_state["video_width"]
        if tuple(any_res_masks.shape[-2:]) == (video_h, video_w):
            video_res_masks = any_res_masks
        else:
            video_res_masks = interpolate(
                any_res_masks.astype(mx.float32),
                size=(video_h, video_w),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks_for_output:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        if self.fill_hole_area > 0:
            video_res_masks = fill_holes_in_mask_scores(
                video_res_masks,
                self.fill_hole_area,
            )
        _eval_tree(any_res_masks, video_res_masks)
        return any_res_masks, video_res_masks

    @overload
    def _consolidate_temp_output_across_obj(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        is_cond: bool,
        run_mem_encoder: bool,
        consolidate_at_video_res: Literal[False] = False,
    ) -> TrackerStoredFrameOutput: ...

    @overload
    def _consolidate_temp_output_across_obj(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        is_cond: bool,
        run_mem_encoder: bool,
        consolidate_at_video_res: Literal[True],
    ) -> _VideoConsolidatedOutput: ...

    def _consolidate_temp_output_across_obj(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        is_cond: bool,
        run_mem_encoder: bool,
        consolidate_at_video_res: bool = False,
    ) -> TrackerStoredFrameOutput | _VideoConsolidatedOutput:
        """Consolidate per-object temporary outputs into one batched output."""
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        if consolidate_at_video_res:
            consolidated_h = inference_state["video_height"]
            consolidated_w = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_h = consolidated_w = self.low_res_mask_size
            consolidated_mask_key = "pred_masks"

        pred_masks: list[mx.array] = []
        obj_ptrs: list[mx.array] = []
        object_scores: list[mx.array] = []
        iou_scores: list[mx.array] = []
        eff_iou_scores: list[mx.array] = []
        empty_mask_ptr: mx.array | None = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx)
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

            if out is None:
                pred_masks.append(
                    mx.full(
                        (1, 1, consolidated_h, consolidated_w),
                        NO_OBJ_SCORE,
                        dtype=mx.float32,
                    )
                )
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state,
                            frame_idx,
                        )
                    obj_ptrs.append(empty_mask_ptr)
                else:
                    obj_ptrs.append(
                        mx.full((1, self.hidden_dim), NO_OBJ_SCORE, dtype=mx.float32)
                    )
                object_scores.append(mx.full((1, 1), 10.0, dtype=mx.float32))
                if self.use_memory_selection:
                    iou_scores.append(mx.zeros((1, 1), dtype=mx.float32))
                continue

            obj_mask = (
                out["pred_masks_video_res"]
                if "pred_masks_video_res" in out
                else out["pred_masks"]
            )
            if obj_mask is None:
                raise RuntimeError(
                    f"No mask output exists for obj_idx={obj_idx} on frame {frame_idx}."
                )
            if tuple(obj_mask.shape[-2:]) != (consolidated_h, consolidated_w):
                obj_mask = interpolate(
                    obj_mask.astype(mx.float32),
                    size=(consolidated_h, consolidated_w),
                    mode="bilinear",
                    align_corners=False,
                )
            pred_masks.append(obj_mask.astype(mx.float32))
            obj_ptrs.append(out["obj_ptr"])
            object_scores.append(out["object_score_logits"])
            if self.use_memory_selection:
                if "iou_score" not in out:
                    raise RuntimeError(
                        "Memory-selection output is missing its IoU score."
                    )
                iou_scores.append(out["iou_score"])
                if "eff_iou_score" in out:
                    eff_iou_scores.append(out["eff_iou_score"])

        consolidated_masks = mx.concat(pred_masks, axis=0)
        if consolidate_at_video_res:
            consolidated_out: TrackerStoredFrameOutput | _VideoConsolidatedOutput = (
                _VideoConsolidatedOutput(
                    maskmem_features=None,
                    maskmem_pos_enc=None,
                    pred_masks_video_res=consolidated_masks,
                    obj_ptr=mx.concat(obj_ptrs, axis=0),
                    object_score_logits=mx.concat(object_scores, axis=0),
                )
            )
        else:
            consolidated_out = TrackerStoredFrameOutput(
                maskmem_features=None,
                maskmem_pos_enc=None,
                pred_masks=consolidated_masks,
                obj_ptr=mx.concat(obj_ptrs, axis=0),
                object_score_logits=mx.concat(object_scores, axis=0),
            )
        if self.use_memory_selection:
            consolidated_out["iou_score"] = mx.concat(iou_scores, axis=0)
            if len(eff_iou_scores) == batch_size:
                consolidated_out["eff_iou_score"] = mx.concat(eff_iou_scores, axis=0)

        if run_mem_encoder:
            if consolidated_mask_key != "pred_masks":
                raise AssertionError("memory encoder cannot run at video resolution")
            if "pred_masks" not in consolidated_out:
                raise AssertionError("memory encoder requires low-resolution masks")
            stored_consolidated_out = consolidated_out
            pred_masks_for_memory = stored_consolidated_out["pred_masks"]
            high_res_masks = interpolate(
                pred_masks_for_memory.astype(mx.float32),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=stored_consolidated_out["object_score_logits"],
                is_mask_from_pts=True,
            )
            stored_consolidated_out["maskmem_features"] = maskmem_features
            stored_consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        _eval_tree(consolidated_out)
        return consolidated_out

    def _get_empty_mask_ptr(
        self, inference_state: InferenceState, frame_idx: int
    ) -> mx.array:
        batch_size = 1
        mask_inputs = mx.zeros(
            (batch_size, 1, self.image_size, self.image_size),
            dtype=mx.float32,
        )
        (
            image,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def propagate_in_video_preflight(
        self, inference_state: InferenceState, run_mem_encoder: bool = True
    ) -> None:
        """Merge temporary prompt outputs into committed tracking state."""
        batch_size = self._get_obj_num(inference_state)
        if batch_size == 0:
            raise RuntimeError("No points are provided; please add points first")
        inference_state["tracking_has_started"] = True

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            temp_frame_inds: set[int] = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=run_mem_encoder,
                )
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds: set[int] = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = min(
                input_frames_inds, default=None
            )
        if (
            inference_state["first_ann_frame_idx"]
            not in output_dict["cond_frame_outputs"]
        ):
            inference_state["first_ann_frame_idx"] = min(
                output_dict["cond_frame_outputs"], default=None
            )

    def _get_processing_order(
        self,
        inference_state: InferenceState,
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> range | list[int]:
        """Return the official frame processing order for propagation."""
        num_frames = inference_state["num_frames"]
        if self.always_start_from_first_ann_frame:
            start_frame_idx = inference_state["first_ann_frame_idx"]
        if start_frame_idx is None:
            start_frame_idx = min(inference_state["output_dict"]["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                return range(start_frame_idx, end_frame_idx - 1, -1)
            return [0]
        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        return range(start_frame_idx, end_frame_idx + 1)

    def _select_propagation_output_obj_ids(
        self,
        inference_state: InferenceState,
        obj_ids: object | None,
    ) -> tuple[list[ObjectId], list[int] | None]:
        if obj_ids is None:
            return list(inference_state["obj_ids"]), None

        requested_obj_ids = _coerce_obj_id_list(obj_ids)
        if not requested_obj_ids:
            raise ValueError("obj_ids must contain at least one object id.")

        duplicates: list[ObjectId] = []
        seen: list[ObjectId] = []
        for obj_id in requested_obj_ids:
            if obj_id in seen:
                duplicates.append(obj_id)
            else:
                seen.append(obj_id)
        if duplicates:
            raise ValueError(f"obj_ids contains duplicate object ids: {duplicates}.")

        obj_id_to_idx = inference_state["obj_id_to_idx"]
        missing = [
            obj_id for obj_id in requested_obj_ids if obj_id not in obj_id_to_idx
        ]
        if missing:
            raise ValueError(
                f"Unknown obj_ids {missing}; active object ids are "
                f"{inference_state['obj_ids']}."
            )
        return requested_obj_ids, [
            obj_id_to_idx[obj_id] for obj_id in requested_obj_ids
        ]

    def propagate_in_video(
        self,
        inference_state: InferenceState,
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        tqdm_disable: bool = False,
        obj_ids: object | None = None,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = False,
    ) -> Iterator[tuple[int, list[ObjectId], mx.array, mx.array, mx.array]]:
        """Propagate prompted objects across cached video features."""
        del tqdm_disable  # no progress-bar dependency in the MLX runtime package
        if propagate_preflight:
            self.propagate_in_video_preflight(
                inference_state, run_mem_encoder=run_mem_encoder
            )

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        batch_size = self._get_obj_num(inference_state)
        output_obj_ids, output_obj_indices = self._select_propagation_output_obj_ids(
            inference_state,
            obj_ids,
        )
        processing_order = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
        )

        for frame_idx in processing_order:
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
                if self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                ):
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
                if not _has_stored_pred_masks(current_out):
                    raise RuntimeError(
                        "Propagation inference returned no low-resolution masks."
                    )
                obj_scores = current_out["object_score_logits"]
                output_dict[storage_key][frame_idx] = current_out

            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            low_res_masks, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            if output_obj_indices is not None:
                low_res_masks = _take_output_indices(low_res_masks, output_obj_indices)
                video_res_masks = _take_output_indices(
                    video_res_masks,
                    output_obj_indices,
                )
                obj_scores = _take_output_indices(obj_scores, output_obj_indices)
                _eval_tree(low_res_masks, video_res_masks, obj_scores)
            yield frame_idx, output_obj_ids, low_res_masks, video_res_masks, obj_scores

    def _add_output_per_object(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        current_out: TrackerStoredFrameOutput,
        storage_key: StorageKey,
    ) -> None:
        """Store per-object slices of a consolidated frame output."""
        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = TrackerStoredFrameOutput(
                maskmem_features=_copy_output_slice(
                    current_out["maskmem_features"], obj_slice
                ),
                maskmem_pos_enc=_copy_output_slice(
                    current_out["maskmem_pos_enc"], obj_slice
                ),
                pred_masks=current_out["pred_masks"][obj_slice],
                obj_ptr=current_out["obj_ptr"][obj_slice],
                object_score_logits=current_out["object_score_logits"][obj_slice],
            )
            if self.use_memory_selection:
                if "iou_score" not in current_out:
                    raise RuntimeError(
                        "Memory-selection output is missing its IoU score."
                    )
                obj_out["iou_score"] = current_out["iou_score"][obj_slice]
                if "eff_iou_score" in current_out:
                    obj_out["eff_iou_score"] = current_out["eff_iou_score"][obj_slice]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def clear_all_points_in_frame(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        obj_id: ObjectId,
        need_output: bool = True,
    ) -> tuple[int, list[ObjectId], None, mx.array] | None:
        """Remove point or mask input for one object on one frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return None
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, None, video_res_masks

    def clear_all_points_in_video(self, inference_state: InferenceState) -> None:
        """Remove all prompt inputs and outputs from the inference state."""
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state: InferenceState) -> None:
        """Reset prompts and tracking outputs while preserving object ids."""
        for value in inference_state["point_inputs_per_obj"].values():
            value.clear()
        for value in inference_state["mask_inputs_per_obj"].values():
            value.clear()
        for value in inference_state["output_dict_per_obj"].values():
            value["cond_frame_outputs"].clear()
            value["non_cond_frame_outputs"].clear()
        for value in inference_state["temp_output_dict_per_obj"].values():
            value["cond_frame_outputs"].clear()
            value["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _get_image_feature(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        batch_size: int,
    ) -> tuple[
        mx.array,
        BackboneFeatureOutput,
        list[mx.array],
        list[mx.array],
        list[tuple[int, int]],
    ]:
        """Retrieve or compute MLX visual features for one frame."""
        cached = inference_state["cached_features"].get(frame_idx)
        if cached is None:
            if self.backbone is None:
                raise RuntimeError(
                    f"Image features for frame {frame_idx} are not cached and "
                    "this tracker was built without a backbone."
                )
            images = inference_state.get("images")
            if images is None:
                raise RuntimeError(
                    f"Image features for frame {frame_idx} are not cached and "
                    "inference_state does not contain images."
                )
            if _is_mlx_array(images):
                image = images[frame_idx : frame_idx + 1].astype(mx.float32)
            else:
                image = _as_mlx_array(images[frame_idx], dtype=mx.float32)
                if len(image.shape) == 3:
                    image = image[None]
            backbone_out = self.forward_image(image)
            inference_state["cached_features"][frame_idx] = (image, backbone_out)
            cached = (image, backbone_out)

        if len(cached) == 5:
            image, backbone_out, vision_feats, vision_pos_embeds, feat_sizes = cached
            image = _broadcast_batch(image, batch_size)
            if "tracker_backbone_out" in backbone_out:
                backbone_out = backbone_out["tracker_backbone_out"]
            backbone_out = _broadcast_backbone_out(backbone_out, batch_size)
            vision_feats = _broadcast_flat_features(vision_feats, batch_size)
            vision_pos_embeds = _broadcast_flat_features(vision_pos_embeds, batch_size)
            _eval_tree(image, vision_feats, vision_pos_embeds)
            return image, backbone_out, vision_feats, vision_pos_embeds, feat_sizes
        if len(cached) != 2:
            raise TypeError(
                "cached_features values must be (image, backbone_out) or "
                "(image, backbone_out, vision_feats, vision_pos_embeds, feat_sizes)."
            )

        image, backbone_out = cached
        image = _broadcast_batch(image, batch_size)
        if "tracker_backbone_out" in backbone_out:
            backbone_out = backbone_out["tracker_backbone_out"]
        backbone_out = _broadcast_backbone_out(backbone_out, batch_size)
        backbone_out, vision_feats, vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )
        _eval_tree(image, vision_feats, vision_pos_embeds)
        return image, backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _run_single_frame_inference(
        self,
        inference_state: InferenceState,
        output_dict: TrackerOutputState,
        frame_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs: PointInputs | None,
        mask_inputs: mx.array | None,
        reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits: mx.array | None = None,
        use_prev_mem_frame: bool = True,
    ) -> tuple[_TemporaryFrameOutput, mx.array]:
        """Run ``track_step`` for one cached frame and compact its output."""
        (
            image,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            use_prev_mem_frame=use_prev_mem_frame,
        )

        pred_masks = current_out["pred_masks"]
        compact_current_out = _TemporaryFrameOutput(
            maskmem_features=current_out["maskmem_features"],
            maskmem_pos_enc=self._get_maskmem_pos_enc(inference_state, current_out),
            pred_masks=pred_masks,
            obj_ptr=current_out["obj_ptr"],
            object_score_logits=current_out["object_score_logits"],
        )
        if self.use_memory_selection:
            if "iou_score" not in current_out or "eff_iou_score" not in current_out:
                raise RuntimeError(
                    "Memory-selection tracking output is missing its score fields."
                )
            compact_current_out["iou_score"] = current_out["iou_score"]
            compact_current_out["eff_iou_score"] = current_out["eff_iou_score"]
        _eval_tree(compact_current_out, pred_masks)
        return compact_current_out, pred_masks

    def _run_memory_encoder(
        self,
        inference_state: InferenceState,
        frame_idx: int,
        batch_size: int,
        high_res_masks: mx.array,
        object_score_logits: mx.array,
        is_mask_from_pts: bool,
    ) -> tuple[mx.array, list[mx.array] | None]:
        """Run the ported memory encoder for a consolidated prompt frame."""
        image, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            image=image,
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
        )
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        _eval_tree(maskmem_features, maskmem_pos_enc)
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(
        self,
        inference_state: InferenceState,
        current_out: (
            _TemporaryFrameOutput
            | TrackerStoredFrameOutput
            | dict[str, list[mx.array] | None]
        ),
    ) -> list[mx.array] | None:
        """Cache one copy of mask memory positional encoding per session."""
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is None:
            return None
        if "maskmem_pos_enc" not in model_constants:
            model_constants["maskmem_pos_enc"] = [
                mx.array(x[0:1]) for x in out_maskmem_pos_enc
            ]
        maskmem_pos_enc = model_constants["maskmem_pos_enc"]
        batch_size = out_maskmem_pos_enc[0].shape[0]
        return [
            mx.broadcast_to(x, (batch_size,) + tuple(x.shape[1:]))
            for x in maskmem_pos_enc
        ]

    def remove_object(
        self,
        inference_state: InferenceState,
        obj_id: ObjectId,
        strict: bool = False,
        need_output: bool = True,
    ) -> tuple[list[ObjectId], list[tuple[int, mx.array]]]:
        """Remove one object id from the predictor state."""
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id)
        updated_frames: list[tuple[int, mx.array]] = []
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.clear_all_points_in_video(inference_state)
            return inference_state["obj_ids"], updated_frames

        obj_input_frames_inds: set[int] = set()
        obj_input_frames_inds.update(
            inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_points_in_frame(
                inference_state,
                frame_idx,
                obj_id,
                need_output=False,
            )

        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = [
            old_idx for old_idx in old_obj_inds if old_idx != old_obj_idx_to_rm
        ]
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = OrderedDict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = OrderedDict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        _remap_integer_keys(
            inference_state["point_inputs_per_obj"],
            old_idx_to_new_idx,
            remain_old_obj_inds,
        )
        _remap_integer_keys(
            inference_state["mask_inputs_per_obj"],
            old_idx_to_new_idx,
            remain_old_obj_inds,
        )
        _remap_integer_keys(
            inference_state["output_dict_per_obj"],
            old_idx_to_new_idx,
            remain_old_obj_inds,
        )
        _remap_integer_keys(
            inference_state["temp_output_dict_per_obj"],
            old_idx_to_new_idx,
            remain_old_obj_inds,
        )

        def slice_state(
            output_dict: TrackerOutputState, storage_key: StorageKey
        ) -> None:
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = _take_output_indices(
                    out["maskmem_features"], remain_old_obj_inds
                )
                out["maskmem_pos_enc"] = _take_output_indices(
                    out["maskmem_pos_enc"], remain_old_obj_inds
                )
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(
                    inference_state,
                    out,
                )
                out["pred_masks"] = _take_output_indices(
                    out["pred_masks"], remain_old_obj_inds
                )
                out["obj_ptr"] = _take_output_indices(
                    out["obj_ptr"], remain_old_obj_inds
                )
                out["object_score_logits"] = _take_output_indices(
                    out["object_score_logits"], remain_old_obj_inds
                )
                if self.use_memory_selection:
                    if "iou_score" not in out:
                        raise RuntimeError(
                            "Memory-selection output is missing its IoU score."
                        )
                    out["iou_score"] = _take_output_indices(
                        out["iou_score"], remain_old_obj_inds
                    )
                    out["eff_iou_score"] = self.cal_mem_score(
                        out["object_score_logits"],
                        out["iou_score"],
                    )
                self._add_output_per_object(
                    inference_state,
                    frame_idx,
                    out,
                    storage_key,
                )

        slice_state(inference_state["output_dict"], "cond_frame_outputs")
        slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    run_mem_encoder=False,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state,
                    consolidated_out["pred_masks_video_res"],
                )
                updated_frames.append((frame_idx, video_res_masks))

        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(
        self, inference_state: InferenceState, frame_idx: int
    ) -> None:
        """Clear nearby non-conditioning memories after a corrective input."""
        radius = self.memory_temporal_stride_for_eval * self.num_maskmem
        frame_idx_begin = int(frame_idx) - radius
        frame_idx_end = int(frame_idx) + radius
        batch_size = self._get_obj_num(inference_state)
        for obj_idx in range(batch_size):
            non_cond_outputs = inference_state["output_dict_per_obj"][obj_idx][
                "non_cond_frame_outputs"
            ]
            for time_idx in range(frame_idx_begin, frame_idx_end + 1):
                non_cond_outputs.pop(time_idx, None)

    def _suppress_shrinked_masks(self, *args: mx.array, **kwargs: float) -> mx.array:
        pred_masks, new_pred_masks = args[:2]
        shrink_threshold = kwargs.pop("shrink_threshold", 0.3)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected _suppress_shrinked_masks kwargs: {names}")
        area_before = mx.sum(pred_masks > 0, axis=(-1, -2))
        area_after = mx.sum(new_pred_masks > 0, axis=(-1, -2))
        area_before = mx.maximum(area_before, mx.array(1.0, dtype=mx.float32))
        area_ratio = area_after.astype(mx.float32) / area_before.astype(mx.float32)
        keep_mask = (area_ratio >= shrink_threshold)[:, :, None, None]
        return mx.where(
            keep_mask,
            pred_masks,
            mx.minimum(pred_masks, mx.array(-10.0, dtype=pred_masks.dtype)),
        )

    def _suppress_object_pw_area_shrinkage(
        self, *args: mx.array, **kwargs: object
    ) -> mx.array:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected _suppress_object_pw_area_shrinkage kwargs: {names}"
            )
        (pred_masks,) = args
        pixel_level_non_overlapping_masks = super()._apply_non_overlapping_constraints(
            pred_masks
        )
        return self._suppress_shrinked_masks(
            pred_masks,
            pixel_level_non_overlapping_masks,
        )

    def _apply_object_wise_non_overlapping_constraints(
        self, *args: mx.array, **kwargs: float
    ) -> mx.array:
        pred_masks, obj_scores = args[:2]
        background_value = kwargs.pop("background_value", -10.0)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                "Unexpected _apply_object_wise_non_overlapping_constraints "
                f"kwargs: {names}"
            )
        background = mx.array(background_value, dtype=pred_masks.dtype)
        pred_masks_single_score = mx.where(
            pred_masks > 0,
            obj_scores[..., None, None],
            background,
        )
        pixel_level_non_overlapping_masks = super()._apply_non_overlapping_constraints(
            pred_masks_single_score
        )
        return mx.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            mx.minimum(pred_masks, background),
        )


__all__ = [
    "NO_OBJ_SCORE",
    "Sam3TrackerPredictor",
    "concat_points",
]
