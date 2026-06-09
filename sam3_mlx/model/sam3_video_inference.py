from __future__ import annotations

from typing import Any

import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.io_utils import load_resource_as_video_frames, masks_to_boxes_xyxy


MLX_PORT_BASE_COMMIT = "ac306ca0fb1c757c00d3c3b2f737ef2f99b45bc3"


class Sam3VideoInference:
    """MLX port of the official SAM3 video-inference model surface.

    The official implementation couples a detector and a tracker. This MLX slice
    owns the same state and method names, but runs framewise image inference until
    the tracker memory encoder/decoder is ported.
    """

    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1

    def __init__(
        self,
        image_model,
        image_size: int = 1008,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        compile_model: bool = False,
        confidence_threshold: float = 0.5,
        processor_factory=None,
        **kwargs,
    ) -> None:
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Sam3VideoInference keyword(s): {unexpected}")
        if compile_model:
            raise_unsupported(
                "sam3_mlx.model.sam3_video_inference.Sam3VideoInference(compile_model=True)",
                reason="torch-compile",
                detail="torch.compile is not part of the sam3_mlx runtime.",
            )
        self.image_model = image_model
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model
        self.confidence_threshold = confidence_threshold
        self.processor_factory = processor_factory

    def eval(self):
        if hasattr(self.image_model, "eval"):
            self.image_model.eval()
        return self

    def to(self, device=None, **kwargs):
        del kwargs
        if device not in (None, "mlx"):
            raise_unsupported(
                f"sam3_mlx.model.sam3_video_inference.Sam3VideoInference.to(device={device!r})",
                reason="unsupported-device",
                detail="sam3_mlx video inference only supports the explicit MLX runtime.",
                alternative="device='mlx'",
            )
        return self

    def init_state(
        self,
        resource_path,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
        video_loader_type: str = "cv2",
    ) -> dict[str, Any]:
        frames = load_resource_as_video_frames(
            resource_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=self.image_mean,
            img_std=self.image_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
        return {
            "image_size": self.image_size,
            "frames": frames,
            "num_frames": len(frames),
            "orig_height": frames.orig_height,
            "orig_width": frames.orig_width,
            "offload_state_to_cpu": offload_state_to_cpu,
            "constants": {},
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "feature_cache": {},
            "cached_frame_outputs": {},
            "action_history": [],
            "text_prompt": None,
            "box_prompt": None,
            "box_labels": None,
            "point_prompt": None,
            "point_labels": None,
            "obj_id": None,
            "prompt_frame_idx": None,
            "removed_obj_ids": set(),
            "previous_stages_out": [None] * len(frames),
            "per_frame_raw_point_input": [None] * len(frames),
            "per_frame_raw_box_input": [None] * len(frames),
            "per_frame_visual_prompt": [None] * len(frames),
            "per_frame_geometric_prompt": [None] * len(frames),
            "per_frame_cur_step": [0] * len(frames),
            "visual_prompt_embed": None,
            "visual_prompt_mask": None,
            "is_image_only": len(frames) == 1,
        }

    def reset_state(self, inference_state: dict[str, Any]) -> None:
        num_frames = inference_state["num_frames"]
        inference_state["text_prompt"] = None
        inference_state["box_prompt"] = None
        inference_state["box_labels"] = None
        inference_state["point_prompt"] = None
        inference_state["point_labels"] = None
        inference_state["obj_id"] = None
        inference_state["prompt_frame_idx"] = None
        inference_state["removed_obj_ids"].clear()
        inference_state["tracker_inference_states"].clear()
        inference_state["tracker_metadata"].clear()
        inference_state["feature_cache"].clear()
        inference_state["cached_frame_outputs"].clear()
        inference_state["action_history"].clear()
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    def add_prompt(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        text_str: str | None = None,
        boxes_xywh=None,
        box_labels=None,
        points=None,
        point_labels=None,
        obj_id: int | None = None,
        rel_coordinates: bool = True,
        output_prob_thresh: float | None = None,
    ) -> tuple[int, dict[str, np.ndarray]]:
        if points is not None and (text_str is not None or boxes_xywh is not None):
            raise AssertionError(
                "When points are provided, text_str and boxes_xywh must be None."
            )
        self._assert_frame_idx(inference_state, frame_idx, name="frame_index")
        if text_str is None and boxes_xywh is None and points is None:
            raise ValueError("add_prompt requires text, boxes_xywh, or points.")

        boxes_cxcywh, labels = _coerce_boxes(
            boxes_xywh=boxes_xywh,
            box_labels=box_labels,
            rel_coordinates=rel_coordinates,
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )
        points_xy, labels_point = _coerce_points(
            points=points,
            point_labels=point_labels,
            rel_coordinates=rel_coordinates,
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )

        self.reset_state(inference_state)
        inference_state["text_prompt"] = text_str
        inference_state["box_prompt"] = boxes_cxcywh
        inference_state["box_labels"] = labels
        inference_state["point_prompt"] = points_xy
        inference_state["point_labels"] = labels_point
        inference_state["obj_id"] = obj_id
        inference_state["prompt_frame_idx"] = frame_idx
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"
        inference_state["per_frame_raw_box_input"][frame_idx] = (
            (boxes_cxcywh, labels) if boxes_cxcywh is not None else None
        )
        inference_state["per_frame_raw_point_input"][frame_idx] = (
            (points_xy, labels_point) if points_xy is not None else None
        )
        obj_ids = [int(obj_id)] if obj_id is not None else None
        inference_state["action_history"].append(
            {
                "type": "add",
                "frame_idx": frame_idx,
                "text": text_str,
                "obj_ids": obj_ids,
            }
        )

        outputs = self._run_frame_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            output_prob_thresh=output_prob_thresh,
        )
        inference_state["cached_frame_outputs"][frame_idx] = outputs
        return frame_idx, outputs

    def propagate_in_video(
        self,
        inference_state: dict[str, Any],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float | None = None,
        reverse: bool = False,
    ):
        processing_order, _ = self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )
        inference_state["feature_cache"]["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
            "framewise_mlx": True,
        }
        for frame_idx in processing_order:
            outputs = self._run_frame_prompt(
                inference_state=inference_state,
                frame_idx=frame_idx,
                output_prob_thresh=output_prob_thresh,
            )
            inference_state["previous_stages_out"][frame_idx] = (
                "_THIS_FRAME_HAS_OUTPUTS_"
            )
            inference_state["cached_frame_outputs"][frame_idx] = outputs
            yield frame_idx, outputs

    def remove_object(
        self,
        inference_state: dict[str, Any],
        obj_id: int,
        frame_idx: int = 0,
        is_user_action: bool = True,
    ):
        del is_user_action
        inference_state["removed_obj_ids"].add(int(obj_id))
        inference_state["action_history"].append(
            {"type": "remove", "frame_idx": frame_idx, "obj_ids": [int(obj_id)]}
        )
        for cached_frame_idx, outputs in list(
            inference_state["cached_frame_outputs"].items()
        ):
            inference_state["cached_frame_outputs"][cached_frame_idx] = (
                _filter_outputs_by_removed_obj_ids(
                    outputs,
                    removed_obj_ids=inference_state["removed_obj_ids"],
                )
            )
        return inference_state["cached_frame_outputs"].get(
            frame_idx,
            _empty_video_outputs(
                orig_height=inference_state["orig_height"],
                orig_width=inference_state["orig_width"],
            ),
        )

    def cancel_propagation(self, inference_state: dict[str, Any]) -> None:
        inference_state["feature_cache"]["cancel_propagation"] = True

    def _get_processing_order(
        self,
        inference_state: dict[str, Any],
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> tuple[range, int]:
        num_frames = inference_state["num_frames"]
        previous_stages_out = inference_state["previous_stages_out"]
        if all(out is None for out in previous_stages_out) and start_frame_idx is None:
            raise RuntimeError(
                "No prompts are received on any frames. Please add prompt on at "
                "least one frame before propagation."
            )
        if start_frame_idx is None:
            start_frame_idx = min(
                t for t, out in enumerate(previous_stages_out) if out is not None
            )
        self._assert_frame_idx(
            inference_state, start_frame_idx, name="start_frame_index"
        )
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if max_frame_num_to_track < 0:
            raise ValueError("max_frame_num_to_track must be non-negative.")
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            processing_order = range(start_frame_idx - 1, end_frame_idx - 1, -1)
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order, end_frame_idx

    def _run_frame_prompt(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        output_prob_thresh: float | None,
    ) -> dict[str, np.ndarray]:
        frame = inference_state["frames"][frame_idx]
        processor = self._make_processor(output_prob_thresh)
        frame_state = processor.set_image(frame)
        text_prompt = inference_state["text_prompt"]
        if text_prompt is not None:
            frame_state = processor.set_text_prompt(
                prompt=text_prompt, state=frame_state
            )
        boxes_cxcywh = inference_state["box_prompt"]
        box_labels = inference_state["box_labels"]
        if boxes_cxcywh is not None:
            for box, label in zip(boxes_cxcywh, box_labels, strict=True):
                frame_state = processor.add_geometric_prompt(
                    box=box.tolist(),
                    label=bool(label),
                    state=frame_state,
                )
        points = inference_state["point_prompt"]
        point_labels = inference_state["point_labels"]
        if points is not None:
            for point, label in zip(points, point_labels, strict=True):
                frame_state = processor.add_point_prompt(
                    point=point.tolist(),
                    label=bool(label),
                    state=frame_state,
                )
        outputs = _state_to_video_outputs(
            frame_state,
            obj_id=inference_state["obj_id"],
            orig_height=inference_state["orig_height"],
            orig_width=inference_state["orig_width"],
        )
        return _filter_outputs_by_removed_obj_ids(
            outputs,
            removed_obj_ids=inference_state["removed_obj_ids"],
        )

    def _make_processor(self, output_prob_thresh: float | None):
        factory = self.processor_factory
        if factory is None:
            from sam3_mlx.model.sam3_image_processor import Sam3Processor

            factory = Sam3Processor
        return factory(
            self.image_model,
            resolution=self.image_size,
            confidence_threshold=(
                self.confidence_threshold
                if output_prob_thresh is None
                else output_prob_thresh
            ),
        )

    @staticmethod
    def _assert_frame_idx(
        inference_state: dict[str, Any],
        frame_idx: int,
        name: str,
    ) -> None:
        num_frames = inference_state["num_frames"]
        if not 0 <= frame_idx < num_frames:
            raise IndexError(
                f"{name} {frame_idx} is out of range for {num_frames} frames."
            )


class Sam3VideoInferenceWithInstanceInteractivity(Sam3VideoInference):
    """SAM3 video inference with the official point-refinement entrypoint name."""

    def add_prompt(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        text_str: str | None = None,
        boxes_xywh=None,
        box_labels=None,
        points=None,
        point_labels=None,
        obj_id: int | None = None,
        rel_coordinates: bool = True,
        output_prob_thresh: float | None = None,
    ):
        if points is not None:
            if text_str is not None or boxes_xywh is not None:
                raise AssertionError(
                    "When points are provided, text_str and boxes_xywh must be None."
                )
            if obj_id is None:
                raise AssertionError(
                    "When points are provided, obj_id must be provided."
                )
            return self.add_tracker_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=point_labels,
                rel_coordinates=rel_coordinates,
                output_prob_thresh=output_prob_thresh,
            )
        return super().add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text_str,
            boxes_xywh=boxes_xywh,
            box_labels=box_labels,
            rel_coordinates=rel_coordinates,
            output_prob_thresh=output_prob_thresh,
        )

    def add_tracker_new_points(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_id: int,
        points,
        labels,
        rel_coordinates: bool = True,
        use_prev_mem_frame: bool = False,
        output_prob_thresh: float | None = None,
    ):
        del use_prev_mem_frame
        return super().add_prompt(
            inference_state=inference_state,
            frame_idx=frame_idx,
            points=points,
            point_labels=labels,
            obj_id=obj_id,
            rel_coordinates=rel_coordinates,
            output_prob_thresh=output_prob_thresh,
        )


def is_image_type(resource_path: str) -> bool:
    """Unsupported shim for the SAM3 video resource-type helper (TorchCodec path)."""
    del resource_path
    raise_unsupported(
        "sam3_mlx.model.sam3_video_inference.is_image_type",
        reason="torchcodec",
        detail=(
            "The official helper belongs to the Torch-only video inference "
            "resource-loading path."
        ),
    )


def _coerce_boxes(
    boxes_xywh,
    box_labels,
    rel_coordinates: bool,
    orig_height: int,
    orig_width: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if boxes_xywh is None:
        if box_labels is not None:
            raise ValueError("bounding_box_labels require bounding_boxes.")
        return None, None

    boxes = np.asarray(boxes_xywh, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes[None, :]
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"bounding_boxes must have shape (N, 4), got {boxes.shape}.")
    if not np.isfinite(boxes).all():
        raise ValueError("bounding_boxes must contain only finite values.")

    if rel_coordinates:
        normalized = boxes
    else:
        scale = np.array([orig_width, orig_height, orig_width, orig_height])
        normalized = boxes / scale[None, :]
    if (
        (normalized[:, :2] < 0).any()
        or (normalized[:, 2:] < 0).any()
        or (normalized[:, 0] + normalized[:, 2] > 1).any()
        or (normalized[:, 1] + normalized[:, 3] > 1).any()
    ):
        raise ValueError("bounding_boxes must be within the image bounds.")

    if box_labels is None:
        labels = np.ones((normalized.shape[0],), dtype=bool)
    else:
        labels = np.asarray(box_labels, dtype=bool)
        if labels.ndim == 0:
            labels = labels[None]
        if labels.shape != (normalized.shape[0],):
            raise ValueError("bounding_box_labels must have one label per box.")

    boxes_cxcywh = normalized.copy()
    boxes_cxcywh[:, 0] = normalized[:, 0] + normalized[:, 2] / 2.0
    boxes_cxcywh[:, 1] = normalized[:, 1] + normalized[:, 3] / 2.0
    return boxes_cxcywh, labels


def _coerce_points(
    points,
    point_labels,
    rel_coordinates: bool,
    orig_height: int,
    orig_width: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if points is None:
        if point_labels is not None:
            raise ValueError("point_labels require points.")
        return None, None

    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim == 1:
        points_array = points_array[None, :]
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points_array.shape}.")
    if not np.isfinite(points_array).all():
        raise ValueError("points must contain only finite values.")

    if rel_coordinates:
        normalized = points_array
    else:
        scale = np.array([orig_width, orig_height], dtype=np.float32)
        normalized = points_array / scale[None, :]
    if (normalized < 0).any() or (normalized > 1).any():
        raise ValueError("points must be within the image bounds.")

    if point_labels is None:
        labels = np.ones((normalized.shape[0],), dtype=bool)
    else:
        labels = np.asarray(point_labels, dtype=bool)
        if labels.ndim == 0:
            labels = labels[None]
        if labels.shape != (normalized.shape[0],):
            raise ValueError("point_labels must have one label per point.")
    return normalized, labels


def _state_to_video_outputs(
    frame_state: dict[str, Any],
    obj_id: int | None,
    orig_height: int,
    orig_width: int,
) -> dict[str, np.ndarray]:
    masks_value = frame_state.get("masks")
    boxes_value = frame_state.get("boxes")
    scores_value = frame_state.get("scores")
    _evaluate_if_mlx(masks_value, boxes_value, scores_value)

    masks = _coerce_masks(masks_value, orig_height, orig_width)
    scores = _coerce_scores(scores_value, masks.shape[0])
    if boxes_value is None:
        boxes_xyxy = masks_to_boxes_xyxy(masks)
    else:
        boxes_xyxy = np.asarray(boxes_value, dtype=np.float32)
        if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
            raise ValueError(f"boxes must have shape (N, 4), got {boxes_xyxy.shape}.")
        if boxes_xyxy.shape[0] != masks.shape[0]:
            raise ValueError(
                "boxes and masks must describe the same number of objects, "
                f"got {boxes_xyxy.shape[0]} boxes and {masks.shape[0]} masks."
            )
    boxes_xywh = _boxes_xyxy_to_normalized_xywh(
        boxes_xyxy,
        orig_height=orig_height,
        orig_width=orig_width,
    )

    count = masks.shape[0]
    if obj_id is not None:
        if count != 1:
            raise_unsupported(
                "sam3_mlx.model.sam3_video_inference.Sam3VideoInference.add_new_points(obj_id,multiple_masks)",
                reason="video-multiplex",
                detail=(
                    "obj_id assignment is only supported when framewise prompting "
                    "returns exactly one mask."
                ),
            )
        out_obj_ids = np.array([obj_id], dtype=np.int64)
    else:
        out_obj_ids = np.arange(count, dtype=np.int64)

    return {
        "out_obj_ids": out_obj_ids,
        "out_probs": scores,
        "out_boxes_xywh": boxes_xywh,
        "out_binary_masks": masks.astype(bool, copy=False),
    }


def _evaluate_if_mlx(*values) -> None:
    live_values = [value for value in values if value is not None]
    if not live_values:
        return
    try:
        import mlx.core as mx
    except ImportError:
        return
    eval_fn = getattr(mx, "eval", None)
    if eval_fn is not None:
        eval_fn(*live_values)


def _coerce_masks(value, orig_height: int, orig_width: int) -> np.ndarray:
    if value is None:
        return np.zeros((0, orig_height, orig_width), dtype=bool)
    masks = np.asarray(value)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape (N, H, W), got {masks.shape}.")
    return masks.astype(bool, copy=False)


def _coerce_scores(value, count: int) -> np.ndarray:
    if value is None:
        return np.zeros((count,), dtype=np.float32)
    scores = np.asarray(value, dtype=np.float32).reshape(-1)
    if scores.shape != (count,):
        raise ValueError(f"scores must have shape ({count},), got {scores.shape}.")
    return scores


def _filter_outputs_by_removed_obj_ids(
    outputs: dict[str, np.ndarray],
    removed_obj_ids: set[int],
) -> dict[str, np.ndarray]:
    if not removed_obj_ids or outputs["out_obj_ids"].size == 0:
        return outputs
    keep = ~np.isin(outputs["out_obj_ids"], np.array(sorted(removed_obj_ids)))
    return {
        "out_obj_ids": outputs["out_obj_ids"][keep],
        "out_probs": outputs["out_probs"][keep],
        "out_boxes_xywh": outputs["out_boxes_xywh"][keep],
        "out_binary_masks": outputs["out_binary_masks"][keep],
    }


def _empty_video_outputs(orig_height: int, orig_width: int) -> dict[str, np.ndarray]:
    return {
        "out_obj_ids": np.zeros((0,), dtype=np.int64),
        "out_probs": np.zeros((0,), dtype=np.float32),
        "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
        "out_binary_masks": np.zeros((0, orig_height, orig_width), dtype=bool),
    }


def _boxes_xyxy_to_normalized_xywh(
    boxes_xyxy: np.ndarray,
    orig_height: int,
    orig_width: int,
) -> np.ndarray:
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {boxes.shape}.")
    out = np.empty_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] / orig_width
    out[:, 1] = boxes[:, 1] / orig_height
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / orig_width
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / orig_height
    return out
