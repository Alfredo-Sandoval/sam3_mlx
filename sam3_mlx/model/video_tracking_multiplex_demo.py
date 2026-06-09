from __future__ import annotations

from collections import OrderedDict
from typing import Any

import mlx.core as mx

from sam3_mlx.model.data_misc import interpolate
from sam3_mlx.model.io_utils import load_resource_as_video_frames
from sam3_mlx.model.multiplex_utils import (
    MultiplexState,
    raise_unsupported_multiplex_runtime,
)
from sam3_mlx.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3_mlx.model.video_tracking_multiplex import (
    NO_OBJ_SCORE,
    VideoTrackingDynamicMultiplex,
    _is_mlx_array,
    concat_points,
)


def _eval_tree(*values: Any) -> None:
    def _eval_value(value: Any) -> None:
        if _is_mlx_array(value):
            mx.eval(value)
        elif isinstance(value, dict):
            for item in value.values():
                _eval_value(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _eval_value(item)

    for value in values:
        _eval_value(value)


def _coerce_obj_id_list(obj_ids: Any) -> list[Any]:
    if isinstance(obj_ids, (str, bytes)):
        return [obj_ids]
    if _is_mlx_array(obj_ids):
        mx.eval(obj_ids)
        return obj_ids.reshape(-1).tolist()
    try:
        return list(obj_ids)
    except TypeError:
        return [obj_ids]


def _take_indices(value: Any, indices: list[int]) -> Any:
    if value is None:
        return None
    index = mx.array(indices, dtype=mx.int64)
    if isinstance(value, list):
        return [mx.take(item, index, axis=0) for item in value]
    return mx.take(value, index, axis=0)


def _replace_indices(value: Any, indices: list[int], replacement: Any) -> Any:
    if len(indices) == 0:
        return value
    if len(indices) != replacement.shape[0]:
        raise ValueError("indices length must match replacement batch")
    replacements = {int(index): row for index, row in zip(indices, replacement)}
    rows = []
    for row_idx in range(value.shape[0]):
        if row_idx in replacements:
            rows.append(replacements[row_idx][None])
        else:
            rows.append(value[row_idx : row_idx + 1])
    return mx.concatenate(rows, axis=0)


def _init_multiplex_demo_state(
    *,
    video_height: int,
    video_width: int,
    num_frames: int,
    images: Any = None,
    cached_features: dict[int, Any] | None = None,
) -> dict[str, Any]:
    if int(video_height) <= 0 or int(video_width) <= 0:
        raise ValueError("video_height and video_width must be positive integers.")
    if int(num_frames) <= 0:
        raise ValueError("num_frames must be a positive integer.")
    if cached_features is None:
        cached_features = {}
    if not isinstance(cached_features, dict):
        raise TypeError("cached_features must be a dict keyed by frame index.")

    return {
        "images": images,
        "num_frames": int(num_frames),
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "video_height": int(video_height),
        "video_width": int(video_width),
        "device": "mlx",
        "storage_device": "mlx",
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": cached_features,
        "constants": {},
        "obj_id_to_idx": OrderedDict(),
        "obj_idx_to_id": OrderedDict(),
        "obj_ids": [],
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        },
        "first_ann_frame_idx": None,
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        },
        "tracking_has_started": False,
        "frames_already_tracked": {},
        "multiplex_state": None,
        "user_refined_frames_per_obj": {},
    }


def _load_multiplex_demo_state_from_resource(
    model: Any,
    *,
    video_path: Any,
    cached_features: dict[int, Any] | None = None,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
    async_loading_frames: bool = False,
    use_torchcodec: bool = False,
    use_cv2: bool = False,
) -> dict[str, Any]:
    if offload_video_to_cpu or offload_state_to_cpu:
        raise_unsupported_multiplex_runtime(
            "VideoTrackingMultiplexDemo.init_state(offload)"
        )
    if async_loading_frames:
        raise_unsupported_multiplex_runtime(
            "VideoTrackingMultiplexDemo.init_state(async_loading_frames)"
        )
    if use_torchcodec:
        raise_unsupported_multiplex_runtime(
            "VideoTrackingMultiplexDemo.init_state(use_torchcodec=True)"
        )
    del use_cv2
    loaded_frames = load_resource_as_video_frames(
        resource_path=video_path,
        image_size=int(model.image_size),
        offload_video_to_cpu=False,
        async_loading_frames=False,
        video_loader_type="cv2",
    )
    images = getattr(loaded_frames, "images", None)
    if images is None:
        raise ValueError("Loaded video frames must expose normalized MLX images.")
    return _init_multiplex_demo_state(
        video_height=int(loaded_frames.orig_height),
        video_width=int(loaded_frames.orig_width),
        num_frames=len(loaded_frames),
        images=images,
        cached_features=cached_features,
    )


class VideoTrackingMultiplexDemo(VideoTrackingDynamicMultiplex):
    def __init__(
        self,
        clear_non_cond_mem_around_input: bool = False,
        clear_non_cond_mem_for_multi_obj: bool = False,
        fill_hole_area: int = 0,
        always_start_from_first_ann_frame: bool = False,
        max_point_num_in_prompt_enc: int = 16,
        non_overlap_masks_for_output: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output

    def init_state(
        self,
        video_path: Any,
        offload_video_to_cpu: bool,
        offload_state_to_cpu: bool,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
    ) -> Any:
        return _load_multiplex_demo_state_from_resource(
            self,
            video_path=video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
            use_torchcodec=use_torchcodec,
            use_cv2=use_cv2,
        )

    def _get_obj_num(self, inference_state: dict[str, Any]) -> int:
        multiplex_state = inference_state.get("multiplex_state")
        if multiplex_state is not None:
            return int(multiplex_state.total_valid_entries)
        return len(inference_state.get("obj_ids", []))

    def _obj_id_to_idx(
        self,
        inference_state: dict[str, Any],
        obj_id: Any,
        error_if_new: bool = False,
    ) -> int:
        obj_id_to_idx = inference_state["obj_id_to_idx"]
        obj_idx = obj_id_to_idx.get(obj_id)
        if obj_idx is not None:
            return int(obj_idx)
        if (
            getattr(self, "is_dynamic_model", False)
            or not inference_state["tracking_has_started"]
        ) and not error_if_new:
            obj_idx = len(obj_id_to_idx)
            obj_id_to_idx[obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(obj_id_to_idx)
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
        raise RuntimeError(
            f"Cannot add new object id {obj_id}. Existing object ids: "
            f"{inference_state['obj_ids']}."
        )

    def _select_propagation_output_obj_ids(
        self,
        inference_state: dict[str, Any],
        obj_ids: Any,
    ) -> tuple[list[Any], list[int] | None]:
        if obj_ids is None:
            return list(inference_state["obj_ids"]), None
        requested_obj_ids = _coerce_obj_id_list(obj_ids)
        seen = set()
        duplicates = []
        for obj_id in requested_obj_ids:
            if obj_id in seen:
                duplicates.append(obj_id)
            seen.add(obj_id)
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
            int(obj_id_to_idx[obj_id]) for obj_id in requested_obj_ids
        ]

    def _get_processing_order(
        self,
        inference_state: dict[str, Any],
        start_frame_idx: int | None,
        max_frame_num_to_track: int | None,
        reverse: bool,
    ) -> Any:
        num_frames = int(inference_state["num_frames"])
        if self.always_start_from_first_ann_frame:
            start_frame_idx = inference_state["first_ann_frame_idx"]
        if start_frame_idx is None:
            start_frame_idx = min(inference_state["output_dict"]["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(int(start_frame_idx) - int(max_frame_num_to_track), 0)
            if int(start_frame_idx) > 0:
                return range(int(start_frame_idx), end_frame_idx - 1, -1)
            return [0]
        end_frame_idx = min(
            int(start_frame_idx) + int(max_frame_num_to_track),
            num_frames - 1,
        )
        return range(int(start_frame_idx), end_frame_idx + 1)

    def _prepare_demo_backbone_features(self, backbone_out: Any) -> dict[str, Any]:
        if (
            isinstance(backbone_out, dict)
            and "interactive" in backbone_out
            and "sam2_backbone_out" in backbone_out
            and isinstance(backbone_out["interactive"], dict)
            and "vision_feats" in backbone_out["interactive"]
        ):
            return backbone_out
        return self._prepare_backbone_features(backbone_out)

    def _get_image_feature(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        batch_size: int,
    ) -> tuple[Any, dict[str, Any]]:
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
                image = mx.array(images[frame_idx], dtype=mx.float32)
                if len(image.shape) == 3:
                    image = image[None]
            backbone_out = self.forward_image(
                image,
                need_sam3_out=True,
                need_interactive_out=True,
                need_propagation_out=True,
            )
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
            cached = (image, backbone_out)

        if not (isinstance(cached, tuple) and len(cached) == 2):
            raise TypeError("cached_features values must be (image, backbone_out).")
        image, backbone_out = cached
        features = self._prepare_demo_backbone_features(backbone_out)
        _eval_tree(image)
        return image, features

    def _run_memory_encoder(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        batch_size: int,
        high_res_masks: Any,
        object_score_logits: Any,
        is_mask_from_pts: bool,
        conditioning_objects: set[int] | None = None,
    ) -> tuple[Any, Any, Any, Any]:
        """Run the memory encoder on ``high_res_masks``.

        Ported from the official ``Sam3VideoTrackingMultiplexDemo._run_memory_encoder``.
        Re-encodes propagated masks into memory. Returns the multiplex
        4-tuple ``(maskmem_features, maskmem_pos_enc, image_features, image_pos_enc)``.
        The official Torch bf16 cast and ``storage_device`` transfers are dropped
        because MLX keeps unified-memory float32 arrays.
        """
        image, backbone_features = self._get_image_feature(
            inference_state, frame_idx, batch_size
        )
        backbone_features_propagation = backbone_features["sam2_backbone_out"]
        propagation_vision_feats = backbone_features_propagation["vision_feats"]
        propagation_vision_pos_embeds = backbone_features_propagation[
            "vision_pos_embeds"
        ]
        propagation_feat_sizes = backbone_features_propagation["feat_sizes"]

        if conditioning_objects is None:
            output_dict = inference_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                storage = output_dict[storage_key]
                if frame_idx not in storage:
                    continue
                conditioning_objects = storage[frame_idx]["conditioning_objects"]
                break
            else:
                raise ValueError(
                    f"conditioning objects not found at frame_idx={frame_idx}"
                )

        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            image=image,
            current_vision_feats=propagation_vision_feats,
            feat_sizes=propagation_feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
            conditioning_objects=conditioning_objects,
            multiplex_state=inference_state["multiplex_state"],
        )
        # "maskmem_pos_enc" is the same across frames, so we cache one copy and
        # re-expand it to the current batch size.
        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
        )
        image_features = propagation_vision_feats[-1]
        image_pos_enc = propagation_vision_pos_embeds[-1]
        return maskmem_features, maskmem_pos_enc, image_features, image_pos_enc

    def _get_maskmem_pos_enc(
        self, inference_state: dict[str, Any], current_out: dict[str, Any]
    ) -> list[Any] | None:
        """Cache ``maskmem_pos_enc`` once and re-expand it to the batch size.

        ``maskmem_pos_enc`` does not depend on the object index, so the demo
        stores a single object's slice in ``inference_state["constants"]`` and
        broadcasts it back up to the actual batch size on each call.
        """
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out.get("maskmem_pos_enc")
        if out_maskmem_pos_enc is None:
            return None
        if "maskmem_pos_enc" not in model_constants:
            assert isinstance(out_maskmem_pos_enc, list)
            # only keep the slice for one object, since it's the same across objects
            maskmem_pos_enc = [x[0:1] for x in out_maskmem_pos_enc]
            model_constants["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            maskmem_pos_enc = model_constants["maskmem_pos_enc"]
        batch_size = out_maskmem_pos_enc[0].shape[0]
        return [mx.broadcast_to(x, (batch_size, *x.shape[1:])) for x in maskmem_pos_enc]

    def _run_single_frame_inference(
        self,
        *,
        inference_state: dict[str, Any],
        output_dict: dict[str, dict[int, Any]],
        frame_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs: Any,
        mask_inputs: Any,
        reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits: Any = None,
        add_to_existing_state: bool = False,
        new_object_masks: Any = None,
        new_object_idxs: list[int] | None = None,
        new_object_ids: list[Any] | None = None,
        are_new_masks_from_pts: bool = False,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
        reconditioning: bool = False,
        objects_to_interact: list[int] | None = None,
    ) -> tuple[dict[str, Any], Any]:
        image, backbone_features = self._get_image_feature(
            inference_state,
            frame_idx,
            batch_size,
        )
        if add_to_existing_state or reconditioning:
            if new_object_idxs is None:
                raise ValueError("new_object_idxs are required when editing state.")
            if new_object_ids is None:
                raise ValueError("new_object_ids are required when editing state.")

            existing_out = output_dict["cond_frame_outputs"].get(frame_idx)
            if existing_out is None:
                existing_out = output_dict["non_cond_frame_outputs"].get(frame_idx)
            if existing_out is None:
                raise RuntimeError(
                    f"No existing output found for frame {frame_idx} in either storage."
                )

            interactive_features = backbone_features["interactive"]
            interactive_pix_feat = self._get_interactive_pix_mem(
                interactive_features["vision_feats"],
                interactive_features["feat_sizes"],
            )
            interactive_high_res_features = [
                x.transpose(1, 2, 0).reshape(x.shape[1], x.shape[2], *feat_size)
                for x, feat_size in zip(
                    interactive_features["vision_feats"][:-1],
                    interactive_features["feat_sizes"][:-1],
                )
            ]
            propagation_features = backbone_features["sam2_backbone_out"]
            propagation_vision_feats = (
                propagation_features["vision_feats"] if run_mem_encoder else None
            )
            propagation_feat_sizes = (
                propagation_features["feat_sizes"] if run_mem_encoder else None
            )

            if (add_to_existing_state or reconditioning) and mask_inputs is None:
                if point_inputs is None:
                    raise ValueError("point_inputs are required to add point objects.")
                sam_mask_inputs = prev_sam_mask_logits
                if (
                    sam_mask_inputs is not None
                    and sam_mask_inputs.shape[0]
                    != point_inputs["point_coords"].shape[0]
                ):
                    sam_mask_inputs = _take_indices(sam_mask_inputs, new_object_idxs)
                interaction_out = self._forward_sam_heads(
                    backbone_features=interactive_pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=sam_mask_inputs,
                    interactive_high_res_features=interactive_high_res_features,
                    multimask_output=self._use_multimask(
                        is_init_cond_frame,
                        point_inputs=point_inputs,
                    ),
                    objects_to_interact=new_object_idxs,
                    multiplex_state=inference_state["multiplex_state"],
                )
                new_object_masks = interaction_out["low_res_masks"]
                are_new_masks_from_pts = True

            edit_kwargs = dict(
                interactive_pix_feat=interactive_pix_feat,
                interactive_high_res_features=interactive_high_res_features,
                propagation_vision_feats=propagation_vision_feats,
                propagation_feat_sizes=propagation_feat_sizes,
                new_masks=mask_inputs if mask_inputs is not None else new_object_masks,
                obj_idxs_in_mask=new_object_idxs,
                obj_ids_in_mask=new_object_ids,
                prev_output=existing_out,
                multiplex_state=inference_state["multiplex_state"],
                add_mask_to_memory=run_mem_encoder,
            )
            if edit_kwargs["new_masks"] is None:
                raise ValueError("new object masks are required when editing state.")
            if reconditioning:
                self.recondition_masks_in_existing_state(
                    **edit_kwargs,
                    are_masks_from_pts=are_new_masks_from_pts,
                )
            else:
                self.add_new_masks_to_existing_state(
                    **edit_kwargs,
                    are_masks_from_pts=are_new_masks_from_pts,
                    allow_new_buckets=allow_new_buckets,
                    prefer_new_buckets=prefer_new_buckets,
                )
            current_out = existing_out
        else:
            current_out = self.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                backbone_features_interactive=backbone_features["interactive"],
                backbone_features_propagation=backbone_features["sam2_backbone_out"],
                image=image,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                gt_masks=None,
                frames_to_add_correction_pt=[],
                output_dict=output_dict,
                num_frames=inference_state["num_frames"],
                track_in_reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                prev_sam_mask_logits=prev_sam_mask_logits,
                multiplex_state=inference_state["multiplex_state"],
                objects_to_interact=objects_to_interact,
                new_object_masks=new_object_masks,
                new_object_idxs=new_object_idxs,
                new_object_ids=new_object_ids,
                are_new_masks_from_pts=are_new_masks_from_pts,
                allow_new_buckets=allow_new_buckets,
                prefer_new_buckets=prefer_new_buckets,
                reconditioning=reconditioning,
            )
        pred_masks = current_out["pred_masks"]
        _eval_tree(current_out, pred_masks)
        return current_out, pred_masks

    def _get_orig_video_res_output(
        self,
        inference_state: dict[str, Any],
        any_res_masks: Any,
    ) -> tuple[Any, Any]:
        video_h = int(inference_state["video_height"])
        video_w = int(inference_state["video_width"])
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

    def _add_output_per_object(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        current_out: dict[str, Any],
        storage_key: str,
    ) -> None:
        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "pred_masks": current_out["pred_masks"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }
            if "pred_masks_video_res" in current_out:
                obj_out["pred_masks_video_res"] = current_out["pred_masks_video_res"][
                    obj_slice
                ]
            if self.use_memory_selection and "iou_score" in current_out:
                obj_out["iou_score"] = current_out["iou_score"][obj_slice]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def _get_or_create_multiplex_state(
        self,
        inference_state: dict[str, Any],
        obj_ids: list[Any],
        *,
        is_new_state: bool,
        reconditioning: bool,
    ) -> Any:
        multiplex_state = inference_state.get("multiplex_state")
        if multiplex_state is not None:
            if not reconditioning and not getattr(self, "is_dynamic_model", False):
                raise AssertionError("New objects are not allowed after state creation")
            return multiplex_state
        if reconditioning:
            raise ValueError("reconditioning requires an existing multiplex_state")
        controller = getattr(self, "multiplex_controller", None)
        if controller is not None:
            multiplex_state = controller.get_state(
                num_valid_entries=len(obj_ids),
                device=inference_state["device"],
                dtype=mx.float32,
                random=False,
                object_ids=obj_ids,
            )
        else:
            multiplex_state = MultiplexState(
                [list(range(len(obj_ids)))],
                dtype=mx.float32,
                allowed_bucket_capacity=len(obj_ids),
                object_ids=obj_ids,
            )
        inference_state["multiplex_state"] = multiplex_state
        return multiplex_state

    def add_new_points(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_id: Any,
        points: Any,
        labels: Any,
        clear_old_points: bool,
        rel_coordinates: bool = True,
        use_prev_mem_frame: bool = False,
    ) -> tuple[int, list[Any], None, Any]:
        del use_prev_mem_frame
        frame_idx = int(frame_idx)
        if not 0 <= frame_idx < int(inference_state["num_frames"]):
            raise ValueError(f"frame_idx {frame_idx} is outside the video frame range.")
        points = (
            points.astype(mx.float32)
            if _is_mlx_array(points)
            else mx.array(
                points,
                dtype=mx.float32,
            )
        )
        labels = (
            labels.astype(mx.int32)
            if _is_mlx_array(labels)
            else mx.array(
                labels,
                dtype=mx.int32,
            )
        )
        if len(points.shape) == 2:
            points = points[None]
        if len(labels.shape) == 1:
            labels = labels[None]
        if tuple(points.shape[:2]) != tuple(labels.shape):
            raise ValueError(
                "points must have shape [B, P, 2] and labels must have shape [B, P]."
            )
        if points.shape[0] != 1:
            raise_unsupported_multiplex_runtime(
                "VideoTrackingMultiplexDemo.add_new_points(batch_size)"
            )
        if rel_coordinates:
            points = points * float(self.image_size)

        multiplex_state = inference_state.get("multiplex_state")
        state_obj_ids = (
            []
            if multiplex_state is None
            else list(getattr(multiplex_state, "object_ids", None) or [])
        )
        is_new_state = multiplex_state is None
        is_existing_object = obj_id in state_obj_ids
        is_dynamic_new_object = not is_new_state and not is_existing_object
        is_refinement = (
            not is_new_state
            and inference_state["tracking_has_started"]
            and is_existing_object
        )
        existing_multi_object_edit = False
        existing_multi_object_gap_fill = False
        if (
            not is_new_state
            and is_existing_object
            and self._get_obj_num(inference_state) != 1
        ):
            existing_multi_object_edit = True
            frame_has_existing_output = (
                frame_idx in inference_state["output_dict"]["cond_frame_outputs"]
                or frame_idx in inference_state["output_dict"]["non_cond_frame_outputs"]
            )
            if not frame_has_existing_output:
                existing_multi_object_gap_fill = True

        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        obj_ids = [obj_id]
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]
        old_point_inputs = (
            None if clear_old_points else point_inputs_per_frame.get(frame_idx)
        )
        point_inputs = concat_points(old_point_inputs, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        if is_new_state:
            multiplex_state = self._get_or_create_multiplex_state(
                inference_state,
                obj_ids,
                is_new_state=True,
                reconditioning=False,
            )

        batch_size = self._get_obj_num(inference_state)
        run_batch_size = (
            1
            if (
                is_dynamic_new_object
                or (existing_multi_object_edit and not existing_multi_object_gap_fill)
            )
            else batch_size
        )
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_init_cond_frame
            else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        )
        user_refined_frames = None
        if is_refinement and not is_init_cond_frame:
            user_refined_frames_map = inference_state.setdefault(
                "user_refined_frames_per_obj",
                {},
            )
            user_refined_frames = user_refined_frames_map.setdefault(obj_id, set())
        is_first_refinement = (
            user_refined_frames is not None and frame_idx not in user_refined_frames
        )
        run_is_init_cond_frame = (
            False
            if existing_multi_object_gap_fill
            else is_dynamic_new_object or is_init_cond_frame or is_first_refinement
        )
        is_cond = (
            is_init_cond_frame
            or is_refinement
            or self.add_all_frames_to_correct_as_cond
        )
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        prev_sam_mask_logits = None
        if user_refined_frames is not None and not is_first_refinement:
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            search_keys = (
                storage_key,
                "cond_frame_outputs",
                "non_cond_frame_outputs",
            )
            prev_out = None
            for container in (obj_temp_output_dict, obj_output_dict):
                searched = set()
                for candidate_key in search_keys:
                    if candidate_key in searched:
                        continue
                    searched.add(candidate_key)
                    prev_out = container[candidate_key].get(frame_idx)
                    if prev_out is not None:
                        break
                if prev_out is not None:
                    break
            if prev_out is not None and prev_out.get("pred_masks") is not None:
                prev_sam_mask_logits = mx.clip(prev_out["pred_masks"], -32.0, 32.0)

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=inference_state["output_dict"],
            frame_idx=frame_idx,
            batch_size=run_batch_size,
            is_init_cond_frame=run_is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=False if is_dynamic_new_object else reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            add_to_existing_state=is_dynamic_new_object,
            new_object_idxs=[obj_idx]
            if is_dynamic_new_object
            or existing_multi_object_edit
            or existing_multi_object_gap_fill
            else None,
            new_object_ids=[obj_id]
            if is_dynamic_new_object
            or existing_multi_object_edit
            or existing_multi_object_gap_fill
            else None,
            allow_new_buckets=is_dynamic_new_object,
            prefer_new_buckets=is_dynamic_new_object,
            reconditioning=(
                existing_multi_object_edit and not existing_multi_object_gap_fill
            ),
            objects_to_interact=[obj_idx]
            if existing_multi_object_edit or existing_multi_object_gap_fill
            else None,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state,
            current_out["pred_masks"],
        )
        current_out["pred_masks_video_res"] = video_res_masks
        current_out["local_obj_id_to_idx"] = OrderedDict(
            inference_state["obj_id_to_idx"]
        )

        if is_cond:
            inference_state["output_dict"]["non_cond_frame_outputs"].pop(
                frame_idx, None
            )
            inference_state["consolidated_frame_inds"][
                "non_cond_frame_outputs"
            ].discard(frame_idx)
        inference_state["output_dict"][storage_key][frame_idx] = current_out
        inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = frame_idx
        if user_refined_frames is not None:
            user_refined_frames.add(frame_idx)

        self._add_output_per_object(
            inference_state, frame_idx, current_out, storage_key
        )
        obj_temp = inference_state["temp_output_dict_per_obj"][obj_idx][storage_key]
        obj_temp[frame_idx] = dict(
            inference_state["output_dict_per_obj"][obj_idx][storage_key][frame_idx]
        )
        _eval_tree(current_out, video_res_masks)
        return frame_idx, list(inference_state["obj_ids"]), None, video_res_masks

    def add_new_masks(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_ids: Any,
        masks: Any,
        add_mask_to_memory: bool = False,
        reconditioning: bool = False,
    ) -> tuple[int, list[Any], None, Any]:
        del add_mask_to_memory
        obj_ids = _coerce_obj_id_list(obj_ids)
        if len(obj_ids) == 0:
            raise ValueError("obj_ids must contain at least one object id.")
        obj_idxs = [
            self._obj_id_to_idx(
                inference_state,
                obj_id,
                error_if_new=reconditioning,
            )
            for obj_id in obj_ids
        ]
        masks = (
            masks.astype(mx.float32)
            if _is_mlx_array(masks)
            else mx.array(
                masks,
                dtype=mx.float32,
            )
        )
        if len(masks.shape) != 3:
            raise ValueError("masks must have shape [N, H, W].")
        num_objects, mask_h, mask_w = masks.shape
        if num_objects != len(obj_ids):
            raise ValueError(
                "masks batch must match obj_ids length; "
                f"got {num_objects} masks for {len(obj_ids)} ids."
            )

        masks_orig = masks[:, None, :, :]
        input_mask_size = int(getattr(self, "input_mask_size", self.image_size))
        if (int(mask_h), int(mask_w)) == (input_mask_size, input_mask_size):
            mask_inputs = masks_orig
        else:
            mask_inputs = interpolate(
                masks_orig,
                size=(input_mask_size, input_mask_size),
                mode="bilinear",
                align_corners=False,
            )

        video_h = int(inference_state["video_height"])
        video_w = int(inference_state["video_width"])
        if (int(mask_h), int(mask_w)) == (video_h, video_w):
            mask_inputs_video_res = masks_orig
        else:
            mask_inputs_video_res = interpolate(
                masks_orig,
                size=(video_h, video_w),
                mode="bilinear",
                align_corners=False,
            )
        mask_inputs_video_res = mask_inputs_video_res > 0.5

        multiplex_state = inference_state.get("multiplex_state")
        is_new_state = multiplex_state is None
        multiplex_state = self._get_or_create_multiplex_state(
            inference_state,
            obj_ids,
            is_new_state=is_new_state,
            reconditioning=reconditioning,
        )

        for mask_idx, obj_idx in enumerate(obj_idxs):
            inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = (
                mask_inputs_video_res[mask_idx : mask_idx + 1]
            )
            inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = (
            False
            if is_init_cond_frame
            else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        )
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        allow_new_buckets = (
            not is_new_state
            and not reconditioning
            and multiplex_state.available_slots < num_objects
        )
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=inference_state["output_dict"],
            frame_idx=frame_idx,
            batch_size=num_objects,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
            add_to_existing_state=not is_new_state and not reconditioning,
            new_object_masks=(None if is_new_state else mask_inputs),
            new_object_idxs=(None if is_new_state else obj_idxs),
            new_object_ids=(None if is_new_state else obj_ids),
            allow_new_buckets=allow_new_buckets,
            reconditioning=reconditioning,
        )

        _, video_res_masks = self._get_orig_video_res_output(
            inference_state,
            current_out["pred_masks"],
        )
        direct_logits = mx.where(
            mask_inputs_video_res,
            mx.array(-NO_OBJ_SCORE, dtype=video_res_masks.dtype),
            mx.array(NO_OBJ_SCORE, dtype=video_res_masks.dtype),
        )
        video_res_masks = _replace_indices(video_res_masks, obj_idxs, direct_logits)
        current_out["pred_masks_video_res"] = video_res_masks
        current_out["local_obj_id_to_idx"] = OrderedDict(
            inference_state["obj_id_to_idx"]
        )

        if is_cond:
            inference_state["output_dict"]["non_cond_frame_outputs"].pop(
                frame_idx, None
            )
            inference_state["consolidated_frame_inds"][
                "non_cond_frame_outputs"
            ].discard(frame_idx)
        inference_state["output_dict"][storage_key][frame_idx] = current_out
        inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = frame_idx

        self._add_output_per_object(
            inference_state, frame_idx, current_out, storage_key
        )
        for obj_idx in obj_idxs:
            obj_temp = inference_state["temp_output_dict_per_obj"][obj_idx][storage_key]
            obj_temp[frame_idx] = dict(
                inference_state["output_dict_per_obj"][obj_idx][storage_key][frame_idx]
            )

        _eval_tree(current_out, video_res_masks)
        return frame_idx, list(inference_state["obj_ids"]), None, video_res_masks

    def propagate_in_video(
        self,
        inference_state: dict[str, Any],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        tqdm_disable: bool = False,
        obj_ids: Any = None,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = False,
    ) -> Any:
        del tqdm_disable
        if propagate_preflight:
            preflight = getattr(self, "propagate_in_video_preflight", None)
            if preflight is not None:
                preflight(inference_state, run_mem_encoder=run_mem_encoder)
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        if inference_state.get("multiplex_state") is None:
            raise ValueError("inference_state['multiplex_state'] is required.")
        inference_state["tracking_has_started"] = True

        batch_size = self._get_obj_num(inference_state)
        output_obj_ids, output_obj_indices = self._select_propagation_output_obj_ids(
            inference_state,
            obj_ids,
        )
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
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
                if clear_non_cond_mem:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
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
                current_out["local_obj_id_to_idx"] = OrderedDict(
                    inference_state["obj_id_to_idx"]
                )
                output_dict[storage_key][frame_idx] = current_out

            self._add_output_per_object(
                inference_state,
                frame_idx,
                current_out,
                storage_key,
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            low_res_masks, video_res_masks = self._get_orig_video_res_output(
                inference_state,
                pred_masks,
            )
            if output_obj_indices is not None:
                low_res_masks = _take_indices(low_res_masks, output_obj_indices)
                video_res_masks = _take_indices(video_res_masks, output_obj_indices)
                _eval_tree(low_res_masks, video_res_masks)
            yield frame_idx, output_obj_ids, low_res_masks, video_res_masks

    def clear_all_points_in_video(self, inference_state: dict[str, Any]) -> None:
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["multiplex_state"] = None
        inference_state.get("user_refined_frames_per_obj", {}).clear()

    def _reset_tracking_results(self, inference_state: dict[str, Any]) -> None:
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

    def _clear_non_cond_mem_around_input(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
    ) -> None:
        radius = int(self.memory_temporal_stride_for_eval) * int(self.num_maskmem)
        frame_idx_begin = int(frame_idx) - radius
        frame_idx_end = int(frame_idx) + radius
        output_dict = inference_state["output_dict"]
        for time_idx in range(frame_idx_begin, frame_idx_end + 1):
            output_dict["non_cond_frame_outputs"].pop(time_idx, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(time_idx, None)
            inference_state["consolidated_frame_inds"][
                "non_cond_frame_outputs"
            ].discard(time_idx)

    def clear_all_points_in_frame(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_id: Any,
        need_output: bool = True,
        preserve_user_refined: bool = False,
    ) -> Any:
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        if not preserve_user_refined:
            inference_state.get("user_refined_frames_per_obj", {}).get(
                obj_id,
                set(),
            ).discard(frame_idx)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        batch_size = len(inference_state["obj_ids"])
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"].get(obj_idx2, {}):
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"].get(obj_idx2, {}):
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
                consolidated_frame_inds["non_cond_frame_outputs"].add(frame_idx)
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"].get(obj_idx2)
                if obj_output_dict is None:
                    continue
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return None
        for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            current_out = inference_state["output_dict"][storage_key].get(frame_idx)
            if current_out is not None:
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state,
                    current_out["pred_masks"],
                )
                return (
                    frame_idx,
                    list(inference_state["obj_ids"]),
                    None,
                    video_res_masks,
                )
        return None

    def remove_object(
        self,
        inference_state: dict[str, Any],
        obj_id: Any,
        strict: bool = False,
        need_output: bool = True,
        clear_user_refined_map: bool = True,
    ) -> Any:
        return self.remove_objects(
            inference_state,
            [obj_id],
            strict=strict,
            need_output=need_output,
            clear_user_refined_map=clear_user_refined_map,
        )

    def remove_objects(
        self,
        inference_state: dict[str, Any],
        obj_ids: Any,
        strict: bool = False,
        need_output: bool = True,
        clear_user_refined_map: bool = True,
    ) -> Any:
        requested_obj_ids = _coerce_obj_id_list(obj_ids)
        obj_id_to_idx = inference_state["obj_id_to_idx"]
        old_obj_idxs_to_rm = [
            obj_id_to_idx.get(obj_id, None) for obj_id in requested_obj_ids
        ]
        missing = [
            obj_id
            for obj_id, old_idx in zip(requested_obj_ids, old_obj_idxs_to_rm)
            if old_idx is None
        ]
        if missing and strict:
            raise ValueError(
                f"Object ids {missing} do not exist in the tracking state."
            )

        used_pairs = [
            (int(old_idx), obj_id)
            for old_idx, obj_id in zip(old_obj_idxs_to_rm, requested_obj_ids)
            if old_idx is not None
        ]
        updated_frames = []
        if not used_pairs:
            return list(inference_state["obj_ids"]), updated_frames

        if len(used_pairs) == len(inference_state["obj_ids"]):
            self.clear_all_points_in_video(inference_state)
            return list(inference_state["obj_ids"]), updated_frames

        old_obj_idxs = [pair[0] for pair in used_pairs]
        removed_obj_ids = [pair[1] for pair in used_pairs]
        removed_obj_id_set = set(removed_obj_ids)

        if clear_user_refined_map:
            user_refined_map = inference_state.get("user_refined_frames_per_obj", {})
            for obj_id in removed_obj_ids:
                user_refined_map.pop(obj_id, None)

        all_obj_input_frames_inds = set()
        for old_obj_idx, obj_id in used_pairs:
            point_frames = set(inference_state["point_inputs_per_obj"][old_obj_idx])
            mask_frames = set(inference_state["mask_inputs_per_obj"][old_obj_idx])
            obj_input_frames = point_frames | mask_frames
            for input_frame_idx in obj_input_frames:
                self.clear_all_points_in_frame(
                    inference_state,
                    input_frame_idx,
                    obj_id,
                    need_output=False,
                    preserve_user_refined=not clear_user_refined_map,
                )
            all_obj_input_frames_inds.update(obj_input_frames)

        old_obj_ids = list(inference_state["obj_ids"])
        remain_old_obj_inds = [
            old_idx
            for old_idx in range(len(old_obj_ids))
            if old_idx not in old_obj_idxs
        ]
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = OrderedDict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = OrderedDict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        def _map_keys(container: dict[int, Any]) -> None:
            old_values = dict(container)
            container.clear()
            for old_idx in remain_old_obj_inds:
                if old_idx in old_values:
                    container[old_idx_to_new_idx[old_idx]] = old_values[old_idx]

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        multiplex_state = inference_state.get("multiplex_state")
        if multiplex_state is None:
            raise ValueError("inference_state['multiplex_state'] is required.")
        buckets_to_keep = multiplex_state.remove_objects(old_obj_idxs, strict=True)

        def _slice_packed_output(out: dict[str, Any]) -> None:
            if buckets_to_keep:
                out["maskmem_features"] = _take_indices(
                    out.get("maskmem_features"),
                    buckets_to_keep,
                )
                out["maskmem_pos_enc"] = _take_indices(
                    out.get("maskmem_pos_enc"),
                    buckets_to_keep,
                )
                out["obj_ptr"] = _take_indices(out.get("obj_ptr"), buckets_to_keep)
            else:
                out["maskmem_features"] = None
                out["maskmem_pos_enc"] = None
                out["obj_ptr"] = None

            local_obj_id_to_idx = OrderedDict(
                out.get("local_obj_id_to_idx", inference_state["obj_id_to_idx"])
            )
            keep_indices = [
                int(obj_idx)
                for obj_id, obj_idx in local_obj_id_to_idx.items()
                if obj_id not in removed_obj_id_set
                and 0 <= int(obj_idx) < out["pred_masks"].shape[0]
                and 0 <= int(obj_idx) < out["object_score_logits"].shape[0]
            ]
            old_to_new = {
                old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)
            }
            out["pred_masks"] = _take_indices(out["pred_masks"], keep_indices)
            out["object_score_logits"] = _take_indices(
                out["object_score_logits"],
                keep_indices,
            )
            if "iou_score" in out:
                out["iou_score"] = _take_indices(out["iou_score"], keep_indices)
            if "eff_iou_score" in out:
                out["eff_iou_score"] = _take_indices(
                    out["eff_iou_score"],
                    keep_indices,
                )
            if "pred_masks_video_res" in out:
                out["pred_masks_video_res"] = _take_indices(
                    out["pred_masks_video_res"],
                    keep_indices,
                )
            out["local_obj_id_to_idx"] = OrderedDict(
                (obj_id, old_to_new[int(obj_idx)])
                for obj_id, obj_idx in local_obj_id_to_idx.items()
                if obj_id not in removed_obj_id_set and int(obj_idx) in old_to_new
            )
            conditioning_objects = set(out.get("conditioning_objects", set()))
            out["conditioning_objects"] = {
                old_to_new[old_idx]
                for old_idx in conditioning_objects
                if old_idx in old_to_new
            }

        def _slice_state(storage_key: str) -> None:
            for frame_idx, out in inference_state["output_dict"][storage_key].items():
                _slice_packed_output(out)
                self._add_output_per_object(
                    inference_state,
                    frame_idx,
                    out,
                    storage_key,
                )

        _slice_state("cond_frame_outputs")
        _slice_state("non_cond_frame_outputs")

        if need_output:
            for frame_idx in sorted(all_obj_input_frames_inds):
                for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                    out = inference_state["output_dict"][storage_key].get(frame_idx)
                    if out is None:
                        continue
                    _, video_res_masks = self._get_orig_video_res_output(
                        inference_state,
                        out["pred_masks"],
                    )
                    updated_frames.append((frame_idx, video_res_masks))
                    break

        return list(inference_state["obj_ids"]), updated_frames


class Sam3VideoTrackingMultiplexDemo(VideoTrackingMultiplexDemo):
    def init_state(
        self,
        video_height: int | None = None,
        video_width: int | None = None,
        num_frames: int | None = None,
        cached_features: Any = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        video_path: Any = None,
        async_loading_frames: bool = False,
        use_torchcodec: bool = False,
        use_cv2: bool = False,
    ) -> Any:
        if video_path is not None:
            return _load_multiplex_demo_state_from_resource(
                self,
                video_path=video_path,
                cached_features=cached_features,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                async_loading_frames=async_loading_frames,
                use_torchcodec=use_torchcodec,
                use_cv2=use_cv2,
            )
        if offload_video_to_cpu or offload_state_to_cpu:
            raise_unsupported_multiplex_runtime(
                "Sam3VideoTrackingMultiplexDemo.init_state(offload)"
            )
        if async_loading_frames:
            raise_unsupported_multiplex_runtime(
                "Sam3VideoTrackingMultiplexDemo.init_state(async_loading_frames)"
            )
        if use_torchcodec:
            raise_unsupported_multiplex_runtime(
                "Sam3VideoTrackingMultiplexDemo.init_state(use_torchcodec=True)"
            )
        if video_height is None or video_width is None or num_frames is None:
            raise ValueError(
                "video_height, video_width, and num_frames are required when "
                "video_path is not provided."
            )
        del use_cv2
        return _init_multiplex_demo_state(
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            cached_features=cached_features,
        )

    def propagate_in_video(
        self,
        inference_state: dict[str, Any],
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        reverse: bool = False,
        tqdm_disable: bool = False,
        obj_ids: Any = None,
        run_mem_encoder: bool = True,
        propagate_preflight: bool = False,
    ) -> Any:
        output_obj_indices = None
        base_outputs = super().propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
            tqdm_disable=tqdm_disable,
            obj_ids=obj_ids,
            run_mem_encoder=run_mem_encoder,
            propagate_preflight=propagate_preflight,
        )
        for frame_idx, output_obj_ids, low_res_masks, video_res_masks in base_outputs:
            if obj_ids is not None and output_obj_indices is None:
                _, output_obj_indices = self._select_propagation_output_obj_ids(
                    inference_state,
                    obj_ids,
                )
            output_dict = inference_state["output_dict"]
            current_out = output_dict["cond_frame_outputs"].get(frame_idx)
            if current_out is None:
                current_out = output_dict["non_cond_frame_outputs"][frame_idx]
            obj_scores = current_out["object_score_logits"]
            if output_obj_indices is not None:
                obj_scores = _take_indices(obj_scores, output_obj_indices)
                _eval_tree(obj_scores)
            yield frame_idx, output_obj_ids, low_res_masks, video_res_masks, obj_scores
