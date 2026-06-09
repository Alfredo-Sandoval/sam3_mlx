from __future__ import annotations

import os
from typing import Any, List

import mlx.core as mx
import numpy as np

from sam3_mlx.model.data_misc import BatchedDatapoint, FindStage
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.model_misc import SAM3Output
from sam3_mlx.model.sam3_multiplex_detector_utils import nms_masks
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.multiplex_utils import raise_unsupported_multiplex_runtime
from sam3_mlx.model.vl_combiner import SAM3VLBackbone

try:
    from sam3_mlx.model.vl_combiner import SAM3VLBackboneTri
except ImportError:  # pragma: no cover - mirrors optional upstream import
    SAM3VLBackboneTri = None


class Sam3MultiplexImageBase(Sam3Image):
    """Image-on-video wrapper around the Sam3Image model."""

    def __init__(
        self,
        *args: Any,
        tracking_score_thresh: float = 0.0,
        offload_outputs_to_cpu_for_eval: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.tracking_score_thresh = tracking_score_thresh
        self.offload_outputs_to_cpu_for_eval = offload_outputs_to_cpu_for_eval
        self.trim_outputs_for_eval = True

    def forward(
        self,
        input: BatchedDatapoint,
        is_inference: bool = False,
    ) -> tuple[SAM3Output, None]:
        del is_inference
        assert not getattr(self, "training", False), (
            "Sam3MultiplexImageBase should only be used in eval mode."
        )

        backbone_out = {"img_batch_all_stages": input.img_batch}
        text_outputs = self.backbone.forward_text(
            input.find_text_batch, device=self.device
        )
        backbone_out.update(text_outputs)

        previous_stages_out = SAM3Output(
            iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE
        )
        for frame_idx, find_input in enumerate(input.find_inputs):
            find_target = input.find_targets[frame_idx]
            geometric_prompt = self._get_geo_prompt_from_find_input(find_input)
            cur_out, _ = self.forward_video_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                find_target=find_target,
                geometric_prompt=geometric_prompt,
            )
            if self.offload_outputs_to_cpu_for_eval:
                cur_out = {key: np.asarray(value) for key, value in cur_out.items()}
            previous_stages_out.append([cur_out])

        return previous_stages_out, None

    def forward_video_grounding(self, *args: Any, **kwargs: Any) -> Any:
        grounding_out = self.forward_grounding(*args, **kwargs)
        out = {
            "pred_logits": grounding_out["pred_logits"],
            "pred_boxes": grounding_out["pred_boxes"],
            "pred_boxes_xyxy": grounding_out["pred_boxes_xyxy"],
            "pred_masks": grounding_out["pred_masks"],
            "pred_object_ids": self._get_dummy_object_ids(grounding_out["pred_logits"]),
        }
        if "prev_encoder_out" in grounding_out:
            out["prev_encoder_out"] = grounding_out["prev_encoder_out"]
        return out, kwargs.get("backbone_out")

    def _get_dummy_object_ids(self, pred_logits: Any) -> Any:
        batch_size, num_queries, _ = pred_logits.shape
        is_above_thresh = pred_logits.squeeze(2) > self.tracking_score_thresh
        dummy_obj_ids = mx.broadcast_to(
            mx.arange(num_queries, dtype=mx.int64)[None, :],
            (batch_size, num_queries),
        )
        return mx.where(is_above_thresh, dummy_obj_ids, mx.array(-1, dtype=mx.int64))

    def _trim_outputs(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    def _batch_find_inputs(
        self,
        find_inputs: List[FindStage],
        chunk_start: int,
        chunk_end: int,
    ) -> FindStage:
        chunk_find_inputs = [
            find_inputs[i % len(find_inputs)] for i in range(chunk_start, chunk_end)
        ]
        dtype = chunk_find_inputs[0].img_ids.dtype
        batched_img_ids = mx.arange(chunk_start, chunk_end, dtype=dtype)
        batched_img_ids_np = np.arange(chunk_start, chunk_end)

        def batch_tensors(values: list[Any], axis: int = 0) -> Any:
            if values[0] is None:
                return None
            return mx.concat(values, axis=axis)

        return FindStage(
            img_ids=batched_img_ids,
            img_ids_np=batched_img_ids_np,
            text_ids=batch_tensors(
                [find_input.text_ids for find_input in chunk_find_inputs]
            ),
            input_boxes=batch_tensors(
                [find_input.input_boxes for find_input in chunk_find_inputs]
            ),
            input_boxes_mask=batch_tensors(
                [find_input.input_boxes_mask for find_input in chunk_find_inputs]
            ),
            input_boxes_label=batch_tensors(
                [find_input.input_boxes_label for find_input in chunk_find_inputs]
            ),
            input_points=batch_tensors(
                [find_input.input_points for find_input in chunk_find_inputs]
            ),
            input_points_mask=batch_tensors(
                [find_input.input_points_mask for find_input in chunk_find_inputs]
            ),
            ptrs=None,
            ptrs_seg=None,
            object_ids=None,
            input_boxes_before_embed=batch_tensors(
                [
                    find_input.input_boxes_before_embed
                    for find_input in chunk_find_inputs
                ]
            ),
            input_points_before_embed=batch_tensors(
                [
                    find_input.input_points_before_embed
                    for find_input in chunk_find_inputs
                ]
            ),
        )

    def _batch_geometric_prompts(
        self,
        geometric_prompts: List[Prompt],
        chunk_start: int,
        chunk_end: int,
    ) -> Prompt:
        chunk_prompts = [geometric_prompts[i] for i in range(chunk_start, chunk_end)]
        return self._batch_geometric_prompts_from_list(chunk_prompts)

    def _batch_geometric_prompts_from_list(self, chunk_prompts: List[Prompt]) -> Prompt:
        def batch_tensors(values: list[Any], axis: int) -> Any:
            if values[0] is None:
                return None
            return mx.concat(values, axis=axis)

        return Prompt(
            box_embeddings=batch_tensors(
                [prompt.box_embeddings for prompt in chunk_prompts],
                axis=1,
            ),
            box_mask=batch_tensors(
                [prompt.box_mask for prompt in chunk_prompts], axis=0
            ),
            box_labels=batch_tensors(
                [prompt.box_labels for prompt in chunk_prompts],
                axis=1,
            ),
            point_embeddings=batch_tensors(
                [prompt.point_embeddings for prompt in chunk_prompts],
                axis=1,
            ),
            point_mask=batch_tensors(
                [prompt.point_mask for prompt in chunk_prompts],
                axis=0,
            ),
            point_labels=batch_tensors(
                [prompt.point_labels for prompt in chunk_prompts],
                axis=1,
            ),
        )


def _is_mlx_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("mlx.")


class Sam3MultiplexDetector(Sam3MultiplexImageBase):
    """Fail-fast shell for the distributed multiplex detector."""

    def __init__(
        self,
        *args: Any,
        async_all_gather: bool = True,
        gather_backbone_out: Any = None,
        is_multiplex: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.async_all_gather = async_all_gather
        if gather_backbone_out is None:
            gather_backbone_out = isinstance(self.backbone, SAM3VLBackbone) or (
                SAM3VLBackboneTri is not None
                and isinstance(self.backbone, SAM3VLBackboneTri)
            )
        self.gather_backbone_out = gather_backbone_out
        self.is_multiplex = is_multiplex

    def forward_video_grounding_multigpu(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            positional_names = (
                "backbone_out",
                "find_inputs",
                "geometric_prompt",
                "frame_idx",
                "num_frames",
                "multigpu_buffer",
            )
            if len(args) > len(positional_names):
                raise TypeError(
                    "forward_video_grounding_multigpu accepts at most six "
                    "positional arguments."
                )
            for name, value in zip(positional_names, args):
                if name in kwargs:
                    raise TypeError(f"{name} passed both positionally and by keyword.")
                kwargs[name] = value

        backbone_out = kwargs.pop("backbone_out")
        find_inputs = kwargs.pop("find_inputs")
        geometric_prompt = kwargs.pop("geometric_prompt")
        frame_idx = int(kwargs.pop("frame_idx"))
        num_frames = int(kwargs.pop("num_frames"))
        multigpu_buffer = kwargs.pop("multigpu_buffer", None)
        if multigpu_buffer is None:
            multigpu_buffer = {}
        track_in_reverse = bool(kwargs.pop("track_in_reverse", False))
        return_sam2_backbone_feats = bool(
            kwargs.pop("return_sam2_backbone_feats", False)
        )
        run_nms = bool(kwargs.pop("run_nms", False))
        nms_prob_thresh = kwargs.pop("nms_prob_thresh", None)
        nms_iou_thresh = kwargs.pop("nms_iou_thresh", None)
        nms_use_iom = bool(kwargs.pop("nms_use_iom", False))
        max_frame_num_to_track = kwargs.pop("max_frame_num_to_track", None)
        propagate_start = kwargs.pop("propagate_in_video_start_frame_idx", None)
        kwargs.pop("feature_cache", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                "Unexpected forward_video_grounding_multigpu keyword "
                f"argument(s): {names}"
            )

        if self.rank != 0 or self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexDetector.forward_video_grounding_multigpu(distributed)"
            )
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )

        valid_start, valid_end = self._grounding_valid_frame_range(
            frame_idx=frame_idx,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            max_frame_num_to_track=max_frame_num_to_track,
            propagate_in_video_start_frame_idx=propagate_start,
        )
        if not valid_start <= frame_idx < valid_end:
            raise ValueError(
                f"frame_idx={frame_idx} is outside the valid grounding range "
                f"[{valid_start}, {valid_end})."
            )

        if frame_idx not in multigpu_buffer:
            self._build_multigpu_buffer_next_chunk(
                backbone_out=backbone_out,
                find_inputs=find_inputs,
                geometric_prompt=geometric_prompt,
                frame_idx_begin=frame_idx,
                frame_idx_end=frame_idx + 1,
                num_frames=num_frames,
                multigpu_buffer=multigpu_buffer,
                run_nms=run_nms,
                nms_prob_thresh=nms_prob_thresh,
                nms_iou_thresh=nms_iou_thresh,
                nms_use_iom=nms_use_iom,
            )

        out = self._read_multigpu_buffer_frame(
            multigpu_buffer[frame_idx],
            return_sam2_backbone_feats=return_sam2_backbone_feats,
        )

        if not track_in_reverse:
            multigpu_buffer.pop(frame_idx - 1, None)
            next_frame_idx = frame_idx + 1
            should_prefetch = next_frame_idx < valid_end
        else:
            multigpu_buffer.pop(frame_idx + 1, None)
            next_frame_idx = frame_idx - 1
            should_prefetch = next_frame_idx >= valid_start
        if should_prefetch and next_frame_idx not in multigpu_buffer:
            self._build_multigpu_buffer_next_chunk(
                backbone_out=backbone_out,
                find_inputs=find_inputs,
                geometric_prompt=geometric_prompt,
                frame_idx_begin=next_frame_idx,
                frame_idx_end=next_frame_idx + 1,
                num_frames=num_frames,
                multigpu_buffer=multigpu_buffer,
                run_nms=run_nms,
                nms_prob_thresh=nms_prob_thresh,
                nms_iou_thresh=nms_iou_thresh,
                nms_use_iom=nms_use_iom,
            )
        return out, backbone_out

    def _build_multigpu_buffer_next_chunk(
        self,
        *,
        backbone_out: dict[str, Any],
        find_inputs: list[FindStage],
        geometric_prompt: Prompt,
        frame_idx_begin: int,
        frame_idx_end: int,
        num_frames: int,
        multigpu_buffer: dict[int, dict[str, tuple[Any, None]]],
        run_nms: bool,
        nms_prob_thresh: Any,
        nms_iou_thresh: Any,
        nms_use_iom: bool,
    ) -> None:
        for frame_idx_to_save in range(frame_idx_begin, min(frame_idx_end, num_frames)):
            out_local, _ = self.forward_video_grounding(
                backbone_out=backbone_out,
                find_input=find_inputs[frame_idx_to_save % len(find_inputs)],
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
            self._attach_single_frame_backbone_outputs(out_local)
            if run_nms:
                out_local = self._suppress_logits_with_nms(
                    out_local,
                    nms_prob_thresh=nms_prob_thresh,
                    nms_iou_thresh=nms_iou_thresh,
                    nms_use_iom=nms_use_iom,
                )
            multigpu_buffer[frame_idx_to_save] = {
                key: (value, None) for key, value in out_local.items()
            }

    def _attach_single_frame_backbone_outputs(self, out: dict[str, Any]) -> None:
        prev_encoder_out = out.get("prev_encoder_out")
        if not isinstance(prev_encoder_out, dict):
            return
        backbone_data = prev_encoder_out.get("backbone_out", {})
        if not isinstance(backbone_data, dict):
            return

        sam2_backbone = backbone_data.get("sam2_backbone_out")
        if isinstance(sam2_backbone, dict):
            for level, feature in enumerate(sam2_backbone.get("backbone_fpn", [])):
                out[f"sam2_backbone_fpn_{level}"] = getattr(
                    feature,
                    "tensors",
                    feature,
                )
            if "vision_pos_enc" in sam2_backbone:
                out["sam2_backbone_pos_enc"] = [
                    getattr(value, "tensors", value)
                    for value in sam2_backbone.get("vision_pos_enc", [])
                ]

        interactive = backbone_data.get("interactive")
        if self.is_multiplex and isinstance(interactive, dict):
            for level, feature in enumerate(interactive.get("backbone_fpn", [])):
                out[f"interactive_backbone_fpn_{level}"] = getattr(
                    feature,
                    "tensors",
                    feature,
                )
            if "vision_pos_enc" in interactive:
                out["interactive_backbone_pos_enc"] = [
                    getattr(value, "tensors", value)
                    for value in interactive.get("vision_pos_enc", [])
                ]

    def _read_multigpu_buffer_frame(
        self,
        frame_buffer: dict[str, tuple[Any, Any]],
        *,
        return_sam2_backbone_feats: bool,
    ) -> dict[str, Any]:
        out = {}
        for key, (value, handle) in frame_buffer.items():
            if (
                key.startswith("sam2_backbone_")
                or key.startswith("interactive_backbone_")
                or key.startswith("propagation_backbone_")
            ) and not return_sam2_backbone_feats:
                continue
            if handle is not None:
                handle.wait()
            out[key] = value
        return out

    def _suppress_logits_with_nms(
        self,
        out: dict[str, Any],
        *,
        nms_prob_thresh: Any,
        nms_iou_thresh: Any,
        nms_use_iom: bool,
    ) -> dict[str, Any]:
        if nms_prob_thresh is None or nms_iou_thresh is None:
            raise ValueError(
                "nms_prob_thresh and nms_iou_thresh are required when run_nms=True."
            )
        out = out.copy()
        pred_probs = mx.sigmoid(out["pred_logits"].squeeze(-1))
        if len(pred_probs.shape) == 1:
            keep = nms_masks(
                pred_probs,
                out["pred_masks"],
                prob_threshold=float(nms_prob_thresh),
                iou_threshold=float(nms_iou_thresh),
                nms_use_iom=bool(nms_use_iom),
                do_compile=False,
            )
        elif len(pred_probs.shape) == 2:
            keep_rows = [
                nms_masks(
                    pred_probs[prompt_idx],
                    out["pred_masks"][prompt_idx],
                    prob_threshold=float(nms_prob_thresh),
                    iou_threshold=float(nms_iou_thresh),
                    nms_use_iom=bool(nms_use_iom),
                    do_compile=False,
                )
                for prompt_idx in range(pred_probs.shape[0])
            ]
            keep = (
                mx.stack(keep_rows, axis=0)
                if keep_rows
                else mx.zeros(pred_probs.shape, dtype=mx.bool_)
            )
        else:
            raise ValueError(
                "pred_logits must squeeze to shape (N,) or (B, N), "
                f"got {pred_probs.shape}."
            )
        out["pred_logits"] = mx.where(
            keep[..., None],
            out["pred_logits"],
            out["pred_logits"] - mx.array(1.0e4, dtype=out["pred_logits"].dtype),
        )
        return out

    def forward_video_grounding_batched_multigpu(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        if args:
            positional_names = (
                "backbone_out",
                "find_inputs",
                "geometric_prompt",
                "frame_idx",
                "num_frames",
                "grounding_cache",
            )
            if len(args) > len(positional_names):
                raise TypeError(
                    "forward_video_grounding_batched_multigpu accepts at most "
                    "six positional arguments."
                )
            for name, value in zip(positional_names, args):
                if name in kwargs:
                    raise TypeError(f"{name} passed both positionally and by keyword.")
                kwargs[name] = value

        backbone_out = kwargs.pop("backbone_out")
        find_inputs = kwargs.pop("find_inputs")
        kwargs.pop("geometric_prompt")
        frame_idx = int(kwargs.pop("frame_idx"))
        num_frames = int(kwargs.pop("num_frames"))
        grounding_cache = kwargs.pop("grounding_cache")
        track_in_reverse = bool(kwargs.pop("track_in_reverse", False))
        return_sam2_backbone_feats = bool(
            kwargs.pop("return_sam2_backbone_feats", False)
        )
        run_nms = bool(kwargs.pop("run_nms", False))
        nms_prob_thresh = kwargs.pop("nms_prob_thresh", None)
        nms_iou_thresh = kwargs.pop("nms_iou_thresh", None)
        nms_use_iom = bool(kwargs.pop("nms_use_iom", False))
        max_frame_num_to_track = kwargs.pop("max_frame_num_to_track", None)
        propagate_start = kwargs.pop("propagate_in_video_start_frame_idx", None)
        kwargs.pop("feature_cache", None)
        batch_size = int(kwargs.pop("batch_size", 16))
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(
                "Unexpected forward_video_grounding_batched_multigpu keyword "
                f"argument(s): {names}"
            )

        if self.rank != 0 or self.world_size != 1:
            raise_unsupported_multiplex_runtime(
                "Sam3MultiplexDetector.forward_video_grounding_batched_multigpu(distributed)"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if not 0 <= frame_idx < num_frames:
            raise ValueError(
                f"frame_idx={frame_idx} is out of range for {num_frames} frames."
            )

        valid_start, valid_end = self._grounding_valid_frame_range(
            frame_idx=frame_idx,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            max_frame_num_to_track=max_frame_num_to_track,
            propagate_in_video_start_frame_idx=propagate_start,
        )
        if not valid_start <= frame_idx < valid_end:
            raise ValueError(
                f"frame_idx={frame_idx} is outside the valid grounding range "
                f"[{valid_start}, {valid_end})."
            )

        grounding_buffer = grounding_cache.setdefault("grounding_buffer", {})
        chunk_start = max(valid_start, (frame_idx // batch_size) * batch_size)
        chunk_end = min(chunk_start + batch_size, valid_end)
        chunk_key = (chunk_start, chunk_end)

        if chunk_key not in grounding_buffer:
            grounding_buffer[chunk_key] = self._process_grounding_chunk_batched(
                backbone_out=backbone_out,
                find_inputs=find_inputs,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                run_nms=run_nms,
                nms_prob_thresh=nms_prob_thresh,
                nms_iou_thresh=nms_iou_thresh,
                nms_use_iom=nms_use_iom,
                return_sam2_backbone_feats=return_sam2_backbone_feats,
            )
            self._cleanup_previous_chunks_multigpu(
                grounding_cache=grounding_cache,
                current_chunk_key=chunk_key,
                batch_size=batch_size,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )

        local_idx = frame_idx - chunk_start
        out = self._slice_batched_output(
            grounding_buffer[chunk_key],
            local_idx,
            return_sam2_backbone_feats=return_sam2_backbone_feats,
        )
        return out, backbone_out

    def _grounding_valid_frame_range(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        track_in_reverse: bool,
        max_frame_num_to_track: Any,
        propagate_in_video_start_frame_idx: Any,
    ) -> tuple[int, int]:
        del frame_idx
        if max_frame_num_to_track is None:
            return 0, num_frames

        start = (
            0
            if propagate_in_video_start_frame_idx is None
            else int(propagate_in_video_start_frame_idx)
        )
        max_frames = int(max_frame_num_to_track)
        if max_frames < 1:
            raise ValueError("max_frame_num_to_track must be >= 1 when provided.")
        if track_in_reverse:
            valid_start = max(0, start - max_frames + 1)
            valid_end = min(num_frames, start + 1)
        else:
            valid_start = max(0, start)
            valid_end = min(num_frames, start + max_frames)
        return valid_start, valid_end

    def _process_grounding_chunk_batched(
        self,
        *,
        backbone_out: dict[str, Any],
        find_inputs: list[FindStage],
        chunk_start: int,
        chunk_end: int,
        run_nms: bool,
        nms_prob_thresh: Any,
        nms_iou_thresh: Any,
        nms_use_iom: bool,
        return_sam2_backbone_feats: bool,
    ) -> dict[str, Any]:
        chunk_size = chunk_end - chunk_start
        if chunk_size < 1:
            raise ValueError("grounding chunks must contain at least one frame.")

        chunk_geo_prompts = [
            self._get_geo_prompt_from_find_input(find_inputs[index % len(find_inputs)])
            for index in range(chunk_start, chunk_end)
        ]
        batched_find_input = self._batch_find_inputs(
            find_inputs,
            chunk_start,
            chunk_end,
        )
        batched_geometric_prompt = self._batch_geometric_prompts_from_list(
            chunk_geo_prompts
        )
        out = self.forward_grounding(
            backbone_out=backbone_out,
            find_input=batched_find_input,
            find_target=None,
            geometric_prompt=batched_geometric_prompt,
        )

        if run_nms:
            out = self._suppress_logits_with_nms(
                out,
                nms_prob_thresh=nms_prob_thresh,
                nms_iou_thresh=nms_iou_thresh,
                nms_use_iom=nms_use_iom,
            )

        if return_sam2_backbone_feats and "prev_encoder_out" in out:
            backbone_data = out["prev_encoder_out"].get("backbone_out", {})
            if self.is_multiplex and "interactive" in backbone_data:
                out["_interactive_backbone"] = backbone_data["interactive"]
            if "sam2_backbone_out" in backbone_data:
                out["_sam2_backbone"] = backbone_data["sam2_backbone_out"]

        out["_chunk_size"] = chunk_size
        return out

    def _slice_batched_output(
        self,
        chunk_outputs: dict[str, Any],
        local_idx: int,
        *,
        return_sam2_backbone_feats: bool,
    ) -> dict[str, Any]:
        batch_dim_keys = {
            "pred_logits",
            "pred_boxes",
            "pred_boxes_xyxy",
            "pred_masks",
            "pred_logits_o2m",
            "pred_boxes_o2m",
            "pred_boxes_xyxy_o2m",
            "pred_masks_o2m",
            "queries",
            "presence_logit_dec",
        }
        skip_keys = {
            "_chunk_size",
            "_interactive_backbone",
            "_sam2_backbone",
            "prev_encoder_out",
            "encoder_hidden_states",
            "aux_outputs",
        }

        out: dict[str, Any] = {}
        for key, value in chunk_outputs.items():
            if key in skip_keys:
                continue
            if _is_mlx_array(value):
                if key in batch_dim_keys or value.ndim > 0:
                    out[key] = value[local_idx : local_idx + 1]
                else:
                    out[key] = value

        if "pred_logits" in out:
            out["pred_object_ids"] = self._get_dummy_object_ids(out["pred_logits"])

        if return_sam2_backbone_feats:
            self._slice_batched_backbone_outputs(chunk_outputs, local_idx, out)

        return out

    def _slice_batched_backbone_outputs(
        self,
        chunk_outputs: dict[str, Any],
        local_idx: int,
        out: dict[str, Any],
    ) -> None:
        if "_sam2_backbone" in chunk_outputs:
            sam2_backbone = chunk_outputs["_sam2_backbone"]
            for level, feature in enumerate(sam2_backbone.get("backbone_fpn", [])):
                tensors = getattr(feature, "tensors", feature)
                out[f"sam2_backbone_fpn_{level}"] = tensors[local_idx : local_idx + 1]
            out["sam2_backbone_pos_enc"] = [
                value[local_idx : local_idx + 1]
                for value in sam2_backbone.get("vision_pos_enc", [])
            ]

        if self.is_multiplex and "_interactive_backbone" in chunk_outputs:
            interactive = chunk_outputs["_interactive_backbone"]
            for level, feature in enumerate(interactive.get("backbone_fpn", [])):
                tensors = getattr(feature, "tensors", feature)
                out[f"interactive_backbone_fpn_{level}"] = tensors[
                    local_idx : local_idx + 1
                ]
            out["interactive_backbone_pos_enc"] = [
                value[local_idx : local_idx + 1]
                for value in interactive.get("vision_pos_enc", [])
            ]

    def _cleanup_previous_chunks_multigpu(
        self,
        *,
        grounding_cache: dict[str, Any],
        current_chunk_key: tuple[int, int],
        batch_size: int,
        num_frames: int,
        track_in_reverse: bool,
    ) -> None:
        chunk_start, chunk_end = current_chunk_key
        grounding_buffer = grounding_cache["grounding_buffer"]
        if not track_in_reverse:
            prev_chunk_key = (chunk_start - batch_size, chunk_start)
            if prev_chunk_key[0] >= 0:
                grounding_buffer.pop(prev_chunk_key, None)
        else:
            next_start = chunk_end
            if next_start < num_frames:
                next_chunk_key = (
                    next_start,
                    min(next_start + batch_size, num_frames),
                )
                grounding_buffer.pop(next_chunk_key, None)

    def _gather_tensor(self, x: Any) -> tuple[list[Any], None]:
        if self.world_size == 1:
            return [x], None
        raise_unsupported_multiplex_runtime("Sam3MultiplexDetector._gather_tensor")
