# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX-safe subset of the official SAM3 video base module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.sam3_tracker_utils import mask_to_box
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.rle import rle_encode


class MaskletConfirmationStatus(Enum):
    UNCONFIRMED = 1
    CONFIRMED = 2


def _is_mlx_array(value: Any) -> bool:
    return type(value).__module__.startswith("mlx.")


def _to_numpy(value: Any) -> np.ndarray:
    return to_numpy(value, copy=False)


def _from_numpy_like(value: np.ndarray, like: Any):
    if _is_mlx_array(like):
        if value.dtype == np.bool_:
            return mx.array(value, dtype=mx.bool_)
        if np.issubdtype(value.dtype, np.integer):
            return mx.array(value, dtype=mx.int64)
        return mx.array(value, dtype=mx.float32)
    return value


def _frame_size(frame: Any) -> tuple[int, int]:
    frame_size = getattr(frame, "size", None)
    if isinstance(frame_size, (tuple, list)) and len(frame_size) == 2:
        width, height = frame_size
        return int(height), int(width)
    shape = getattr(frame, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    array = np.asarray(frame)
    if array.ndim < 2:
        raise ValueError("video frame must expose PIL size or array H/W dimensions.")
    return int(array.shape[0]), int(array.shape[1])


def _scalar_float(value: Any) -> float:
    return float(_to_numpy(value).reshape(-1)[0])


def _scalar_int(value: Any) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


def _raise_video_base_unsupported(
    feature: str,
    *,
    reason: str = "video-multiplex",
    detail: str,
    alternative: str | None = None,
):
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
        alternative=alternative,
    )


def _mask_intersection_np(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    assert masks1.shape[1:] == masks2.shape[1:]
    m1_flat = masks1.reshape(masks1.shape[0], -1).astype(np.float32)
    m2_flat = masks2.reshape(masks2.shape[0], -1).astype(np.float32)
    return m1_flat @ m2_flat.T


def _mask_iou_np(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    intersection = _mask_intersection_np(masks1, masks2)
    area1 = masks1.reshape(masks1.shape[0], -1).sum(axis=1)
    area2 = masks2.reshape(masks2.shape[0], -1).sum(axis=1)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / np.clip(union, 1.0, None)


def _mask_iom_np(masks1: np.ndarray, masks2: np.ndarray) -> np.ndarray:
    intersection = _mask_intersection_np(masks1, masks2)
    area1 = masks1.reshape(masks1.shape[0], -1).sum(axis=1)
    area2 = masks2.reshape(masks2.shape[0], -1).sum(axis=1)
    min_area = np.minimum(area1[:, None], area2[None, :])
    return intersection / (min_area + 1.0e-8)


@dataclass
class RealizedAssociateDetTrkresult:
    new_det_fa_inds: np.array
    unmatched_trk_obj_ids: np.array
    det_to_matched_trk_obj_ids: Dict[int, np.array]
    trk_id_to_max_iou_high_conf_det: Dict[int, int]
    empty_trk_obj_ids: np.array
    new_det_obj_ids: Optional[np.array] = None
    new_det_gpu_ids: Optional[np.array] = None
    num_obj_dropped_due_to_limit: Optional[int] = None

    def get_new_det_gpu_ids(
        self, tracker_metadata_prev, is_image_only, det_scores, tracking_obj
    ):
        if self.new_det_obj_ids is None:
            det_scores_np = _to_numpy(det_scores)
            prev_obj_num = np.sum(tracker_metadata_prev["num_obj_per_gpu"])
            new_det_num = len(self.new_det_fa_inds)
            num_obj_dropped_due_to_limit = 0
            if (
                not is_image_only
                and prev_obj_num + new_det_num > tracking_obj.max_num_objects
            ):
                new_det_num_to_keep = tracking_obj.max_num_objects - prev_obj_num
                num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
                self.new_det_fa_inds = tracking_obj._drop_new_det_with_obj_limit(
                    self.new_det_fa_inds, det_scores_np, new_det_num_to_keep
                )
                assert len(self.new_det_fa_inds) == new_det_num_to_keep
                new_det_num = len(self.new_det_fa_inds)
            new_det_start_obj_id = tracker_metadata_prev["max_obj_id"] + 1
            new_det_obj_ids = new_det_start_obj_id + np.arange(new_det_num)
            if tracking_obj.is_multiplex:
                prev_workload_per_gpu = tracker_metadata_prev["num_buc_per_gpu"]
            else:
                prev_workload_per_gpu = tracker_metadata_prev["num_obj_per_gpu"]
            new_det_gpu_ids = tracking_obj._assign_new_det_to_gpus(
                new_det_num=new_det_num,
                prev_workload_per_gpu=prev_workload_per_gpu,
            )
            self.new_det_obj_ids = new_det_obj_ids
            self.new_det_gpu_ids = new_det_gpu_ids
            self.num_obj_dropped_due_to_limit = num_obj_dropped_due_to_limit
        return (
            self.new_det_obj_ids,
            self.new_det_gpu_ids,
            self.num_obj_dropped_due_to_limit,
        )


def realize_adt_result(adt_lazy_result, tracker_metadata_prev, det_mask_preds):
    if isinstance(adt_lazy_result, LazyAssociateDetTrkResult):
        adt_lazy_result._convert_to_numpy()
        return adt_lazy_result._create_cpu_metadata(
            tracker_metadata_prev["obj_ids_all_gpu"], det_mask_preds
        )
    return adt_lazy_result


class LazyAssociateDetTrkResult:
    def __init__(
        self,
        trk_is_unmatched,
        trk_is_nonempty,
        is_new_det,
        det_to_max_iou_trk_idx,
        det_is_high_conf,
        det_is_high_iou,
        det_keep,
        im_mask,
    ):
        self.trk_is_unmatched = trk_is_unmatched
        self.trk_is_nonempty = trk_is_nonempty
        self.is_new_det = is_new_det
        self.det_to_max_iou_trk_idx = det_to_max_iou_trk_idx
        self.det_is_high_conf = det_is_high_conf
        self.det_is_high_iou = det_is_high_iou
        self.det_keep = det_keep
        self.im_mask = im_mask

    def _convert_to_numpy(self):
        self.trk_is_unmatched = _to_numpy(self.trk_is_unmatched).astype(bool)
        self.trk_is_nonempty = _to_numpy(self.trk_is_nonempty).astype(bool)
        self.is_new_det = _to_numpy(self.is_new_det).astype(bool)
        self.det_to_max_iou_trk_idx = _to_numpy(self.det_to_max_iou_trk_idx)
        self.det_is_high_conf = _to_numpy(self.det_is_high_conf).astype(bool)
        self.det_is_high_iou = _to_numpy(self.det_is_high_iou).astype(bool)
        self.det_keep = _to_numpy(self.det_keep).astype(bool).tolist()
        self.im_mask = _to_numpy(self.im_mask).astype(bool)

    def _create_cpu_metadata(self, trk_obj_ids, det_masks):
        trk_obj_ids = _to_numpy(trk_obj_ids)
        det_masks_np = _to_numpy(det_masks)
        unmatched_trk_obj_ids = trk_obj_ids[self.trk_is_unmatched]
        empty_trk_obj_ids = trk_obj_ids[~self.trk_is_nonempty]
        new_det_fa_inds = np.nonzero(self.is_new_det)[0]
        det_is_high_conf_and_iou = set(
            np.nonzero(self.det_is_high_conf & self.det_is_high_iou)[0]
        )
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}
        for det_idx in range(det_masks_np.shape[0]):
            if self.det_keep[det_idx]:
                det_to_matched_trk_obj_ids[det_idx] = trk_obj_ids[
                    self.im_mask[det_idx, :]
                ]
                if det_idx in det_is_high_conf_and_iou:
                    trk_obj_id = trk_obj_ids[
                        self.det_to_max_iou_trk_idx[det_idx]
                    ].item()
                    trk_id_to_max_iou_high_conf_det[trk_obj_id] = det_idx
        return RealizedAssociateDetTrkresult(
            new_det_fa_inds=new_det_fa_inds,
            unmatched_trk_obj_ids=unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det=trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids=empty_trk_obj_ids,
        )


def _associate_det_trk_compilable(
    det_masks,
    det_scores,
    det_keep,
    trk_masks,
    new_det_thresh,
    iou_threshold_trk,
    iou_threshold,
    HIGH_CONF_THRESH,
    use_iom_recondition,
    o2o_matching_masklets_enable,
    iom_thresh_recondition,
    iou_thresh_recondition,
):
    if o2o_matching_masklets_enable:
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base._associate_det_trk_compilable(o2o_matching_masklets_enable=True)",
            detail=(
                "o2o_matching_masklets_enable requires the official Torch/SciPy "
                "Hungarian matching path and is not ported in sam3_mlx."
            ),
            alternative="o2o_matching_masklets_enable=False",
        )

    det_masks_np = _to_numpy(det_masks)
    trk_masks_np = _to_numpy(trk_masks)
    det_scores_np = _to_numpy(det_scores)
    det_keep_np = _to_numpy(det_keep).astype(bool)

    det_masks_binary = det_masks_np > 0
    det_masks_binary = det_masks_binary.copy()
    det_masks_binary[~det_keep_np] = False
    trk_masks_binary = trk_masks_np > 0

    if use_iom_recondition:
        intersection_metric = _mask_iom_np(det_masks_binary, trk_masks_binary)
    else:
        intersection_metric = _mask_iou_np(det_masks_binary, trk_masks_binary)

    if det_masks_binary.shape[0] == 0 or trk_masks_binary.shape[0] == 0:
        trk_is_nonempty = np.any(trk_masks_binary, axis=(1, 2))
        trk_is_unmatched = trk_is_nonempty.copy()
        is_new_det = (det_scores_np >= new_det_thresh) & det_keep_np
        det_to_max_iou_trk_idx = np.zeros(det_masks_binary.shape[0], dtype=np.int64)
        det_is_high_conf = (
            (det_scores_np >= HIGH_CONF_THRESH) & det_keep_np
        ) & ~is_new_det
        det_is_high_iou = np.zeros(det_masks_binary.shape[0], dtype=bool)
        im_mask = np.zeros(
            (det_masks_binary.shape[0], trk_masks_binary.shape[0]), dtype=bool
        )
        return (
            _from_numpy_like(trk_is_unmatched, det_masks),
            _from_numpy_like(trk_is_nonempty, det_masks),
            _from_numpy_like(is_new_det, det_masks),
            _from_numpy_like(det_to_max_iou_trk_idx, det_masks),
            _from_numpy_like(det_is_high_conf, det_masks),
            _from_numpy_like(det_is_high_iou, det_masks),
            _from_numpy_like(det_keep_np, det_masks),
            _from_numpy_like(im_mask, det_masks),
        )

    trk_is_matched = np.any(intersection_metric >= iou_threshold_trk, axis=0)
    trk_is_nonempty = np.any(trk_masks_binary, axis=(1, 2))
    trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)

    is_new_det = np.logical_and(
        np.logical_and(det_scores_np >= new_det_thresh, det_keep_np),
        ~np.any(intersection_metric >= iou_threshold, axis=1),
    )

    intersection_thresh_recond = (
        iom_thresh_recondition if use_iom_recondition else iou_thresh_recondition
    )
    det_match_to_many_trk = (
        np.sum(intersection_metric >= intersection_thresh_recond, axis=1) > 1
    )
    trk_match_to_many_det = (
        np.sum(intersection_metric >= intersection_thresh_recond, axis=0) > 1
    )
    intersection_metric = np.where(
        trk_match_to_many_det[None, :],
        np.zeros_like(intersection_metric),
        intersection_metric,
    )
    intersection_metric = np.where(
        det_match_to_many_trk[:, None],
        np.zeros_like(intersection_metric),
        intersection_metric,
    )

    det_to_max_iou_trk_idx = np.argmax(intersection_metric, axis=1)
    det_is_high_conf = ((det_scores_np >= HIGH_CONF_THRESH) & det_keep_np) & ~is_new_det
    det_is_high_iou = np.max(intersection_metric, axis=1) >= intersection_thresh_recond
    im_mask = intersection_metric >= iou_threshold

    return (
        _from_numpy_like(trk_is_unmatched, det_masks),
        _from_numpy_like(trk_is_nonempty, det_masks),
        _from_numpy_like(is_new_det, det_masks),
        _from_numpy_like(det_to_max_iou_trk_idx, det_masks),
        _from_numpy_like(det_is_high_conf, det_masks),
        _from_numpy_like(det_is_high_iou, det_masks),
        _from_numpy_like(det_keep_np, det_masks),
        _from_numpy_like(im_mask, det_masks),
    )


class Sam3VideoBase(nn.Module):
    def __init__(
        self,
        detector: nn.Module,
        tracker: nn.Module,
        score_threshold_detection=0.5,
        det_nms_thresh=0.0,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.0,
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        suppress_unmatched_only_within_hotstart=True,
        init_trk_keep_alive=0,
        max_trk_keep_alive=8,
        min_trk_keep_alive=-4,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=False,
        o2o_matching_masklets_enable=False,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        max_num_objects=-1,
        recondition_every_nth_frame=-1,
        masklet_confirmation_enable=False,
        masklet_confirmation_consecutive_det_thresh=3,
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.0,
    ):
        super().__init__()
        self.detector = detector
        self.tracker = tracker
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = (
            suppress_unmatched_only_within_hotstart
        )
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = (
            decrease_trk_keep_alive_for_empty_masklets
        )
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self.eval()
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self._dist_pg_cpu = None

        if max_num_objects > 0:
            num_obj_for_compile = math.ceil(max_num_objects / self.world_size)
        else:
            max_num_objects = 10000
            num_obj_for_compile = 16
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = (
            masklet_confirmation_consecutive_det_thresh
        )
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

    @property
    def device(self):
        return "mlx"

    def _init_dist_pg_cpu(self):
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base.Sam3VideoBase._init_dist_pg_cpu",
            reason="torch-distributed",
            detail="Torch distributed CPU process groups are not used in sam3_mlx.",
        )

    def broadcast_python_obj_cpu(self, *args, **kwargs):
        del args, kwargs
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base.Sam3VideoBase.broadcast_python_obj_cpu",
            reason="torch-distributed",
            detail="Torch distributed object broadcast is not ported to MLX.",
        )

    def forward(self, *args, **kwargs):
        del args, kwargs
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base.Sam3VideoBase.forward",
            detail=(
                "Full detector/tracker orchestration depends on the unported "
                "official tracker-memory runtime."
            ),
            alternative="sam3_mlx.model.sam3_video_inference.Sam3VideoInference",
        )

    def _load_checkpoint(self, *args, **kwargs):
        del args, kwargs
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base.Sam3VideoBase._load_checkpoint",
            detail="Torch checkpoint loading is not part of this MLX video-base slice.",
            alternative="sam3_mlx.model_builder.build_sam3_image_model",
        )

    def _drop_new_det_with_obj_limit(self, new_det_fa_inds, det_scores_np, num_to_keep):
        if num_to_keep <= 0:
            return np.array([], dtype=np.asarray(new_det_fa_inds).dtype)
        new_det_fa_inds = np.asarray(new_det_fa_inds)
        det_scores_np = np.asarray(det_scores_np)
        order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        keep = np.sort(new_det_fa_inds[order[:num_to_keep]])
        return keep

    def _assign_new_det_to_gpus(self, new_det_num, prev_workload_per_gpu):
        prev_workload_per_gpu = np.asarray(prev_workload_per_gpu)
        if prev_workload_per_gpu.size == 0:
            return np.zeros(new_det_num, dtype=np.int64)
        workloads = prev_workload_per_gpu.astype(np.int64, copy=True)
        gpu_ids = []
        for _ in range(new_det_num):
            gpu_id = int(np.argmin(workloads))
            gpu_ids.append(gpu_id)
            workloads[gpu_id] += 1
        return np.asarray(gpu_ids, dtype=np.int64)

    def _suppress_detections_close_to_boundary(self, boxes, margin=0.025):
        """Return a keep mask for normalized xyxy boxes whose centers are in-bounds."""
        x_min = boxes[..., 0]
        y_min = boxes[..., 1]
        x_max = boxes[..., 2]
        y_max = boxes[..., 3]
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        return (
            (x_c > margin)
            & (x_c < 1.0 - margin)
            & (y_c > margin)
            & (y_c < 1.0 - margin)
        )

    def _process_hotstart(self, *args, **kwargs):
        del args, kwargs
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base.Sam3VideoBase._process_hotstart",
            detail="Hotstart tracking heuristics require the unported tracker state.",
        )

    def update_masklet_confirmation_status(self, *args, **kwargs):
        del args, kwargs
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base.Sam3VideoBase.update_masklet_confirmation_status",
            detail="Masklet confirmation requires the unported tracker state.",
        )

    def prep_for_evaluator(self, video_frames, tracking_res, scores_labels):
        # Evaluator-export boundary: downstream COCO/RLE payloads are host-side
        # NumPy/Python structures, not MLX model-runtime tensors.
        video_frames = tuple(video_frames)
        if not video_frames:
            raise ValueError("prep_for_evaluator requires at least one video frame.")
        num_frames = len(video_frames)
        height, width = _frame_size(video_frames[0])
        zero_mask = np.zeros((1, height, width), dtype=bool)
        object_ids = list(scores_labels.keys())
        preds: dict[str, Any] = {
            "scores": [],
            "labels": [],
            "boxes": [],
            "masks_rle": [],
        }
        for object_id in object_ids:
            object_masks = []
            score, label = scores_labels[object_id]
            for frame_idx in range(num_frames):
                frame_masks = tracking_res.get(frame_idx, {})
                mask = frame_masks.get(object_id, zero_mask)
                mask_np = _to_numpy(mask).astype(bool, copy=False)
                if mask_np.ndim == 2:
                    mask_np = mask_np[None, :, :]
                if mask_np.shape != (1, height, width):
                    raise ValueError(
                        "tracking_res masks must have shape (1, H, W), "
                        f"got {mask_np.shape} for frame {frame_idx}."
                    )
                object_masks.append(mask_np)
            object_masks_np = np.concatenate(object_masks, axis=0)
            preds["scores"].append(_scalar_float(score))
            preds["labels"].append(_scalar_int(label))
            boxes = mask_to_box(object_masks_np[:, None, :, :]).reshape(num_frames, 4)
            preds["boxes"].append(boxes.astype(np.float32, copy=False))
            preds["masks_rle"].append(rle_encode(object_masks_np, return_areas=True))

        preds["boxes"] = (
            np.stack(preds["boxes"], axis=0).astype(np.float32, copy=False)
            if preds["boxes"]
            else np.zeros((0, num_frames, 4), dtype=np.float32)
        )
        preds["scores"] = np.array(preds["scores"], dtype=np.float32)
        preds["per_frame_scores"] = preds["scores"]
        preds["labels"] = np.array(preds["labels"], dtype=np.int64)
        return preds

    def _encode_prompt(self, **kwargs):
        del kwargs
        _raise_video_base_unsupported(
            "sam3_mlx.model.sam3_video_base.Sam3VideoBase._encode_prompt",
            detail="Prompt encoding in Sam3VideoBase requires the unported tracker path.",
            alternative="sam3_mlx.model.sam3_video_inference.Sam3VideoInference.add_prompt",
        )


__all__ = [
    "LazyAssociateDetTrkResult",
    "MaskletConfirmationStatus",
    "RealizedAssociateDetTrkresult",
    "Sam3VideoBase",
    "_associate_det_trk_compilable",
    "realize_adt_result",
]
