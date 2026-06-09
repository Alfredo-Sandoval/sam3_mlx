import json
from types import SimpleNamespace

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sam3_mlx.model.multiplex_utils import (
    MultiplexState,
    UnsupportedMultiplexRuntimeError,
)
from sam3_mlx.model.sam3_multiplex_base import (
    MaskletConfirmationStatus,
    Sam3MultiplexBase,
    Sam3MultiplexPredictorWrapper,
)
from sam3_mlx.model.sam3_video_base import LazyAssociateDetTrkResult, realize_adt_result
from sam3_mlx.model.video_tracking_multiplex import VideoTrackingMultiplex
from sam3_mlx.model.video_tracking_multiplex_demo import VideoTrackingMultiplexDemo
from tests._paths import PORT_TRACKER_FIXTURE_ROOT

OFFICIAL_SAM3_MULTIPLEX_BASE_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"
TRACKER_ADD_PARITY_FIXTURE = (
    PORT_TRACKER_FIXTURE_ROOT / "multiplex_tracker_add_parity.json"
)
MEMORY_UPDATE_PARITY_FIXTURE = (
    PORT_TRACKER_FIXTURE_ROOT / "multiplex_memory_update_parity.json"
)
DETECTOR_FRAME_PARITY_FIXTURE = (
    PORT_TRACKER_FIXTURE_ROOT / "multiplex_detector_frame_parity.json"
)


class _DummyController:
    allowed_bucket_capacity = 4
    training = False


class _DummyTracker:
    is_multiplex = True
    multiplex_controller = _DummyController()


class _DummyDetector:
    is_multiplex = True
    running_in_prod = False


class _PlainTracker:
    is_multiplex = False


class _PlainDetector:
    is_multiplex = False
    running_in_prod = False


class _TextBackbone:
    def __init__(self):
        self.calls = []

    def forward_text(self, find_text_batch, device=None):
        self.calls.append((tuple(find_text_batch), device))
        return {
            "language_features": mx.ones((1, 1, 4), dtype=mx.float32),
            "language_mask": mx.ones((1, 1), dtype=mx.bool_),
        }


class _FeatureDetector(_DummyDetector):
    def __init__(
        self,
        *,
        incomplete_backbone: bool = False,
        incomplete_interactive_backbone: bool = False,
    ):
        self.backbone = _TextBackbone()
        self.incomplete_backbone = incomplete_backbone
        self.incomplete_interactive_backbone = incomplete_interactive_backbone
        self.calls = []

    def forward_video_grounding_multigpu(self, **kwargs):
        self.calls.append(kwargs)
        output = {
            "pred_logits": mx.array([[[2.0], [-2.0]]], dtype=mx.float32),
            "pred_boxes_xyxy": mx.zeros((1, 2, 4), dtype=mx.float32),
            "pred_masks": mx.ones((1, 2, 2, 2), dtype=mx.float32),
            "sam2_backbone_fpn_0": mx.ones((1, 2, 2, 2), dtype=mx.float32),
            "sam2_backbone_fpn_1": mx.ones((1, 2, 1, 1), dtype=mx.float32) * 2,
            "sam2_backbone_fpn_2": mx.ones((1, 2, 1, 1), dtype=mx.float32) * 3,
            "sam2_backbone_pos_enc": [
                mx.ones((1, 2, 2, 2), dtype=mx.float32) * 4,
                mx.ones((1, 2, 1, 1), dtype=mx.float32) * 5,
                mx.ones((1, 2, 1, 1), dtype=mx.float32) * 6,
            ],
            "interactive_backbone_fpn_0": mx.ones(
                (1, 2, 2, 2),
                dtype=mx.float32,
            )
            * 7,
            "interactive_backbone_fpn_1": mx.ones(
                (1, 2, 1, 1),
                dtype=mx.float32,
            )
            * 8,
            "interactive_backbone_fpn_2": mx.ones(
                (1, 2, 1, 1),
                dtype=mx.float32,
            )
            * 9,
            "interactive_backbone_pos_enc": [
                mx.ones((1, 2, 2, 2), dtype=mx.float32) * 10,
                mx.ones((1, 2, 1, 1), dtype=mx.float32) * 11,
                mx.ones((1, 2, 1, 1), dtype=mx.float32) * 12,
            ],
        }
        if self.incomplete_backbone:
            output.pop("sam2_backbone_fpn_1")
        if self.incomplete_interactive_backbone:
            output.pop("interactive_backbone_fpn_1")
        return output, kwargs["backbone_out"]


class _MaskDecoder:
    def conv_s0(self, value):
        return value + 10

    def conv_s1(self, value):
        return value + 20


class _FeatureTracker(_DummyTracker):
    sam_mask_decoder = _MaskDecoder()
    interactive_sam_mask_decoder = _MaskDecoder()


class _ObjectRemovalTracker(_DummyTracker):
    def __init__(self):
        self.remove_calls = []

    def remove_object(self, inference_state, obj_id, strict=False, need_output=True):
        self.remove_calls.append(
            {
                "state": inference_state,
                "obj_id": obj_id,
                "strict": strict,
                "need_output": need_output,
            }
        )
        inference_state["obj_ids"] = [
            existing_obj_id
            for existing_obj_id in inference_state.get("obj_ids", [])
            if existing_obj_id != obj_id
        ]
        return inference_state["obj_ids"], []


class _MemoryUpdateTracker(_DummyTracker):
    def __init__(self, *, dynamic: bool = False):
        self.is_multiplex_dynamic = dynamic
        self.object_score_logit_threshold = 0.0
        self.maskmem_backbone = SimpleNamespace(
            mask_downsampler=SimpleNamespace(interpol_size=(4, 4))
        )
        self.encoder_calls = []
        self.add_output_calls = []
        self.suppress_calls = []

    def _suppress_object_pw_area_shrinkage(self, high_res_masks):
        self.suppress_calls.append(high_res_masks)
        return high_res_masks

    def _run_memory_encoder(
        self,
        sam2_state,
        frame_idx,
        local_batch_size,
        local_high_res_masks,
        local_object_score_logits,
        is_mask_from_pts,
    ):
        call_id = len(self.encoder_calls) + 1
        self.encoder_calls.append(
            {
                "state": sam2_state["name"],
                "frame_idx": frame_idx,
                "local_batch_size": local_batch_size,
                "high_res_masks": np.asarray(local_high_res_masks),
                "mask_sample": np.asarray(local_high_res_masks[:, 0, 0, 0]),
                "score_logits": np.asarray(local_object_score_logits),
                "is_mask_from_pts": is_mask_from_pts,
            }
        )
        maskmem_features = mx.full((local_batch_size, 2), call_id, dtype=mx.float32)
        maskmem_pos_enc = [
            mx.full((local_batch_size, 1), call_id + 10, dtype=mx.float32)
        ]
        image_features = mx.full((local_batch_size, 3), call_id + 20, dtype=mx.float32)
        image_pos_enc = mx.full((local_batch_size, 4), call_id + 30, dtype=mx.float32)
        return maskmem_features, maskmem_pos_enc, image_features, image_pos_enc

    def add_output_per_object(
        self,
        *,
        inference_state,
        frame_idx,
        current_out,
        storage_key,
    ):
        self.add_output_calls.append(
            {
                "state": inference_state["name"],
                "frame_idx": frame_idx,
                "storage_key": storage_key,
                "current_out": current_out,
            }
        )


class _NoObjectPointerMemoryUpdateTracker(_MemoryUpdateTracker):
    def no_obj_ptr_linear(self, pointers):
        return pointers + 100.0


def _lazy_hotstart_assoc(
    *,
    trk_is_unmatched,
    trk_is_nonempty,
    im_mask,
) -> LazyAssociateDetTrkResult:
    im_mask_mx = mx.array(im_mask, dtype=mx.bool_)
    num_det = int(im_mask_mx.shape[0])
    return LazyAssociateDetTrkResult(
        trk_is_unmatched=mx.array(trk_is_unmatched, dtype=mx.bool_),
        trk_is_nonempty=mx.array(trk_is_nonempty, dtype=mx.bool_),
        is_new_det=mx.zeros((num_det,), dtype=mx.bool_),
        det_to_max_iou_trk_idx=mx.zeros((num_det,), dtype=mx.int64),
        det_is_high_conf=mx.zeros((num_det,), dtype=mx.bool_),
        det_is_high_iou=mx.zeros((num_det,), dtype=mx.bool_),
        det_keep=mx.ones((num_det,), dtype=mx.bool_),
        im_mask=im_mask_mx,
    )


class _ReconditionTracker(_DummyTracker):
    input_mask_size = 2

    def __init__(self):
        self.add_calls = []
        self.preflight_calls = []

    def add_new_masks(
        self,
        *,
        inference_state,
        frame_idx,
        obj_ids,
        masks,
        reconditioning=False,
    ):
        self.add_calls.append(
            {
                "state": inference_state,
                "frame_idx": frame_idx,
                "obj_ids": list(obj_ids),
                "masks": masks,
                "reconditioning": reconditioning,
            }
        )
        inference_state.setdefault("obj_ids", [])
        for obj_id in obj_ids:
            if obj_id not in inference_state["obj_ids"]:
                inference_state["obj_ids"].append(obj_id)
        inference_state.setdefault(
            "obj_idx_to_id",
            {idx: obj_id for idx, obj_id in enumerate(inference_state["obj_ids"])},
        )
        return frame_idx, inference_state["obj_ids"], None, None

    def propagate_in_video_preflight(self, sam2_state, run_mem_encoder=True):
        self.preflight_calls.append((sam2_state, run_mem_encoder))


class _SingleMaskReconditionTracker(_DummyTracker):
    input_mask_size = 2

    def __init__(self):
        self.add_calls = []

    def add_new_mask(self, *, inference_state, frame_idx, obj_id, mask):
        self.add_calls.append(
            {
                "state": inference_state,
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "mask": mask,
            }
        )
        return frame_idx, inference_state.setdefault("obj_ids", []), None, None


class _NonDynamicAddTracker(_DummyTracker):
    input_mask_size = 2
    is_multiplex_dynamic = False

    def __init__(self, *, per_obj_inference: bool):
        self.per_obj_inference = per_obj_inference
        self.init_calls = []
        self.add_calls = []
        self.preflight_calls = []

    def init_state(self, *, cached_features, video_height, video_width, num_frames):
        sam2_state = {
            "name": f"init-{len(self.init_calls) + 1}",
            "obj_ids": [],
        }
        self.init_calls.append(
            {
                "cached_features": cached_features,
                "video_height": video_height,
                "video_width": video_width,
                "num_frames": num_frames,
                "state": sam2_state,
            }
        )
        return sam2_state

    def add_new_mask(
        self,
        *,
        inference_state,
        frame_idx,
        obj_id,
        mask,
        add_mask_to_memory=False,
    ):
        self.add_calls.append(
            {
                "state": inference_state,
                "frame_idx": frame_idx,
                "obj_id": int(obj_id),
                "mask": mask,
                "add_mask_to_memory": add_mask_to_memory,
            }
        )
        inference_state.setdefault("obj_ids", []).append(int(obj_id))
        return frame_idx, inference_state["obj_ids"], None, None

    def propagate_in_video_preflight(self, sam2_state, run_mem_encoder=True):
        self.preflight_calls.append((sam2_state, run_mem_encoder))


def _dynamic_tracker_state(
    name: str,
    assignments: list[list[int]],
    object_ids: list[int],
    *,
    allowed_bucket_capacity: int,
):
    return {
        "name": name,
        "obj_ids": object_ids.copy(),
        "multiplex_state": MultiplexState(
            assignments,
            allowed_bucket_capacity=allowed_bucket_capacity,
            object_ids=object_ids.copy(),
        ),
    }


class _DynamicAddTracker(_DummyTracker):
    input_mask_size = 2
    is_multiplex_dynamic = True

    def __init__(self):
        self.init_calls = []
        self.add_calls = []
        self.preflight_calls = []

    def init_state(self, *, cached_features, video_height, video_width, num_frames):
        sam2_state = _dynamic_tracker_state(
            f"init-{len(self.init_calls) + 1}",
            [[-1, -1, -1]],
            [],
            allowed_bucket_capacity=3,
        )
        self.init_calls.append(
            {
                "cached_features": cached_features,
                "video_height": video_height,
                "video_width": video_width,
                "num_frames": num_frames,
                "state": sam2_state,
            }
        )
        return sam2_state

    def add_new_masks(
        self,
        *,
        inference_state,
        frame_idx,
        obj_ids,
        masks,
        add_mask_to_memory=False,
    ):
        obj_ids = [int(obj_id) for obj_id in obj_ids]
        self.add_calls.append(
            {
                "state": inference_state,
                "frame_idx": frame_idx,
                "obj_ids": obj_ids.copy(),
                "masks": masks,
                "add_mask_to_memory": add_mask_to_memory,
            }
        )
        inference_state.setdefault("obj_ids", [])
        for obj_id in obj_ids:
            if obj_id not in inference_state["obj_ids"]:
                inference_state["obj_ids"].append(obj_id)

        multiplex_state = inference_state["multiplex_state"]
        start_idx = multiplex_state.total_valid_entries
        multiplex_state.add_objects(
            list(range(start_idx, start_idx + len(obj_ids))),
            object_ids=obj_ids,
            allow_new_buckets=False,
        )
        return frame_idx, inference_state["obj_ids"], None, None

    def propagate_in_video_preflight(self, sam2_state, run_mem_encoder=True):
        self.preflight_calls.append((sam2_state, run_mem_encoder))
        sam2_state["tracking_has_started"] = True


class _PackedAddAdapterTracker(_DummyTracker):
    add_new_masks = VideoTrackingMultiplex.add_new_masks
    add_new_masks_to_existing_state = (
        VideoTrackingMultiplex.add_new_masks_to_existing_state
    )
    _current_frame_output = staticmethod(VideoTrackingMultiplex._current_frame_output)
    _prepared_features_from_state = VideoTrackingMultiplex._prepared_features_from_state
    propagate_in_video_preflight = VideoTrackingMultiplex.propagate_in_video_preflight

    input_mask_size = 4
    is_multiplex_dynamic = True

    def __init__(self):
        self.use_mask_input_as_output_without_sam = True
        self.use_memory_selection = False
        self.use_obj_ptrs_in_encoder = False
        self.save_image_features = False
        self.mask_calls = []
        self.memory_calls = []
        self.pix_mem_calls = []

    def _get_interactive_pix_mem(self, vision_feats, feat_sizes):
        self.pix_mem_calls.append(
            {
                "vision_feats": vision_feats,
                "feat_sizes": feat_sizes,
            }
        )
        return mx.array([[1.0]], dtype=mx.float32)

    def _use_mask_as_output(
        self,
        *,
        backbone_features,
        high_res_features,
        mask_inputs,
        multiplex_state,
        objects_in_mask,
    ):
        self.mask_calls.append(
            {
                "backbone_features": backbone_features,
                "high_res_features": high_res_features,
                "mask_inputs": mask_inputs,
                "multiplex_state": multiplex_state,
                "objects_in_mask": objects_in_mask,
            }
        )
        batch = int(mask_inputs.shape[0])
        values = mx.arange(batch, dtype=mx.float32).reshape(batch, 1, 1, 1)
        return {
            "low_res_masks": mx.ones((batch, 1, 2, 2), dtype=mx.float32)
            * (7.0 + values),
            "high_res_masks": mx.ones((batch, 1, 4, 4), dtype=mx.float32)
            * (70.0 + values),
            "object_score_logits": mx.ones((batch, 1), dtype=mx.float32) * 101.0,
            "ious": mx.ones((batch, 1), dtype=mx.float32),
        }

    def _encode_new_memory(
        self,
        *,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        conditioning_objects,
        is_mask_from_pts,
        multiplex_state,
    ):
        self.memory_calls.append(
            {
                "image": image,
                "current_vision_feats": current_vision_feats,
                "feat_sizes": feat_sizes,
                "pred_masks_high_res": pred_masks_high_res,
                "object_score_logits": object_score_logits,
                "conditioning_objects": conditioning_objects.copy(),
                "is_mask_from_pts": is_mask_from_pts,
                "multiplex_state": multiplex_state,
            }
        )
        return (
            mx.array([[9.0]], dtype=mx.float32),
            [mx.array([[8.0]], dtype=mx.float32)],
        )


class _PackedDetectorFrameTracker(_PackedAddAdapterTracker):
    sam_mask_decoder = _MaskDecoder()
    interactive_sam_mask_decoder = _MaskDecoder()
    _prepare_backbone_features = VideoTrackingMultiplex._prepare_backbone_features
    num_feature_levels = 3

    def __init__(self):
        super().__init__()
        self.propagate_calls = []

    def propagate_in_video(
        self,
        inference_state,
        *,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
        run_mem_encoder,
        propagate_preflight,
    ):
        self.propagate_calls.append(
            {
                "state": inference_state,
                "start_frame_idx": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
                "reverse": reverse,
                "run_mem_encoder": run_mem_encoder,
                "propagate_preflight": propagate_preflight,
            }
        )
        yield (
            start_frame_idx,
            list(inference_state["obj_ids"]),
            mx.array([[[2.0, -2.0], [-2.0, -2.0]]], dtype=mx.float32),
            None,
            mx.ones((len(inference_state["obj_ids"]),), dtype=mx.float32),
        )


class _PackedDetectorFrameDetector(_FeatureDetector):
    def forward_video_grounding_multigpu(self, **kwargs):
        output, backbone_out = super().forward_video_grounding_multigpu(**kwargs)
        output["pred_logits"] = mx.array([[[2.0], [2.5]]], dtype=mx.float32)
        output["pred_masks"] = mx.array(
            [
                [
                    [[2.0, -2.0], [-2.0, -2.0]],
                    [[-2.0, -2.0], [-2.0, 2.0]],
                ]
            ],
            dtype=mx.float32,
        )
        return output, backbone_out


class _PackedStartupTracker(_PackedDetectorFrameTracker):
    def __init__(self):
        super().__init__()
        self.init_calls = []
        self.preflight_calls = []

    def init_state(self, *, cached_features, video_height, video_width, num_frames):
        sam2_state = {
            "obj_ids": [],
            "multiplex_state": MultiplexState(
                [[-1, -1]],
                dtype=mx.float32,
                allowed_bucket_capacity=2,
                object_ids=[],
            ),
        }
        self.init_calls.append(
            {
                "cached_features": cached_features,
                "video_height": video_height,
                "video_width": video_width,
                "num_frames": num_frames,
                "state": sam2_state,
            }
        )
        return sam2_state

    def propagate_in_video_preflight(self, sam2_state, run_mem_encoder=True):
        self.preflight_calls.append((sam2_state, run_mem_encoder))
        VideoTrackingMultiplex.propagate_in_video_preflight(
            self,
            sam2_state,
            run_mem_encoder=run_mem_encoder,
        )


def test_sam3_multiplex_base_ports_constructor_state_for_multiplex_metadata(
    monkeypatch,
):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("ENABLE_PROFILING", "1")
    detector = _DummyDetector()

    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=detector,
        is_multiplex=True,
        max_num_objects=17,
        max_num_kboxes=9,
        running_in_prod=True,
        use_batched_grounding=True,
        batched_grounding_batch_size=3,
    )

    assert base.rank == 1
    assert base.world_size == 2
    assert base.max_num_objects == 17
    assert base.num_obj_for_compile == 3
    assert base.bucket_capacity == 4
    assert base.max_num_kboxes == 9
    assert base.running_in_prod is True
    assert detector.running_in_prod is True
    assert base.use_batched_grounding is True
    assert base.batched_grounding_batch_size == 3
    assert base._profiling_enabled is True


def test_sam3_multiplex_base_rejects_mismatched_multiplex_flags():
    tracker = _DummyTracker()
    detector = _DummyDetector()
    detector.is_multiplex = False

    with pytest.raises(AssertionError, match="is_multiplex must be the same"):
        Sam3MultiplexBase(
            tracker=tracker,
            detector=detector,
            is_multiplex=True,
        )


def test_sam3_multiplex_base_keeps_checkpoint_loading_fail_fast():
    with pytest.raises(UnsupportedMultiplexRuntimeError, match="checkpoint loading"):
        Sam3MultiplexBase(
            tracker=_DummyTracker(),
            detector=_DummyDetector(),
            ckpt_path="tracker.pt",
        )


def test_sam3_multiplex_base_updates_masklet_confirmation_status():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        masklet_confirmation_enable=True,
        masklet_confirmation_consecutive_det_thresh=2,
    )
    rank0_metadata = base._initialize_metadata()["rank0_metadata"]
    confirmation = rank0_metadata["masklet_confirmation"]
    confirmation["status"] = np.array(
        [
            MaskletConfirmationStatus.UNCONFIRMED.value,
            MaskletConfirmationStatus.CONFIRMED.value,
        ],
        dtype=np.int64,
    )
    confirmation["consecutive_det_num"] = np.array([1, 3], dtype=np.int64)

    updated = base.update_masklet_confirmation_status(
        rank0_metadata=rank0_metadata,
        obj_ids_all_gpu_prev=np.array([10, 11], dtype=np.int64),
        obj_ids_all_gpu_updated=np.array([11, 10, 12], dtype=np.int64),
        det_to_matched_trk_obj_ids={0: np.array([10, 11], dtype=np.int64)},
        new_det_obj_ids=np.array([12], dtype=np.int64),
    )
    updated_confirmation = updated["masklet_confirmation"]

    np.testing.assert_array_equal(
        updated_confirmation["status"],
        np.array(
            [
                MaskletConfirmationStatus.CONFIRMED.value,
                MaskletConfirmationStatus.CONFIRMED.value,
                MaskletConfirmationStatus.UNCONFIRMED.value,
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        updated_confirmation["consecutive_det_num"],
        np.array([4, 2, 1], dtype=np.int64),
    )


def test_sam3_multiplex_base_masklet_confirmation_persists_and_resets_counts():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        masklet_confirmation_enable=True,
        masklet_confirmation_consecutive_det_thresh=2,
    )
    rank0_metadata = base._initialize_metadata()["rank0_metadata"]
    rank0_metadata["masklet_confirmation"]["status"] = np.array(
        [
            MaskletConfirmationStatus.CONFIRMED.value,
            MaskletConfirmationStatus.UNCONFIRMED.value,
        ],
        dtype=np.int64,
    )
    rank0_metadata["masklet_confirmation"]["consecutive_det_num"] = np.array(
        [2, 1],
        dtype=np.int64,
    )

    updated = base.update_masklet_confirmation_status(
        rank0_metadata=rank0_metadata,
        obj_ids_all_gpu_prev=np.array([1, 2], dtype=np.int64),
        obj_ids_all_gpu_updated=np.array([1, 2, 3], dtype=np.int64),
        det_to_matched_trk_obj_ids={},
        new_det_obj_ids=np.array([], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        updated["masklet_confirmation"]["status"],
        np.array(
            [
                MaskletConfirmationStatus.CONFIRMED.value,
                MaskletConfirmationStatus.UNCONFIRMED.value,
                MaskletConfirmationStatus.UNCONFIRMED.value,
            ],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        updated["masklet_confirmation"]["consecutive_det_num"],
        np.array([0, 0, 0], dtype=np.int64),
    )


def test_sam3_multiplex_base_masklet_confirmation_rejects_shape_drift():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        masklet_confirmation_enable=True,
    )
    rank0_metadata = base._initialize_metadata()["rank0_metadata"]
    rank0_metadata["masklet_confirmation"]["status"] = np.array([1, 1])
    rank0_metadata["masklet_confirmation"]["consecutive_det_num"] = np.array([0])

    with pytest.raises(ValueError, match="status length"):
        base.update_masklet_confirmation_status(
            rank0_metadata=rank0_metadata,
            obj_ids_all_gpu_prev=np.array([7], dtype=np.int64),
            obj_ids_all_gpu_updated=np.array([7], dtype=np.int64),
            det_to_matched_trk_obj_ids={},
            new_det_obj_ids=np.array([], dtype=np.int64),
        )


def test_sam3_multiplex_base_hotstart_records_new_tracklets_with_defaults():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        init_trk_keep_alive=2,
    )
    rank0_metadata = {}

    removed, updated = base._process_hotstart(
        frame_idx=4,
        num_frames=8,
        reverse=False,
        det_to_matched_trk_obj_ids={},
        new_det_obj_ids=[5],
        empty_trk_obj_ids=[],
        unmatched_trk_obj_ids=[],
        rank0_metadata=rank0_metadata,
        tracker_metadata={},
    )

    assert removed == set()
    assert updated is rank0_metadata
    assert rank0_metadata["obj_first_frame_idx"] == {5: 4}
    assert rank0_metadata["trk_keep_alive"][5] == 2
    assert rank0_metadata["removed_obj_ids"] == set()
    assert rank0_metadata["suppressed_obj_ids"][4] == set()


def test_sam3_multiplex_base_hotstart_removes_unmatched_tracklets():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        hotstart_delay=3,
        hotstart_unmatch_thresh=2,
        hotstart_dup_thresh=2,
    )
    rank0_metadata = base._initialize_metadata()["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"][7] = 0
    rank0_metadata["trk_keep_alive"][7] = 0
    rank0_metadata["unmatched_frame_inds"][7].append(1)

    removed, updated = base._process_hotstart(
        frame_idx=2,
        num_frames=5,
        reverse=False,
        det_to_matched_trk_obj_ids={},
        new_det_obj_ids=np.array([], dtype=np.int64),
        empty_trk_obj_ids=np.array([], dtype=np.int64),
        unmatched_trk_obj_ids=np.array([7], dtype=np.int64),
        rank0_metadata=rank0_metadata,
        tracker_metadata={},
    )

    assert removed == {7}
    assert updated is rank0_metadata
    assert rank0_metadata["removed_obj_ids"] == {7}
    assert rank0_metadata["unmatched_frame_inds"][7] == [1, 2]
    assert rank0_metadata["trk_keep_alive"][7] == -1
    assert rank0_metadata["suppressed_obj_ids"][2] == set()


def test_sam3_multiplex_base_hotstart_suppresses_stale_unmatched_tracklets():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        hotstart_delay=0,
        suppress_unmatched_only_within_hotstart=False,
    )
    rank0_metadata = base._initialize_metadata()["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"][8] = 0
    rank0_metadata["trk_keep_alive"][8] = 0

    removed, _ = base._process_hotstart(
        frame_idx=10,
        num_frames=12,
        reverse=False,
        det_to_matched_trk_obj_ids={},
        new_det_obj_ids=[],
        empty_trk_obj_ids=[],
        unmatched_trk_obj_ids=[8],
        rank0_metadata=rank0_metadata,
        tracker_metadata={},
    )

    assert removed == set()
    assert rank0_metadata["trk_keep_alive"][8] == -1
    assert rank0_metadata["suppressed_obj_ids"][10] == {8}


def test_sam3_multiplex_base_hotstart_decrements_empty_masklet_keep_alive():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        decrease_trk_keep_alive_for_empty_masklets=True,
    )
    rank0_metadata = base._initialize_metadata()["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"][9] = 0
    rank0_metadata["trk_keep_alive"][9] = 0

    removed, _ = base._process_hotstart(
        frame_idx=5,
        num_frames=8,
        reverse=False,
        det_to_matched_trk_obj_ids={},
        new_det_obj_ids=[],
        empty_trk_obj_ids=[9],
        unmatched_trk_obj_ids=[],
        rank0_metadata=rank0_metadata,
        tracker_metadata={},
    )

    assert removed == set()
    assert rank0_metadata["trk_keep_alive"][9] == -1
    assert 9 not in rank0_metadata["suppressed_obj_ids"][5]


def test_sam3_multiplex_base_hotstart_removes_later_duplicate_tracklets():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        hotstart_delay=3,
        hotstart_unmatch_thresh=2,
        hotstart_dup_thresh=2,
    )
    rank0_metadata = base._initialize_metadata()["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"].update({1: 0, 2: 1})
    rank0_metadata["trk_keep_alive"].update({1: 0, 2: 0})
    rank0_metadata["overlap_pair_to_frame_inds"][(1, 2)].append(1)

    removed, _ = base._process_hotstart(
        frame_idx=2,
        num_frames=5,
        reverse=False,
        det_to_matched_trk_obj_ids={0: [1, 2]},
        new_det_obj_ids=[],
        empty_trk_obj_ids=[],
        unmatched_trk_obj_ids=[],
        rank0_metadata=rank0_metadata,
        tracker_metadata={},
    )

    assert removed == {2}
    assert rank0_metadata["removed_obj_ids"] == {2}
    assert rank0_metadata["overlap_pair_to_frame_inds"][(1, 2)] == [1, 2]
    assert rank0_metadata["trk_keep_alive"][1] == 1
    assert rank0_metadata["trk_keep_alive"][2] == 1


def test_sam3_multiplex_base_hotstart_reverse_removes_later_duplicate_tracklets():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        hotstart_delay=3,
        hotstart_unmatch_thresh=2,
        hotstart_dup_thresh=2,
    )
    rank0_metadata = base._initialize_metadata()["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"].update({10: 6, 11: 4})
    rank0_metadata["trk_keep_alive"].update({10: 0, 11: 0})
    rank0_metadata["overlap_pair_to_frame_inds"][(10, 11)].append(4)

    removed, _ = base._process_hotstart(
        frame_idx=3,
        num_frames=8,
        reverse=True,
        det_to_matched_trk_obj_ids={0: np.array([10, 11], dtype=np.int64)},
        new_det_obj_ids=[],
        empty_trk_obj_ids=[],
        unmatched_trk_obj_ids=[],
        rank0_metadata=rank0_metadata,
        tracker_metadata={},
    )

    assert removed == {11}
    assert rank0_metadata["removed_obj_ids"] == {11}
    assert rank0_metadata["overlap_pair_to_frame_inds"][(10, 11)] == [4, 3]


def test_sam3_multiplex_base_hotstart_gpu_updates_position_metadata_from_rank0():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        hotstart_delay=3,
        hotstart_unmatch_thresh=2,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10, 11], dtype=np.int64)
    rank0_metadata = tracker_metadata_prev["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"].update({10: 0, 11: 1})
    rank0_metadata["unmatched_frame_inds"][11].append(1)
    rank0_metadata["trk_keep_alive"].update({10: 0, 11: 0})
    adt_result = _lazy_hotstart_assoc(
        trk_is_unmatched=[False, True],
        trk_is_nonempty=[True, True],
        im_mask=[[True, False]],
    )

    to_remove, to_suppress, gpu_metadata = base._process_hotstart_gpu(
        frame_idx=2,
        reverse=False,
        adt_result=adt_result,
        tracker_metadata_prev=tracker_metadata_prev,
        gpu_metadata_prev={"N_obj": 0},
    )

    np.testing.assert_array_equal(
        np.asarray(to_remove),
        np.array([False, True]),
    )
    np.testing.assert_array_equal(
        np.asarray(to_suppress),
        np.array([False, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["obj_first_frame"]),
        np.array([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["consecutive_unmatch_count"]),
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["trk_keep_alive"]),
        np.array([1, -1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["removed_mask"]),
        np.array([False, True]),
    )


def test_sam3_multiplex_base_hotstart_gpu_suppresses_stale_unmatched():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        hotstart_unmatch_thresh=5,
        suppress_unmatched_only_within_hotstart=False,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([8], dtype=np.int64)
    rank0_metadata = tracker_metadata_prev["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"][8] = 0
    rank0_metadata["trk_keep_alive"][8] = 0
    adt_result = _lazy_hotstart_assoc(
        trk_is_unmatched=[True],
        trk_is_nonempty=[True],
        im_mask=np.zeros((0, 1), dtype=bool),
    )

    to_remove, to_suppress, gpu_metadata = base._process_hotstart_gpu(
        frame_idx=10,
        reverse=False,
        adt_result=adt_result,
        tracker_metadata_prev=tracker_metadata_prev,
        gpu_metadata_prev={"N_obj": 0},
    )

    np.testing.assert_array_equal(np.asarray(to_remove), np.array([False]))
    np.testing.assert_array_equal(np.asarray(to_suppress), np.array([True]))
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["trk_keep_alive"]),
        np.array([-1], dtype=np.int64),
    )


def test_sam3_multiplex_base_hotstart_gpu_rejects_association_shape_drift():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([8], dtype=np.int64)
    adt_result = _lazy_hotstart_assoc(
        trk_is_unmatched=[False],
        trk_is_nonempty=[True],
        im_mask=[[True, False]],
    )

    with pytest.raises(ValueError, match="im_mask must have shape"):
        base._process_hotstart_gpu(
            frame_idx=1,
            reverse=False,
            adt_result=adt_result,
            tracker_metadata_prev=tracker_metadata_prev,
            gpu_metadata_prev={"N_obj": 0},
        )


def test_sam3_multiplex_base_hotstart_gpu_removes_later_duplicate():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        hotstart_delay=3,
        hotstart_dup_thresh=1,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([1, 2], dtype=np.int64)
    rank0_metadata = tracker_metadata_prev["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"].update({1: 0, 2: 1})
    rank0_metadata["trk_keep_alive"].update({1: 0, 2: 0})
    adt_result = _lazy_hotstart_assoc(
        trk_is_unmatched=[False, False],
        trk_is_nonempty=[True, True],
        im_mask=[[True, True]],
    )

    to_remove, to_suppress, gpu_metadata = base._process_hotstart_gpu(
        frame_idx=2,
        reverse=False,
        adt_result=adt_result,
        tracker_metadata_prev=tracker_metadata_prev,
        gpu_metadata_prev={"N_obj": 0},
    )

    np.testing.assert_array_equal(
        np.asarray(to_remove),
        np.array([False, True]),
    )
    np.testing.assert_array_equal(
        np.asarray(to_suppress),
        np.array([False, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["overlap_pair_counts"]),
        np.array([[0, 1], [0, 0]], dtype=np.int64),
    )


def test_sam3_multiplex_base_hotstart_gpu_compacts_and_extends_metadata():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        init_trk_keep_alive=3,
    )
    gpu_metadata = {
        "N_obj": 3,
        "obj_first_frame": mx.array([0, 1, 2], dtype=mx.int64),
        "consecutive_unmatch_count": mx.array([0, 4, 1], dtype=mx.int64),
        "trk_keep_alive": mx.array([3, -1, 2], dtype=mx.int64),
        "removed_mask": mx.array([False, True, False], dtype=mx.bool_),
        "overlap_pair_counts": mx.array(
            [
                [0, 7, 8],
                [0, 0, 9],
                [0, 0, 0],
            ],
            dtype=mx.int64,
        ),
        "last_occluded_tensor": mx.array([-1, 5, 6], dtype=mx.int64),
    }

    compacted = base._compact_hotstart_gpu_metadata(gpu_metadata)
    np.testing.assert_array_equal(
        np.asarray(compacted["obj_first_frame"]),
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(compacted["overlap_pair_counts"]),
        np.array([[0, 8], [0, 0]], dtype=np.int64),
    )

    extended = base._extend_hotstart_gpu_metadata_for_new_objects(
        compacted,
        frame_idx=7,
        num_new_objects=2,
    )
    assert extended["N_obj"] == 4
    np.testing.assert_array_equal(
        np.asarray(extended["obj_first_frame"]),
        np.array([0, 2, 7, 7], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(extended["trk_keep_alive"]),
        np.array([3, 2, 3, 3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(extended["overlap_pair_counts"]),
        np.array(
            [
                [0, 8, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
    )


def test_sam3_multiplex_base_planning_skips_hotstart_during_warm_up():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        hotstart_delay=3,
        hotstart_unmatch_thresh=1,
    )
    base._warm_up_complete = False
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_per_gpu"][0] = np.array([7], dtype=np.int64)
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([7], dtype=np.int64)
    tracker_metadata_prev["num_obj_per_gpu"][0] = 1
    tracker_metadata_prev["num_buc_per_gpu"][0] = 1
    tracker_metadata_prev["max_obj_id"] = 7
    tracker_metadata_prev["obj_id_to_score"][7] = 0.7
    rank0_metadata = tracker_metadata_prev["rank0_metadata"]
    rank0_metadata["obj_first_frame_idx"][7] = 0
    rank0_metadata["trk_keep_alive"][7] = 0
    tracker_metadata_prev["gpu_metadata"] = {
        "N_obj": 1,
        "obj_first_frame": mx.array([0], dtype=mx.int64),
        "consecutive_unmatch_count": mx.array([0], dtype=mx.int64),
        "trk_keep_alive": mx.array([0], dtype=mx.int64),
        "removed_mask": mx.array([False], dtype=mx.bool_),
        "overlap_pair_counts": mx.zeros((1, 1), dtype=mx.int64),
        "last_occluded_tensor": mx.array([-1], dtype=mx.int64),
    }

    def associate_det_trk(**kwargs):
        del kwargs
        return _lazy_hotstart_assoc(
            trk_is_unmatched=[True],
            trk_is_nonempty=[True],
            im_mask=np.zeros((0, 1), dtype=bool),
        )

    base._associate_det_trk = associate_det_trk

    update_plan, metadata = base.run_tracker_update_planning_phase(
        frame_idx=1,
        num_frames=4,
        reverse=False,
        det_out={
            "mask": mx.zeros((0, 2, 2), dtype=mx.float32),
            "scores": mx.zeros((0,), dtype=mx.float32),
        },
        det_keep=mx.zeros((0,), dtype=mx.bool_),
        tracker_low_res_masks_global=mx.ones((1, 2, 2), dtype=mx.float32),
        tracker_obj_scores_global=mx.zeros((1,), dtype=mx.float32),
        tracker_metadata_prev=tracker_metadata_prev,
        tracker_states_local=[],
    )

    assert update_plan["obj_ids_newly_removed"] == set()
    np.testing.assert_array_equal(
        metadata["obj_ids_all_gpu"],
        np.array([7], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        metadata["num_obj_per_gpu"],
        np.array([1], dtype=np.int64),
    )
    assert metadata["obj_id_to_score"][7] == 0.7
    assert metadata["rank0_metadata"]["removed_obj_ids"] == set()
    assert metadata["rank0_metadata"]["unmatched_frame_inds"].get(7, []) == []
    np.testing.assert_array_equal(
        np.asarray(metadata["gpu_metadata"]["removed_mask"]),
        np.array([False]),
    )
    np.testing.assert_array_equal(
        np.asarray(metadata["gpu_metadata"]["consecutive_unmatch_count"]),
        np.array([0], dtype=np.int64),
    )


def test_sam3_multiplex_base_creates_planning_metadata_without_aliasing_cpu_state():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    previous = base._initialize_metadata()
    previous["obj_ids_per_gpu"][0] = np.array([3], dtype=np.int64)
    previous["obj_ids_all_gpu"] = np.array([3], dtype=np.int64)
    previous["num_obj_per_gpu"][0] = 1
    previous["num_buc_per_gpu"][0] = 1
    previous["max_obj_id"] = 3
    previous["obj_id_to_score"][3] = 0.75
    previous["obj_id_to_sam2_score_frame_wise"][5][3] = mx.array(
        0.25,
        dtype=mx.float32,
    )
    previous["rank0_metadata"]["obj_first_frame_idx"][3] = 0
    previous["rank0_metadata"]["removed_obj_ids"].add(99)
    previous["gpu_metadata"]["N_obj"] = 1

    metadata = base._create_planning_metadata(previous)

    assert metadata["gpu_metadata"] is previous["gpu_metadata"]
    np.testing.assert_array_equal(metadata["obj_ids_per_gpu"][0], np.array([3]))
    assert metadata["obj_ids_all_gpu"] is None
    np.testing.assert_array_equal(metadata["num_obj_per_gpu"], np.array([1]))
    assert metadata["obj_id_to_score"] == {3: 0.75}
    assert metadata["obj_id_to_last_occluded"] == {}
    assert metadata["rank0_metadata"]["obj_first_frame_idx"] == {3: 0}
    assert metadata["rank0_metadata"]["removed_obj_ids"] == {99}

    metadata["obj_ids_per_gpu"][0][0] = 10
    metadata["rank0_metadata"]["removed_obj_ids"].add(3)

    np.testing.assert_array_equal(previous["obj_ids_per_gpu"][0], np.array([3]))
    assert previous["rank0_metadata"]["removed_obj_ids"] == {99}


def test_sam3_multiplex_base_post_execution_counts_dynamic_multiplex_buckets():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    tracker_metadata = base._initialize_metadata()
    tracker_metadata["num_buc_per_gpu"][0] = 99
    tracker_states = [
        {"name": "plain-state", "obj_ids": [1]},
        {"multiplex_state": SimpleNamespace(num_buckets=2)},
        {"multiplex_state": SimpleNamespace(num_buckets=3), "obj_ids": [2, 3]},
    ]

    assert base._count_buckets_in_states(tracker_states) == 5

    base._post_execution_phase_hook(tracker_states, tracker_metadata)

    np.testing.assert_array_equal(
        tracker_metadata["num_buc_per_gpu"],
        np.array([5], dtype=np.int64),
    )


def test_sam3_multiplex_base_post_execution_bucket_hook_noops_without_metadata():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )

    base._post_execution_phase_hook(
        [{"multiplex_state": SimpleNamespace(num_buckets=2)}],
        None,
    )


def test_sam3_multiplex_base_recent_occlusion_suppression_selects_newer_overlap():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.5,
    )
    masks = mx.array(
        [
            [[True, True], [False, False]],
            [[True, True], [False, False]],
        ],
        dtype=mx.bool_,
    )

    forward = base._get_objects_to_suppress_based_on_most_recently_occluded(
        masks,
        mx.array([4, 1], dtype=mx.int64),
        np.array([10, 11], dtype=np.int64),
        frame_idx=5,
        reverse=False,
    )
    reverse = base._get_objects_to_suppress_based_on_most_recently_occluded(
        masks,
        mx.array([4, 1], dtype=mx.int64),
        np.array([10, 11], dtype=np.int64),
        frame_idx=5,
        reverse=True,
    )

    np.testing.assert_array_equal(np.asarray(forward), np.array([True, False]))
    np.testing.assert_array_equal(np.asarray(reverse), np.array([False, True]))


def test_sam3_multiplex_base_recent_occlusion_preserves_unoccluded_suppressor_rule():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.5,
    )
    allow_base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.5,
        allow_unoccluded_to_suppress=True,
    )
    masks = mx.array(
        [
            [[True, True], [False, False]],
            [[True, True], [False, False]],
        ],
        dtype=mx.bool_,
    )

    default_result = base._get_objects_to_suppress_based_on_most_recently_occluded(
        masks,
        mx.array([4, -1], dtype=mx.int64),
        np.array([10, 11], dtype=np.int64),
    )
    allowed_result = (
        allow_base._get_objects_to_suppress_based_on_most_recently_occluded(
            masks,
            mx.array([4, -1], dtype=mx.int64),
            np.array([10, 11], dtype=np.int64),
        )
    )

    np.testing.assert_array_equal(
        np.asarray(default_result),
        np.array([False, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(allowed_result),
        np.array([True, False]),
    )


def test_sam3_multiplex_base_suppresses_recent_occlusion_masks_and_metadata():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.5,
        allow_unoccluded_to_suppress=True,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_per_gpu"][0] = np.array([10, 11], dtype=np.int64)
    tracker_metadata_prev["num_obj_per_gpu"][0] = 2
    tracker_metadata_prev["num_buc_per_gpu"][0] = 1
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10, 11], dtype=np.int64)
    tracker_metadata_prev["max_obj_id"] = 11
    tracker_metadata_prev["obj_id_to_last_occluded"] = {10: 4, 11: 1}
    tracker_metadata_new = base._create_planning_metadata(tracker_metadata_prev)
    masks = mx.array(
        [
            [[2.0, 2.0], [-1.0, -1.0]],
            [[2.0, 2.0], [-1.0, -1.0]],
        ],
        dtype=mx.float32,
    )

    suppressed = base._suppress_overlapping_based_on_recent_occlusion(
        frame_idx=5,
        tracker_low_res_masks_global=masks,
        tracker_metadata_prev=tracker_metadata_prev,
        tracker_metadata_new=tracker_metadata_new,
        to_remove_mask=mx.array([False, False], dtype=mx.bool_),
        reverse=False,
    )

    np.testing.assert_array_equal(
        np.asarray(suppressed),
        np.array(
            [
                [[-10.0, -10.0], [-10.0, -10.0]],
                [[2.0, 2.0], [-1.0, -1.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert tracker_metadata_new["obj_id_to_last_occluded"] == {10: 5, 11: 1}


def test_sam3_multiplex_base_recent_occlusion_rejects_object_id_remove_set():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.5,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10, 11], dtype=np.int64)
    tracker_metadata_new = base._create_planning_metadata(tracker_metadata_prev)

    with pytest.raises(TypeError, match="boolean vector aligned with obj_ids"):
        base._suppress_overlapping_based_on_recent_occlusion(
            frame_idx=5,
            tracker_low_res_masks_global=mx.ones((2, 2, 2), dtype=mx.float32),
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_metadata_new=tracker_metadata_new,
            to_remove_mask={10},
        )


def test_sam3_multiplex_base_counts_zero_buckets_for_non_multiplex_model():
    base = Sam3MultiplexBase(
        tracker=_PlainTracker(),
        detector=_PlainDetector(),
        is_multiplex=False,
    )

    assert (
        base._count_buckets_in_states(
            [{"multiplex_state": SimpleNamespace(num_buckets=2)}]
        )
        == 0
    )


def test_sam3_multiplex_base_tracker_remove_objects_accepts_hotstart_sets():
    tracker = _ObjectRemovalTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    tracker_states = [{"name": "state", "obj_ids": [1, 2, 3]}]

    base._tracker_remove_objects(tracker_states, {2, 3})

    assert tracker_states == [{"name": "state", "obj_ids": [1]}]
    assert [call["obj_id"] for call in tracker.remove_calls] == [2, 3]
    assert all(call["strict"] is False for call in tracker.remove_calls)
    assert all(call["need_output"] is False for call in tracker.remove_calls)


def test_sam3_multiplex_base_tracker_remove_objects_maps_packed_object_ids():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        allowed_bucket_capacity=2,
        object_ids=[10, 11, 12],
    )
    tracker_states = [
        {
            "name": "packed-state",
            "obj_ids": [10, 11, 12],
            "multiplex_state": multiplex_state,
        }
    ]

    base._tracker_remove_objects(tracker_states, {11})

    assert len(tracker_states) == 1
    assert tracker_states[0]["obj_ids"] == [10, 12]
    assert multiplex_state.object_ids == [10, 12]
    assert multiplex_state.total_valid_entries == 2
    assert multiplex_state.num_buckets == 2
    assert multiplex_state.get_all_valid_object_idx() == {0, 1}
    data = np.array([[5.0], [9.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        multiplex_state.demux(multiplex_state.mux(data)),
        data,
    )


def test_sam3_multiplex_base_tracker_remove_objects_drops_empty_packed_states():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    multiplex_state = MultiplexState(
        [[0, 1]],
        allowed_bucket_capacity=2,
        object_ids=[20, 21],
    )
    tracker_states = [
        {
            "obj_ids": np.array([20, 21], dtype=np.int64),
            "multiplex_state": multiplex_state,
        }
    ]

    base._tracker_remove_objects(tracker_states, mx.array([20, 21], dtype=mx.int64))

    assert tracker_states == []
    assert multiplex_state.object_ids == []
    assert multiplex_state.assignments is None


def test_sam3_multiplex_base_tracker_update_memories_writes_encoded_outputs():
    tracker = _MemoryUpdateTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    state_a = {
        "name": "state-a",
        "obj_ids": [10, 11],
        "output_dict": {
            "cond_frame_outputs": {3: {"object_score_logits": mx.ones((2, 1))}},
            "non_cond_frame_outputs": {3: {"object_score_logits": mx.ones((2, 1))}},
        },
    }
    state_b = {
        "name": "state-b",
        "obj_ids": [12],
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {3: {"object_score_logits": mx.ones((1, 1))}},
        },
    }
    low_res_masks = mx.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[-1.0, -1.0], [-1.0, -1.0]],
        ],
        dtype=mx.float32,
    )

    base._tracker_update_memories(
        [state_a, state_b],
        frame_idx=3,
        tracker_metadata={"num_obj_per_gpu": np.array([3], dtype=np.int64)},
        low_res_masks=low_res_masks,
    )

    assert len(tracker.suppress_calls) == 1
    assert [
        {
            "state": call["state"],
            "frame_idx": call["frame_idx"],
            "local_batch_size": call["local_batch_size"],
            "is_mask_from_pts": call["is_mask_from_pts"],
        }
        for call in tracker.encoder_calls
    ] == [
        {
            "state": "state-a",
            "frame_idx": 3,
            "local_batch_size": 2,
            "is_mask_from_pts": False,
        },
        {
            "state": "state-b",
            "frame_idx": 3,
            "local_batch_size": 1,
            "is_mask_from_pts": False,
        },
    ]
    np.testing.assert_array_equal(
        tracker.encoder_calls[0]["mask_sample"],
        np.array([1.0, 2.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tracker.encoder_calls[0]["score_logits"],
        np.array([[10.0], [10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tracker.encoder_calls[1]["mask_sample"],
        np.array([-1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tracker.encoder_calls[1]["score_logits"],
        np.array([[-10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(state_a["output_dict"]["cond_frame_outputs"][3]["maskmem_features"]),
        np.ones((2, 2), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(
            state_a["output_dict"]["non_cond_frame_outputs"][3]["image_pos_enc"]
        ),
        np.full((2, 4), 31.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(
            state_b["output_dict"]["non_cond_frame_outputs"][3]["maskmem_features"]
        ),
        np.full((1, 2), 2.0, dtype=np.float32),
    )
    assert [call["storage_key"] for call in tracker.add_output_calls] == [
        "cond_frame_outputs",
        "non_cond_frame_outputs",
        "non_cond_frame_outputs",
    ]


def test_sam3_multiplex_base_tracker_update_memories_sorts_dynamic_objects():
    tracker = _MemoryUpdateTracker(dynamic=True)
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    state_a = {
        "name": "state-a",
        "obj_ids": [30, 10],
        "output_dict": {
            "cond_frame_outputs": {4: {"object_score_logits": mx.ones((2, 1))}},
            "non_cond_frame_outputs": {},
        },
    }
    state_b = {
        "name": "state-b",
        "obj_ids": [20],
        "output_dict": {
            "cond_frame_outputs": {4: {"object_score_logits": mx.ones((1, 1))}},
            "non_cond_frame_outputs": {},
        },
    }
    low_res_masks_sorted_by_obj_id = mx.array(
        [
            [[10.0, 10.0], [10.0, 10.0]],
            [[20.0, 20.0], [20.0, 20.0]],
            [[30.0, 30.0], [30.0, 30.0]],
        ],
        dtype=mx.float32,
    )

    base._tracker_update_memories(
        [state_a, state_b],
        frame_idx=4,
        tracker_metadata={"num_obj_per_gpu": np.array([3], dtype=np.int64)},
        low_res_masks=low_res_masks_sorted_by_obj_id,
    )

    assert tracker.encoder_calls[0]["state"] == "state-a"
    np.testing.assert_array_equal(
        tracker.encoder_calls[0]["mask_sample"],
        np.array([10.0, 30.0], dtype=np.float32),
    )
    assert tracker.encoder_calls[1]["state"] == "state-b"
    np.testing.assert_array_equal(
        tracker.encoder_calls[1]["mask_sample"],
        np.array([20.0], dtype=np.float32),
    )


def test_sam3_multiplex_base_tracker_update_memories_matches_official_fixture():
    fixture = json.loads(MEMORY_UPDATE_PARITY_FIXTURE.read_text())
    assert fixture["official_commit"] == OFFICIAL_SAM3_MULTIPLEX_BASE_COMMIT
    assert fixture["case"] == "memory_encoder_slicing_and_dynamic_sorting"
    assert fixture["component"] == "Sam3MultiplexBase._tracker_update_memories"
    for case_payload in fixture["cases"].values():
        assert all(case_payload["scalar_matches"].values())
        for metric in case_payload["metrics"].values():
            assert metric["max_abs"] <= fixture["atol"]

    def make_state(name, obj_ids):
        return {
            "name": name,
            "obj_ids": obj_ids.copy(),
            "output_dict": {
                "cond_frame_outputs": {
                    4: {"object_score_logits": mx.ones((len(obj_ids), 1))}
                },
                "non_cond_frame_outputs": {
                    4: {"object_score_logits": mx.ones((len(obj_ids), 1))}
                },
            },
        }

    def assert_payload_matches(observed, expected_payload):
        observed_np = np.asarray(observed)
        assert list(observed_np.shape) == expected_payload["shape"]
        assert str(observed_np.dtype) == expected_payload["dtype"]
        np.testing.assert_allclose(
            observed_np,
            np.asarray(expected_payload["values"], dtype=np.float32),
            rtol=0.0,
            atol=fixture["atol"],
        )

    def run_case(case_name, *, dynamic, states, low_res_values):
        tracker = _MemoryUpdateTracker(dynamic=dynamic)
        base = Sam3MultiplexBase(
            tracker=tracker,
            detector=_DummyDetector(),
            is_multiplex=True,
        )
        low_res_masks = mx.array(
            np.stack(
                [np.full((2, 2), value, dtype=np.float32) for value in low_res_values],
                axis=0,
            ),
            dtype=mx.float32,
        )

        base._tracker_update_memories(
            states,
            frame_idx=4,
            tracker_metadata={
                "num_obj_per_gpu": np.array(
                    [sum(len(state["obj_ids"]) for state in states)],
                    dtype=np.int64,
                )
            },
            low_res_masks=low_res_masks,
        )

        expected = fixture["cases"][case_name]["official"]
        assert [
            {
                key: value
                for key, value in call.items()
                if key
                not in {
                    "high_res_masks",
                    "mask_sample",
                    "score_logits",
                }
            }
            for call in tracker.encoder_calls
        ] == [
            {
                key: value
                for key, value in call.items()
                if key
                not in {
                    "local_high_res_masks",
                    "local_object_score_logits",
                }
            }
            for call in expected["encoder_calls"]
        ]
        assert [
            {
                "state": call["state"],
                "frame_idx": call["frame_idx"],
                "storage_key": call["storage_key"],
            }
            for call in tracker.add_output_calls
        ] == expected["add_output_calls"]

        for call, expected_call in zip(
            tracker.encoder_calls,
            expected["encoder_calls"],
            strict=True,
        ):
            assert_payload_matches(
                call["high_res_masks"],
                expected_call["local_high_res_masks"],
            )
            assert_payload_matches(
                call["score_logits"],
                expected_call["local_object_score_logits"],
            )
        assert len(tracker.suppress_calls) == len(expected["suppress_calls"])
        for call, expected_call in zip(
            tracker.suppress_calls,
            expected["suppress_calls"],
            strict=True,
        ):
            assert_payload_matches(
                call,
                expected_call["high_res_masks"],
            )

        for state, expected_state in zip(states, expected["states"], strict=True):
            assert state["name"] == expected_state["name"]
            assert state["obj_ids"] == expected_state["obj_ids"]
            for storage_key, frame_key in [
                ("cond_frame_outputs", "cond"),
                ("non_cond_frame_outputs", "non_cond"),
            ]:
                frame_out = state["output_dict"][storage_key][4]
                expected_frame = expected_state[frame_key]
                assert_payload_matches(
                    frame_out["maskmem_features"],
                    expected_frame["maskmem_features"],
                )
                assert_payload_matches(
                    frame_out["maskmem_pos_enc"][0],
                    expected_frame["maskmem_pos_enc_0"],
                )
                assert_payload_matches(
                    frame_out["image_features"],
                    expected_frame["image_features"],
                )
                assert_payload_matches(
                    frame_out["image_pos_enc"],
                    expected_frame["image_pos_enc"],
                )

    run_case(
        "static_state_slicing",
        dynamic=False,
        states=[
            make_state("state-a", [10, 11]),
            make_state("state-b", [12]),
        ],
        low_res_values=[1.0, 2.0, -1.0],
    )
    run_case(
        "dynamic_object_id_sorting",
        dynamic=True,
        states=[
            make_state("state-a", [30, 10]),
            make_state("state-b", [20]),
        ],
        low_res_values=[10.0, 20.0, 30.0],
    )


def test_sam3_multiplex_base_tracker_update_memories_reapplies_no_object_pointer():
    tracker = _NoObjectPointerMemoryUpdateTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        reapply_no_object_pointer=True,
    )
    multiplex_state = MultiplexState(
        [[0, 1]],
        allowed_bucket_capacity=2,
        object_ids=[10, 20],
    )
    ptr_values = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)
    state = {
        "name": "state-a",
        "obj_ids": [10, 20],
        "multiplex_state": multiplex_state,
        "output_dict": {
            "cond_frame_outputs": {
                6: {
                    "object_score_logits": mx.ones((2, 1), dtype=mx.float32),
                    "obj_ptr": multiplex_state.mux(ptr_values),
                }
            },
            "non_cond_frame_outputs": {},
        },
    }
    low_res_masks = mx.array(
        [
            [[3.0, 3.0], [3.0, 3.0]],
            [[-2.0, -2.0], [-2.0, -2.0]],
        ],
        dtype=mx.float32,
    )

    base._tracker_update_memories(
        [state],
        frame_idx=6,
        tracker_metadata={"num_obj_per_gpu": np.array([2], dtype=np.int64)},
        low_res_masks=low_res_masks,
    )

    frame_out = state["output_dict"]["cond_frame_outputs"][6]
    np.testing.assert_array_equal(
        np.asarray(multiplex_state.demux(frame_out["obj_ptr"])),
        np.array(
            [
                [1.0, 2.0],
                [103.0, 104.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        tracker.encoder_calls[0]["score_logits"],
        np.array([[10.0], [-10.0]], dtype=np.float32),
    )
    assert tracker.add_output_calls[0]["current_out"] is frame_out


def test_sam3_multiplex_base_tracker_update_memories_keeps_unsuppressed_pointers():
    tracker = _NoObjectPointerMemoryUpdateTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        reapply_no_object_pointer=True,
    )
    multiplex_state = MultiplexState(
        [[0, -1], [1, -1]],
        allowed_bucket_capacity=1,
        object_ids=[10, 20],
    )
    existing_ptr = mx.array(
        [
            [[1.0, 2.0], [90.0, 91.0]],
            [[3.0, 4.0], [92.0, 93.0]],
        ],
        dtype=mx.float32,
    )
    state = {
        "name": "state-a",
        "obj_ids": [10, 20],
        "multiplex_state": multiplex_state,
        "output_dict": {
            "cond_frame_outputs": {
                8: {
                    "object_score_logits": mx.ones((2, 1), dtype=mx.float32),
                    "obj_ptr": existing_ptr,
                }
            },
            "non_cond_frame_outputs": {},
        },
    }
    low_res_masks = mx.ones((2, 2, 2), dtype=mx.float32)

    base._tracker_update_memories(
        [state],
        frame_idx=8,
        tracker_metadata={"num_obj_per_gpu": np.array([2], dtype=np.int64)},
        low_res_masks=low_res_masks,
    )

    frame_out = state["output_dict"]["cond_frame_outputs"][8]
    assert frame_out["obj_ptr"] is existing_ptr
    np.testing.assert_array_equal(
        np.asarray(frame_out["obj_ptr"]),
        np.array(
            [
                [[1.0, 2.0], [90.0, 91.0]],
                [[3.0, 4.0], [92.0, 93.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_sam3_multiplex_base_tracker_update_memories_requires_no_object_linear():
    tracker = _MemoryUpdateTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        reapply_no_object_pointer=True,
    )
    multiplex_state = MultiplexState(
        [[0]],
        allowed_bucket_capacity=1,
        object_ids=[10],
    )
    state = {
        "name": "state-a",
        "obj_ids": [10],
        "multiplex_state": multiplex_state,
        "output_dict": {
            "cond_frame_outputs": {
                9: {
                    "object_score_logits": mx.ones((1, 1), dtype=mx.float32),
                    "obj_ptr": multiplex_state.mux(mx.ones((1, 2), dtype=mx.float32)),
                }
            },
            "non_cond_frame_outputs": {},
        },
    }

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="no_obj_ptr_linear"):
        base._tracker_update_memories(
            [state],
            frame_idx=9,
            tracker_metadata={"num_obj_per_gpu": np.array([1], dtype=np.int64)},
            low_res_masks=-mx.ones((1, 2, 2), dtype=mx.float32),
        )


def test_sam3_multiplex_base_tracker_update_memories_requires_pointer_scores():
    tracker = _NoObjectPointerMemoryUpdateTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        reapply_no_object_pointer=True,
    )
    multiplex_state = MultiplexState(
        [[0]],
        allowed_bucket_capacity=1,
        object_ids=[10],
    )
    state = {
        "name": "state-a",
        "obj_ids": [10],
        "multiplex_state": multiplex_state,
        "output_dict": {
            "cond_frame_outputs": {
                10: {
                    "obj_ptr": multiplex_state.mux(mx.ones((1, 2), dtype=mx.float32)),
                }
            },
            "non_cond_frame_outputs": {},
        },
    }

    with pytest.raises(ValueError, match="object_score_logits"):
        base._tracker_update_memories(
            [state],
            frame_idx=10,
            tracker_metadata={"num_obj_per_gpu": np.array([1], dtype=np.int64)},
            low_res_masks=-mx.ones((1, 2, 2), dtype=mx.float32),
        )


def test_sam3_multiplex_base_execution_updates_tracker_memories_when_available():
    tracker = _MemoryUpdateTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    tracker_state = {
        "name": "state-a",
        "obj_ids": [10],
        "output_dict": {
            "cond_frame_outputs": {5: {"object_score_logits": mx.ones((1, 1))}},
            "non_cond_frame_outputs": {},
        },
    }
    tracker_metadata_new = base._initialize_metadata()
    tracker_metadata_new["num_obj_per_gpu"][0] = 1
    tracker_metadata_new["num_buc_per_gpu"][0] = 1
    tracker_metadata_new["obj_ids_per_gpu"][0] = np.array([10], dtype=np.int64)
    tracker_metadata_new["obj_ids_all_gpu"] = np.array([10], dtype=np.int64)

    base.run_tracker_update_execution_phase(
        frame_idx=5,
        num_frames=6,
        reverse=False,
        det_out={"mask": mx.zeros((0, 2, 2), dtype=mx.float32)},
        tracker_states_local=[tracker_state],
        tracker_update_plan={
            "new_det_fa_inds": np.array([], dtype=np.int64),
            "new_det_obj_ids": np.array([], dtype=np.int64),
            "new_det_gpu_ids": np.array([], dtype=np.int64),
            "obj_ids_newly_removed": set(),
            "tracker_low_res_masks_global": mx.array(
                [[[3.0, 3.0], [3.0, 3.0]]],
                dtype=mx.float32,
            ),
        },
        orig_vid_height=2,
        orig_vid_width=2,
        feature_cache={},
        tracker_metadata_new=tracker_metadata_new,
    )

    assert len(tracker.encoder_calls) == 1
    assert tracker.encoder_calls[0]["state"] == "state-a"
    np.testing.assert_array_equal(
        tracker.encoder_calls[0]["mask_sample"],
        np.array([3.0], dtype=np.float32),
    )
    assert tracker.add_output_calls[0]["storage_key"] == "cond_frame_outputs"
    np.testing.assert_array_equal(
        np.asarray(
            tracker_state["output_dict"]["cond_frame_outputs"][5]["maskmem_features"]
        ),
        np.ones((1, 2), dtype=np.float32),
    )


def test_sam3_multiplex_base_execution_updates_old_memory_before_local_add():
    class _ExecutionPhaseTracker(_DynamicAddTracker):
        input_mask_size = 2

        def __init__(self):
            super().__init__()
            self.events = []
            self.maskmem_backbone = SimpleNamespace(
                mask_downsampler=SimpleNamespace(interpol_size=(2, 2))
            )
            self.encoder_calls = []
            self.add_output_calls = []
            self.suppress_calls = []

        def _suppress_object_pw_area_shrinkage(self, high_res_masks):
            self.events.append("suppress")
            self.suppress_calls.append(high_res_masks)
            return high_res_masks

        def _run_memory_encoder(
            self,
            sam2_state,
            frame_idx,
            local_batch_size,
            local_high_res_masks,
            local_object_score_logits,
            is_mask_from_pts,
        ):
            self.events.append(f"memory:{sam2_state['obj_ids']}")
            self.encoder_calls.append(
                {
                    "state": sam2_state["name"],
                    "frame_idx": frame_idx,
                    "local_batch_size": local_batch_size,
                    "high_res_masks": local_high_res_masks,
                    "score_logits": local_object_score_logits,
                    "is_mask_from_pts": is_mask_from_pts,
                }
            )
            return (
                mx.ones((local_batch_size, 2), dtype=mx.float32),
                [mx.ones((local_batch_size, 1), dtype=mx.float32) * 2.0],
                mx.ones((local_batch_size, 3), dtype=mx.float32) * 3.0,
                mx.ones((local_batch_size, 4), dtype=mx.float32) * 4.0,
            )

        def add_new_masks(self, **kwargs):
            self.events.append(f"add:{list(kwargs['obj_ids'])}")
            return super().add_new_masks(**kwargs)

        def propagate_in_video_preflight(self, sam2_state, run_mem_encoder=True):
            self.events.append("preflight")
            super().propagate_in_video_preflight(
                sam2_state,
                run_mem_encoder=run_mem_encoder,
            )

        def add_output_per_object(
            self,
            *,
            inference_state,
            frame_idx,
            current_out,
            storage_key,
        ):
            self.events.append(f"add_output:{storage_key}")
            self.add_output_calls.append(
                {
                    "state": inference_state["name"],
                    "frame_idx": frame_idx,
                    "current_out": current_out,
                    "storage_key": storage_key,
                }
            )

    tracker = _ExecutionPhaseTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    multiplex_state = MultiplexState(
        [[0, -1]],
        allowed_bucket_capacity=2,
        object_ids=[20],
    )
    tracker_state = {
        "name": "state-a",
        "obj_ids": [20],
        "multiplex_state": multiplex_state,
        "output_dict": {
            "cond_frame_outputs": {5: {"object_score_logits": mx.ones((1, 1))}},
            "non_cond_frame_outputs": {},
        },
    }
    tracker_metadata_new = base._initialize_metadata()

    returned_states = base.run_tracker_update_execution_phase(
        frame_idx=5,
        num_frames=6,
        reverse=False,
        det_out={
            "mask": mx.array(
                [
                    [[1.0, -1.0], [-1.0, -1.0]],
                    [[-1.0, 2.0], [2.0, -1.0]],
                ],
                dtype=mx.float32,
            )
        },
        tracker_states_local=[tracker_state],
        tracker_update_plan={
            "new_det_fa_inds": np.array([0, 1], dtype=np.int64),
            "new_det_obj_ids": np.array([30, 31], dtype=np.int64),
            "new_det_gpu_ids": np.array([1, 0], dtype=np.int64),
            "obj_ids_newly_removed": {20},
            "tracker_low_res_masks_global": mx.array(
                [[[5.0, 5.0], [5.0, 5.0]]],
                dtype=mx.float32,
            ),
        },
        orig_vid_height=2,
        orig_vid_width=2,
        feature_cache={"cached": "features"},
        tracker_metadata_new=tracker_metadata_new,
    )

    assert returned_states == [tracker_state]
    assert tracker.events == [
        "suppress",
        "memory:[20]",
        "add_output:cond_frame_outputs",
        "add:[31]",
        "preflight",
    ]
    assert tracker.add_calls[0]["obj_ids"] == [31]
    np.testing.assert_array_equal(
        np.asarray(tracker.add_calls[0]["masks"]),
        np.array([[[False, True], [True, False]]], dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(tracker.encoder_calls[0]["high_res_masks"]),
        np.full((1, 1, 2, 2), 5.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(tracker.encoder_calls[0]["score_logits"]),
        np.array([[10.0]], dtype=np.float32),
    )
    assert tracker_state["obj_ids"] == [31]
    assert multiplex_state.object_ids == [31]
    np.testing.assert_array_equal(
        tracker_metadata_new["obj_ids_all_gpu"],
        np.array([31], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["num_obj_per_gpu"],
        np.array([1], dtype=np.int64),
    )
    assert tracker_metadata_new["gpu_metadata"]["N_obj"] == 1
    np.testing.assert_array_equal(
        np.asarray(tracker_metadata_new["gpu_metadata"]["obj_first_frame"]),
        np.array([5], dtype=np.int64),
    )


def test_sam3_multiplex_base_execution_counts_sparse_packed_buckets_after_removal():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    multiplex_state = MultiplexState(
        [[0, 1], [2, -1]],
        allowed_bucket_capacity=2,
        object_ids=[10, 11, 12],
    )
    tracker_states = [
        {
            "name": "packed-state",
            "obj_ids": [10, 11, 12],
            "multiplex_state": multiplex_state,
        }
    ]
    tracker_metadata_new = base._initialize_metadata()
    tracker_metadata_new["num_obj_per_gpu"][0] = 2
    tracker_metadata_new["num_buc_per_gpu"][0] = 99
    tracker_metadata_new["obj_ids_per_gpu"][0] = np.array([10, 12], dtype=np.int64)
    tracker_metadata_new["obj_ids_all_gpu"] = np.array([10, 12], dtype=np.int64)

    base.run_tracker_update_execution_phase(
        frame_idx=5,
        num_frames=6,
        reverse=False,
        det_out={"mask": mx.zeros((0, 2, 2), dtype=mx.float32)},
        tracker_states_local=tracker_states,
        tracker_update_plan={
            "new_det_fa_inds": np.array([], dtype=np.int64),
            "new_det_obj_ids": np.array([], dtype=np.int64),
            "new_det_gpu_ids": np.array([], dtype=np.int64),
            "obj_ids_newly_removed": {11},
        },
        orig_vid_height=2,
        orig_vid_width=2,
        feature_cache={},
        tracker_metadata_new=tracker_metadata_new,
    )

    assert tracker_states[0]["obj_ids"] == [10, 12]
    assert multiplex_state.object_ids == [10, 12]
    assert multiplex_state.num_buckets == 2
    assert tracker_metadata_new["num_buc_per_gpu"][0] == 2
    assert tracker_metadata_new["gpu_metadata"]["N_obj"] == 2


def test_sam3_multiplex_base_execution_compacts_gpu_metadata_after_removal():
    tracker = _ObjectRemovalTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    tracker_state = {"obj_ids": [7, 9, 11]}
    tracker_metadata_new = base._initialize_metadata()
    tracker_metadata_new["num_obj_per_gpu"][0] = 3
    tracker_metadata_new["num_buc_per_gpu"][0] = 1
    tracker_metadata_new["obj_ids_per_gpu"][0] = np.array([7, 9, 11], dtype=np.int64)
    tracker_metadata_new["obj_ids_all_gpu"] = np.array([7, 9, 11], dtype=np.int64)
    tracker_metadata_new["gpu_metadata"] = {
        "N_obj": 3,
        "obj_first_frame": mx.array([0, 1, 2], dtype=mx.int64),
        "consecutive_unmatch_count": mx.array([3, 4, 5], dtype=mx.int64),
        "trk_keep_alive": mx.array([6, 7, 8], dtype=mx.int64),
        "removed_mask": mx.array([False, False, False], dtype=mx.bool_),
        "overlap_pair_counts": mx.array(
            [
                [0, 12, 13],
                [0, 0, 23],
                [0, 0, 0],
            ],
            dtype=mx.int64,
        ),
        "last_occluded_tensor": mx.array([-1, 5, 6], dtype=mx.int64),
    }

    base.run_tracker_update_execution_phase(
        frame_idx=5,
        num_frames=6,
        reverse=False,
        det_out={"mask": mx.zeros((0, 2, 2), dtype=mx.float32)},
        tracker_states_local=[tracker_state],
        tracker_update_plan={
            "new_det_fa_inds": np.array([], dtype=np.int64),
            "new_det_obj_ids": np.array([], dtype=np.int64),
            "new_det_gpu_ids": np.array([], dtype=np.int64),
            "obj_ids_newly_removed": {9},
            "tracker_low_res_masks_global": None,
        },
        orig_vid_height=2,
        orig_vid_width=2,
        feature_cache={},
        tracker_metadata_new=tracker_metadata_new,
    )

    assert tracker_state["obj_ids"] == [7, 11]
    np.testing.assert_array_equal(
        tracker_metadata_new["obj_ids_per_gpu"][0],
        np.array([7, 11], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["obj_ids_all_gpu"],
        np.array([7, 11], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["num_obj_per_gpu"],
        np.array([2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["num_buc_per_gpu"],
        np.array([1], dtype=np.int64),
    )

    gpu_metadata = tracker_metadata_new["gpu_metadata"]
    assert gpu_metadata["N_obj"] == 2
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["obj_first_frame"]),
        np.array([0, 2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["consecutive_unmatch_count"]),
        np.array([3, 5], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["trk_keep_alive"]),
        np.array([6, 8], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["removed_mask"]),
        np.array([False, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["overlap_pair_counts"]),
        np.array([[0, 13], [0, 0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["last_occluded_tensor"]),
        np.array([-1, 6], dtype=np.int64),
    )


def test_sam3_multiplex_base_recent_occlusion_suppresses_newer_overlap():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.2,
    )
    masks = mx.array(
        [
            [[True, True], [False, False]],
            [[True, True], [False, False]],
            [[False, False], [True, True]],
        ],
        dtype=mx.bool_,
    )

    to_suppress = base._get_objects_to_suppress_based_on_most_recently_occluded(
        masks,
        mx.array([5, 2, -1], dtype=mx.int64),
        np.array([10, 20, 30], dtype=np.int64),
    )

    np.testing.assert_array_equal(
        np.asarray(to_suppress),
        np.array([True, False, False]),
    )


def test_sam3_multiplex_base_recent_occlusion_reverse_suppresses_lower_frame():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.2,
    )
    masks = mx.array(
        [
            [[True, True], [False, False]],
            [[True, True], [False, False]],
        ],
        dtype=mx.bool_,
    )

    to_suppress = base._get_objects_to_suppress_based_on_most_recently_occluded(
        masks,
        mx.array([2, 5], dtype=mx.int64),
        np.array([10, 20], dtype=np.int64),
        reverse=True,
    )

    np.testing.assert_array_equal(np.asarray(to_suppress), np.array([True, False]))


def test_sam3_multiplex_base_suppresses_recent_overlap_and_updates_metadata():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.2,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10, 20], dtype=np.int64)
    tracker_metadata_prev["obj_id_to_last_occluded"] = {10: 5, 20: 1}
    tracker_metadata_new = base._create_planning_metadata(tracker_metadata_prev)
    low_res_masks = mx.array(
        [
            [[2.0, 2.0], [-1.0, -1.0]],
            [[2.0, 2.0], [-1.0, -1.0]],
        ],
        dtype=mx.float32,
    )

    suppressed = base._suppress_overlapping_based_on_recent_occlusion(
        frame_idx=6,
        tracker_low_res_masks_global=low_res_masks,
        tracker_metadata_prev=tracker_metadata_prev,
        tracker_metadata_new=tracker_metadata_new,
    )

    np.testing.assert_array_equal(
        np.asarray(suppressed),
        np.array(
            [
                [[-10.0, -10.0], [-10.0, -10.0]],
                [[2.0, 2.0], [-1.0, -1.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert tracker_metadata_new["obj_id_to_last_occluded"] == {10: 6, 20: 1}
    np.testing.assert_array_equal(
        np.asarray(tracker_metadata_new["gpu_metadata"]["last_occluded_tensor"]),
        np.array([6, 1], dtype=np.int64),
    )


def test_sam3_multiplex_base_execution_suppresses_recent_overlap_before_memory_update():
    tracker = _MemoryUpdateTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.2,
    )
    tracker_states = [
        {
            "name": "state-a",
            "obj_ids": [10, 20],
            "output_dict": {
                "cond_frame_outputs": {7: {"object_score_logits": mx.ones((2, 1))}},
                "non_cond_frame_outputs": {},
            },
        }
    ]
    tracker_metadata_new = base._initialize_metadata()
    tracker_metadata_new["num_obj_per_gpu"][0] = 2
    tracker_metadata_new["num_buc_per_gpu"][0] = 1
    tracker_metadata_new["obj_ids_per_gpu"][0] = np.array([10, 20], dtype=np.int64)
    tracker_metadata_new["obj_ids_all_gpu"] = np.array([10, 20], dtype=np.int64)
    tracker_metadata_new["obj_id_to_last_occluded"] = {10: 5, 20: 1}

    base.run_tracker_update_execution_phase(
        frame_idx=7,
        num_frames=8,
        reverse=False,
        det_out={"mask": mx.zeros((0, 2, 2), dtype=mx.float32)},
        tracker_states_local=tracker_states,
        tracker_update_plan={
            "new_det_fa_inds": np.array([], dtype=np.int64),
            "new_det_obj_ids": np.array([], dtype=np.int64),
            "new_det_gpu_ids": np.array([], dtype=np.int64),
            "obj_ids_newly_removed": set(),
            "tracker_low_res_masks_global": mx.array(
                [
                    [[3.0, 3.0], [-1.0, -1.0]],
                    [[3.0, 3.0], [-1.0, -1.0]],
                ],
                dtype=mx.float32,
            ),
        },
        orig_vid_height=2,
        orig_vid_width=2,
        feature_cache={},
        tracker_metadata_new=tracker_metadata_new,
    )

    np.testing.assert_array_equal(
        tracker.encoder_calls[0]["mask_sample"],
        np.array([-10.0, 3.0], dtype=np.float32),
    )
    assert tracker_metadata_new["obj_id_to_last_occluded"] == {10: 7, 20: 1}


def test_sam3_multiplex_base_tracker_add_new_objects_uses_tightest_dynamic_state():
    tracker = _DynamicAddTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    loose_state = _dynamic_tracker_state(
        "loose",
        [[0, -1, -1], [1, -1, -1]],
        [100, 101],
        allowed_bucket_capacity=3,
    )
    tight_state = _dynamic_tracker_state(
        "tight",
        [[0, -1, -1]],
        [200],
        allowed_bucket_capacity=3,
    )
    tracker_states = [loose_state, tight_state]

    returned_states = base._tracker_add_new_objects(
        frame_idx=4,
        num_frames=8,
        new_obj_ids=mx.array([30, 31], dtype=mx.int64),
        new_obj_masks=mx.array(
            [
                [[1.0, -1.0], [-1.0, 0.5]],
                [[-1.0, 2.0], [0.0, -3.0]],
            ],
            dtype=mx.float32,
        ),
        tracker_states_local=tracker_states,
        orig_vid_height=2,
        orig_vid_width=2,
        feature_cache={"cached": "features"},
    )

    assert returned_states is tracker_states
    assert tracker.init_calls == []
    assert len(tracker.add_calls) == 1
    add_call = tracker.add_calls[0]
    assert add_call["state"] is tight_state
    assert add_call["frame_idx"] == 4
    assert add_call["obj_ids"] == [30, 31]
    assert add_call["add_mask_to_memory"] is True
    np.testing.assert_array_equal(
        np.asarray(add_call["masks"]),
        np.array(
            [
                [[True, False], [False, True]],
                [[False, True], [False, False]],
            ],
            dtype=bool,
        ),
    )
    assert tight_state["obj_ids"] == [200, 30, 31]
    assert tight_state["multiplex_state"].object_ids == [200, 30, 31]
    assert tight_state["multiplex_state"].available_slots == 0
    assert loose_state["multiplex_state"].available_slots == 4
    assert tracker.preflight_calls == [(tight_state, True)]


def test_sam3_multiplex_base_tracker_add_new_objects_creates_dynamic_state_when_full():
    tracker = _DynamicAddTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    cached_backbone = object()
    full_state = _dynamic_tracker_state(
        "full",
        [[0, -1]],
        [10],
        allowed_bucket_capacity=1,
    )
    full_state["backbone_out"] = cached_backbone
    tracker_states = [full_state]
    feature_cache = {"frame": "features"}

    returned_states = base._tracker_add_new_objects(
        frame_idx=5,
        num_frames=9,
        new_obj_ids=np.array([40, 41], dtype=np.int64),
        new_obj_masks=mx.array(
            [
                [[2.0, -2.0], [-2.0, -2.0]],
                [[-2.0, -2.0], [2.0, -2.0]],
            ],
            dtype=mx.float32,
        ),
        tracker_states_local=tracker_states,
        orig_vid_height=6,
        orig_vid_width=7,
        feature_cache=feature_cache,
    )

    assert returned_states is tracker_states
    assert len(tracker_states) == 2
    assert len(tracker.init_calls) == 1
    assert tracker.init_calls[0]["cached_features"] is feature_cache
    assert tracker.init_calls[0]["video_height"] == 6
    assert tracker.init_calls[0]["video_width"] == 7
    assert tracker.init_calls[0]["num_frames"] == 9
    new_state = tracker_states[1]
    assert new_state["backbone_out"] is cached_backbone

    assert len(tracker.add_calls) == 1
    add_call = tracker.add_calls[0]
    assert add_call["state"] is new_state
    assert add_call["obj_ids"] == [40, 41]
    assert add_call["add_mask_to_memory"] is True
    np.testing.assert_array_equal(
        np.asarray(add_call["masks"]),
        np.array(
            [
                [[True, False], [False, False]],
                [[False, False], [True, False]],
            ],
            dtype=bool,
        ),
    )
    assert new_state["obj_ids"] == [40, 41]
    assert new_state["multiplex_state"].object_ids == [40, 41]
    assert new_state["multiplex_state"].available_slots == 1
    assert full_state["obj_ids"] == [10]
    assert tracker.preflight_calls == [(new_state, True)]


def test_sam3_multiplex_base_tracker_add_matches_official_fixture():
    fixture = json.loads(TRACKER_ADD_PARITY_FIXTURE.read_text())
    assert fixture["official_commit"] == OFFICIAL_SAM3_MULTIPLEX_BASE_COMMIT
    assert fixture["case"] == "dynamic_state_selection_and_mask_preprocess"
    assert fixture["component"] == "Sam3MultiplexBase._tracker_add_new_objects"
    assert fixture["mlx_runtime_guardrails"] == {
        "cached_features_attached_to_selected_state": True,
        "empty_current_output_seeded_for_new_packed_state": True,
    }
    for case_payload in fixture["cases"].values():
        assert all(case_payload["scalar_matches"].values())
        for metric in case_payload["metrics"].values():
            assert metric["max_abs"] <= fixture["atol"]

    def run_case(case_name, tracker_states, obj_ids):
        tracker = _DynamicAddTracker()
        tracker.input_mask_size = 4
        base = Sam3MultiplexBase(
            tracker=tracker,
            detector=_DummyDetector(),
            is_multiplex=True,
        )
        returned_states = base._tracker_add_new_objects(
            frame_idx=4,
            num_frames=8,
            new_obj_ids=mx.array(obj_ids, dtype=mx.int64),
            new_obj_masks=mx.array(
                [
                    [[2.0, -2.0], [-2.0, 2.0]],
                    [[-2.0, 2.0], [0.0, -3.0]],
                ],
                dtype=mx.float32,
            ),
            tracker_states_local=tracker_states,
            orig_vid_height=6,
            orig_vid_width=7,
            feature_cache={"frame": "features"},
        )
        expected = fixture["cases"][case_name]["official"]
        assert [state["name"] for state in returned_states] == expected[
            "returned_state_names"
        ]
        assert [
            {
                "state_name": call["state"]["name"],
                "frame_idx": call["frame_idx"],
                "obj_ids": call["obj_ids"],
                "add_mask_to_memory": call["add_mask_to_memory"],
            }
            for call in tracker.add_calls
        ] == [
            {k: v for k, v in call.items() if k != "masks"}
            for call in expected["add_calls"]
        ]
        assert [
            {"state_name": state["name"], "run_mem_encoder": run_mem_encoder}
            for state, run_mem_encoder in tracker.preflight_calls
        ] == expected["preflight_calls"]
        assert [
            {
                "state_name": call["state"]["name"],
                "video_height": call["video_height"],
                "video_width": call["video_width"],
                "num_frames": call["num_frames"],
            }
            for call in tracker.init_calls
        ] == expected["init_calls"]

        for add_call, expected_call in zip(
            tracker.add_calls,
            expected["add_calls"],
            strict=True,
        ):
            np.testing.assert_array_equal(
                np.asarray(add_call["masks"]),
                np.asarray(expected_call["masks"]["values"], dtype=bool),
            )

        observed_contract = [
            {
                "name": state["name"],
                "obj_ids": state["obj_ids"],
                "object_ids": state["multiplex_state"].object_ids,
                "available_slots": state["multiplex_state"].available_slots,
                "backbone_out": state.get("backbone_out"),
                "tracking_has_started": bool(state.get("tracking_has_started", False)),
            }
            for state in tracker_states
        ]
        expected_contract = [
            {
                k: v
                for k, v in state.items()
                if k
                not in {
                    "has_cached_features",
                    "empty_output_seeded",
                }
            }
            for state in expected["states"]
        ]
        assert observed_contract == expected_contract
        return tracker_states

    reuse_states = [
        _dynamic_tracker_state(
            "loose",
            [[0, -1, -1], [1, -1, -1]],
            [100, 101],
            allowed_bucket_capacity=3,
        ),
        _dynamic_tracker_state(
            "tight",
            [[0, -1, -1]],
            [200],
            allowed_bucket_capacity=3,
        ),
    ]
    run_case("reuse_tightest_dynamic_state", reuse_states, [300, 301])
    assert reuse_states[1]["cached_features"] == {"frame": "features"}

    create_states = [
        _dynamic_tracker_state(
            "full",
            [[0, -1]],
            [100],
            allowed_bucket_capacity=1,
        )
    ]
    create_states[0]["backbone_out"] = "cached-backbone"
    returned_create_states = run_case(
        "create_dynamic_state_when_full",
        create_states,
        [300, 301],
    )
    created_state = returned_create_states[1]
    assert created_state["cached_features"] == {"frame": "features"}
    assert 4 in created_state["output_dict"]["cond_frame_outputs"]


def test_sam3_multiplex_base_detector_frame_matches_official_fixture():
    fixture = json.loads(DETECTOR_FRAME_PARITY_FIXTURE.read_text())
    assert fixture["official_commit"] == OFFICIAL_SAM3_MULTIPLEX_BASE_COMMIT
    assert fixture["case"] == "detector_frame_startup_and_existing_state"
    assert fixture["component"] == "Sam3MultiplexBase._det_track_one_frame"

    assert set(fixture["cases"]) == {
        "startup_frame",
        "two_frame_detector_tracker_loop",
    }
    for case_payload in fixture["cases"].values():
        assert all(case_payload["scalar_matches"].values())
        for metric in case_payload["metrics"].values():
            assert metric["max_abs"] <= fixture["atol"]

    startup = fixture["cases"]["startup_frame"]
    assert startup["metrics"]["startup_frame.add_calls.0.masks"]["max_abs"] == 0.0
    assert startup["mlx"]["calls"]["add_calls"][0]["obj_ids"] == [0, 1]
    assert startup["mlx"]["frames"][0]["obj_id_to_mask"]["0"]["area"] == 1
    assert startup["mlx"]["frames"][0]["obj_id_to_mask"]["1"]["area"] == 1

    two_frame = fixture["cases"]["two_frame_detector_tracker_loop"]
    assert [event["event"] for event in two_frame["mlx"]["event_log"]] == [
        "forward_text",
        "detector",
        "init_state",
        "add_new_masks",
        "preflight",
        "detector",
        "propagate",
        "suppress_masks",
        "memory_encoder",
        "add_output_per_object",
    ]
    assert two_frame["mlx"]["calls"]["propagate_calls"][0]["run_mem_encoder"] is False
    np.testing.assert_allclose(
        np.array(
            two_frame["mlx"]["frames"][1]["tracker_obj_scores_global"]["values"],
            dtype=np.float32,
        ),
        np.full((2,), 0.7310586, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_sam3_multiplex_base_non_dynamic_add_creates_new_state_without_per_obj():
    tracker = _NonDynamicAddTracker(per_obj_inference=False)
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    cached_backbone = object()
    existing_state = {
        "name": "existing",
        "obj_ids": [10],
        "backbone_out": cached_backbone,
    }
    tracker_states = [existing_state]
    feature_cache = {"frame": "features"}

    returned_states = base._tracker_add_new_objects(
        frame_idx=6,
        num_frames=9,
        new_obj_ids=np.array([20], dtype=np.int64),
        new_obj_masks=mx.array([[[2.0, -2.0], [-2.0, -2.0]]], dtype=mx.float32),
        tracker_states_local=tracker_states,
        orig_vid_height=5,
        orig_vid_width=6,
        feature_cache=feature_cache,
    )

    assert returned_states is tracker_states
    assert len(tracker_states) == 2
    new_state = tracker_states[1]
    assert new_state is tracker.init_calls[0]["state"]
    assert tracker.init_calls[0]["cached_features"] is feature_cache
    assert new_state["backbone_out"] is cached_backbone
    assert existing_state["obj_ids"] == [10]
    assert new_state["obj_ids"] == [20]
    assert tracker.add_calls[0]["state"] is new_state
    assert tracker.add_calls[0]["obj_id"] == 20
    assert tracker.add_calls[0]["add_mask_to_memory"] is True
    assert tracker.preflight_calls == [(new_state, True)]


def test_sam3_multiplex_base_non_dynamic_add_reuses_state_with_per_obj():
    tracker = _NonDynamicAddTracker(per_obj_inference=True)
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    existing_state = {
        "name": "existing",
        "obj_ids": [10],
    }
    tracker_states = [existing_state]

    returned_states = base._tracker_add_new_objects(
        frame_idx=6,
        num_frames=9,
        new_obj_ids=np.array([20], dtype=np.int64),
        new_obj_masks=mx.array([[[2.0, -2.0], [-2.0, -2.0]]], dtype=mx.float32),
        tracker_states_local=tracker_states,
        orig_vid_height=5,
        orig_vid_width=6,
        feature_cache={"frame": "features"},
    )

    assert returned_states is tracker_states
    assert tracker_states == [existing_state]
    assert tracker.init_calls == []
    assert existing_state["obj_ids"] == [10, 20]
    assert tracker.add_calls[0]["state"] is existing_state
    assert tracker.preflight_calls == [(existing_state, True)]


def test_sam3_multiplex_base_execution_adds_detector_masks_to_packed_state():
    tracker = _PackedAddAdapterTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    multiplex_state = MultiplexState(
        [[0, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
        object_ids=[10],
    )
    prev_output = {
        "conditioning_objects": {0},
        "pred_masks": mx.ones((1, 1, 2, 2), dtype=mx.float32) * 10.0,
        "pred_masks_high_res": mx.ones((1, 1, 4, 4), dtype=mx.float32) * 11.0,
        "object_score_logits": mx.array([[1.0]], dtype=mx.float32),
    }
    interactive_vision_feats = [
        mx.arange(4, dtype=mx.float32).reshape(4, 1, 1),
        mx.ones((1, 1, 1), dtype=mx.float32),
    ]
    propagation_vision_feats = [mx.array([[3.0]], dtype=mx.float32)]
    tracker_state = {
        "obj_ids": [10],
        "multiplex_state": multiplex_state,
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {3: prev_output},
        },
        "backbone_out": {
            "interactive": {
                "vision_feats": interactive_vision_feats,
                "feat_sizes": [(2, 2), (1, 1)],
            },
            "sam2_backbone_out": {
                "vision_feats": propagation_vision_feats,
                "feat_sizes": [(2, 2)],
            },
        },
    }

    returned_states = base.run_tracker_update_execution_phase(
        frame_idx=3,
        num_frames=5,
        reverse=False,
        det_out={
            "mask": mx.array(
                [
                    [[2.0, -2.0], [-2.0, 2.0]],
                ],
                dtype=mx.float32,
            )
        },
        tracker_states_local=[tracker_state],
        tracker_update_plan={
            "new_det_fa_inds": np.array([0], dtype=np.int64),
            "new_det_obj_ids": np.array([20], dtype=np.int64),
            "new_det_gpu_ids": np.array([0], dtype=np.int64),
            "obj_ids_newly_removed": set(),
            "tracker_low_res_masks_global": None,
        },
        orig_vid_height=4,
        orig_vid_width=4,
        feature_cache={},
    )

    assert returned_states == [tracker_state]
    assert tracker_state["tracking_has_started"] is True
    assert tracker_state["obj_ids"] == [10, 20]
    assert multiplex_state.assignments == [[0, 1]]
    assert multiplex_state.object_ids == [10, 20]
    assert tracker.pix_mem_calls[0]["vision_feats"] is interactive_vision_feats
    assert tracker.mask_calls[0]["objects_in_mask"] == [1]
    np.testing.assert_array_equal(
        np.asarray(tracker.mask_calls[0]["mask_inputs"]),
        np.array(
            [
                [
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(prev_output["pred_masks"]),
        np.concatenate(
            [
                np.ones((1, 1, 2, 2), dtype=np.float32) * 10.0,
                np.ones((1, 1, 2, 2), dtype=np.float32) * 7.0,
            ],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(prev_output["pred_masks_high_res"]),
        np.concatenate(
            [
                np.ones((1, 1, 4, 4), dtype=np.float32) * 11.0,
                np.ones((1, 1, 4, 4), dtype=np.float32) * 70.0,
            ],
            axis=0,
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(prev_output["object_score_logits"]),
        np.array([[1.0], [101.0]], dtype=np.float32),
    )
    assert prev_output["conditioning_objects"] == {0, 1}
    assert tracker.memory_calls[0]["current_vision_feats"] is propagation_vision_feats
    assert tracker.memory_calls[0]["conditioning_objects"] == {0, 1}
    np.testing.assert_array_equal(
        np.asarray(prev_output["maskmem_features"]),
        np.array([[9.0]], dtype=np.float32),
    )


def test_sam3_multiplex_base_full_detector_frame_adds_to_packed_cached_state():
    tracker = _PackedDetectorFrameTracker()
    detector = _PackedDetectorFrameDetector()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=detector,
        is_multiplex=True,
        score_threshold_detection=0.5,
        new_det_thresh=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
        max_num_objects=4,
    )
    multiplex_state = MultiplexState(
        [[0, -1]],
        dtype=mx.float32,
        allowed_bucket_capacity=2,
        object_ids=[10],
    )
    prev_output = {
        "conditioning_objects": {0},
        "pred_masks": mx.ones((1, 1, 2, 2), dtype=mx.float32) * 10.0,
        "pred_masks_high_res": mx.ones((1, 1, 4, 4), dtype=mx.float32) * 11.0,
        "object_score_logits": mx.array([[1.0]], dtype=mx.float32),
    }
    tracker_state = {
        "obj_ids": [10],
        "multiplex_state": multiplex_state,
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {3: prev_output},
        },
    }
    feature_cache = {}
    tracker_metadata_prev = {}
    input_batch = SimpleNamespace(
        img_batch=mx.zeros((5, 3, 4, 4), dtype=mx.float32),
        find_text_batch=["shoe"],
        find_inputs=["frame-0", "frame-1", "frame-2", "frame-3", "frame-4"],
    )

    (
        obj_id_to_mask,
        obj_id_to_score,
        tracker_states_new,
        tracker_metadata_new,
        frame_stats,
        tracker_obj_scores,
    ) = base._run_local_tracker_detector_frame(
        frame_idx=3,
        num_frames=5,
        reverse=False,
        input_batch=input_batch,
        geometric_prompt=None,
        tracker_states_local=[tracker_state],
        tracker_metadata_prev=tracker_metadata_prev,
        feature_cache=feature_cache,
        orig_vid_height=2,
        orig_vid_width=2,
    )

    assert tracker_states_new == [tracker_state]
    assert tracker_state["cached_features"] is feature_cache
    assert 3 in feature_cache
    assert tracker_state["backbone_out"] is feature_cache[3][1]
    assert tracker_state["obj_ids"] == [10, 11]
    assert multiplex_state.assignments == [[0, 1]]
    assert multiplex_state.object_ids == [10, 11]
    assert tracker.propagate_calls[0]["state"] is tracker_state
    assert tracker.propagate_calls[0]["run_mem_encoder"] is False
    assert tracker.pix_mem_calls[0]["feat_sizes"] == [(2, 2), (1, 1), (1, 1)]
    assert tracker.mask_calls[0]["objects_in_mask"] == [1]
    assert tracker.memory_calls[0]["multiplex_state"] is multiplex_state
    assert tracker.memory_calls[0]["conditioning_objects"] == {0, 1}
    assert set(obj_id_to_mask) == {10, 11}
    assert obj_id_to_mask[10].shape == (1, 2, 2)
    assert obj_id_to_mask[11].shape == (1, 2, 2)
    np.testing.assert_allclose(
        np.asarray(obj_id_to_score[11]),
        np.array(0.9241418, dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["obj_ids_all_gpu"],
        np.array([10, 11], dtype=np.int64),
    )
    assert tracker_metadata_new["max_obj_id"] == 11
    assert frame_stats == {"num_obj_tracked": 2, "num_obj_dropped": 0}
    np.testing.assert_allclose(
        np.asarray(tracker_obj_scores),
        np.array([0.7310586], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_sam3_multiplex_base_detector_startup_seeds_packed_dynamic_state():
    tracker = _PackedStartupTracker()
    detector = _PackedDetectorFrameDetector()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=detector,
        is_multiplex=True,
        score_threshold_detection=0.5,
        image_only_det_thresh=0.5,
        new_det_thresh=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
        max_num_objects=4,
    )
    feature_cache = {}
    tracker_metadata_prev = {}
    input_batch = SimpleNamespace(
        img_batch=mx.zeros((3, 3, 4, 4), dtype=mx.float32),
        find_text_batch=["shoe"],
        find_inputs=["frame-0", "frame-1", "frame-2"],
    )

    (
        obj_id_to_mask,
        obj_id_to_score,
        tracker_states_new,
        tracker_metadata_new,
        frame_stats,
        tracker_obj_scores,
    ) = base._det_track_one_frame(
        frame_idx=0,
        num_frames=3,
        reverse=False,
        input_batch=input_batch,
        geometric_prompt=None,
        tracker_states_local=[],
        tracker_metadata_prev=tracker_metadata_prev,
        feature_cache=feature_cache,
        orig_vid_height=2,
        orig_vid_width=2,
        is_image_only=False,
    )

    assert len(tracker_states_new) == 1
    startup_state = tracker_states_new[0]
    assert tracker.init_calls[0]["state"] is startup_state
    assert tracker.init_calls[0]["cached_features"] is feature_cache
    assert startup_state["cached_features"] is feature_cache
    assert 0 in feature_cache
    assert set(feature_cache[0][1]) == {"interactive", "sam2_backbone_out"}
    assert startup_state["backbone_out"] is feature_cache[0][1]
    assert startup_state["obj_ids"] == [0, 1]
    assert startup_state["tracking_has_started"] is True
    assert tracker.preflight_calls == [(startup_state, True)]
    multiplex_state = startup_state["multiplex_state"]
    assert multiplex_state.assignments == [[0, 1]]

    assert multiplex_state.object_ids == [0, 1]

    current_out = startup_state["output_dict"]["cond_frame_outputs"][0]
    assert current_out["conditioning_objects"] == {0, 1}
    assert current_out["pred_masks"].shape == (2, 1, 2, 2)
    assert current_out["pred_masks_high_res"].shape == (2, 1, 4, 4)
    assert current_out["object_score_logits"].shape == (2, 1)
    np.testing.assert_array_equal(
        np.asarray(current_out["maskmem_features"]),
        np.array([[9.0]], dtype=np.float32),
    )
    assert tracker.pix_mem_calls[0]["feat_sizes"] == [(2, 2), (1, 1), (1, 1)]
    assert tracker.mask_calls[0]["objects_in_mask"] == [0, 1]
    np.testing.assert_array_equal(
        np.asarray(tracker.mask_calls[0]["mask_inputs"]),
        np.array(
            [
                [
                    [
                        [1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ],
                [
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ]
                ],
            ],
            dtype=np.float32,
        ),
    )
    assert tracker.memory_calls[0]["conditioning_objects"] == {0, 1}
    assert tracker.memory_calls[0]["multiplex_state"] is multiplex_state

    assert set(obj_id_to_mask) == {0, 1}
    np.testing.assert_allclose(
        np.asarray(obj_id_to_score[0]),
        np.array(0.880797, dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(obj_id_to_score[1]),
        np.array(0.9241418, dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["obj_ids_all_gpu"],
        np.array([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["num_obj_per_gpu"],
        np.array([2], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["num_buc_per_gpu"],
        np.array([1], dtype=np.int64),
    )
    assert tracker_metadata_new["max_obj_id"] == 1
    assert frame_stats == {"num_obj_tracked": 2, "num_obj_dropped": 0}
    np.testing.assert_array_equal(
        np.asarray(tracker_obj_scores),
        np.zeros((0,), dtype=np.float32),
    )


def test_sam3_multiplex_base_initial_detection_materialization_cleans_sprinkles():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        image_only_det_thresh=0.5,
        fill_hole_area=1,
        sprinkle_removal_area=1,
    )

    obj_id_to_mask, obj_id_to_score, tracker_metadata, frame_stats, _ = (
        base._materialize_initial_detection_frame(
            frame_idx=0,
            pred_scores=mx.array([0.9], dtype=mx.float32),
            pred_masks=mx.array(
                [
                    [
                        [1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0],
                        [-1.0, -1.0, 1.0, -1.0],
                        [-1.0, -1.0, -1.0, -1.0],
                    ]
                ],
                dtype=mx.float32,
            ),
            tracker_metadata_prev={},
            orig_vid_height=4,
            orig_vid_width=4,
        )
    )

    assert sorted(obj_id_to_mask) == [0]
    np.testing.assert_array_equal(
        np.asarray(obj_id_to_mask[0]),
        np.zeros((1, 4, 4), dtype=bool),
    )
    assert obj_id_to_score == {0: pytest.approx(0.9)}
    np.testing.assert_array_equal(
        tracker_metadata["obj_ids_all_gpu"],
        np.array([0], dtype=np.int64),
    )
    assert frame_stats == {"num_obj_tracked": 1, "num_obj_dropped": 0}


def test_sam3_multiplex_base_two_frame_packed_detector_tracker_loop():
    class _SequentialPackedTracker(_PackedStartupTracker):
        def __init__(self):
            super().__init__()
            self.maskmem_backbone = SimpleNamespace(
                mask_downsampler=SimpleNamespace(interpol_size=(4, 4))
            )
            self.memory_update_calls = []
            self.add_output_calls = []

        def propagate_in_video(
            self,
            inference_state,
            *,
            start_frame_idx,
            max_frame_num_to_track,
            reverse,
            run_mem_encoder,
            propagate_preflight,
        ):
            self.propagate_calls.append(
                {
                    "state": inference_state,
                    "start_frame_idx": start_frame_idx,
                    "max_frame_num_to_track": max_frame_num_to_track,
                    "reverse": reverse,
                    "run_mem_encoder": run_mem_encoder,
                    "propagate_preflight": propagate_preflight,
                }
            )
            obj_ids = list(inference_state["obj_ids"])
            masks = mx.array(
                [
                    [[2.0, -2.0], [-2.0, -2.0]],
                    [[-2.0, -2.0], [-2.0, 2.0]],
                ],
                dtype=mx.float32,
            )[: len(obj_ids)]
            scores = mx.ones((len(obj_ids),), dtype=mx.float32)
            frame_out = {
                "conditioning_objects": set(range(len(obj_ids))),
                "pred_masks": masks[:, None, :, :],
                "pred_masks_high_res": mx.ones(
                    (len(obj_ids), 1, 4, 4),
                    dtype=mx.float32,
                ),
                "object_score_logits": scores[:, None],
            }
            inference_state["output_dict"]["non_cond_frame_outputs"][
                start_frame_idx
            ] = frame_out
            yield (
                start_frame_idx,
                obj_ids,
                masks,
                None,
                scores,
            )

        def _suppress_object_pw_area_shrinkage(self, high_res_masks):
            return high_res_masks

        def _run_memory_encoder(
            self,
            sam2_state,
            frame_idx,
            local_batch_size,
            local_high_res_masks,
            local_object_score_logits,
            is_mask_from_pts,
        ):
            self.memory_update_calls.append(
                {
                    "state": sam2_state,
                    "frame_idx": frame_idx,
                    "local_batch_size": local_batch_size,
                    "high_res_masks": local_high_res_masks,
                    "score_logits": local_object_score_logits,
                    "is_mask_from_pts": is_mask_from_pts,
                }
            )
            return (
                mx.full((local_batch_size, 2), 42.0, dtype=mx.float32),
                [mx.full((local_batch_size, 1), 43.0, dtype=mx.float32)],
                mx.full((local_batch_size, 3), 44.0, dtype=mx.float32),
                mx.full((local_batch_size, 4), 45.0, dtype=mx.float32),
            )

        def add_output_per_object(
            self,
            *,
            inference_state,
            frame_idx,
            current_out,
            storage_key,
        ):
            self.add_output_calls.append(
                {
                    "state": inference_state,
                    "frame_idx": frame_idx,
                    "current_out": current_out,
                    "storage_key": storage_key,
                }
            )

    tracker = _SequentialPackedTracker()
    detector = _PackedDetectorFrameDetector()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=detector,
        is_multiplex=True,
        score_threshold_detection=0.5,
        image_only_det_thresh=0.5,
        new_det_thresh=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
        max_num_objects=4,
    )
    feature_cache = {}
    tracker_metadata_prev = {}
    input_batch = SimpleNamespace(
        img_batch=mx.zeros((3, 3, 4, 4), dtype=mx.float32),
        find_text_batch=["shoe"],
        find_inputs=["frame-0", "frame-1", "frame-2"],
    )

    (
        _frame0_masks,
        _frame0_scores,
        tracker_states,
        tracker_metadata,
        frame0_stats,
        _frame0_tracker_scores,
    ) = base._det_track_one_frame(
        frame_idx=0,
        num_frames=3,
        reverse=False,
        input_batch=input_batch,
        geometric_prompt=None,
        tracker_states_local=[],
        tracker_metadata_prev=tracker_metadata_prev,
        feature_cache=feature_cache,
        orig_vid_height=2,
        orig_vid_width=2,
        is_image_only=False,
    )
    assert frame0_stats == {"num_obj_tracked": 2, "num_obj_dropped": 0}
    assert len(tracker_states) == 1
    assert tracker_states[0]["obj_ids"] == [0, 1]

    (
        frame1_masks,
        frame1_scores,
        tracker_states_next,
        tracker_metadata_next,
        frame1_stats,
        frame1_tracker_scores,
    ) = base._det_track_one_frame(
        frame_idx=1,
        num_frames=3,
        reverse=False,
        input_batch=input_batch,
        geometric_prompt=None,
        tracker_states_local=tracker_states,
        tracker_metadata_prev=tracker_metadata,
        feature_cache=feature_cache,
        orig_vid_height=2,
        orig_vid_width=2,
        is_image_only=False,
    )

    assert tracker_states_next == tracker_states
    assert tracker_states_next[0]["obj_ids"] == [0, 1]
    assert tracker_states_next[0]["multiplex_state"].object_ids == [0, 1]
    assert set(frame1_masks) == {0, 1}
    assert set(frame1_scores) == {0, 1}
    assert frame1_stats == {"num_obj_tracked": 2, "num_obj_dropped": 0}
    np.testing.assert_array_equal(
        tracker_metadata_next["obj_ids_all_gpu"],
        np.array([0, 1], dtype=np.int64),
    )
    assert tracker_states_next[0]["cached_features"] is feature_cache
    assert "backbone_out" in tracker_states_next[0]
    assert tracker.propagate_calls[-1]["state"] is tracker_states_next[0]
    assert tracker.propagate_calls[-1]["start_frame_idx"] == 1
    assert len(tracker.memory_update_calls) == 1
    memory_call = tracker.memory_update_calls[0]
    assert memory_call["frame_idx"] == 1
    assert memory_call["local_batch_size"] == 2
    np.testing.assert_array_equal(
        np.asarray(memory_call["score_logits"]),
        np.array([[10.0], [10.0]], dtype=np.float32),
    )
    frame1_out = tracker_states_next[0]["output_dict"]["non_cond_frame_outputs"][1]
    np.testing.assert_array_equal(
        np.asarray(frame1_out["maskmem_features"]),
        np.full((2, 2), 42.0, dtype=np.float32),
    )
    assert tracker.add_output_calls[-1]["current_out"] is frame1_out
    np.testing.assert_allclose(
        np.asarray(frame1_tracker_scores),
        np.full((2,), 0.7310586, dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_sam3_multiplex_base_reconditions_high_confidence_masklet():
    tracker = _ReconditionTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_state = {"obj_ids": [10], "obj_idx_to_id": {0: 10}}
    tracker_low_res_masks = mx.array(
        [[[2.0, -1.0], [-1.0, -1.0]]],
        dtype=mx.float32,
    )
    det_out = {
        "mask": mx.array(
            [
                [[-1.0, -1.0], [-1.0, -1.0]],
                [[-1.0, 2.0], [-1.0, -1.0]],
            ],
            dtype=mx.float32,
        )
    }
    tracker_metadata = base._initialize_metadata()
    tracker_metadata["obj_ids_all_gpu"] = np.array([10], dtype=np.int64)

    tracker_states, reconditioned_obj_ids, updated_masks = base._recondition_masklets(
        frame_idx=2,
        det_out=det_out,
        trk_id_to_max_iou_high_conf_det={10: 1},
        tracker_states_local=[tracker_state],
        tracker_metadata=tracker_metadata,
        tracker_obj_scores_global=mx.array([2.0], dtype=mx.float32),
        tracker_low_res_masks_global=tracker_low_res_masks,
    )

    assert tracker_states == [tracker_state]
    assert reconditioned_obj_ids == {10}
    assert len(tracker.add_calls) == 1
    add_call = tracker.add_calls[0]
    assert add_call["state"] is tracker_state
    assert add_call["frame_idx"] == 2
    assert add_call["obj_ids"] == [10]
    assert add_call["reconditioning"] is True
    assert tracker.preflight_calls == [(tracker_state, True)]
    np.testing.assert_array_equal(
        np.asarray(add_call["masks"]),
        np.array([[[False, True], [False, False]]], dtype=bool),
    )
    np.testing.assert_array_equal(
        np.asarray(updated_masks),
        np.array([[[-1.0, 2.0], [-1.0, -1.0]]], dtype=np.float32),
    )


def test_sam3_multiplex_base_reconditioning_tracks_only_updated_object_ids():
    tracker = _ReconditionTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_state = {"obj_ids": [10, 11], "obj_idx_to_id": {0: 10, 1: 11}}
    tracker_metadata = base._initialize_metadata()
    tracker_metadata["obj_ids_all_gpu"] = np.array([10, 11], dtype=np.int64)

    _, reconditioned_obj_ids, updated_masks = base._recondition_masklets(
        frame_idx=2,
        det_out={
            "mask": mx.array(
                [[[1.0, -1.0], [-1.0, -1.0]]],
                dtype=mx.float32,
            )
        },
        trk_id_to_max_iou_high_conf_det={10: 0},
        tracker_states_local=[tracker_state],
        tracker_metadata=tracker_metadata,
        tracker_obj_scores_global=mx.array([2.0, 2.0], dtype=mx.float32),
        tracker_low_res_masks_global=mx.array(
            [
                [[-1.0, 1.0], [-1.0, -1.0]],
                [[-1.0, -1.0], [1.0, -1.0]],
            ],
            dtype=mx.float32,
        ),
    )

    assert reconditioned_obj_ids == {10}
    assert len(tracker.add_calls) == 1
    assert tracker.add_calls[0]["obj_ids"] == [10]
    np.testing.assert_array_equal(
        np.asarray(updated_masks),
        np.array(
            [
                [[1.0, -1.0], [-1.0, -1.0]],
                [[-1.0, -1.0], [1.0, -1.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_sam3_multiplex_base_reconditioning_supports_single_mask_helper():
    tracker = _SingleMaskReconditionTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_state = {"obj_ids": [10]}
    tracker_metadata = base._initialize_metadata()
    tracker_metadata["obj_ids_all_gpu"] = np.array([10], dtype=np.int64)

    _, reconditioned_obj_ids, _ = base._recondition_masklets(
        frame_idx=2,
        det_out={
            "mask": mx.array(
                [[[1.0, -1.0], [-1.0, -1.0]]],
                dtype=mx.float32,
            )
        },
        trk_id_to_max_iou_high_conf_det={10: 0},
        tracker_states_local=[tracker_state],
        tracker_metadata=tracker_metadata,
        tracker_obj_scores_global=mx.array([2.0], dtype=mx.float32),
        tracker_low_res_masks_global=mx.array(
            [[[1.0, -1.0], [-1.0, -1.0]]],
            dtype=mx.float32,
        ),
    )

    assert reconditioned_obj_ids == {10}
    assert len(tracker.add_calls) == 1
    assert tracker.add_calls[0]["obj_id"] == 10
    np.testing.assert_array_equal(
        np.asarray(tracker.add_calls[0]["mask"]),
        np.array([[True, False], [False, False]], dtype=bool),
    )


def test_sam3_multiplex_base_periodic_planning_reconditions_masklet():
    tracker = _ReconditionTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        recondition_every_nth_frame=1,
        new_det_thresh=0.95,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        iou_thresh_recondition=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_state = {"obj_ids": [10], "obj_idx_to_id": {0: 10}}
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_per_gpu"][0] = np.array([10], dtype=np.int64)
    tracker_metadata_prev["num_obj_per_gpu"][0] = 1
    tracker_metadata_prev["num_buc_per_gpu"][0] = 1
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10], dtype=np.int64)
    tracker_metadata_prev["max_obj_id"] = 10
    tracker_metadata_prev["rank0_metadata"]["obj_first_frame_idx"][10] = 0
    tracker_metadata_prev["rank0_metadata"]["trk_keep_alive"][10] = 1
    tracker_low_res_masks = mx.array(
        [[[2.0, -1.0], [-1.0, -1.0]]],
        dtype=mx.float32,
    )

    update_plan, tracker_metadata_new = base.run_tracker_update_planning_phase(
        frame_idx=2,
        num_frames=4,
        reverse=False,
        det_out={
            "mask": tracker_low_res_masks,
            "scores": mx.array([0.9], dtype=mx.float32),
        },
        det_keep=mx.array([True], dtype=mx.bool_),
        tracker_low_res_masks_global=tracker_low_res_masks,
        tracker_obj_scores_global=mx.array([2.0], dtype=mx.float32),
        tracker_metadata_prev=tracker_metadata_prev,
        tracker_states_local=[tracker_state],
    )

    assert update_plan["reconditioned_obj_ids"] == {10}
    np.testing.assert_array_equal(
        update_plan["new_det_fa_inds"],
        np.array([], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["obj_ids_all_gpu"],
        np.array([10], dtype=np.int64),
    )
    assert len(tracker.add_calls) == 1
    assert tracker.add_calls[0]["reconditioning"] is True
    assert tracker.preflight_calls == [(tracker_state, True)]


def test_sam3_multiplex_base_planning_suppresses_recent_overlap_masks():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.5,
        allow_unoccluded_to_suppress=True,
        new_det_thresh=0.95,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
    )
    tracker_state = {"obj_ids": [10, 11]}
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_per_gpu"][0] = np.array([10, 11], dtype=np.int64)
    tracker_metadata_prev["num_obj_per_gpu"][0] = 2
    tracker_metadata_prev["num_buc_per_gpu"][0] = 1
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10, 11], dtype=np.int64)
    tracker_metadata_prev["max_obj_id"] = 11
    tracker_metadata_prev["obj_id_to_last_occluded"] = {10: 4, 11: 1}
    tracker_metadata_prev["rank0_metadata"]["obj_first_frame_idx"].update(
        {10: 0, 11: 1}
    )
    tracker_metadata_prev["rank0_metadata"]["trk_keep_alive"].update({10: 1, 11: 1})
    tracker_low_res_masks = mx.array(
        [
            [[2.0, 2.0], [-1.0, -1.0]],
            [[2.0, 2.0], [-1.0, -1.0]],
        ],
        dtype=mx.float32,
    )

    update_plan, tracker_metadata_new = base.run_tracker_update_planning_phase(
        frame_idx=5,
        num_frames=8,
        reverse=False,
        det_out={
            "mask": mx.array([[[2.0, 2.0], [-1.0, -1.0]]], dtype=mx.float32),
            "scores": mx.array([0.9], dtype=mx.float32),
        },
        det_keep=mx.array([True], dtype=mx.bool_),
        tracker_low_res_masks_global=tracker_low_res_masks,
        tracker_obj_scores_global=mx.array([2.0, 2.0], dtype=mx.float32),
        tracker_metadata_prev=tracker_metadata_prev,
        tracker_states_local=[tracker_state],
    )

    np.testing.assert_array_equal(
        update_plan["new_det_fa_inds"],
        np.array([], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(update_plan["tracker_low_res_masks_global"]),
        np.array(
            [
                [[-10.0, -10.0], [-10.0, -10.0]],
                [[2.0, 2.0], [-1.0, -1.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert tracker_metadata_new["obj_id_to_last_occluded"] == {10: 5, 11: 1}
    gpu_metadata = tracker_metadata_new["gpu_metadata"]
    assert gpu_metadata["N_obj"] == 2
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["obj_first_frame"]),
        np.array([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["trk_keep_alive"]),
        np.array([0, 0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["overlap_pair_counts"]),
        np.zeros((2, 2), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.asarray(gpu_metadata["last_occluded_tensor"]),
        np.array([5, 1], dtype=np.int64),
    )


def test_sam3_multiplex_base_bbox_planning_reconditions_mismatched_masklet():
    tracker = _ReconditionTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        reconstruction_bbox_iou_thresh=0.5,
        reconstruction_bbox_det_score=0.8,
        new_det_thresh=0.95,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        iou_thresh_recondition=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_state = {"obj_ids": [10], "obj_idx_to_id": {0: 10}}
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_per_gpu"][0] = np.array([10], dtype=np.int64)
    tracker_metadata_prev["num_obj_per_gpu"][0] = 1
    tracker_metadata_prev["num_buc_per_gpu"][0] = 1
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10], dtype=np.int64)
    tracker_metadata_prev["max_obj_id"] = 10
    tracker_metadata_prev["rank0_metadata"]["obj_first_frame_idx"][10] = 0
    tracker_metadata_prev["rank0_metadata"]["trk_keep_alive"][10] = 1
    tracker_low_res_masks = mx.array(
        [
            [
                [2.0, 2.0, -1.0, -1.0],
                [2.0, 2.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ],
        dtype=mx.float32,
    )

    update_plan, tracker_metadata_new = base.run_tracker_update_planning_phase(
        frame_idx=2,
        num_frames=4,
        reverse=False,
        det_out={
            "mask": tracker_low_res_masks,
            "scores": mx.array([0.9], dtype=mx.float32),
            "bbox": mx.array([[0.75, 0.75, 1.0, 1.0]], dtype=mx.float32),
        },
        det_keep=mx.array([True], dtype=mx.bool_),
        tracker_low_res_masks_global=tracker_low_res_masks,
        tracker_obj_scores_global=mx.array([2.0], dtype=mx.float32),
        tracker_metadata_prev=tracker_metadata_prev,
        tracker_states_local=[tracker_state],
    )

    assert update_plan["reconditioned_obj_ids"] == {10}
    np.testing.assert_array_equal(
        update_plan["new_det_fa_inds"],
        np.array([], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        tracker_metadata_new["obj_ids_all_gpu"],
        np.array([10], dtype=np.int64),
    )
    assert len(tracker.add_calls) == 1
    assert tracker.add_calls[0]["reconditioning"] is True
    assert tracker.preflight_calls == [(tracker_state, True)]


def test_sam3_multiplex_base_bbox_planning_respects_detection_score_threshold():
    tracker = _ReconditionTracker()
    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
        reconstruction_bbox_iou_thresh=0.5,
        reconstruction_bbox_det_score=0.9,
        new_det_thresh=0.95,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        iou_thresh_recondition=0.5,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_state = {"obj_ids": [10], "obj_idx_to_id": {0: 10}}
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_per_gpu"][0] = np.array([10], dtype=np.int64)
    tracker_metadata_prev["num_obj_per_gpu"][0] = 1
    tracker_metadata_prev["num_buc_per_gpu"][0] = 1
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10], dtype=np.int64)
    tracker_metadata_prev["max_obj_id"] = 10
    tracker_metadata_prev["rank0_metadata"]["obj_first_frame_idx"][10] = 0
    tracker_metadata_prev["rank0_metadata"]["trk_keep_alive"][10] = 1
    tracker_low_res_masks = mx.array(
        [
            [
                [2.0, 2.0, -1.0, -1.0],
                [2.0, 2.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ]
        ],
        dtype=mx.float32,
    )

    update_plan, _ = base.run_tracker_update_planning_phase(
        frame_idx=2,
        num_frames=4,
        reverse=False,
        det_out={
            "mask": tracker_low_res_masks,
            "scores": mx.array([0.85], dtype=mx.float32),
            "bbox": mx.array([[0.75, 0.75, 1.0, 1.0]], dtype=mx.float32),
        },
        det_keep=mx.array([True], dtype=mx.bool_),
        tracker_low_res_masks_global=tracker_low_res_masks,
        tracker_obj_scores_global=mx.array([2.0], dtype=mx.float32),
        tracker_metadata_prev=tracker_metadata_prev,
        tracker_states_local=[tracker_state],
    )

    assert update_plan["reconditioned_obj_ids"] == set()
    assert tracker.add_calls == []
    assert tracker.preflight_calls == []


def test_sam3_multiplex_base_build_outputs_merges_tracker_and_detection_masks():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10, 11, 12], dtype=np.int64)
    tracker_metadata_prev["obj_ids_per_gpu"][0] = np.array([10, 12], dtype=np.int64)
    tracker_low_res_masks = mx.array(
        [[[1.0, -1.0], [-1.0, 1.0]]],
        dtype=mx.float32,
    )
    det_out = {
        "mask": mx.array(
            [
                [[-1.0, -1.0], [-1.0, -1.0]],
                [[-1.0, 2.0], [3.0, -1.0]],
            ],
            dtype=mx.float32,
        )
    }

    outputs = base.build_outputs(
        frame_idx=4,
        num_frames=8,
        reverse=False,
        det_out=det_out,
        tracker_low_res_masks_global=tracker_low_res_masks,
        tracker_obj_scores_global=mx.zeros((1,), dtype=mx.float32),
        tracker_metadata_prev=tracker_metadata_prev,
        sam2_update_plan={
            "new_det_fa_inds": np.array([1], dtype=np.int64),
            "new_det_obj_ids": np.array([13], dtype=np.int64),
        },
        orig_vid_height=2,
        orig_vid_width=2,
    )

    assert set(outputs) == {10, 12, 13}
    np.testing.assert_array_equal(
        np.asarray(outputs[10]),
        np.array([[[True, False], [False, True]]]),
    )
    np.testing.assert_array_equal(
        np.asarray(outputs[12]),
        np.array([[[False, False], [False, False]]]),
    )
    np.testing.assert_array_equal(
        np.asarray(outputs[13]),
        np.array([[[False, True], [True, False]]]),
    )


def test_sam3_multiplex_base_build_outputs_overrides_reconditioned_masks():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_metadata_prev = base._initialize_metadata()
    tracker_metadata_prev["obj_ids_all_gpu"] = np.array([10], dtype=np.int64)
    tracker_metadata_prev["obj_ids_per_gpu"][0] = np.array([10], dtype=np.int64)
    tracker_low_res_masks = mx.array(
        [[[1.0, -1.0], [-1.0, 1.0]]],
        dtype=mx.float32,
    )
    det_out = {
        "mask": mx.array(
            [
                [[-1.0, -1.0], [-1.0, -1.0]],
                [[-1.0, 2.0], [3.0, -1.0]],
            ],
            dtype=mx.float32,
        )
    }

    outputs = base.build_outputs(
        frame_idx=4,
        num_frames=8,
        reverse=False,
        det_out=det_out,
        tracker_low_res_masks_global=tracker_low_res_masks,
        tracker_obj_scores_global=mx.zeros((1,), dtype=mx.float32),
        tracker_metadata_prev=tracker_metadata_prev,
        sam2_update_plan={
            "new_det_fa_inds": np.array([], dtype=np.int64),
            "new_det_obj_ids": np.array([], dtype=np.int64),
            "trk_id_to_max_iou_high_conf_det": {10: 1},
        },
        orig_vid_height=2,
        orig_vid_width=2,
        reconditioned_obj_ids={10},
    )

    assert set(outputs) == {10}
    np.testing.assert_array_equal(
        np.asarray(outputs[10]),
        np.array([[[False, True], [True, False]]]),
    )


def test_sam3_multiplex_base_build_outputs_rejects_bad_update_plan():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        fill_hole_area=0,
        sprinkle_removal_area=0,
    )
    tracker_metadata_prev = base._initialize_metadata()

    with pytest.raises(ValueError, match="same length"):
        base.build_outputs(
            frame_idx=0,
            num_frames=1,
            reverse=False,
            det_out={"mask": mx.zeros((1, 2, 2), dtype=mx.float32)},
            tracker_low_res_masks_global=mx.zeros((0, 2, 2), dtype=mx.float32),
            tracker_obj_scores_global=mx.zeros((0,), dtype=mx.float32),
            tracker_metadata_prev=tracker_metadata_prev,
            sam2_update_plan={
                "new_det_fa_inds": np.array([0], dtype=np.int64),
                "new_det_obj_ids": np.array([], dtype=np.int64),
            },
            orig_vid_height=2,
            orig_vid_width=2,
        )

    with pytest.raises(ValueError, match="outside"):
        base.build_outputs(
            frame_idx=0,
            num_frames=1,
            reverse=False,
            det_out={"mask": mx.zeros((1, 2, 2), dtype=mx.float32)},
            tracker_low_res_masks_global=mx.zeros((0, 2, 2), dtype=mx.float32),
            tracker_obj_scores_global=mx.zeros((0,), dtype=mx.float32),
            tracker_metadata_prev=tracker_metadata_prev,
            sam2_update_plan={
                "new_det_fa_inds": np.array([2], dtype=np.int64),
                "new_det_obj_ids": np.array([1], dtype=np.int64),
            },
            orig_vid_height=2,
            orig_vid_width=2,
        )

    with pytest.raises(ValueError, match="outside"):
        base.build_outputs(
            frame_idx=0,
            num_frames=1,
            reverse=False,
            det_out={"mask": mx.zeros((1, 2, 2), dtype=mx.float32)},
            tracker_low_res_masks_global=mx.zeros((0, 2, 2), dtype=mx.float32),
            tracker_obj_scores_global=mx.zeros((0,), dtype=mx.float32),
            tracker_metadata_prev=tracker_metadata_prev,
            sam2_update_plan={
                "new_det_fa_inds": np.array([], dtype=np.int64),
                "new_det_obj_ids": np.array([], dtype=np.int64),
                "trk_id_to_max_iou_high_conf_det": {1: 2},
            },
            orig_vid_height=2,
            orig_vid_width=2,
            reconditioned_obj_ids={1},
        )


def test_sam3_multiplex_base_runs_backbone_detection_and_caches_tracker_features():
    detector = _FeatureDetector()
    base = Sam3MultiplexBase(
        tracker=_FeatureTracker(),
        detector=detector,
        is_multiplex=True,
        score_threshold_detection=0.5,
    )
    input_batch = SimpleNamespace(
        img_batch=mx.zeros((2, 3, 4, 4), dtype=mx.float32),
        find_text_batch=["shoe", "visual", "geometric"],
        find_inputs=["frame-0", "frame-1"],
    )
    feature_cache = {
        "tracking_bounds": {
            "max_frame_num_to_track": 2,
            "propagate_in_video_start_frame_idx": 0,
        }
    }

    det_out, det_keep = base.run_backbone_and_detection(
        frame_idx=0,
        num_frames=2,
        input_batch=input_batch,
        geometric_prompt="prompt",
        feature_cache=feature_cache,
        reverse=False,
    )

    assert detector.backbone.calls == [(("shoe", "visual", "geometric"), "mlx")]
    assert len(detector.calls) == 1
    detector_call = detector.calls[0]
    assert detector_call["find_inputs"] == ["frame-0", "frame-1"]
    assert detector_call["return_sam2_backbone_feats"] is True
    assert detector_call["max_frame_num_to_track"] == 2
    assert detector_call["propagate_in_video_start_frame_idx"] == 0
    np.testing.assert_allclose(
        np.asarray(det_out["scores"]),
        np.array([[0.880797, 0.119203]], dtype=np.float32),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(np.asarray(det_keep), np.array([[True, False]]))

    cached_image, cached_backbone = feature_cache[0]
    assert cached_image.shape == (1, 3, 4, 4)
    assert set(cached_backbone) == {"interactive", "sam2_backbone_out"}
    cached_interactive = cached_backbone["interactive"]
    cached_sam2 = cached_backbone["sam2_backbone_out"]
    np.testing.assert_array_equal(
        np.asarray(cached_sam2["backbone_fpn"][0]),
        np.ones((1, 2, 2, 2), dtype=np.float32) + 10,
    )
    np.testing.assert_array_equal(
        np.asarray(cached_sam2["backbone_fpn"][1]),
        np.ones((1, 2, 1, 1), dtype=np.float32) * 22,
    )
    np.testing.assert_array_equal(
        np.asarray(cached_sam2["backbone_fpn"][2]),
        np.ones((1, 2, 1, 1), dtype=np.float32) * 3,
    )
    assert cached_sam2["vision_features"] is cached_sam2["backbone_fpn"][-1]
    np.testing.assert_array_equal(
        np.asarray(cached_interactive["backbone_fpn"][0]),
        np.ones((1, 2, 2, 2), dtype=np.float32) * 17,
    )
    np.testing.assert_array_equal(
        np.asarray(cached_interactive["backbone_fpn"][1]),
        np.ones((1, 2, 1, 1), dtype=np.float32) * 28,
    )
    np.testing.assert_array_equal(
        np.asarray(cached_interactive["backbone_fpn"][2]),
        np.ones((1, 2, 1, 1), dtype=np.float32) * 9,
    )
    assert (
        cached_interactive["vision_features"] is cached_interactive["backbone_fpn"][-1]
    )


def test_sam3_video_base_suppresses_boxes_close_to_boundary_centers():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    boxes = mx.array(
        [
            [
                [0.00, 0.20, 0.05, 0.40],
                [0.03, 0.03, 0.05, 0.05],
                [0.95, 0.20, 1.00, 0.40],
            ],
            [
                [0.20, 0.00, 0.40, 0.05],
                [0.20, 0.95, 0.40, 1.00],
                [0.20, 0.20, 0.40, 0.40],
            ],
        ],
        dtype=mx.float32,
    )

    keep = base._suppress_detections_close_to_boundary(boxes)

    np.testing.assert_array_equal(
        np.asarray(keep),
        np.array(
            [
                [False, True, False],
                [False, False, True],
            ]
        ),
    )


def test_sam3_multiplex_base_boundary_suppression_filters_detector_keep_mask():
    class _BoundaryFeatureDetector(_FeatureDetector):
        def forward_video_grounding_multigpu(self, **kwargs):
            output, backbone_out = super().forward_video_grounding_multigpu(**kwargs)
            output["pred_logits"] = mx.ones((1, 4, 1), dtype=mx.float32) * 2
            output["pred_boxes_xyxy"] = mx.array(
                [
                    [
                        [0.00, 0.20, 0.05, 0.40],
                        [0.10, 0.10, 0.20, 0.20],
                        [0.95, 0.20, 1.00, 0.40],
                        [0.20, 0.95, 0.40, 1.00],
                    ]
                ],
                dtype=mx.float32,
            )
            output["pred_masks"] = mx.ones((1, 4, 2, 2), dtype=mx.float32)
            return output, backbone_out

    detector = _BoundaryFeatureDetector()
    base = Sam3MultiplexBase(
        tracker=_FeatureTracker(),
        detector=detector,
        is_multiplex=True,
        score_threshold_detection=0.5,
        suppress_det_close_to_boundary=True,
    )
    input_batch = SimpleNamespace(
        img_batch=mx.zeros((1, 3, 4, 4), dtype=mx.float32),
        find_text_batch=["shoe"],
        find_inputs=["frame-0"],
    )

    det_out, det_keep = base.run_backbone_and_detection(
        frame_idx=0,
        num_frames=1,
        input_batch=input_batch,
        geometric_prompt=None,
        feature_cache={},
        reverse=False,
    )

    np.testing.assert_array_equal(
        np.asarray(det_keep),
        np.array([[False, True, False, False]]),
    )
    assert det_out["bbox"].shape == (1, 4, 4)
    assert det_out["mask"].shape == (1, 4, 2, 2)


def test_sam3_multiplex_base_rejects_incomplete_sam2_backbone_cache():
    base = Sam3MultiplexBase(
        tracker=_FeatureTracker(),
        detector=_FeatureDetector(incomplete_backbone=True),
        is_multiplex=True,
    )
    input_batch = SimpleNamespace(
        img_batch=mx.zeros((1, 3, 4, 4), dtype=mx.float32),
        find_text_batch=["shoe"],
        find_inputs=["frame-0"],
    )

    with pytest.raises(ValueError, match="SAM2 backbone feature cache is incomplete"):
        base.run_backbone_and_detection(
            frame_idx=0,
            num_frames=1,
            input_batch=input_batch,
            geometric_prompt=None,
            feature_cache={},
            reverse=False,
        )


def test_sam3_multiplex_base_rejects_incomplete_interactive_backbone_cache():
    base = Sam3MultiplexBase(
        tracker=_FeatureTracker(),
        detector=_FeatureDetector(incomplete_interactive_backbone=True),
        is_multiplex=True,
    )
    input_batch = SimpleNamespace(
        img_batch=mx.zeros((1, 3, 4, 4), dtype=mx.float32),
        find_text_batch=["shoe"],
        find_inputs=["frame-0"],
    )

    with pytest.raises(
        ValueError, match="Interactive backbone feature cache is incomplete"
    ):
        base.run_backbone_and_detection(
            frame_idx=0,
            num_frames=1,
            input_batch=input_batch,
            geometric_prompt=None,
            feature_cache={},
            reverse=False,
        )


def test_sam3_multiplex_base_associates_detections_with_tracklets():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        max_num_objects=4,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.5,
    )
    det_masks = mx.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ],
        dtype=mx.float32,
    )
    det_scores = mx.array([0.9, 0.7, 0.4], dtype=mx.float32)
    det_keep = mx.array([True, True, True], dtype=mx.bool_)
    trk_masks = mx.array(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ],
        dtype=mx.float32,
    )
    trk_obj_ids = np.array([101, 102], dtype=np.int64)

    result = base._associate_det_trk(
        det_masks=det_masks,
        det_scores=det_scores,
        det_keep=det_keep,
        trk_masks=trk_masks,
        trk_obj_ids=trk_obj_ids,
    )
    realized = realize_adt_result(
        result,
        {"obj_ids_all_gpu": trk_obj_ids},
        det_masks,
    )

    np.testing.assert_array_equal(realized.new_det_fa_inds, np.array([1]))
    np.testing.assert_array_equal(
        realized.unmatched_trk_obj_ids, np.zeros(0, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        realized.empty_trk_obj_ids, np.zeros(0, dtype=np.int64)
    )
    np.testing.assert_array_equal(
        realized.det_to_matched_trk_obj_ids[0],
        np.array([101], dtype=np.int64),
    )


def test_sam3_multiplex_base_association_resizes_and_validates_inputs():
    base = Sam3MultiplexBase(
        tracker=_DummyTracker(),
        detector=_DummyDetector(),
        is_multiplex=True,
        max_num_objects=3,
    )
    det_masks = mx.ones((1, 4, 4), dtype=mx.float32)
    det_scores = mx.array([0.9], dtype=mx.float32)
    det_keep = mx.array([True], dtype=mx.bool_)
    trk_masks = mx.ones((1, 2, 2), dtype=mx.float32)

    result = base._associate_det_trk(
        det_masks=det_masks,
        det_scores=det_scores,
        det_keep=det_keep,
        trk_masks=trk_masks,
        trk_obj_ids=np.array([5], dtype=np.int64),
    )
    realized = realize_adt_result(
        result,
        {"obj_ids_all_gpu": np.array([5], dtype=np.int64)},
        det_masks,
    )
    np.testing.assert_array_equal(realized.new_det_fa_inds, np.zeros(0, dtype=np.int64))
    np.testing.assert_array_equal(
        realized.unmatched_trk_obj_ids, np.zeros(0, dtype=np.int64)
    )

    with pytest.raises(TypeError, match="det_masks must be floating"):
        base._associate_det_trk(
            det_masks=mx.ones((1, 2, 2), dtype=mx.int64),
            det_scores=det_scores,
            det_keep=det_keep,
            trk_masks=trk_masks,
            trk_obj_ids=np.array([5], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="same length"):
        base._associate_det_trk(
            det_masks=mx.ones((1, 2, 2), dtype=mx.float32),
            det_scores=det_scores,
            det_keep=det_keep,
            trk_masks=mx.ones((2, 2, 2), dtype=mx.float32),
            trk_obj_ids=np.array([5], dtype=np.int64),
        )


def test_sam3_multiplex_predictor_wrapper_delegates_mlx_child_module():
    class _WrappedModel(nn.Module):
        is_multiplex = True

        def __init__(self):
            super().__init__()
            self.multiplex_controller = _DummyController()
            self.calls = []

        def _add_output_per_object(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "stored"

    model = _WrappedModel()
    wrapper = Sam3MultiplexPredictorWrapper(model=model, per_obj_inference=False)

    assert wrapper.model is model
    assert wrapper.multiplex_controller is model.multiplex_controller
    assert wrapper.add_output_per_object(1, frame_idx=2) == "stored"
    assert model.calls == [((1,), {"frame_idx": 2})]


def test_sam3_multiplex_predictor_wrapper_skips_per_object_inference_storage():
    class _PerObjectModel(nn.Module):
        is_multiplex = True
        multiplex_controller = _DummyController()

        def __init__(self):
            super().__init__()
            self.calls = []

        def _add_output_per_object(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "should-not-run"

    model = _PerObjectModel()
    wrapper = Sam3MultiplexPredictorWrapper(model=model, per_obj_inference=True)

    assert wrapper.add_output_per_object("state", frame_idx=4) is None
    assert model.calls == []


def test_sam3_multiplex_predictor_wrapper_requires_batched_storage_helper():
    class _MissingStorageModel(nn.Module):
        is_multiplex = True
        multiplex_controller = _DummyController()

    wrapper = Sam3MultiplexPredictorWrapper(
        model=_MissingStorageModel(),
        per_obj_inference=False,
    )

    with pytest.raises(
        UnsupportedMultiplexRuntimeError,
        match="add_output_per_object",
    ):
        wrapper.add_output_per_object("state")


class _MemoryEncoderHarness:
    """Bare object to bind the real demo memory-encoder helpers onto, with the
    image-feature and SAM2 memory-encoder dependencies stubbed."""

    def __init__(self, vision_feats, vision_pos_embeds, feat_sizes, encoded):
        self._sam2_out = {
            "vision_feats": vision_feats,
            "vision_pos_embeds": vision_pos_embeds,
            "feat_sizes": feat_sizes,
        }
        self._encoded = encoded
        self.encode_calls: list[dict] = []
        self._get_maskmem_pos_enc = (
            VideoTrackingMultiplexDemo._get_maskmem_pos_enc.__get__(self)
        )

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        return "IMAGE", {"sam2_backbone_out": self._sam2_out}

    def _encode_new_memory(self, **kwargs):
        self.encode_calls.append(kwargs)
        return self._encoded


def test_run_memory_encoder_composes_official_4tuple_contract():
    """W1: VideoTrackingMultiplexDemo._run_memory_encoder composes
    _get_image_feature + _encode_new_memory + _get_maskmem_pos_enc into the
    official multiplex 4-tuple, forwards args, and resolves conditioning objects
    from the frame output entry. Lightweight stub harness, no model build."""
    vision_feats = [mx.ones((4, 2, 8)), mx.ones((4, 2, 16))]
    vision_pos_embeds = [mx.zeros((4, 2, 8)), mx.full((2, 256, 4, 4), 7.0)]
    feat_sizes = [(2, 2), (4, 4)]
    harness = _MemoryEncoderHarness(
        vision_feats,
        vision_pos_embeds,
        feat_sizes,
        encoded=(mx.full((2, 64), 5.0), [mx.full((2, 256, 4, 4), 9.0)]),
    )

    inference_state = {
        "constants": {},
        "multiplex_state": "MUX",
        "output_dict": {
            "cond_frame_outputs": {0: {"conditioning_objects": {1, 2}}},
            "non_cond_frame_outputs": {},
        },
    }
    high_res_masks = mx.zeros((2, 1, 1152, 1152))
    object_score_logits = mx.array([[1.0], [2.0]])

    run = VideoTrackingMultiplexDemo._run_memory_encoder.__get__(harness)
    result = run(inference_state, 0, 2, high_res_masks, object_score_logits, False)

    assert isinstance(result, tuple) and len(result) == 4
    maskmem_features, maskmem_pos_enc, image_features, image_pos_enc = result

    # args forwarded into _encode_new_memory
    forwarded = harness.encode_calls[0]
    assert forwarded["current_vision_feats"] is vision_feats
    assert forwarded["feat_sizes"] is feat_sizes
    assert forwarded["pred_masks_high_res"] is high_res_masks
    assert forwarded["object_score_logits"] is object_score_logits
    assert forwarded["is_mask_from_pts"] is False
    assert forwarded["multiplex_state"] == "MUX"
    # conditioning objects resolved from the frame's output entry
    assert forwarded["conditioning_objects"] == {1, 2}

    # image features/pos are the last-level propagation tensors, unchanged
    assert mx.array_equal(image_features, vision_feats[-1])
    assert mx.array_equal(image_pos_enc, vision_pos_embeds[-1])

    # maskmem_pos_enc caches the one-object slice and re-expands to batch size
    assert isinstance(maskmem_pos_enc, list)
    assert maskmem_pos_enc[0].shape == (2, 256, 4, 4)
    cached = inference_state["constants"]["maskmem_pos_enc"]
    assert cached[0].shape == (1, 256, 4, 4)


def test_run_memory_encoder_accepts_explicit_conditioning_objects():
    """W1: an explicit conditioning_objects argument bypasses the output_dict
    lookup and is forwarded as-is, alongside is_mask_from_pts."""
    harness = _MemoryEncoderHarness(
        vision_feats=[mx.ones((1, 1, 4))],
        vision_pos_embeds=[mx.zeros((1, 1, 4))],
        feat_sizes=[(2, 2)],
        encoded=(mx.zeros((1, 4)), [mx.zeros((1, 4, 2, 2))]),
    )
    inference_state = {
        "constants": {},
        "multiplex_state": "MUX",
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
    }
    run = VideoTrackingMultiplexDemo._run_memory_encoder.__get__(harness)
    run(
        inference_state,
        0,
        1,
        mx.zeros((1, 1, 1152, 1152)),
        mx.zeros((1, 1)),
        True,
        conditioning_objects={9},
    )
    assert harness.encode_calls[0]["conditioning_objects"] == {9}
    assert harness.encode_calls[0]["is_mask_from_pts"] is True


def test_run_memory_encoder_raises_when_conditioning_objects_missing():
    """W1: with no explicit conditioning_objects and no frame output entry, the
    encoder fails fast, matching the official ValueError."""
    harness = _MemoryEncoderHarness(
        vision_feats=[mx.ones((1, 1, 4))],
        vision_pos_embeds=[mx.zeros((1, 1, 4))],
        feat_sizes=[(2, 2)],
        encoded=(mx.zeros((1, 4)), [mx.zeros((1, 4, 2, 2))]),
    )
    inference_state = {
        "constants": {},
        "multiplex_state": "MUX",
        "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
    }
    run = VideoTrackingMultiplexDemo._run_memory_encoder.__get__(harness)
    with pytest.raises(ValueError, match="conditioning objects not found"):
        run(
            inference_state, 7, 1, mx.zeros((1, 1, 1152, 1152)), mx.zeros((1, 1)), False
        )


def test_can_update_tracker_memories_true_for_built_packed_tracker():
    """W1/W3: a real built packed tracker (demo wrapped in
    Sam3MultiplexPredictorWrapper) satisfies _can_update_tracker_memories, so
    Route A (_tracker_update_memories) is reachable in the packed loop. Guards
    against regressions that remove _run_memory_encoder (W1) or break the
    wrapper's add_output_per_object delegation (W2)."""
    from sam3_mlx.model_builder import build_sam3_multiplex_video_model

    demo = build_sam3_multiplex_video_model(
        load_from_HF=False,
        checkpoint_path=None,
        multiplex_count=4,
        use_fa3=False,
        use_rope_real=False,
        device="mlx",
    )
    tracker = Sam3MultiplexPredictorWrapper(
        model=demo,
        per_obj_inference=False,
        is_multiplex=True,
        is_multiplex_dynamic=True,
    )
    # _run_memory_encoder resolves through the wrapper __getattr__ to the demo (W1)
    assert getattr(tracker, "_run_memory_encoder", None) is not None
    # add_output_per_object is provided by the wrapper itself (W2, already wired)
    assert getattr(tracker, "add_output_per_object", None) is not None

    base = Sam3MultiplexBase(
        tracker=tracker,
        detector=_DummyDetector(),
        is_multiplex=True,
    )
    assert base._can_update_tracker_memories() is True
