import numpy as np
import mlx.core as mx
import pytest

from sam3_mlx.model.data_misc import FindStage
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.multiplex_utils import UnsupportedMultiplexRuntimeError
from sam3_mlx.model.sam3_multiplex_detector import (
    Sam3MultiplexDetector,
    Sam3MultiplexImageBase,
)


class _DummyDecoder:
    num_queries = 1
    num_o2m_queries = 0
    dac = False


class _DummyTransformer:
    d_model = 4
    decoder = _DummyDecoder()


class _ForwardingDetector(Sam3MultiplexDetector):
    def __init__(self, output):
        self.rank = 0
        self.world_size = 1
        self.output = output
        self.calls = []

    def forward_video_grounding(
        self,
        *,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt,
    ):
        self.calls.append(
            {
                "backbone_out": backbone_out,
                "find_input": find_input,
                "find_target": find_target,
                "geometric_prompt": geometric_prompt,
            }
        )
        return self.output.copy(), backbone_out


class _BatchedForwardingDetector(Sam3MultiplexDetector):
    def __init__(self, *, num_queries=2):
        self.rank = 0
        self.world_size = 1
        self.is_multiplex = True
        self.tracking_score_thresh = 0.25
        self.num_queries = num_queries
        self.calls = []

    def forward_grounding(
        self,
        *,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt,
    ):
        self.calls.append(
            {
                "img_ids": np.asarray(find_input.img_ids),
                "text_ids": np.asarray(find_input.text_ids),
                "find_target": find_target,
                "box_batch": geometric_prompt.box_mask.shape[0],
            }
        )
        img_ids = find_input.img_ids.astype(mx.float32)
        batch_size = img_ids.shape[0]
        query_offsets = mx.arange(self.num_queries, dtype=mx.float32)[None, :]
        pred_logits = (img_ids[:, None] + query_offsets + 1.0)[..., None]
        pred_boxes = mx.zeros((batch_size, self.num_queries, 4), dtype=mx.float32)
        pred_masks = mx.ones((batch_size, self.num_queries, 2, 2), dtype=mx.float32)
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_boxes_xyxy": pred_boxes,
            "pred_masks": pred_masks,
        }


class _BatchedNmsDetector(_BatchedForwardingDetector):
    def __init__(self):
        super().__init__(num_queries=3)

    def forward_grounding(
        self,
        *,
        backbone_out,
        find_input,
        find_target,
        geometric_prompt,
    ):
        self.calls.append({"img_ids": np.asarray(find_input.img_ids)})
        batch_size = find_input.img_ids.shape[0]
        logits = mx.array([2.0, 1.0, 1.5], dtype=mx.float32)
        masks = mx.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
            ],
            dtype=mx.float32,
        )
        pred_logits = mx.broadcast_to(logits[None, :, None], (batch_size, 3, 1))
        pred_masks = mx.broadcast_to(masks[None, :, :, :], (batch_size, 3, 2, 2))
        pred_boxes = mx.zeros((batch_size, 3, 4), dtype=mx.float32)
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_boxes_xyxy": pred_boxes,
            "pred_masks": pred_masks,
        }


def _find_stage(img_id: int, text_id: int) -> FindStage:
    return FindStage(
        img_ids=mx.array([img_id], dtype=mx.int64),
        text_ids=mx.array([text_id], dtype=mx.int64),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
        input_boxes_before_embed=None,
        input_points_before_embed=None,
    )


def _find_stage_with_empty_prompt(img_id: int, text_id: int) -> FindStage:
    stage = _find_stage(img_id, text_id)
    stage.input_boxes_before_embed = mx.zeros((0, 1, 4), dtype=mx.float32)
    stage.input_boxes_mask = mx.zeros((1, 0), dtype=mx.bool_)
    stage.input_boxes_label = mx.zeros((0, 1), dtype=mx.int64)
    stage.input_points_before_embed = mx.zeros((1, 0, 3), dtype=mx.float32)
    stage.input_points_mask = mx.zeros((1, 0), dtype=mx.bool_)
    return stage


def _blank_instance() -> Sam3MultiplexImageBase:
    instance = Sam3MultiplexImageBase.__new__(Sam3MultiplexImageBase)
    instance.tracking_score_thresh = 0.25
    return instance


def test_get_dummy_object_ids_uses_query_indices_above_tracking_threshold():
    instance = _blank_instance()
    logits = mx.array(
        [
            [[0.1], [0.3], [0.25]],
            [[0.9], [0.0], [0.4]],
        ],
        dtype=mx.float32,
    )

    object_ids = instance._get_dummy_object_ids(logits)

    np.testing.assert_array_equal(
        np.asarray(object_ids),
        np.array([[-1, 1, -1], [0, -1, 2]], dtype=np.int64),
    )


def test_batch_find_inputs_uses_chunk_frame_ids_and_modulo_source_fields():
    instance = _blank_instance()
    batched = instance._batch_find_inputs(
        [_find_stage(0, 10), _find_stage(1, 20)],
        chunk_start=1,
        chunk_end=4,
    )

    np.testing.assert_array_equal(np.asarray(batched.img_ids), np.array([1, 2, 3]))
    np.testing.assert_array_equal(batched.img_ids_np, np.array([1, 2, 3]))
    np.testing.assert_array_equal(np.asarray(batched.text_ids), np.array([20, 10, 20]))
    assert batched.input_boxes is None
    assert batched.input_points is None


def test_batch_geometric_prompts_preserves_official_prompt_batch_axes():
    instance = _blank_instance()
    prompts = [
        Prompt(
            box_embeddings=mx.ones((1, 1, 2), dtype=mx.float32),
            box_mask=mx.array([[False]]),
            box_labels=mx.array([[1]], dtype=mx.int64),
            point_embeddings=mx.ones((1, 1, 2), dtype=mx.float32) * 2,
            point_mask=mx.array([[False]]),
            point_labels=mx.array([[1]], dtype=mx.int64),
        ),
        Prompt(
            box_embeddings=mx.ones((1, 1, 2), dtype=mx.float32) * 3,
            box_mask=mx.array([[True]]),
            box_labels=mx.array([[0]], dtype=mx.int64),
            point_embeddings=mx.ones((1, 1, 2), dtype=mx.float32) * 4,
            point_mask=mx.array([[True]]),
            point_labels=mx.array([[0]], dtype=mx.int64),
        ),
    ]

    batched = instance._batch_geometric_prompts_from_list(prompts)

    assert batched.box_embeddings.shape == (1, 2, 2)
    assert batched.box_mask.shape == (2, 1)
    assert batched.box_labels.shape == (1, 2)
    assert batched.point_embeddings.shape == (1, 2, 2)
    assert batched.point_mask.shape == (2, 1)
    assert batched.point_labels.shape == (1, 2)
    np.testing.assert_array_equal(np.asarray(batched.box_labels), np.array([[1, 0]]))


def test_multiplex_detector_constructor_ports_rank_and_gather_state(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "1")

    detector = Sam3MultiplexDetector(
        backbone=object(),
        transformer=_DummyTransformer(),
        input_geometry_encoder=object(),
        use_dot_prod_scoring=False,
        async_all_gather=False,
        gather_backbone_out=False,
        is_multiplex=True,
    )
    tensor = mx.array([1.0, 2.0], dtype=mx.float32)

    gathered, handle = detector._gather_tensor(tensor)

    assert detector.rank == 2
    assert detector.world_size == 1
    assert detector.async_all_gather is False
    assert detector.gather_backbone_out is False
    assert detector.is_multiplex is True
    assert gathered == [tensor]
    assert handle is None


def test_multiplex_detector_gather_rejects_multi_process_runtime():
    detector = Sam3MultiplexDetector.__new__(Sam3MultiplexDetector)
    detector.world_size = 2
    detector.async_all_gather = True

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="_gather_tensor"):
        detector._gather_tensor(mx.array([1.0], dtype=mx.float32))


def test_forward_video_grounding_multigpu_delegates_in_single_process_runtime():
    output = {
        "pred_logits": mx.array([[[2.0], [1.0]]], dtype=mx.float32),
        "pred_masks": mx.ones((1, 2, 2, 2), dtype=mx.float32),
    }
    detector = _ForwardingDetector(output)
    prompt = Prompt(
        box_embeddings=mx.zeros((0, 1, 4), dtype=mx.float32),
        box_mask=mx.zeros((1, 0), dtype=mx.bool_),
        box_labels=mx.zeros((0, 1), dtype=mx.int64),
        point_embeddings=None,
        point_mask=None,
    )
    backbone_out = {"img_batch_all_stages": mx.zeros((2, 3, 4, 4), dtype=mx.float32)}
    find_inputs = [_find_stage(0, 10), _find_stage(1, 20)]

    out, returned_backbone = detector.forward_video_grounding_multigpu(
        backbone_out=backbone_out,
        find_inputs=find_inputs,
        geometric_prompt=prompt,
        frame_idx=1,
        num_frames=2,
        multigpu_buffer={},
    )

    assert returned_backbone is backbone_out
    assert detector.calls == [
        {
            "backbone_out": backbone_out,
            "find_input": find_inputs[1],
            "find_target": None,
            "geometric_prompt": prompt,
        }
    ]
    np.testing.assert_array_equal(
        np.asarray(out["pred_logits"]), np.asarray(output["pred_logits"])
    )


def test_forward_video_grounding_multigpu_reads_cached_frame_without_recompute():
    output = {
        "pred_logits": mx.array([[[2.0], [1.0]]], dtype=mx.float32),
        "pred_masks": mx.ones((1, 2, 2, 2), dtype=mx.float32),
    }
    detector = _ForwardingDetector(output)
    backbone_out = {"img_batch_all_stages": mx.zeros((1, 3, 4, 4), dtype=mx.float32)}
    multigpu_buffer = {
        0: {
            "pred_logits": (output["pred_logits"], None),
            "pred_masks": (output["pred_masks"], None),
        }
    }

    out, _ = detector.forward_video_grounding_multigpu(
        backbone_out=backbone_out,
        find_inputs=[_find_stage(0, 10)],
        geometric_prompt=None,
        frame_idx=0,
        num_frames=1,
        multigpu_buffer=multigpu_buffer,
    )

    assert detector.calls == []
    assert list(multigpu_buffer) == [0]
    np.testing.assert_array_equal(
        np.asarray(out["pred_logits"]), np.asarray(output["pred_logits"])
    )


def test_forward_video_grounding_multigpu_prefetches_and_cleans_buffer_slots():
    output = {
        "pred_logits": mx.array([[[2.0], [1.0]]], dtype=mx.float32),
        "pred_masks": mx.ones((1, 2, 2, 2), dtype=mx.float32),
    }
    detector = _ForwardingDetector(output)
    backbone_out = {"img_batch_all_stages": mx.zeros((2, 3, 4, 4), dtype=mx.float32)}
    find_inputs = [_find_stage(0, 10), _find_stage(1, 20)]
    multigpu_buffer = {}

    detector.forward_video_grounding_multigpu(
        backbone_out=backbone_out,
        find_inputs=find_inputs,
        geometric_prompt=None,
        frame_idx=0,
        num_frames=2,
        multigpu_buffer=multigpu_buffer,
    )
    detector.forward_video_grounding_multigpu(
        backbone_out=backbone_out,
        find_inputs=find_inputs,
        geometric_prompt=None,
        frame_idx=1,
        num_frames=2,
        multigpu_buffer=multigpu_buffer,
    )

    assert [call["find_input"] for call in detector.calls] == find_inputs
    assert list(multigpu_buffer) == [1]


def test_forward_video_grounding_multigpu_applies_mlx_nms_score_suppression():
    output = {
        "pred_logits": mx.array([[[2.0], [1.0], [1.5]]], dtype=mx.float32),
        "pred_masks": mx.array(
            [
                [
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 1.0], [0.0, 0.0]],
                ]
            ],
            dtype=mx.float32,
        ),
    }
    detector = _ForwardingDetector(output)
    prompt = Prompt(
        box_embeddings=mx.zeros((0, 1, 4), dtype=mx.float32),
        box_mask=mx.zeros((1, 0), dtype=mx.bool_),
        box_labels=mx.zeros((0, 1), dtype=mx.int64),
        point_embeddings=None,
        point_mask=None,
    )

    out, _ = detector.forward_video_grounding_multigpu(
        backbone_out={"img_batch_all_stages": mx.zeros((1, 3, 4, 4), dtype=mx.float32)},
        find_inputs=[_find_stage(0, 10)],
        geometric_prompt=prompt,
        frame_idx=0,
        num_frames=1,
        multigpu_buffer={},
        run_nms=True,
        nms_prob_thresh=0.0,
        nms_iou_thresh=0.9999995,
    )

    logits = np.asarray(out["pred_logits"])
    assert logits[0, 0, 0] == pytest.approx(2.0)
    assert logits[0, 1, 0] < -9000.0
    assert logits[0, 2, 0] == pytest.approx(1.5)


def test_forward_video_grounding_multigpu_rejects_distributed_runtime():
    output = {
        "pred_logits": mx.zeros((1, 1, 1), dtype=mx.float32),
        "pred_masks": mx.zeros((1, 1, 2, 2), dtype=mx.float32),
    }
    detector = _ForwardingDetector(output)
    detector.world_size = 2

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="distributed"):
        detector.forward_video_grounding_multigpu(
            backbone_out={},
            find_inputs=[_find_stage(0, 10)],
            geometric_prompt=None,
            frame_idx=0,
            num_frames=1,
            multigpu_buffer={},
        )


def test_forward_video_grounding_batched_multigpu_processes_and_caches_chunk():
    detector = _BatchedForwardingDetector()
    prompt = Prompt(
        box_embeddings=mx.zeros((0, 1, 4), dtype=mx.float32),
        box_mask=mx.zeros((1, 0), dtype=mx.bool_),
        box_labels=mx.zeros((0, 1), dtype=mx.int64),
        point_embeddings=None,
        point_mask=None,
    )
    backbone_out = {"img_batch_all_stages": mx.zeros((3, 3, 4, 4), dtype=mx.float32)}
    find_inputs = [
        _find_stage_with_empty_prompt(0, 10),
        _find_stage_with_empty_prompt(1, 20),
        _find_stage_with_empty_prompt(2, 30),
    ]
    grounding_cache = {}

    out, returned_backbone = detector.forward_video_grounding_batched_multigpu(
        backbone_out=backbone_out,
        find_inputs=find_inputs,
        geometric_prompt=prompt,
        frame_idx=1,
        num_frames=3,
        grounding_cache=grounding_cache,
        batch_size=2,
    )
    cached_out, _ = detector.forward_video_grounding_batched_multigpu(
        backbone_out=backbone_out,
        find_inputs=find_inputs,
        geometric_prompt=prompt,
        frame_idx=0,
        num_frames=3,
        grounding_cache=grounding_cache,
        batch_size=2,
    )

    assert returned_backbone is backbone_out
    assert len(detector.calls) == 1
    np.testing.assert_array_equal(detector.calls[0]["img_ids"], np.array([0, 1]))
    np.testing.assert_array_equal(detector.calls[0]["text_ids"], np.array([10, 20]))
    assert detector.calls[0]["find_target"] is None
    assert detector.calls[0]["box_batch"] == 2
    assert list(grounding_cache["grounding_buffer"]) == [(0, 2)]
    np.testing.assert_allclose(
        np.asarray(out["pred_logits"]),
        np.array([[[2.0], [3.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(cached_out["pred_logits"]),
        np.array([[[1.0], [2.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(out["pred_object_ids"]),
        np.array([[0, 1]], dtype=np.int64),
    )


def test_forward_video_grounding_batched_multigpu_cleans_previous_chunk():
    detector = _BatchedForwardingDetector()
    backbone_out = {"img_batch_all_stages": mx.zeros((3, 3, 4, 4), dtype=mx.float32)}
    find_inputs = [
        _find_stage_with_empty_prompt(0, 10),
        _find_stage_with_empty_prompt(1, 20),
        _find_stage_with_empty_prompt(2, 30),
    ]
    grounding_cache = {}

    detector.forward_video_grounding_batched_multigpu(
        backbone_out=backbone_out,
        find_inputs=find_inputs,
        geometric_prompt=None,
        frame_idx=0,
        num_frames=3,
        grounding_cache=grounding_cache,
        batch_size=2,
    )
    out, _ = detector.forward_video_grounding_batched_multigpu(
        backbone_out=backbone_out,
        find_inputs=find_inputs,
        geometric_prompt=None,
        frame_idx=2,
        num_frames=3,
        grounding_cache=grounding_cache,
        batch_size=2,
    )

    assert list(grounding_cache["grounding_buffer"]) == [(2, 3)]
    np.testing.assert_allclose(
        np.asarray(out["pred_logits"]),
        np.array([[[3.0], [4.0]]], dtype=np.float32),
    )


def test_forward_video_grounding_batched_multigpu_applies_batched_mlx_nms():
    detector = _BatchedNmsDetector()

    out, _ = detector.forward_video_grounding_batched_multigpu(
        backbone_out={"img_batch_all_stages": mx.zeros((2, 3, 4, 4), dtype=mx.float32)},
        find_inputs=[
            _find_stage_with_empty_prompt(0, 10),
            _find_stage_with_empty_prompt(1, 20),
        ],
        geometric_prompt=None,
        frame_idx=0,
        num_frames=2,
        grounding_cache={},
        batch_size=2,
        run_nms=True,
        nms_prob_thresh=0.0,
        nms_iou_thresh=0.9999995,
    )

    logits = np.asarray(out["pred_logits"])
    assert logits[0, 0, 0] == pytest.approx(2.0)
    assert logits[0, 1, 0] < -9000.0
    assert logits[0, 2, 0] == pytest.approx(1.5)


def test_forward_video_grounding_batched_multigpu_rejects_distributed_runtime():
    detector = _BatchedForwardingDetector()
    detector.world_size = 2

    with pytest.raises(UnsupportedMultiplexRuntimeError, match="distributed"):
        detector.forward_video_grounding_batched_multigpu(
            backbone_out={},
            find_inputs=[_find_stage(0, 10)],
            geometric_prompt=None,
            frame_idx=0,
            num_frames=1,
            grounding_cache={},
        )


def test_forward_video_grounding_batched_multigpu_rejects_bad_batch_size():
    detector = _BatchedForwardingDetector()

    with pytest.raises(ValueError, match="batch_size"):
        detector.forward_video_grounding_batched_multigpu(
            backbone_out={},
            find_inputs=[_find_stage(0, 10)],
            geometric_prompt=None,
            frame_idx=0,
            num_frames=1,
            grounding_cache={},
            batch_size=0,
        )
