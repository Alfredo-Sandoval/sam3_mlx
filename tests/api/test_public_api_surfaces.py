from __future__ import annotations

import tomllib

import numpy as np
import pytest

import sam3_mlx
from sam3_mlx import model_builder
from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.geometry_encoders import SequenceGeometryEncoder
from sam3_mlx.model.model_misc import DotProductScoring, TransformerWrapper
from sam3_mlx.model.necks import Sam3DualViTDetNeck
from sam3_mlx.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_multiplex_video_predictor import Sam3MultiplexVideoPredictor
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor
from sam3_mlx.model.text_encoder_ve import VETextEncoder
from sam3_mlx.model.vl_combiner import SAM3VLBackbone
from tests._paths import REPO_ROOT


class _FakeImageProcessor:
    def __init__(self, image_model, resolution=1008, confidence_threshold=0.5):
        self.image_model = image_model
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold

    def set_image(self, image):
        del image
        mask = np.zeros((1, 480, 640), dtype=bool)
        mask[:, 10:20, 30:45] = True
        return {
            "masks": mask,
            "boxes": np.array([[30.0, 10.0, 45.0, 20.0]], dtype=np.float32),
            "scores": np.array([0.91], dtype=np.float32),
        }

    def set_text_prompt(self, prompt, state):
        state = dict(state)
        state["text_prompt"] = prompt
        return state

    def add_geometric_prompt(self, box, label, state):
        state = dict(state)
        state["box_prompt"] = (box, label)
        return state

    def add_point_prompt(self, point, label, state):
        state = dict(state)
        state["point_prompt"] = (point, label)
        return state


class _WarmUpModel:
    def __init__(self):
        self.calls = []

    def warm_up_compilation(self):
        self.calls.append(self._warm_up_complete)


class _NoWarmHookModel:
    pass


def test_public_package_exports_image_video_and_multiplex_builders():
    expected_exports = {
        "build_tracker": model_builder.build_tracker,
        "build_sam3_image_model": model_builder.build_sam3_image_model,
        "build_sam3_multiplex_video_model": (
            model_builder.build_sam3_multiplex_video_model
        ),
        "build_sam3_multiplex_video_predictor": (
            model_builder.build_sam3_multiplex_video_predictor
        ),
        "build_sam3_predictor": model_builder.build_sam3_predictor,
        "build_sam3_video_model": model_builder.build_sam3_video_model,
        "build_sam3_video_predictor": model_builder.build_sam3_video_predictor,
        "download_ckpt_from_hf": model_builder.download_ckpt_from_hf,
    }
    expected_all = [
        "Sam3MlxUnsupportedError",
        *expected_exports,
    ]

    assert sam3_mlx.__all__ == expected_all
    assert sam3_mlx.Sam3MlxUnsupportedError is Sam3MlxUnsupportedError
    for name, expected in expected_exports.items():
        assert getattr(sam3_mlx, name) is expected


def test_distribution_excludes_unsupported_source_surfaces():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
    package_data = pyproject["tool"]["setuptools"]["package-data"]["sam3_mlx"]

    assert package_find["include"] == ["sam3_mlx", "sam3_mlx.*"]
    assert set(package_find["exclude"]) == {
        "sam3_mlx.agent",
        "sam3_mlx.agent.*",
        "sam3_mlx.eval",
        "sam3_mlx.eval.*",
        "sam3_mlx.train",
        "sam3_mlx.train.*",
        "sam3_mlx.perflib.triton",
        "sam3_mlx.perflib.triton.*",
    }
    assert package_data == ["assets/*.txt.gz"]


def test_image_builder_threads_checkpoint_free_architecture_toggles():
    model = sam3_mlx.build_sam3_image_model(
        load_from_HF=False,
        enable_segmentation=False,
        enable_inst_interactivity=True,
    )

    assert isinstance(model, Sam3Image)
    assert model.device == "mlx"
    assert model.training is False
    assert model.segmentation_head is None
    assert model.num_feature_levels == 1
    assert model.hidden_dim == 256
    assert model.use_dot_prod_scoring is True
    assert model.multimask_output is True
    assert isinstance(model.backbone, SAM3VLBackbone)
    assert isinstance(model.backbone.vision_backbone, Sam3DualViTDetNeck)
    assert isinstance(model.backbone.language_backbone, VETextEncoder)
    assert isinstance(model.geometry_encoder, SequenceGeometryEncoder)
    assert isinstance(model.transformer, TransformerWrapper)
    assert isinstance(model.dot_prod_scoring, DotProductScoring)
    assert isinstance(model.inst_interactive_predictor, SAM3InteractiveImagePredictor)


def test_video_predictor_request_api_uses_official_output_schema():
    predictor = sam3_mlx.build_sam3_predictor(
        version="sam3",
        model=object(),
        load_from_HF=False,
        resolution=16,
        processor_factory=_FakeImageProcessor,
    )

    assert isinstance(predictor, Sam3VideoPredictor)
    start = predictor.handle_request(
        {"type": "start_session", "resource_path": "<load-dummy-video-2>"}
    )
    add = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": start["session_id"],
            "frame_index": 0,
            "text": "object",
        }
    )
    stream = list(
        predictor.handle_stream_request(
            {"type": "propagate_in_video", "session_id": start["session_id"]}
        )
    )

    assert set(add["outputs"]) == {
        "out_binary_masks",
        "out_boxes_xywh",
        "out_obj_ids",
        "out_probs",
    }
    np.testing.assert_array_equal(add["outputs"]["out_obj_ids"], np.array([0]))
    np.testing.assert_allclose(
        add["outputs"]["out_probs"], np.array([0.91], dtype=np.float32)
    )
    assert add["outputs"]["out_binary_masks"].shape == (1, 480, 640)
    assert [(item["frame_index"], set(item["outputs"])) for item in stream] == [
        (0, set(add["outputs"])),
        (1, set(add["outputs"])),
    ]
    assert predictor.handle_request(
        {"type": "remove_object", "session_id": start["session_id"], "obj_id": 0}
    ) == {"is_success": True}
    assert predictor.handle_request(
        {"type": "close_session", "session_id": start["session_id"]}
    ) == {"is_success": True}


def test_multiplex_predictor_builder_constructs_checkpoint_free_mlx_stack():
    predictor = sam3_mlx.build_sam3_multiplex_video_predictor(
        load_from_HF=False,
        use_fa3=False,
        use_rope_real=False,
        max_num_objects=4,
        multiplex_count=4,
        async_loading_frames=False,
        session_expiration_sec=33,
        default_output_prob_thresh=0.25,
        warm_up=True,
    )

    assert isinstance(predictor, Sam3MultiplexVideoPredictor)
    assert predictor.warm_up is True
    assert predictor.async_loading_frames is False
    assert predictor.session_expiration_sec == 33
    assert predictor.default_output_prob_thresh == 0.25
    assert (
        predictor.model.__class__.__name__ == "Sam3MultiplexTrackingWithInteractivity"
    )
    assert predictor.model._warm_up_complete is True
    assert predictor.model.max_num_objects == 4
    assert predictor.model.bucket_capacity == 4
    assert predictor.model.score_threshold_detection == 0.4
    assert predictor.model.image_only_det_thresh == 0.5
    assert predictor.model.suppress_det_close_to_boundary is True
    assert predictor.model.tracker.__class__.__name__ == "Sam3MultiplexPredictorWrapper"
    assert predictor.model.detector.__class__.__name__ == "Sam3MultiplexDetector"

    start = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": "<load-dummy-video-1>",
            "session_id": "mux-session",
        }
    )
    assert start == {"session_id": "mux-session"}
    state = predictor._all_inference_states["mux-session"]["state"]
    assert state["num_frames"] == 1
    assert state["device"] == "mlx"
    assert state["is_image_only"] is True
    assert state["input_batch"].img_batch.shape == (1, 3, 1008, 1008)
    assert predictor.handle_request(
        {"type": "close_session", "session_id": "mux-session"}
    ) == {"is_success": True}


def test_multiplex_predictor_builder_threads_detection_thresholds():
    predictor = sam3_mlx.build_sam3_multiplex_video_predictor(
        load_from_HF=False,
        use_fa3=False,
        use_rope_real=False,
        max_num_objects=4,
        multiplex_count=4,
        async_loading_frames=False,
        score_threshold_detection=0.125,
        image_only_det_thresh=0.25,
        suppress_det_close_to_boundary=False,
    )

    assert predictor.model.score_threshold_detection == 0.125
    assert predictor.model.image_only_det_thresh == 0.25
    assert predictor.model.suppress_det_close_to_boundary is False


def test_multiplex_predictor_warm_up_uses_optional_mlx_model_hook():
    model = _WarmUpModel()

    predictor = Sam3MultiplexVideoPredictor(model=model, warm_up=True)

    assert predictor.warm_up is True
    assert model.calls == [False]
    assert model._warm_up_complete is True


def test_multiplex_predictor_warm_up_without_hook_is_noop_marker():
    model = _NoWarmHookModel()

    predictor = Sam3MultiplexVideoPredictor(model=model, warm_up=True)

    assert predictor.warm_up is True
    assert model._warm_up_complete is True


@pytest.mark.parametrize(
    ("call", "feature_fragment"),
    [
        (
            lambda: sam3_mlx.build_sam3_predictor(),
            "build_sam3_multiplex_video_predictor",
        ),
        (
            sam3_mlx.build_sam3_multiplex_video_predictor,
            "build_sam3_multiplex_video_predictor",
        ),
    ],
)
def test_multiplex_public_api_fails_fast_until_runtime_is_ported(
    call, feature_fragment
):
    with pytest.raises(Sam3MlxUnsupportedError) as exc_info:
        call()

    assert exc_info.value.reason == "video-multiplex"
    assert feature_fragment in exc_info.value.feature
