import inspect

import numpy as np
import pytest

import sam3_mlx
from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.maskformer_segmentation import UniversalSegmentationHead
from sam3_mlx.model.sam3_image import Sam3Image
from sam3_mlx.model.sam3_image_processor import Sam3Processor
from sam3_mlx.model.sam3_video_inference import Sam3VideoInference
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor


BLOCKED_ACCELERATOR = "cu" + "da"


UPSTREAM_BUILDER_PARAM_ORDER = {
    "build_tracker": [
        "apply_temporal_disambiguation",
        "with_backbone",
        "compile_mode",
    ],
    "build_sam3_image_model": [
        "bpe_path",
        "device",
        "eval_mode",
        "checkpoint_path",
        "load_from_HF",
        "enable_segmentation",
        "enable_inst_interactivity",
        "compile",
    ],
    "build_sam3_video_model": [
        "checkpoint_path",
        "load_from_HF",
        "bpe_path",
        "has_presence_token",
        "geo_encoder_use_img_cross_attn",
        "strict_state_dict_loading",
        "apply_temporal_disambiguation",
        "device",
        "compile",
    ],
    "build_sam3_multiplex_video_model": [
        "checkpoint_path",
        "load_from_HF",
        "multiplex_count",
        "use_fa3",
        "use_rope_real",
        "strict_state_dict_loading",
        "device",
        "compile",
    ],
    "build_sam3_multiplex_video_predictor": [
        "checkpoint_path",
        "bpe_path",
        "max_num_objects",
        "multiplex_count",
        "use_fa3",
        "use_rope_real",
        "compile",
        "warm_up",
        "session_expiration_sec",
        "default_output_prob_thresh",
        "async_loading_frames",
    ],
    "build_sam3_predictor": [
        "checkpoint_path",
        "bpe_path",
        "version",
        "compile",
        "warm_up",
        "max_num_objects",
        "multiplex_count",
        "use_fa3",
        "use_rope_real",
        "async_loading_frames",
    ],
}


class _FakeBackbone:
    def forward_image(self, image):
        return {"image_batch": image}

    def forward_text(self, prompts, device=None):
        return {
            "language_features": np.zeros((1, len(prompts), 1), dtype=np.float32),
            "language_mask": np.zeros((len(prompts), 1), dtype=bool),
        }


class _FakeModel:
    inst_interactive_predictor = None
    backbone = _FakeBackbone()

    def _get_dummy_prompt(self, num_prompts=1):
        return {"num_prompts": num_prompts}


class _FakeImageProcessor:
    def __init__(self, image_model, resolution=1008, confidence_threshold=0.5):
        self.image_model = image_model
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold

    def set_image(self, image):
        del image
        return {
            "masks": np.zeros((0, 480, 640), dtype=bool),
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
        }

    def set_text_prompt(self, prompt, state):
        del prompt
        return state


def test_public_builders_keep_upstream_parameter_names_and_order():
    for name, expected_order in UPSTREAM_BUILDER_PARAM_ORDER.items():
        signature = inspect.signature(getattr(sam3_mlx, name))
        assert list(signature.parameters)[: len(expected_order)] == expected_order


def test_video_predictor_builder_keeps_upstream_vararg_shape():
    signature = inspect.signature(sam3_mlx.build_sam3_video_predictor)
    kinds = {name: param.kind for name, param in signature.parameters.items()}

    assert kinds["model_args"] is inspect.Parameter.VAR_POSITIONAL
    assert kinds["model_kwargs"] is inspect.Parameter.VAR_KEYWORD
    assert "gpus_to_use" in kinds


def test_image_builder_uses_explicit_mlx_default():
    model = sam3_mlx.build_sam3_image_model(load_from_HF=False)

    assert isinstance(model, Sam3Image)
    assert model.device == "mlx"
    assert model.hidden_dim == 256
    assert model.inst_interactive_predictor is None
    assert isinstance(model.segmentation_head, UniversalSegmentationHead)


@pytest.mark.parametrize("device", ["mlx", None])
def test_image_processor_accepts_only_explicit_mlx_device(device):
    processor = Sam3Processor(_FakeModel(), resolution=14, device=device)

    assert processor.device == "mlx"


@pytest.mark.parametrize(
    "device",
    [
        BLOCKED_ACCELERATOR,
        f"{BLOCKED_ACCELERATOR}:0",
        f"{BLOCKED_ACCELERATOR}:",
        f"{BLOCKED_ACCELERATOR}:abc",
        "cpu",
        "tpu",
        42,
    ],
)
def test_non_mlx_devices_fail_fast(device):
    with pytest.raises(Sam3MlxUnsupportedError) as builder_exc:
        sam3_mlx.build_sam3_image_model(device=device, load_from_HF=False)
    assert builder_exc.value.reason == "unsupported-device"

    with pytest.raises(Sam3MlxUnsupportedError) as processor_exc:
        Sam3Processor(_FakeModel(), resolution=14, device=device)
    assert processor_exc.value.reason == "unsupported-device"


def test_video_model_builder_uses_explicit_mlx_device():
    image_model = object()
    model = sam3_mlx.build_sam3_video_model(
        device="mlx",
        image_model=image_model,
        load_from_HF=False,
        image_size=14,
        processor_factory=_FakeImageProcessor,
    )

    assert isinstance(model, Sam3VideoInference)
    assert model.image_model is image_model
    assert model.image_size == 14
    assert model.image_mean == (0.5, 0.5, 0.5)
    assert model.image_std == (0.5, 0.5, 0.5)
    assert model.confidence_threshold == 0.5
    assert model.processor_factory is _FakeImageProcessor


def test_video_predictor_accepts_upstream_positional_builder_args():
    predictor = sam3_mlx.build_sam3_video_predictor(
        None,
        None,
        True,
        True,
        True,
        True,
        "imageio",
        True,
        False,
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=_FakeImageProcessor,
    )

    assert isinstance(predictor, Sam3VideoPredictor)
    assert predictor.async_loading_frames is True
    assert predictor.video_loader_type == "imageio"
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": "<load-dummy-video-1>"}
    )
    assert sorted(response) == ["session_id"]


def test_sam3_predictor_version_sam3_uses_explicit_mlx_device():
    image_model = object()
    predictor = sam3_mlx.build_sam3_predictor(
        version="sam3",
        device="mlx",
        model=image_model,
        load_from_HF=False,
        resolution=14,
        processor_factory=_FakeImageProcessor,
    )

    assert isinstance(predictor, Sam3VideoPredictor)
    assert predictor.async_loading_frames is True
    assert predictor.model.image_model is image_model
    assert predictor.model.image_size == 14
    assert predictor.model.processor_factory is _FakeImageProcessor


def test_sam3_predictor_version_sam3_propagates_compile_fail_fast():
    with pytest.raises(Sam3MlxUnsupportedError) as exc:
        sam3_mlx.build_sam3_predictor(
            version="sam3",
            compile=True,
            model=object(),
            load_from_HF=False,
            resolution=14,
            processor_factory=_FakeImageProcessor,
        )

    assert exc.value.reason == "torch-compile"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"compile": True}, "torch-compile"),
        ({"device": "tpu"}, "unsupported-device"),
        ({"has_presence_token": False}, "video-multiplex"),
        ({"geo_encoder_use_img_cross_attn": False}, "video-multiplex"),
        ({"strict_state_dict_loading": False}, "video-multiplex"),
        ({"apply_temporal_disambiguation": False}, "video-multiplex"),
    ],
)
def test_video_predictor_video_model_shortcut_keeps_fail_fast_guards(kwargs, reason):
    with pytest.raises(Sam3MlxUnsupportedError) as exc:
        Sam3VideoPredictor(video_model=object(), **kwargs)

    assert exc.value.reason == reason


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_model": object()},
        {"model": object()},
        {"video_model": object()},
    ],
)
def test_video_predictor_rejects_checkpoint_with_injected_model(kwargs):
    with pytest.raises(ValueError, match="checkpoint_path cannot be used"):
        Sam3VideoPredictor(checkpoint_path="weights.pt", **kwargs)


def test_video_predictor_close_session_accepts_upstream_cache_threshold():
    predictor = sam3_mlx.build_sam3_video_predictor(
        model=object(),
        load_from_HF=False,
        resolution=14,
        processor_factory=_FakeImageProcessor,
    )
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": "<load-dummy-video-1>"}
    )

    assert predictor.handle_request(
        {
            "type": "close_session",
            "session_id": response["session_id"],
            "run_gc_collect": False,
            "clear_cache_threshold": 90,
        }
    ) == {"is_success": True}
