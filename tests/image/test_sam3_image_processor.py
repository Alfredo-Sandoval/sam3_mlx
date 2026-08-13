from collections.abc import Callable, Mapping
import importlib
import os
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
from PIL import Image
import pytest
import mlx.core as mx

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.data_misc import FindStage, ResizeWeights
from sam3_mlx.model.geometry_encoders import Prompt
import sam3_mlx.model.sam3_image_processor as image_processor
from sam3_mlx.model.sam3_image_processor import (
    OriginalSizes,
    ProcessorState,
    Sam3Processor,
    transform,
)
from sam3_mlx.model.sam3_image import Output


def _private_callable(name: str) -> object:
    value = getattr(image_processor, name)
    if not callable(value):
        raise TypeError(f"sam3_image_processor.{name} must be callable")
    return value


_resize_weights_1d = cast(
    Callable[[int, int], ResizeWeights],
    _private_callable("_resize_weights_1d"),
)
_resize_uint8_bilinear_like_torchvision = cast(
    Callable[
        [npt.NDArray[np.uint8], tuple[int, int]],
        npt.NDArray[np.uint8],
    ],
    _private_callable("_resize_uint8_bilinear_like_torchvision"),
)
_single_image_keep_indices = cast(
    Callable[[mx.array, float], mx.array],
    _private_callable("_single_image_keep_indices"),
)
_batch_original_sizes = cast(
    Callable[[ProcessorState], OriginalSizes | None],
    _private_callable("_batch_original_sizes"),
)


class _GroundingCall(TypedDict):
    backbone_out: Mapping[str, object]
    find_input: FindStage
    geometric_prompt: Prompt
    find_target: object | None


class _FakeBackbone:
    def __init__(self) -> None:
        self.forward_image_inputs: list[mx.array] = []
        self.forward_text_calls: list[tuple[list[str], object | None]] = []

    def forward_image(self, image: mx.array) -> Mapping[str, object]:
        self.forward_image_inputs.append(image)
        return {"image_batch": image}

    def forward_text(
        self,
        prompts: list[str],
        device: object | None = None,
    ) -> dict[str, mx.array]:
        self.forward_text_calls.append((prompts, device))
        return {
            "language_features": mx.zeros((1, len(prompts), 1), dtype=mx.float32),
            "language_mask": mx.zeros((len(prompts), 1), dtype=mx.bool_),
        }


class _FakeModel:
    def __init__(self, outputs: Mapping[str, object] | None = None) -> None:
        self.backbone = _FakeBackbone()
        self.inst_interactive_predictor = None
        self.outputs = None if outputs is None else Output(outputs)
        self.dummy_prompt_sizes: list[int] = []
        self.forward_grounding_calls: list[_GroundingCall] = []

    def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
        self.dummy_prompt_sizes.append(num_prompts)
        return Prompt()

    def forward_grounding(
        self,
        backbone_out: Mapping[str, object],
        find_input: object,
        find_target: object | None,
        geometric_prompt: Prompt,
    ) -> Output:
        if not isinstance(find_input, FindStage):
            raise TypeError("find_input must be a FindStage")
        self.forward_grounding_calls.append(
            {
                "backbone_out": backbone_out,
                "find_input": find_input,
                "geometric_prompt": geometric_prompt,
                "find_target": find_target,
            }
        )
        if self.outputs is None:
            raise RuntimeError("scripted grounding outputs were not configured")
        return self.outputs


def _logit(probabilities: object) -> npt.NDArray[np.float32]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def _state_mapping(state: ProcessorState, key: str) -> Mapping[str, object]:
    value = state.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"processor state {key} must be a mapping")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(item, str) for item in mapping):
        raise TypeError(f"processor state {key} must use string keys")
    return cast(Mapping[str, object], mapping)


def _mapping_array(mapping: Mapping[str, object], key: str) -> mx.array:
    value = mapping.get(key)
    if not isinstance(value, mx.array):
        raise TypeError(f"{key} must be an MLX array")
    return value


def _state_array_list(state: ProcessorState, key: str) -> list[mx.array]:
    value = state.get(key)
    if not isinstance(value, list):
        raise TypeError(f"processor state {key} must be a list")
    arrays: list[mx.array] = []
    for item in cast(list[object], value):
        if not isinstance(item, mx.array):
            raise TypeError(f"processor state {key} must contain MLX arrays")
        arrays.append(item)
    return arrays


def test_single_image_keep_indices_returns_ordered_indices():
    scores = mx.array([[0.1, 0.6, 0.2, 0.9]], dtype=mx.float32)

    indices = _single_image_keep_indices(scores, 0.5)

    assert indices.tolist() == [1, 3]
    assert indices.dtype == mx.int64


def test_single_image_keep_indices_uses_strict_threshold():
    scores = mx.array([[0.5, 0.5001, 0.8, 0.49]], dtype=mx.float32)

    indices = _single_image_keep_indices(scores, 0.5)

    assert indices.tolist() == [1, 2]


def test_single_image_keep_indices_handles_empty_result():
    scores = mx.array([[0.1, 0.2, 0.3]], dtype=mx.float32)

    indices = _single_image_keep_indices(scores, 0.5)

    assert indices.tolist() == []
    assert indices.shape == (0,)
    assert indices.dtype == mx.int64


def test_single_image_keep_indices_does_not_export_full_keep_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_asarray(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("full keep-mask NumPy export should not be used")

    monkeypatch.setattr(image_processor.np, "asarray", fail_asarray)
    scores = mx.array([[0.1, 0.6, 0.2, 0.9]], dtype=mx.float32)

    indices = _single_image_keep_indices(scores, 0.5)

    assert indices.tolist() == [1, 3]


@pytest.mark.parametrize("resolution", [0, -14, 15, 1007, 14.0, True])
def test_processor_resolution_must_be_positive_integer_multiple_of_patch_size(
    resolution: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer multiple of 14"):
        Sam3Processor(_FakeModel(), resolution=resolution)


def test_transform_enforces_resolution_patch_multiple_for_direct_callers():
    image = Image.new("RGB", (4, 2), color=(255, 0, 0))

    with pytest.raises(ValueError, match="positive integer multiple of 14"):
        transform(image, resolution=13)


def test_resize_uint8_bilinear_matches_torchvision_upsample_literal():
    image = np.array(
        [
            [[0, 0, 0], [255, 0, 0]],
            [[0, 255, 0], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )

    resized = _resize_uint8_bilinear_like_torchvision(image, (4, 4))

    np.testing.assert_array_equal(
        resized,
        np.array(
            [
                [[0, 0, 0], [64, 0, 0], [191, 0, 0], [255, 0, 0]],
                [[0, 64, 0], [48, 48, 16], [143, 16, 48], [191, 0, 64]],
                [[0, 191, 0], [16, 143, 48], [48, 48, 143], [64, 0, 191]],
                [[0, 255, 0], [0, 191, 64], [0, 64, 191], [0, 0, 255]],
            ],
            dtype=np.uint8,
        ),
    )


def test_resize_uint8_bilinear_matches_torchvision_float_tie_direction():
    image = np.zeros((1, 256, 3), dtype=np.uint8)
    image[:, 21, 2] = 174
    image[:, 22, 2] = 183

    resized = _resize_uint8_bilinear_like_torchvision(image, (4, 1008))

    assert resized[0, 87, 2] == 181


def test_resize_uint8_bilinear_matches_torchvision_antialiased_downsample_literal():
    image = np.array(
        [
            [[0, 5, 10], [15, 20, 25], [30, 35, 40], [45, 50, 55]],
            [[60, 65, 70], [75, 80, 85], [90, 95, 100], [105, 110, 115]],
            [[120, 125, 130], [135, 140, 145], [150, 155, 160], [165, 170, 175]],
            [[180, 185, 190], [195, 200, 205], [210, 215, 220], [225, 230, 235]],
            [[240, 245, 250], [255, 0, 5], [10, 15, 20], [25, 30, 35]],
        ],
        dtype=np.uint8,
    )

    resized = _resize_uint8_bilinear_like_torchvision(image, (2, 3))

    np.testing.assert_allclose(
        resized,
        np.array(
            [
                [[64, 70, 74], [82, 88, 92], [101, 106, 111]],
                [[184, 165, 170], [161, 125, 130], [138, 143, 148]],
            ],
            dtype=np.uint8,
        ),
        rtol=0.0,
        atol=1,
    )


def test_resize_weights_1d_are_cached_and_immutable():
    first = _resize_weights_1d(5, 2)
    second = _resize_weights_1d(5, 2)

    assert first is second
    assert len(first) == 2
    for indices, weights in first:
        assert not indices.flags.writeable
        assert not weights.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        first[0][1][0] = 0.0


def test_transform_uses_torchvision_tensor_resize_contract():
    image = Image.fromarray(
        np.array(
            [
                [[0, 0, 0], [255, 0, 0]],
                [[0, 255, 0], [0, 0, 255]],
            ],
            dtype=np.uint8,
        )
    )

    transformed = to_numpy(transform(image, resolution=14))

    expected_image = _resize_uint8_bilinear_like_torchvision(
        np.asarray(image, dtype=np.uint8),
        (14, 14),
    )
    expected = expected_image.astype(np.float32) / 255.0
    expected = ((expected - 0.5) / 0.5).transpose(2, 0, 1)
    np.testing.assert_allclose(transformed, expected, rtol=0.0, atol=1e-6)


def _synthetic_rgb_image(width: int, height: int) -> Image.Image:
    y, x = np.mgrid[:height, :width]
    image = np.stack(
        [
            (17 * x + 3 * y) % 256,
            (5 * x + 29 * y) % 256,
            (11 * x + 7 * y) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def _attribute(value: object, name: str) -> object:
    return getattr(value, name)


def _require_callable(value: object, context: str) -> Callable[..., object]:
    if not callable(value):
        raise TypeError(f"{context} must be callable")
    return value


def _callable_attribute(value: object, name: str) -> Callable[..., object]:
    return _require_callable(_attribute(value, name), name)


def _call_method(value: object, name: str, *args: object) -> object:
    return _callable_attribute(value, name)(*args)


def test_transform_matches_torchvision_on_synthetic_aspect_ratios_when_available():
    if os.environ.get("SAM3_MLX_REQUIRE_TORCHVISION") == "1":
        torch_module = importlib.import_module("torch")
        importlib.import_module("torchvision")
    else:
        torch_module = pytest.importorskip("torch")
        pytest.importorskip("torchvision")
    v2_module = importlib.import_module("torchvision.transforms.v2")
    functional_module = importlib.import_module("torchvision.transforms.v2.functional")

    compose = _callable_attribute(v2_module, "Compose")
    to_dtype = _callable_attribute(v2_module, "ToDtype")
    resize = _callable_attribute(v2_module, "Resize")
    normalize = _callable_attribute(v2_module, "Normalize")
    to_image = _callable_attribute(functional_module, "to_image")

    resolution = 42
    official_transform = _require_callable(
        compose(
            [
                to_dtype(_attribute(torch_module, "uint8"), scale=True),
                resize(size=(resolution, resolution)),
                to_dtype(_attribute(torch_module, "float32"), scale=True),
                normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
        "torchvision Compose result",
    )
    resize_transform = _require_callable(
        resize(size=(resolution, resolution)),
        "torchvision Resize result",
    )

    for image in (
        _synthetic_rgb_image(7, 5),
        _synthetic_rgb_image(64, 21),
        _synthetic_rgb_image(21, 64),
    ):
        official_tensor = resize_transform(to_image(image))
        official_hwc = _call_method(official_tensor, "permute", 1, 2, 0)
        official_uint8 = np.asarray(
            _call_method(official_hwc, "numpy"),
            dtype=np.uint8,
        )
        local_uint8 = _resize_uint8_bilinear_like_torchvision(
            np.asarray(image, dtype=np.uint8),
            (resolution, resolution),
        )
        np.testing.assert_array_equal(local_uint8, official_uint8)

        transformed_tensor = official_transform(to_image(image))
        official = np.asarray(
            _call_method(transformed_tensor, "numpy"),
            dtype=np.float32,
        )
        local = to_numpy(transform(image, resolution=resolution))

        np.testing.assert_allclose(local, official, rtol=0.0, atol=1e-6)


def test_set_image_batch_records_two_image_sizes_and_batched_mlx_tensor():
    model = _FakeModel()
    processor = Sam3Processor(model, resolution=14)
    images = [
        Image.new("RGB", (4, 2), color=(255, 0, 0)),
        np.zeros((3, 5, 3), dtype=np.uint8),
    ]

    state = processor.set_image_batch(images)

    assert state["original_heights"] == [2, 3]
    assert state["original_widths"] == [4, 5]
    image_batch = model.backbone.forward_image_inputs[-1]
    assert image_batch.shape == (2, 3, 14, 14)
    assert image_batch.dtype == mx.float32
    assert _mapping_array(
        _state_mapping(state, "backbone_out"), "image_batch"
    ).shape == (
        2,
        3,
        14,
        14,
    )


def test_batch_text_prompt_returns_per_image_outputs_with_sizes_and_thresholding():
    presence_logit = 10.0
    outputs = {
        "pred_boxes": mx.array(
            [
                [
                    [0.5, 0.5, 0.2, 0.2],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.1, 0.1, 0.2, 0.2],
                ],
                [
                    [0.5, 0.5, 1.0, 0.5],
                    [0.5, 0.5, 0.2, 0.2],
                    [0.25, 0.25, 0.5, 0.5],
                ],
            ],
            dtype=mx.float32,
        ),
        "pred_logits": mx.array(
            _logit(
                [
                    [[0.2], [0.8], [0.5]],
                    [[0.7], [0.49], [0.95]],
                ]
            ),
            dtype=mx.float32,
        ),
        "pred_masks": mx.array(
            [
                [[[-4.0]], [[2.0]], [[4.0]]],
                [[[2.0]], [[2.0]], [[-4.0]]],
            ],
            dtype=mx.float32,
        ),
        "presence_logit_dec": mx.array(
            [[presence_logit], [presence_logit]],
            dtype=mx.float32,
        ),
    }
    model = _FakeModel(outputs=outputs)
    processor = Sam3Processor(model, resolution=14, confidence_threshold=0.5)
    state = processor.set_image_batch(
        [
            Image.new("RGB", (4, 2), color=(255, 0, 0)),
            Image.new("RGB", (2, 4), color=(0, 255, 0)),
        ]
    )

    result = processor.set_text_prompt("truck", state)

    find_input = model.forward_grounding_calls[-1]["find_input"]
    np.testing.assert_array_equal(to_numpy(find_input.img_ids), np.array([0, 1]))
    np.testing.assert_array_equal(to_numpy(find_input.text_ids), np.array([0, 0]))
    assert model.backbone.forward_text_calls == [(["truck"], "mlx")]
    assert model.dummy_prompt_sizes == [2]

    boxes = _state_array_list(result, "boxes")
    scores = _state_array_list(result, "scores")
    masks = _state_array_list(result, "masks")
    masks_logits = _state_array_list(result, "masks_logits")
    assert len(boxes) == 2
    assert len(scores) == 2
    assert len(masks) == 2
    assert masks_logits[0].shape == (1, 1, 2, 4)
    assert masks_logits[1].shape == (2, 1, 4, 2)

    np.testing.assert_allclose(
        to_numpy(boxes[0]),
        np.array([[1.0, 0.5, 3.0, 1.5]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(boxes[1]),
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-6,
    )
    presence = 1.0 / (1.0 + np.exp(-presence_logit))
    np.testing.assert_allclose(
        to_numpy(scores[0]),
        np.array([0.8 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(scores[1]),
        np.array([0.7 * presence, 0.95 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    assert to_numpy(masks[0]).all()
    assert to_numpy(masks[1][0]).all()
    assert not to_numpy(masks[1][1]).any()


def test_set_text_prompt_validates_caller_supplied_text_outputs() -> None:
    model = _FakeModel()
    processor = Sam3Processor(model, resolution=14)
    state = processor.set_image(Image.new("RGB", (4, 4), color=(0, 0, 0)))

    with pytest.raises(TypeError, match="map string keys to MLX arrays"):
        processor.set_text_prompt(
            "truck",
            state,
            run_grounding=False,
            text_outputs={"language_features": object()},
        )

    backbone_out = _state_mapping(state, "backbone_out")
    assert "language_features" not in backbone_out
    assert model.backbone.forward_text_calls == []


def test_batch_geometric_prompt_fails_fast_until_interactive_batch_contract_exists():
    processor = Sam3Processor(_FakeModel())
    state: ProcessorState = {
        "backbone_out": {"image_batch": mx.zeros((2, 3, 4, 4), dtype=mx.float32)},
        "original_heights": [2, 4],
        "original_widths": [4, 2],
    }

    with pytest.raises(Sam3MlxUnsupportedError, match="Batch geometric prompts") as exc:
        processor.add_geometric_prompt([0.5, 0.5, 0.25, 0.25], True, state)

    assert exc.value.reason == "image-interactivity"
    assert "add_geometric_prompt(batch_state)" in exc.value.feature


def test_set_image_after_batch_clears_batch_metadata():
    model = _FakeModel()
    processor = Sam3Processor(model, resolution=14)
    state = processor.set_image_batch(
        [
            Image.new("RGB", (4, 2), color=(255, 0, 0)),
            Image.new("RGB", (2, 4), color=(0, 255, 0)),
        ]
    )
    state["geometric_prompt"] = {"stale": True}
    state["boxes"] = mx.zeros((1, 4), dtype=mx.float32)
    state["scores"] = mx.zeros((1,), dtype=mx.float32)

    state = processor.set_image(Image.new("RGB", (6, 8), color=(0, 0, 255)), state)

    assert "original_heights" not in state
    assert "original_widths" not in state
    assert state["original_height"] == 8
    assert state["original_width"] == 6
    assert "geometric_prompt" not in state
    assert "boxes" not in state
    assert "scores" not in state
    assert _batch_original_sizes(state) is None
    assert _mapping_array(
        _state_mapping(state, "backbone_out"), "image_batch"
    ).shape == (
        1,
        3,
        14,
        14,
    )


def test_batch_original_sizes_rejects_boolean_dimensions() -> None:
    state: ProcessorState = {
        "original_heights": [True],
        "original_widths": [4],
    }

    with pytest.raises(TypeError, match="original_heights"):
        _batch_original_sizes(state)


def test_reset_prompts_rejects_non_mapping_backbone_state() -> None:
    processor = Sam3Processor(_FakeModel(), resolution=14)
    state: ProcessorState = {"backbone_out": object()}

    with pytest.raises(TypeError, match="backbone_out"):
        processor.reset_all_prompts(state)


def test_run_grounding_rejects_non_prompt_geometric_state() -> None:
    processor = Sam3Processor(_FakeModel(), resolution=14)
    state = processor.set_image(Image.new("RGB", (4, 4), color=(0, 0, 0)))
    state["geometric_prompt"] = object()

    with pytest.raises(TypeError, match="geometric_prompt"):
        processor.run_grounding(state)


def test_set_image_clears_prior_geometric_prompt():
    class _RecordingPrompt(Prompt):
        def __init__(self) -> None:
            super().__init__()
            self.boxes: list[tuple[mx.array, mx.array]] = []

        def append_boxes(
            self,
            boxes: mx.array,
            labels: mx.array,
            mask: mx.array | None = None,
        ) -> None:
            del mask
            self.boxes.append((boxes, labels))

    class _PromptModel(_FakeModel):
        def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
            self.dummy_prompt_sizes.append(num_prompts)
            return _RecordingPrompt()

    model = _PromptModel(
        outputs={
            "pred_boxes": mx.zeros((1, 1, 4), dtype=mx.float32),
            "pred_logits": mx.full((1, 1, 1), -10.0, dtype=mx.float32),
            "pred_masks": mx.zeros((1, 1, 1, 1), dtype=mx.float32),
            "presence_logit_dec": mx.zeros((1, 1), dtype=mx.float32),
        }
    )
    processor = Sam3Processor(model, resolution=14)
    state = processor.set_image(Image.new("RGB", (4, 4), color=(255, 0, 0)))
    processor.add_geometric_prompt(
        [0.5, 0.5, 0.25, 0.25],
        True,
        state,
        run_grounding=False,
    )
    assert "geometric_prompt" in state
    geometric_prompt = state["geometric_prompt"]
    assert isinstance(geometric_prompt, _RecordingPrompt)
    assert len(geometric_prompt.boxes) == 1

    state = processor.set_image(Image.new("RGB", (5, 5), color=(0, 255, 0)), state)
    assert "geometric_prompt" not in state

    state = processor.set_text_prompt("visual", state, run_grounding=False)
    geometric_prompt = state["geometric_prompt"]
    assert isinstance(geometric_prompt, _RecordingPrompt)
    assert geometric_prompt.boxes == []
    assert model.dummy_prompt_sizes[-1] == 1


def test_reset_all_prompts_removes_semantic_seg():
    processor = Sam3Processor(_FakeModel(), resolution=14)
    state: ProcessorState = {
        "backbone_out": {
            "image_batch": mx.zeros((1, 3, 4, 4), dtype=mx.float32),
            "language_features": mx.zeros((1, 1, 1), dtype=mx.float32),
            "language_mask": mx.zeros((1, 1), dtype=mx.bool_),
            "language_embeds": mx.zeros((1, 1, 1), dtype=mx.float32),
        },
        "geometric_prompt": object(),
        "boxes": mx.zeros((1, 4), dtype=mx.float32),
        "masks": mx.zeros((1, 1, 1), dtype=mx.bool_),
        "masks_logits": mx.zeros((1, 1, 1), dtype=mx.float32),
        "mask_logits": mx.zeros((1, 1, 1), dtype=mx.float32),
        "semantic_seg": mx.zeros((1, 1, 4, 4), dtype=mx.float32),
        "scores": mx.zeros((1,), dtype=mx.float32),
        "original_height": 4,
        "original_width": 4,
    }

    processor.reset_all_prompts(state)

    for key in (
        "geometric_prompt",
        "boxes",
        "masks",
        "masks_logits",
        "mask_logits",
        "semantic_seg",
        "scores",
    ):
        assert key not in state
    backbone_out = _state_mapping(state, "backbone_out")
    for key in ("language_features", "language_mask", "language_embeds"):
        assert key not in backbone_out
    # Size metadata is image identity, not a prompt/result; reset leaves it.
    assert state["original_height"] == 4
    assert state["original_width"] == 4
