import numpy as np
from PIL import Image
import pytest
import mlx.core as mx

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import to_numpy
import sam3_mlx.model.sam3_image_processor as image_processor
from sam3_mlx.model.sam3_image_processor import (
    Sam3Processor,
    _resize_weights_1d,
    _resize_uint8_bilinear_like_torchvision,
    _single_image_keep_indices,
    transform,
)


class _FakeBackbone:
    def __init__(self):
        self.forward_image_inputs = []
        self.forward_text_calls = []

    def forward_image(self, image):
        self.forward_image_inputs.append(image)
        return {"image_batch": image}

    def forward_text(self, prompts, device=None):
        self.forward_text_calls.append((prompts, device))
        return {
            "language_features": mx.zeros((1, len(prompts), 1), dtype=mx.float32),
            "language_mask": mx.zeros((len(prompts), 1), dtype=mx.bool_),
        }


class _FakeModel:
    def __init__(self, outputs=None):
        self.backbone = _FakeBackbone()
        self.inst_interactive_predictor = None
        self.outputs = outputs
        self.dummy_prompt_sizes = []
        self.forward_grounding_calls = []

    def _get_dummy_prompt(self, num_prompts=1):
        self.dummy_prompt_sizes.append(num_prompts)
        return {"num_prompts": num_prompts}

    def forward_grounding(
        self, *, backbone_out, find_input, geometric_prompt, find_target
    ):
        self.forward_grounding_calls.append(
            {
                "backbone_out": backbone_out,
                "find_input": find_input,
                "geometric_prompt": geometric_prompt,
                "find_target": find_target,
            }
        )
        return self.outputs


def _logit(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def test_single_image_keep_indices_returns_ordered_indices():
    scores = mx.array([[0.1, 0.6, 0.2, 0.9]], dtype=mx.float32)

    indices = _single_image_keep_indices(scores, threshold=0.5)

    assert indices.tolist() == [1, 3]
    assert indices.dtype == mx.int64


def test_single_image_keep_indices_uses_strict_threshold():
    scores = mx.array([[0.5, 0.5001, 0.8, 0.49]], dtype=mx.float32)

    indices = _single_image_keep_indices(scores, threshold=0.5)

    assert indices.tolist() == [1, 2]


def test_single_image_keep_indices_handles_empty_result():
    scores = mx.array([[0.1, 0.2, 0.3]], dtype=mx.float32)

    indices = _single_image_keep_indices(scores, threshold=0.5)

    assert indices.tolist() == []
    assert indices.shape == (0,)
    assert indices.dtype == mx.int64


def test_single_image_keep_indices_does_not_export_full_keep_mask(monkeypatch):
    def fail_asarray(*args, **kwargs):
        raise AssertionError("full keep-mask NumPy export should not be used")

    monkeypatch.setattr(image_processor.np, "asarray", fail_asarray)
    scores = mx.array([[0.1, 0.6, 0.2, 0.9]], dtype=mx.float32)

    indices = _single_image_keep_indices(scores, threshold=0.5)

    assert indices.tolist() == [1, 3]


@pytest.mark.parametrize("resolution", [0, -14, 15, 1007, 14.0, True])
def test_processor_resolution_must_be_positive_integer_multiple_of_patch_size(
    resolution,
):
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


def _synthetic_rgb_image(width, height):
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


def test_transform_matches_torchvision_on_synthetic_aspect_ratios_when_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.transforms import v2
    from torchvision.transforms.v2 import functional as torch_vision_functional

    resolution = 42
    official_transform = v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    for image in (
        _synthetic_rgb_image(7, 5),
        _synthetic_rgb_image(64, 21),
        _synthetic_rgb_image(21, 64),
    ):
        official_uint8 = (
            v2.Resize(size=(resolution, resolution))(
                torch_vision_functional.to_image(image)
            )
            .permute(1, 2, 0)
            .numpy()
        )
        local_uint8 = _resize_uint8_bilinear_like_torchvision(
            np.asarray(image, dtype=np.uint8),
            (resolution, resolution),
        )
        np.testing.assert_array_equal(local_uint8, official_uint8)

        official = official_transform(torch_vision_functional.to_image(image)).numpy()
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
    assert state["backbone_out"]["image_batch"].shape == (2, 3, 14, 14)


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

    assert len(result["boxes"]) == 2
    assert len(result["scores"]) == 2
    assert len(result["masks"]) == 2
    assert result["masks_logits"][0].shape == (1, 1, 2, 4)
    assert result["masks_logits"][1].shape == (2, 1, 4, 2)

    np.testing.assert_allclose(
        to_numpy(result["boxes"][0]),
        np.array([[1.0, 0.5, 3.0, 1.5]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        to_numpy(result["boxes"][1]),
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
        to_numpy(result["scores"][0]),
        np.array([0.8 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        to_numpy(result["scores"][1]),
        np.array([0.7 * presence, 0.95 * presence], dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )
    assert to_numpy(result["masks"][0]).all()
    assert to_numpy(result["masks"][1][0]).all()
    assert not to_numpy(result["masks"][1][1]).any()


def test_batch_geometric_prompt_fails_fast_until_interactive_batch_contract_exists():
    processor = Sam3Processor(_FakeModel())
    state = {
        "backbone_out": {"image_batch": mx.zeros((2, 3, 4, 4), dtype=mx.float32)},
        "original_heights": [2, 4],
        "original_widths": [4, 2],
    }

    with pytest.raises(Sam3MlxUnsupportedError, match="Batch geometric prompts") as exc:
        processor.add_geometric_prompt([0.5, 0.5, 0.25, 0.25], True, state)

    assert exc.value.reason == "image-interactivity"
    assert "add_geometric_prompt(batch_state)" in exc.value.feature
