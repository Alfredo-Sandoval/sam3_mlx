from __future__ import annotations

from typing import cast

import mlx.core as mx
from mlx import nn
import pytest

from sam3_mlx.precision import (
    apply_precision_policy,
    cast_checkpoint_weights,
    cast_visual_input,
    checkpoint_dtype_for_policy,
    compute_dtype_for_policy,
    parse_checkpoint_dtype,
    parse_precision,
)
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.sam3_image import Output
from sam3_mlx.model.sam3_image_processor import Sam3Processor, transform
from PIL import Image


class _FakeBackbone:
    def __init__(self) -> None:
        self.forward_image_inputs: list[mx.array] = []

    def forward_image(self, image: mx.array) -> dict[str, object]:
        self.forward_image_inputs.append(image)
        return {"image_batch": image}

    def forward_text(
        self,
        prompts: list[str],
        device: object | None = None,
    ) -> dict[str, mx.array]:
        del prompts, device
        return {
            "language_features": mx.zeros((1, 1, 1), dtype=mx.float32),
            "language_mask": mx.zeros((1, 1), dtype=mx.bool_),
        }


class _FakeModel:
    def __init__(self) -> None:
        self.backbone = _FakeBackbone()
        self.inst_interactive_predictor = None
        self.precision = "fp32"

    def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt:
        del num_prompts
        return Prompt()

    def forward_grounding(self, **kwargs: object) -> Output:
        del kwargs
        raise RuntimeError("grounding is not used in precision boundary tests")


def test_parse_precision_accepts_aliases_and_rejects_unknown() -> None:
    assert parse_precision("fp32") == "fp32"
    assert parse_precision("float16") == "fp16"
    assert parse_precision("bfloat16") == "bf16"
    assert parse_precision("mixed") == "mixed"
    with pytest.raises(ValueError, match="fp32, fp16, bf16, mixed"):
        parse_precision("int8")
    with pytest.raises(TypeError, match="precision must be a string"):
        parse_precision(16)


def test_checkpoint_dtype_follows_policy() -> None:
    assert checkpoint_dtype_for_policy("fp32") == "float32"
    assert checkpoint_dtype_for_policy("fp16") == "float16"
    assert checkpoint_dtype_for_policy("bf16") == "bfloat16"
    assert checkpoint_dtype_for_policy("mixed") == "float16"
    assert parse_checkpoint_dtype("float16") == "float16"
    with pytest.raises(ValueError, match="float32, float16, bfloat16"):
        parse_checkpoint_dtype("fp16")


def test_cast_checkpoint_weights_leaves_complex_and_integers() -> None:
    weights = {
        "linear.weight": mx.ones((2, 2), dtype=mx.float32),
        "rope.freqs": mx.ones((2,), dtype=mx.complex64),
        "ids": mx.arange(3, dtype=mx.int64),
    }

    converted = cast_checkpoint_weights(weights, "float16")

    assert converted["linear.weight"].dtype == mx.float16
    assert converted["rope.freqs"].dtype == mx.complex64
    assert converted["ids"].dtype == mx.int64
    mx.eval(converted["linear.weight"])
    assert mx.array_equal(
        converted["linear.weight"].astype(mx.float32), weights["linear.weight"]
    ).item()


class _TinyPrecisionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)


def test_apply_precision_policy_keeps_layernorm_fp32_in_mixed() -> None:
    model = _TinyPrecisionModule()
    apply_precision_policy(model, "mixed")

    assert model.precision == "mixed"
    assert model.compute_dtype == mx.float16
    assert cast(mx.array, model.linear.weight).dtype == mx.float16
    assert cast(mx.array, model.norm.weight).dtype == mx.float32


def test_apply_precision_policy_casts_full_fp16_and_bf16() -> None:
    fp16_model = _TinyPrecisionModule()
    apply_precision_policy(fp16_model, "fp16")
    assert cast(mx.array, fp16_model.linear.weight).dtype == mx.float16
    assert cast(mx.array, fp16_model.norm.weight).dtype == mx.float16

    bf16_model = _TinyPrecisionModule()
    apply_precision_policy(bf16_model, "bf16")
    assert compute_dtype_for_policy("bf16") == mx.bfloat16
    assert cast(mx.array, bf16_model.linear.weight).dtype == mx.bfloat16
    assert cast(mx.array, bf16_model.norm.weight).dtype == mx.bfloat16


def test_transform_stays_fp32_and_visual_boundary_casts_once() -> None:
    image = Image.new("RGB", (4, 4), color=(12, 34, 56))
    tensor = transform(image, resolution=14)

    assert tensor.dtype == mx.float32
    assert cast_visual_input(tensor, "fp32").dtype == mx.float32
    assert cast_visual_input(tensor, "fp16").dtype == mx.float16
    assert cast_visual_input(tensor, "bf16").dtype == mx.bfloat16
    assert cast_visual_input(tensor, "mixed").dtype == mx.float16


def test_builder_applies_precision_before_compile_flags() -> None:
    import sam3_mlx

    model = sam3_mlx.build_sam3_image_model(
        load_from_HF=False,
        enable_segmentation=False,
        precision="bf16",
        compile=True,
    )

    assert model.precision == "bf16"
    assert model.compute_dtype == mx.bfloat16
    assert model.compile_grounding is True
    assert model.backbone._compile_visual is True


def test_processor_casts_at_visual_boundary_not_in_transform() -> None:
    model = _FakeModel()
    model.precision = "fp16"
    processor = Sam3Processor(model, resolution=14)
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))

    processor.set_image(image)

    forwarded = model.backbone.forward_image_inputs[-1]
    assert processor.transform(image).dtype == mx.float32
    assert forwarded.dtype == mx.float16
    assert forwarded.shape[0] == 1
