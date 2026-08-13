from __future__ import annotations

import mlx.core as mx
import pytest

from sam3_mlx.model.sam3_image import (
    Output,
    grounding_compile_key,
    raw_prediction_from_output,
)
from sam3_mlx.precision import apply_precision_policy
from mlx import nn


def test_raw_prediction_from_output_requires_fixed_shape_arrays() -> None:
    out = Output(
        {
            "pred_logits": mx.ones((1, 2, 1), dtype=mx.float32),
            "pred_boxes": mx.ones((1, 2, 4), dtype=mx.float32),
            "pred_masks": mx.ones((1, 2, 2, 2), dtype=mx.float32),
            "presence_logit_dec": mx.ones((1, 1), dtype=mx.float32),
            "pred_boxes_xyxy": mx.ones((1, 2, 4), dtype=mx.float32),
        }
    )

    raw = raw_prediction_from_output(out)

    assert raw["pred_logits"].shape == (1, 2, 1)
    assert raw["pred_boxes"].shape == (1, 2, 4)
    assert raw["pred_masks"].shape == (1, 2, 2, 2)
    assert raw["presence_logit_dec"].shape == (1, 1)
    assert raw["pred_boxes_xyxy"].shape == (1, 2, 4)


def test_raw_prediction_from_output_rejects_missing_masks() -> None:
    with pytest.raises(KeyError, match="pred_masks"):
        raw_prediction_from_output(
            {
                "pred_logits": mx.ones((1, 1, 1)),
                "pred_boxes": mx.ones((1, 1, 4)),
                "presence_logit_dec": mx.ones((1, 1)),
            }
        )


def test_grounding_compile_key_includes_dtype_shape_and_prompt_geometry() -> None:
    fp32_key = grounding_compile_key(
        dtype=mx.float32,
        feat_shape=(72, 1, 256),
        prompt_shape=(32, 1, 256),
        prompt_mask_shape=(1, 32),
        vis_feat_sizes=((72, 72),),
        fpn_shapes=((1, 256, 72, 72),),
        batch_size=1,
    )
    fp16_key = grounding_compile_key(
        dtype=mx.float16,
        feat_shape=(72, 1, 256),
        prompt_shape=(32, 1, 256),
        prompt_mask_shape=(1, 32),
        vis_feat_sizes=((72, 72),),
        fpn_shapes=((1, 256, 72, 72),),
        batch_size=1,
    )
    prompt_key = grounding_compile_key(
        dtype=mx.float32,
        feat_shape=(72, 1, 256),
        prompt_shape=(40, 1, 256),
        prompt_mask_shape=(1, 40),
        vis_feat_sizes=((72, 72),),
        fpn_shapes=((1, 256, 72, 72),),
        batch_size=1,
    )

    assert fp32_key != fp16_key
    assert fp32_key != prompt_key
    assert fp32_key[0] == "float32"
    assert fp16_key[0] == "float16"
    assert prompt_key[2] == (40, 1, 256)


class _Clearable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.cleared: list[str] = []

    def clear_compiled_visual(self) -> None:
        self.cleared.append("visual")

    def clear_compiled_grounding(self) -> None:
        self.cleared.append("grounding")


def test_apply_precision_policy_clears_compiled_visual_and_grounding() -> None:
    model = _Clearable()
    apply_precision_policy(model, "fp16")
    assert model.cleared == ["visual", "grounding"]
