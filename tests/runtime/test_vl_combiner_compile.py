from __future__ import annotations

import mlx.core as mx
from mlx import nn

from sam3_mlx.model.vl_combiner import SAM3VLBackbone


class _TinyVisualBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = mx.array(2.0)
        self.calls = 0

    def forward(
        self,
        samples: mx.array,
        *,
        output_levels: int | None = None,
    ) -> tuple[list[mx.array], list[mx.array], None, None]:
        del output_levels
        self.calls += 1
        feature = samples * self.scale
        position = mx.zeros_like(feature)
        return [feature], [position], None, None


def test_compiled_visual_backbone_matches_eager_and_reuses_compilation() -> None:
    samples = mx.arange(4, dtype=mx.float32).reshape(1, 1, 2, 2)
    eager_visual = _TinyVisualBackbone()
    compiled_visual = _TinyVisualBackbone()
    eager = SAM3VLBackbone(visual=eager_visual, text=None)
    compiled = SAM3VLBackbone(
        visual=compiled_visual,
        text=None,
        compile_visual=True,
    )

    eager_output = eager.forward_image(samples)
    first_output = compiled.forward_image(samples)
    compiled_function = compiled._compiled_forward_image
    second_output = compiled.forward_image(samples + 1)

    mx.eval(eager_output, first_output, second_output)
    assert mx.array_equal(
        first_output["vision_features"], eager_output["vision_features"]
    ).item()
    assert mx.array_equal(second_output["vision_features"], (samples + 1) * 2).item()
    assert compiled_function is not None
    assert compiled._compiled_forward_image is compiled_function
    assert compiled_visual.calls == 1
