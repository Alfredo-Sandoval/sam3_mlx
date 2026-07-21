import os

import numpy as np
import pytest
from PIL import Image

from sam3_mlx import build_sam3_image_model
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.sam3_image_processor import Sam3Processor

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("SAM3_MLX_RUN_WEIGHTED_INTEGRATION") != "1",
        reason="set SAM3_MLX_RUN_WEIGHTED_INTEGRATION=1 to run real weights",
    ),
]


def test_pinned_mlx_checkpoint_executes_end_to_end_image_inference():
    model = build_sam3_image_model()
    processor = Sam3Processor(model, resolution=224, confidence_threshold=0.0)
    image = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    )

    state = processor.set_image(image)
    output = processor.set_text_prompt(prompt="object", state=state)

    assert {"backbone_out", "masks", "boxes", "scores"} <= output.keys()
    masks = to_numpy(output["masks"])
    boxes = to_numpy(output["boxes"])
    scores = to_numpy(output["scores"])
    assert masks.ndim == 4 and masks.shape[1:] == (1, 96, 128)
    assert boxes.shape == (masks.shape[0], 4)
    assert scores.shape == (masks.shape[0],)
    assert np.isfinite(boxes).all()
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
