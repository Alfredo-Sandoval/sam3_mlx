from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sam3_mlx.agent.agent_core import agent_inference
from sam3_mlx.agent.client_llm import send_generate_request
from sam3_mlx.agent.contracts import Message
from sam3_mlx.agent.helpers.keypoints import Keypoints
from sam3_mlx.agent.helpers.masks import BitMasks, ROIMasks
from sam3_mlx.agent.helpers.rle import rle_decode, rle_encode
from sam3_mlx.agent.helpers.som_utils import Color, ColorPalette, rgb_to_hex


def test_agent_rle_helpers_preserve_uncompressed_counts_contract() -> None:
    mask = np.array([[[False, True, False], [True, True, False]]], dtype=bool)

    encoded = rle_encode(mask, return_areas=True)

    assert encoded == [{"counts": [1, 3, 2], "size": [2, 3], "area": 3}]
    np.testing.assert_array_equal(rle_decode(encoded[0]), mask[0])


def test_bitmasks_from_roi_masks_uses_explicit_boxes() -> None:
    roi_masks = ROIMasks(np.ones((1, 2, 2), dtype=np.float32))

    bitmasks = BitMasks.from_roi_masks(
        roi_masks,
        boxes=np.array([[1.0, 1.0, 3.0, 3.0]], dtype=np.float32),
        height=4,
        width=4,
    )

    expected = np.zeros((1, 4, 4), dtype=bool)
    expected[:, 1:3, 1:3] = True
    np.testing.assert_array_equal(bitmasks.tensor, expected)


def test_keypoint_heatmap_rejects_boolean_size_and_maps_visible_point() -> None:
    keypoints = Keypoints(np.array([[[1.0, 1.0, 2.0]]], dtype=np.float32))
    boxes = np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32)

    linear, valid = keypoints.to_heatmap(boxes, 2)

    np.testing.assert_array_equal(linear, np.array([[3]], dtype=np.int64))
    np.testing.assert_array_equal(valid, np.array([[True]], dtype=bool))
    with pytest.raises(TypeError, match="heatmap_size"):
        keypoints.to_heatmap(boxes, True)


def test_agent_color_helpers_keep_palette_contract() -> None:
    color = Color.from_hex("#00ff7f")

    assert color.as_rgb() == (0, 255, 127)
    assert rgb_to_hex(color.as_rgb()) == "#00ff7f"
    assert ColorPalette([color]).by_idx(3) is color


class _NoMaskGenerator:
    def __call__(self, messages: Sequence[Message]) -> str:
        assert messages[0]["role"] == "system"
        return '<tool>{"name":"report_no_mask","parameters":{}}</tool>'


def test_agent_inference_report_no_mask_stays_offline(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (5, 3), "white").save(image_path)

    messages, output, rendered = agent_inference(
        str(image_path),
        "missing object",
        send_generate_request=_NoMaskGenerator(),
        call_sam_service=object(),
        output_dir=tmp_path / "output",
    )

    assert messages[-1]["role"] == "assistant"
    assert output["orig_img_h"] == 3
    assert output["orig_img_w"] == 5
    assert output["pred_masks"] == []
    assert rendered.size == (5, 3)


def test_agent_integer_limits_reject_booleans_before_external_calls(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (1, 1), "white").save(image_path)

    with pytest.raises(TypeError, match="max_generations"):
        agent_inference(
            str(image_path),
            "object",
            send_generate_request=_NoMaskGenerator(),
            call_sam_service=object(),
            max_generations=True,
            output_dir=tmp_path / "output",
        )
    with pytest.raises(TypeError, match="max_tokens"):
        send_generate_request([], max_tokens=True)
