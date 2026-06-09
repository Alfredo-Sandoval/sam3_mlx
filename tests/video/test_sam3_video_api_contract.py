import numpy as np
import pytest
import mlx.core as mx
from PIL import Image

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.io_utils import load_resource_as_video_frames


def test_single_image_path_loads_as_one_frame_video(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), "red").save(image_path)

    frames = load_resource_as_video_frames(image_path, image_size=14)

    assert len(frames) == 1
    assert (frames.orig_height, frames.orig_width) == (3, 4)
    assert frames.frame_paths == (image_path,)
    assert frames.images.shape == (1, 3, 14, 14)
    assert frames.images.dtype == mx.float32


def test_image_folder_frames_follow_official_numeric_sort(tmp_path):
    Image.new("RGB", (8, 6), "red").save(tmp_path / "10.jpg")
    Image.new("RGB", (8, 6), "blue").save(tmp_path / "2.jpg")
    Image.new("RGB", (8, 6), "green").save(tmp_path / "1.jpg")

    frames = load_resource_as_video_frames(tmp_path, image_size=1008)

    assert [path.name for path in frames.frame_paths] == ["1.jpg", "2.jpg", "10.jpg"]
    assert len(frames) == 3
    assert (frames.orig_height, frames.orig_width) == (6, 8)


def test_pil_sequence_loads_as_video_frames_without_frame_paths():
    frames = load_resource_as_video_frames(
        [
            Image.new("RGB", (3, 2), "red"),
            Image.new("RGB", (3, 2), "blue"),
        ],
        image_size=14,
    )

    assert len(frames) == 2
    assert (frames.orig_height, frames.orig_width) == (2, 3)
    assert frames.frame_paths == ()
    assert frames.images.shape == (2, 3, 14, 14)
    assert np.isfinite(to_numpy(frames.images)).all()


def test_pil_sequence_preserves_normalized_rgb_channel_values():
    frames = load_resource_as_video_frames(
        [Image.new("RGB", (1, 1), (255, 128, 0))],
        image_size=1,
    )

    expected_mid = np.float16(np.float16(128.0 / 255.0) - np.float16(0.5)) / np.float16(
        0.5
    )
    expected = np.array([[[[1.0]], [[expected_mid]], [[-1.0]]]], dtype=np.float32)
    np.testing.assert_allclose(to_numpy(frames.images), expected, rtol=0, atol=1e-7)


def test_unknown_resource_fails_fast_with_path_context(tmp_path):
    missing = tmp_path / "not-a-video.resource"

    with pytest.raises(Sam3MlxUnsupportedError, match="not-a-video.resource") as exc:
        load_resource_as_video_frames(missing, image_size=14)

    assert exc.value.reason == "torchcodec"
    assert "unknown_resource" in exc.value.feature
