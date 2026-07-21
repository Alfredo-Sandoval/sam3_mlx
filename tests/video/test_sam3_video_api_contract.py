import sys

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.io_utils import (
    load_resource_as_video_frames,
    load_video_frames_from_video_file,
    load_video_frames_from_video_file_using_cv2,
)
from sam3_mlx.model.sam3_video_inference import Sam3VideoInference


def test_single_image_path_loads_as_one_frame_video(tmp_path):
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), "red").save(image_path)

    frames = load_resource_as_video_frames(image_path, image_size=14)

    assert len(frames) == 1
    assert (frames.orig_height, frames.orig_width) == (3, 4)
    assert frames.frame_paths == (image_path,)
    assert frames.images is not None
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
    assert frames.images is not None
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


def test_selected_frame_runtime_does_not_materialize_unused_tensor_stack():
    model = Sam3VideoInference(image_model=object(), image_size=14)

    state = model.init_state("<load-dummy-video-2>")

    assert state["frames"].images is None
    assert len(state["frames"]) == 2


@pytest.mark.parametrize(
    "offload_name",
    ["offload_video_to_cpu", "offload_state_to_cpu"],
)
def test_selected_frame_runtime_rejects_unimplemented_offload_modes(offload_name):
    model = Sam3VideoInference(image_model=object(), image_size=14)

    with pytest.raises(Sam3MlxUnsupportedError, match=offload_name) as exc:
        if offload_name == "offload_video_to_cpu":
            model.init_state("<load-dummy-video-1>", offload_video_to_cpu=True)
        else:
            model.init_state("<load-dummy-video-1>", offload_state_to_cpu=True)

    assert exc.value.reason == "video-offload"


def test_encoded_video_rejects_silently_ignored_async_loading():
    with pytest.raises(Sam3MlxUnsupportedError, match="async_loading_frames") as exc:
        load_video_frames_from_video_file(
            "video.mp4",
            image_size=14,
            async_loading_frames=True,
        )

    assert exc.value.reason == "video-async-loading"


def test_cv2_selected_frame_loader_decodes_only_requested_frames(monkeypatch):
    class FakeCapture:
        instances = []

        def __init__(self, path):
            self.path = path
            self.position = 0
            self.read_count = 0
            self.released = False
            self.instances.append(self)

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                fake_cv2.CAP_PROP_FRAME_HEIGHT: 2,
                fake_cv2.CAP_PROP_FRAME_WIDTH: 3,
                fake_cv2.CAP_PROP_FRAME_COUNT: 3,
            }[prop]

        def set(self, prop, value):
            assert prop == fake_cv2.CAP_PROP_POS_FRAMES
            self.position = int(value)
            return True

        def read(self):
            self.read_count += 1
            frame = np.empty((2, 3, 3), dtype=np.uint8)
            frame[:] = [self.position, 10, 20]
            return True, frame

        def release(self):
            self.released = True

    class FakeCv2:
        CAP_PROP_FRAME_HEIGHT = 1
        CAP_PROP_FRAME_WIDTH = 2
        CAP_PROP_FRAME_COUNT = 3
        CAP_PROP_POS_FRAMES = 4
        COLOR_BGR2RGB = 5
        INTER_CUBIC = 6
        VideoCapture = FakeCapture

        @staticmethod
        def cvtColor(frame, code):
            assert code == FakeCv2.COLOR_BGR2RGB
            return frame[..., ::-1]

    fake_cv2 = FakeCv2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    frames = load_video_frames_from_video_file_using_cv2(
        "video.mp4",
        image_size=14,
        materialize_images=False,
    )

    assert frames.images is None
    assert len(frames) == 3
    assert sum(instance.read_count for instance in FakeCapture.instances) == 0
    assert frames[1].getpixel((0, 0)) == (20, 10, 1)
    assert sum(instance.read_count for instance in FakeCapture.instances) == 1
