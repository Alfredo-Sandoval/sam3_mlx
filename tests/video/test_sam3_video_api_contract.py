from pathlib import Path
import sys
from types import ModuleType
from typing import Never, cast

import mlx.core as mx
import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.mlx_runtime import to_numpy
from sam3_mlx.model.io_utils import load_resource_as_video_frames


def _bounded_cache_state(value: object) -> tuple[dict[object, object], int]:
    cache: object = getattr(value, "_cache", None)
    cache_size: object = getattr(value, "_cache_size", None)
    if not isinstance(cache, dict):
        raise AssertionError("lazy frame cache must be a dictionary")
    if isinstance(cache_size, bool) or not isinstance(cache_size, int):
        raise AssertionError("lazy frame cache size must be an integer")
    return cast(dict[object, object], cache), cache_size


def test_single_image_path_loads_as_one_frame_video(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), "red").save(image_path)

    frames = load_resource_as_video_frames(image_path, image_size=14)

    assert len(frames) == 1
    assert (frames.orig_height, frames.orig_width) == (3, 4)
    assert frames.frame_paths == (image_path,)
    assert frames.images is not None
    assert frames.images.shape == (1, 3, 14, 14)
    assert frames.images.dtype == mx.float32


def test_image_folder_frames_follow_official_numeric_sort(tmp_path: Path) -> None:
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


def test_image_folder_rejects_mixed_frame_dimensions(tmp_path: Path) -> None:
    Image.new("RGB", (8, 6), "red").save(tmp_path / "1.jpg")
    Image.new("RGB", (4, 3), "blue").save(tmp_path / "2.jpg")

    with pytest.raises(ValueError, match="mixed frame dimensions"):
        load_resource_as_video_frames(tmp_path, image_size=14)


def test_pil_sequence_rejects_mixed_frame_dimensions():
    with pytest.raises(ValueError, match="mixed frame dimensions"):
        load_resource_as_video_frames(
            [
                Image.new("RGB", (8, 6), "red"),
                Image.new("RGB", (4, 3), "blue"),
            ],
            image_size=14,
        )


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


def test_unknown_resource_fails_fast_with_path_context(tmp_path: Path) -> None:
    missing = tmp_path / "not-a-video.resource"

    with pytest.raises(Sam3MlxUnsupportedError, match="not-a-video.resource") as exc:
        load_resource_as_video_frames(missing, image_size=14)

    assert exc.value.reason == "port-gap"
    assert "unknown_resource" in exc.value.feature


def test_async_image_frame_loader_close_stops_background_thread(tmp_path: Path) -> None:
    from sam3_mlx.model.io_utils import AsyncImageFrameLoader

    for index in range(8):
        Image.new("RGB", (4, 3), "red").save(tmp_path / f"{index:02d}.jpg")

    loader = AsyncImageFrameLoader(
        sorted(tmp_path.glob("*.jpg")),
        image_size=14,
        offload_video_to_cpu=False,
        materialize_mlx_frames=False,
    )
    loader.close(join_timeout=2.0)

    assert loader.thread is None or not loader.thread.is_alive()
    with pytest.raises(RuntimeError, match="closed"):
        loader[0]


def test_selected_frame_image_folder_uses_bounded_lazy_host_cache(
    tmp_path: Path,
) -> None:
    from sam3_mlx.model.io_utils import (
        LazyImageFolderFrames,
        load_resource_as_video_frames,
    )

    for index in range(12):
        Image.new("RGB", (6, 4), (index, 0, 0)).save(tmp_path / f"{index:02d}.jpg")

    frames = load_resource_as_video_frames(
        tmp_path,
        image_size=14,
        materialize_mlx_frames=False,
    )
    assert isinstance(frames, LazyImageFolderFrames)
    assert len(frames) == 12
    assert frames.images is None
    assert (frames.orig_height, frames.orig_width) == (4, 6)

    # Access more frames than the default cache size; cache stays bounded.
    for index in range(12):
        frame = frames[index]
        assert frame.size == (6, 4)
    cache, cache_size = _bounded_cache_state(frames)
    assert len(cache) <= cache_size

    frames.close()
    with pytest.raises(RuntimeError, match="closed"):
        frames[0]


def test_selected_frame_video_file_uses_bounded_indexed_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sam3_mlx.model.io_utils import LazyVideoFileFrames

    decoded = [
        np.full((3, 4, 3), fill_value=index, dtype=np.uint8) for index in range(12)
    ]

    class _Capture:
        def __init__(self, path: str) -> None:
            self.path = path
            self.position = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, field: int) -> float:
            return float(
                {
                    1: len(decoded),
                    2: 3,
                    3: 4,
                }[field]
            )

        def set(self, field: int, value: float) -> None:
            assert field == 4
            self.position = int(value)

        def read(self) -> tuple[bool, NDArray[np.uint8]]:
            frame = decoded[self.position]
            self.position += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    def _cvt_color(frame: NDArray[np.uint8], code: int) -> NDArray[np.uint8]:
        assert code == 5
        return frame

    fake_cv2 = ModuleType("cv2")
    setattr(fake_cv2, "VideoCapture", _Capture)
    setattr(fake_cv2, "CAP_PROP_FRAME_COUNT", 1)
    setattr(fake_cv2, "CAP_PROP_FRAME_HEIGHT", 2)
    setattr(fake_cv2, "CAP_PROP_FRAME_WIDTH", 3)
    setattr(fake_cv2, "CAP_PROP_POS_FRAMES", 4)
    setattr(fake_cv2, "COLOR_BGR2RGB", 5)
    setattr(fake_cv2, "cvtColor", _cvt_color)
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    frames = load_resource_as_video_frames(
        tmp_path / "video.mp4",
        image_size=14,
        materialize_mlx_frames=False,
    )
    assert isinstance(frames, LazyVideoFileFrames)
    setattr(frames, "_cache_size", 3)
    for index in range(len(frames)):
        assert np.asarray(frames[index])[0, 0, 0] == index
    cache, cache_size = _bounded_cache_state(frames)
    assert cache_size == 3
    assert len(cache) == 3
    assert 0 not in cache
    frames.close()
    with pytest.raises(RuntimeError, match="closed"):
        frames[0]


def test_invalid_video_loader_type_rejected_before_resource_dispatch(
    tmp_path: Path,
) -> None:
    Image.new("RGB", (4, 3), "red").save(tmp_path / "0.jpg")

    with pytest.raises(RuntimeError, match="video_loader_type"):
        load_resource_as_video_frames(
            tmp_path,
            image_size=14,
            video_loader_type="not-a-backend",
        )

    with pytest.raises(RuntimeError, match="video_loader_type"):
        load_resource_as_video_frames(
            "<load-dummy-video-2>",
            image_size=14,
            video_loader_type="decord",
        )


def test_unknown_resource_preserves_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked = tmp_path / "secret-no-ext"
    blocked.write_bytes(b"not-a-video")

    real_open = Path.exists

    def _exists_then_permission(self: Path) -> bool:
        # Force the extensionless path to look present so the loader attempts
        # decoding rather than the missing-resource unsupported surface.
        if self == blocked:
            return True
        return real_open(self)

    monkeypatch.setattr(Path, "exists", _exists_then_permission)

    def _raise_permission(*_args: object, **_kwargs: object) -> Never:
        raise PermissionError("permission denied for test resource")

    monkeypatch.setattr(
        "sam3_mlx.model.io_utils.load_video_frames_from_video_file",
        _raise_permission,
    )

    with pytest.raises(PermissionError, match="permission denied"):
        load_resource_as_video_frames(blocked, image_size=14)


def test_legacy_and_canonical_loaders_share_frame_order_and_pixels(
    tmp_path: Path,
) -> None:
    from sam3_mlx.model.utils import sam2_utils

    Image.new("RGB", (8, 6), "red").save(tmp_path / "10.jpg")
    Image.new("RGB", (8, 6), "blue").save(tmp_path / "2.jpg")
    Image.new("RGB", (8, 6), "green").save(tmp_path / "1.jpg")

    canonical = load_resource_as_video_frames(tmp_path, image_size=14)
    legacy_images, legacy_h, legacy_w = sam2_utils.load_video_frames(
        str(tmp_path),
        image_size=14,
        offload_video_to_cpu=False,
    )

    assert [path.name for path in canonical.frame_paths] == ["1.jpg", "2.jpg", "10.jpg"]
    assert canonical.images is not None
    assert isinstance(legacy_images, mx.array)
    assert (legacy_h, legacy_w) == (canonical.orig_height, canonical.orig_width)
    assert legacy_images.shape == canonical.images.shape
    np.testing.assert_allclose(
        to_numpy(legacy_images),
        to_numpy(canonical.images),
        rtol=0,
        atol=1e-6,
    )
