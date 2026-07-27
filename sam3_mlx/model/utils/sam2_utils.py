# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Legacy SAM2-style video loading helpers.

This module is a thin compatibility adapter over
``sam3_mlx.model.io_utils``. New code should import the canonical loaders from
``io_utils`` directly. The public return shape remains the official
``(images, video_height, video_width)`` triple for folder/video loads.
"""

from __future__ import annotations

from pathlib import Path
from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model import io_utils


def _validate_compute_device(compute_device) -> None:
    if compute_device not in (None, "mlx"):
        raise_unsupported(
            f"sam3_mlx.model.utils.sam2_utils._validate_compute_device(compute_device={compute_device!r})",
            reason="unsupported-device",
            detail="sam3_mlx targets the explicit MLX runtime; pass compute_device='mlx' or None.",
        )


def _validate_offload_video_to_cpu(offload_video_to_cpu) -> None:
    if offload_video_to_cpu not in (False, True):
        raise TypeError("offload_video_to_cpu must be a bool.")


def _load_img_as_tensor(img_path, image_size):
    return io_utils._load_img_as_tensor(img_path, image_size)


class AsyncVideoFrameLoader:
    """Legacy async JPEG-folder loader returning normalized MLX CHW tensors.

    Prefer ``io_utils.AsyncImageFrameLoader`` for new selected-frame sessions.
    """

    def __init__(
        self,
        img_paths,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        compute_device,
    ):
        _validate_compute_device(compute_device)
        _validate_offload_video_to_cpu(offload_video_to_cpu)
        self._loader = io_utils.AsyncImageFrameLoader(
            [Path(path) for path in img_paths],
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=tuple(img_mean),
            img_std=tuple(img_std),
            materialize_mlx_frames=True,
        )
        self.img_paths = list(img_paths)
        self.video_height = self._loader.video_height
        self.video_width = self._loader.video_width
        self.compute_device = compute_device

    def __getitem__(self, index):
        return self._loader.get_image_tensor(index)

    def __len__(self):
        return len(self._loader)

    @property
    def images(self):
        return [self[index] for index in range(len(self))]

    @property
    def thread(self):
        return self._loader.thread

    def close(self, *, join_timeout: float | None = None) -> None:
        self._loader.close(join_timeout=join_timeout)


def _as_legacy_triple(video_frames):
    """Convert a canonical ``VideoFrames`` object to the legacy return triple."""
    if video_frames.images is None:
        raise RuntimeError(
            "Legacy sam2_utils loaders require materialize_mlx_frames=True so "
            "normalized tensors are available."
        )
    return video_frames.images, video_frames.orig_height, video_frames.orig_width


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    compute_device=None,
    video_loader_type="cv2",
):
    """Load video frames via the canonical ``io_utils`` stack.

    Returns the legacy ``(images, video_height, video_width)`` triple (or an
    async loader triple) rather than a ``VideoFrames`` dataclass.
    """
    _validate_compute_device(compute_device)
    _validate_offload_video_to_cpu(offload_video_to_cpu)
    io_utils._validate_video_loader_type(video_loader_type)

    if isinstance(video_path, (str, Path)) and Path(video_path).is_dir():
        return load_video_frames_from_jpg_images(
            video_path=str(video_path),
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )

    # Video files and other path types: delegate to the canonical OpenCV path.
    frames = io_utils.load_video_frames(
        video_path=video_path,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        async_loading_frames=async_loading_frames,
        video_loader_type=video_loader_type,
        materialize_mlx_frames=True,
    )
    return _as_legacy_triple(frames)


def load_video_frames_from_jpg_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    compute_device=None,
):
    """Load an image folder, preserving the legacy tensor-triple return shape."""
    _validate_compute_device(compute_device)
    _validate_offload_video_to_cpu(offload_video_to_cpu)
    if not (isinstance(video_path, str) and Path(video_path).is_dir()):
        raise_unsupported(
            "sam3_mlx.model.utils.sam2_utils.load_video_frames_from_jpg_images(non_directory)",
            reason="port-gap",
            detail="Only image-frame directories are supported by this legacy helper.",
            alternative="sam3_mlx.model.io_utils.load_resource_as_video_frames",
        )

    if async_loading_frames:
        # Restrict to JPEG-like names for legacy sorting semantics when possible,
        # but fall through to the canonical path list for broader formats.
        folder = Path(video_path)
        frame_paths = [
            p
            for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
        ]
        if not frame_paths:
            raise RuntimeError(f"no images found in {folder}")
        frame_paths = io_utils._sort_frame_paths(frame_paths)
        loader = AsyncVideoFrameLoader(
            [str(p) for p in frame_paths],
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
        )
        return loader, loader.video_height, loader.video_width

    frames = io_utils.load_video_frames_from_image_folder(
        video_path,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        async_loading_frames=False,
        materialize_mlx_frames=True,
    )
    return _as_legacy_triple(frames)


def load_video_frames_from_video_file(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    compute_device=None,
    video_loader_type="cv2",
):
    """Decode a video file through the canonical OpenCV loader."""
    _validate_compute_device(compute_device)
    _validate_offload_video_to_cpu(offload_video_to_cpu)
    frames = io_utils.load_video_frames_from_video_file(
        video_path=video_path,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        video_loader_type=video_loader_type,
        materialize_mlx_frames=True,
    )
    return _as_legacy_triple(frames)
