# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""MLX port of ``sam3.model.utils.sam2_utils`` from the official SAM3 tree."""

from __future__ import annotations

import os
from threading import Thread

import numpy as np
from PIL import Image

import mlx.core as mx

from sam3_mlx._unsupported import raise_unsupported


try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - progress display is optional.

    def tqdm(iterable, **kwargs):
        del kwargs
        return iterable


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
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype != np.uint8:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = (mx.array(img_np, dtype=mx.float32) / 255.0).transpose(2, 0, 1)
    video_width, video_height = img_pil.size
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    A list of video frames loaded asynchronously without blocking session start.

    This mirrors the official JPEG-folder loader, but returns normalized MLX
    arrays and does not perform PyTorch device transfers.
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
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.images = [None] * len(img_paths)
        self.exception = None
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device

        self.__getitem__(0)

        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        img = (img - self.img_mean) / self.img_std
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


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
    """
    Load video frames from a path following the official SAM3 utility contract.

    JPEG folders are ported to MLX arrays. Direct video-file decoding in the
    official helper depends on Decord's PyTorch bridge, so this MLX port fails
    explicitly for those inputs instead of silently using Torch.
    """
    del video_loader_type
    _validate_compute_device(compute_device)
    _validate_offload_video_to_cpu(offload_video_to_cpu)
    is_bytes = isinstance(video_path, bytes)
    is_str = isinstance(video_path, str)
    is_mp4_path = is_str and os.path.splitext(video_path)[-1] in [".mp4", ".MP4"]
    if is_bytes or is_mp4_path:
        return load_video_frames_from_video_file(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            compute_device=compute_device,
        )
    if is_str and os.path.isdir(video_path):
        return load_video_frames_from_jpg_images(
            video_path=video_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
    raise_unsupported(
        "sam3_mlx.model.utils.sam2_utils.load_video_frames(unsupported_resource)",
        reason="torchcodec",
        detail="Only MP4 video and JPEG folders are supported by this MLX helper.",
        alternative="a JPEG-frame folder or the repo-local io_utils loader",
    )


def load_video_frames_from_jpg_images(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False,
    compute_device=None,
):
    _validate_compute_device(compute_device)
    _validate_offload_video_to_cpu(offload_video_to_cpu)
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise_unsupported(
            "sam3_mlx.model.utils.sam2_utils.load_video_frames_from_jpg_images(non_directory)",
            reason="torchcodec",
            detail="Only JPEG-frame directories are supported by this MLX helper.",
            alternative=(
                "Extract frames with ffmpeg, e.g. "
                "ffmpeg -i <video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'"
            ),
        )

    frame_names = [
        p
        for p in os.listdir(jpg_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = mx.array(img_mean, dtype=mx.float32).reshape(3, 1, 1)
    img_std = mx.array(img_std, dtype=mx.float32).reshape(3, 1, 1)

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = []
    for img_path in tqdm(img_paths, desc="frame loading (JPEG)"):
        img, video_height, video_width = _load_img_as_tensor(img_path, image_size)
        images.append(img)
    images = mx.stack(images, axis=0)
    images = (images - img_mean) / img_std
    return images, video_height, video_width


def load_video_frames_from_video_file(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    compute_device=None,
    video_loader_type="cv2",
):
    del image_size, offload_video_to_cpu, img_mean, img_std, video_loader_type
    _validate_compute_device(compute_device)
    raise_unsupported(
        "sam3_mlx.model.utils.sam2_utils.load_video_frames_from_video_file",
        reason="torchcodec",
        detail=(
            "Direct video-file decoding uses Decord's PyTorch bridge upstream and "
            f"is not ported. Extract JPEG frames instead: {video_path!r}."
        ),
        alternative="a JPEG-frame folder via load_video_frames_from_jpg_images",
    )
