from __future__ import annotations

import queue
from dataclasses import dataclass
import os
from pathlib import Path
import re
from threading import Condition, Lock, Thread, get_ident
from types import TracebackType
from typing import Any, Sequence

import numpy as np
from PIL import Image

import mlx.core as mx

from sam3_mlx._unsupported import raise_unsupported


def _raise_io_unsupported(feature: str, *, reason: str, detail: str, alternative=None):
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
        alternative=alternative,
    )


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _uint8_hwc_to_mlx_chw_float32(array: np.ndarray):
    """Move RGB uint8 pixels into MLX as normalized CHW float32."""
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HWC RGB uint8 image array.")
    return (mx.array(array, dtype=mx.float32) / 255.0).transpose(2, 0, 1)


def _uint8_hwc_to_official_normalized_mlx_chw_float32(
    array: np.ndarray,
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
):
    """Mirror official SAM3's float16 image-storage normalization."""
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an HWC RGB uint8 image array.")
    image = (array.astype(np.float32) / 255.0).astype(np.float16)
    chw = np.transpose(image, (2, 0, 1))
    mean = np.asarray(img_mean, dtype=np.float16).reshape(3, 1, 1)
    std = np.asarray(img_std, dtype=np.float16).reshape(3, 1, 1)
    normalized = ((chw - mean) / std).astype(np.float16)
    return mx.array(normalized.astype(np.float32), dtype=mx.float32)


@dataclass(frozen=True)
class VideoFrames:
    frames: tuple[Image.Image, ...]
    orig_height: int
    orig_width: int
    frame_paths: tuple[Path, ...] = ()
    images: Any | None = None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Image.Image:
        return self.frames[index]


def load_resource_as_video_frames(
    resource_path: str | Path | Sequence[Image.Image],
    image_size: int,
    offload_video_to_cpu: bool = False,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
    video_loader_type: str = "cv2",
) -> VideoFrames | AsyncImageFrameLoader:
    """Load image-safe resources into host frames plus normalized MLX arrays.

    The official Torch path returns normalized tensors. The MLX port keeps RGB
    PIL frames available for the selected-frame processor while exposing the
    image tensor stack explicitly as ``VideoFrames.images``.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if offload_video_to_cpu not in (False, True):
        raise TypeError("offload_video_to_cpu must be a bool.")

    if isinstance(resource_path, Sequence) and not isinstance(
        resource_path, (str, bytes, Path)
    ):
        return _load_pil_sequence(
            resource_path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
        )

    resource_path_str = os.fspath(resource_path)
    path = Path(resource_path_str)
    if path.suffix.lower() in IMAGE_EXTS:
        return load_image_as_single_frame_video(
            image_path=path,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
        )
    return load_video_frames(
        video_path=resource_path_str,
        image_size=image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean,
        img_std=img_std,
        async_loading_frames=async_loading_frames,
        video_loader_type=video_loader_type,
    )


def load_image_as_single_frame_video(
    image_path: str | Path,
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> VideoFrames:
    """Load an image path using the official single-frame video contract."""
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    path = Path(image_path)
    frame, tensor = _load_rgb_image_and_tensor(
        path,
        image_size=image_size,
        img_mean=img_mean,
        img_std=img_std,
    )
    return VideoFrames(
        frames=(frame,),
        orig_height=frame.height,
        orig_width=frame.width,
        frame_paths=(path,),
        images=tensor[None, ...],
    )


def load_video_frames(
    video_path: str | os.PathLike[str],
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
    video_loader_type: str = "cv2",
) -> VideoFrames | AsyncImageFrameLoader:
    """Route video-like resources following the official SAM3 loader contract."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if offload_video_to_cpu not in (False, True):
        raise TypeError("offload_video_to_cpu must be a bool.")

    video_path_str = os.fspath(video_path)
    dummy_match = re.fullmatch(r"<load-(dummy|zero)-video-(\d+)>", video_path_str)
    if dummy_match is not None:
        kind, num_frames_text = dummy_match.groups()
        return load_dummy_video(
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            num_frames=int(num_frames_text),
            do_zeros=kind == "zero",
        )
    if video_path_str.startswith("<load-dummy-video"):
        return load_dummy_video(
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            num_frames=60,
        )
    if video_path_str.startswith("<load-zero-video"):
        return load_dummy_video(
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            num_frames=60,
            do_zeros=True,
        )

    if os.path.isdir(video_path_str):
        return load_video_frames_from_image_folder(
            video_path_str,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
        )

    suffix = os.path.splitext(video_path_str)[-1].lower()
    if suffix in VIDEO_EXTS:
        return load_video_frames_from_video_file(
            video_path_str,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )

    try:
        return load_video_frames_from_video_file(
            video_path_str,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
    except Exception as exc:
        _raise_io_unsupported(
            "sam3_mlx.model.io_utils.load_video_frames(unknown_resource)",
            reason="torchcodec",
            detail=(
                "Only video files and image folders are supported; "
                f"failed to load {video_path_str!r} as video: {exc}"
            ),
            alternative="Use an image path, image folder, PIL sequence, or cv2-decodable video.",
        )


def load_dummy_video(
    image_size: int,
    offload_video_to_cpu: bool = False,
    num_frames: int = 60,
    do_zeros: bool = False,
) -> VideoFrames:
    """Return deterministic dummy frames for API tests and warmup paths."""
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative.")

    video_height, video_width = 480, 640
    if do_zeros:
        arrays = np.zeros((num_frames, image_size, image_size, 3), dtype=np.uint8)
    else:
        rng = np.random.default_rng(0)
        arrays = rng.integers(
            0,
            256,
            size=(num_frames, image_size, image_size, 3),
            dtype=np.uint8,
        )
    frames = tuple(Image.fromarray(array) for array in arrays)
    if num_frames:
        images = mx.stack(
            [_pil_to_mlx_image(frame, image_size) for frame in frames],
            axis=0,
        )
    else:
        images = mx.zeros((0, 3, image_size, image_size), dtype=mx.float32)
    return VideoFrames(
        frames=frames,
        orig_height=video_height,
        orig_width=video_width,
        images=images,
    )


def load_video_frames_from_image_folder(
    image_folder: str | Path,
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
) -> VideoFrames | AsyncImageFrameLoader:
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    folder = Path(image_folder)
    if not folder.is_dir():
        raise NotADirectoryError(
            f"Image-folder video path is not a directory: {folder}"
        )

    frame_paths = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not frame_paths:
        raise RuntimeError(f"no images found in {folder}")
    frame_paths = _sort_frame_paths(frame_paths)
    if async_loading_frames:
        return AsyncImageFrameLoader(
            frame_paths,
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
        )
    payloads = [
        _load_rgb_image_and_tensor(
            path,
            image_size=image_size,
            img_mean=img_mean,
            img_std=img_std,
        )
        for path in frame_paths
    ]
    frames = tuple(frame for frame, _ in payloads)
    images = mx.stack([tensor for _, tensor in payloads], axis=0)
    first = frames[0]
    return VideoFrames(
        frames=frames,
        orig_height=first.height,
        orig_width=first.width,
        frame_paths=tuple(frame_paths),
        images=images,
    )


def load_video_frames_from_video_file(
    video_path: str | Path,
    image_size: int = 1008,
    offload_video_to_cpu: bool = False,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    async_loading_frames: bool = False,
    gpu_acceleration: bool = False,
    gpu_device: Any | None = None,
    video_loader_type: str = "cv2",
) -> VideoFrames:
    """Load frames from a video file using an explicitly supported backend."""
    del async_loading_frames
    if video_loader_type == "cv2":
        return load_video_frames_from_video_file_using_cv2(
            video_path=str(video_path),
            image_size=image_size,
            img_mean=img_mean,
            img_std=img_std,
            offload_video_to_cpu=offload_video_to_cpu,
        )
    if video_loader_type == "torchcodec":
        return AsyncVideoFileLoaderWithTorchCodec(
            video_path=str(video_path),
            image_size=image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=img_mean,
            img_std=img_std,
            gpu_acceleration=gpu_acceleration,
            gpu_device=gpu_device,
        )
    raise RuntimeError("video_loader_type must be either 'cv2' or 'torchcodec'")


def load_video_frames_from_video_file_using_cv2(
    video_path: str,
    image_size: int,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    offload_video_to_cpu: bool = False,
) -> VideoFrames:
    """Decode a video file with OpenCV and expose normalized MLX image tensors."""
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    try:
        import cv2
    except ModuleNotFoundError:
        _raise_io_unsupported(
            "sam3_mlx.model.io_utils.load_video_frames_from_video_file_using_cv2",
            reason="torchcodec",
            detail="Video-file decoding requires optional OpenCV support.",
            alternative="Use an image folder or install OpenCV for the MLX video loader.",
        )

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frames: list[Image.Image] = []
    tensors: list[Any] = []
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    mean, std = _mean_std_arrays(img_mean, img_std)
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb).copy())
            frame_resized = cv2.resize(
                frame_rgb,
                (image_size, image_size),
                interpolation=cv2.INTER_CUBIC,
            )
            tensor = _uint8_hwc_to_mlx_chw_float32(frame_resized)
            tensors.append((tensor - mean) / std)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames could be decoded from video: {path}. "
            "The file may be empty, corrupted, or encoded with an unsupported codec."
        )
    if orig_height <= 0 or orig_width <= 0:
        first = frames[0]
        orig_height, orig_width = first.height, first.width
    return VideoFrames(
        frames=tuple(frames),
        orig_height=orig_height,
        orig_width=orig_width,
        frame_paths=(path,),
        images=mx.stack(tensors, axis=0),
    )


class AsyncImageFrameLoader:
    """Image-folder frame loader matching the official async session contract."""

    def __init__(
        self,
        frame_paths: Sequence[Path],
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    ) -> None:
        if not frame_paths:
            raise RuntimeError("no images found in image-folder video")
        _validate_image_loading_args(image_size, offload_video_to_cpu)
        self.frame_paths = tuple(frame_paths)
        self.image_size = image_size
        self.img_mean = img_mean
        self.img_std = img_std
        self._frames: list[Image.Image | None] = [None] * len(self.frame_paths)
        self._image_tensors: list[Any | None] = [None] * len(self.frame_paths)
        self._lock = Lock()
        self.exception: BaseException | None = None

        first = self.__getitem__(0)
        self.orig_height = first.height
        self.orig_width = first.width
        self.video_height = self.orig_height
        self.video_width = self.orig_width

        def _load_frames() -> None:
            try:
                for index in range(len(self.frame_paths)):
                    self.__getitem__(index)
            except BaseException as exc:
                self.exception = exc

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, index: int) -> Image.Image:
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception
        with self._lock:
            frame = self._frames[index]
            if frame is None:
                frame, tensor = _load_rgb_image_and_tensor(
                    self.frame_paths[index],
                    image_size=self.image_size,
                    img_mean=self.img_mean,
                    img_std=self.img_std,
                )
                self._frames[index] = frame
                self._image_tensors[index] = tensor
            return frame

    @property
    def frames(self) -> tuple[Image.Image, ...]:
        self.wait()
        return tuple(frame for frame in self._frames if frame is not None)

    @property
    def images(self):
        self.wait()
        tensors = [tensor for tensor in self._image_tensors if tensor is not None]
        if len(tensors) != len(self._image_tensors):
            raise RuntimeError("not all image-folder frames were loaded")
        return mx.stack(tensors, axis=0)

    def wait(self) -> None:
        thread = self.thread
        if thread is not None:
            thread.join()
            self.thread = None
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

    def __getstate__(self) -> dict[str, Any]:
        self.wait()
        state = self.__dict__.copy()
        state["_lock"] = None
        state["thread"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = Lock()


class TorchCodecDecoder:
    """Fail-fast placeholder for the official TorchCodec decoder surface."""

    def __init__(
        self,
        source: str | bytes,
        dimension_order: str = "NCHW",
        device: str = "cpu",
        num_threads: int = 1,
    ) -> None:
        del source, dimension_order, device, num_threads
        _raise_io_unsupported(
            "sam3_mlx.model.io_utils.TorchCodecDecoder",
            reason="torchcodec",
            detail=(
                "TorchCodec video decoding is a Torch-only surface in official SAM3. "
                "The MLX port supports video-file decoding through OpenCV only."
            ),
            alternative="video_loader_type='cv2'",
        )


class FIFOLock:
    """A lock that serves waiting threads in acquisition order."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._waiters: queue.Queue[int] = queue.Queue()
        self._condition = Condition()

    def acquire(self) -> None:
        ident = get_ident()
        with self._condition:
            self._waiters.put(ident)
            while self._waiters.queue[0] != ident or not self._lock.acquire(
                blocking=False
            ):
                self._condition.wait()

    def release(self) -> None:
        with self._condition:
            self._lock.release()
            self._waiters.get()
            self._condition.notify_all()

    def __enter__(self) -> FIFOLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.release()


class AsyncVideoFileLoaderWithTorchCodec:
    """Official TorchCodec async loader surface, unavailable in the MLX port."""

    def __init__(
        self,
        video_path: str,
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean: tuple[float, float, float] | Any,
        img_std: tuple[float, float, float] | Any,
        gpu_acceleration: bool = True,
        gpu_device: Any | None = None,
        use_rand_seek_in_loading: bool = False,
    ) -> None:
        del (
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            gpu_acceleration,
            gpu_device,
            use_rand_seek_in_loading,
        )
        _raise_io_unsupported(
            "sam3_mlx.model.io_utils.AsyncVideoFrameLoader",
            reason="torchcodec",
            detail=(
                "video_loader_type='torchcodec' depends on official SAM3's "
                "TorchCodec/Torch-only path and is not implemented for Apple "
                "Silicon MLX."
            ),
            alternative="video_loader_type='cv2' or an image folder",
        )


def _load_pil_sequence(
    images: Sequence[Image.Image],
    image_size: int,
    offload_video_to_cpu: bool,
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
) -> VideoFrames:
    _validate_image_loading_args(image_size, offload_video_to_cpu)
    if not images:
        raise RuntimeError("no images found in PIL image sequence")
    if not all(isinstance(image, Image.Image) for image in images):
        raise TypeError("resource_path image sequences must contain only PIL images.")
    frames = tuple(image.convert("RGB").copy() for image in images)
    image_tensors = mx.stack(
        [
            _pil_to_mlx_image(
                frame,
                image_size=image_size,
                img_mean=img_mean,
                img_std=img_std,
            )
            for frame in frames
        ],
        axis=0,
    )
    first = frames[0]
    return VideoFrames(
        frames=frames,
        orig_height=first.height,
        orig_width=first.width,
        images=image_tensors,
    )


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _load_img_as_tensor(img_path: str | Path, image_size: int) -> tuple[Any, int, int]:
    """Load and resize an image into an unnormalized MLX CHW float32 tensor."""
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    frame = _load_rgb_image(Path(img_path))
    orig_width, orig_height = frame.width, frame.height
    resized = frame.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    tensor = _uint8_hwc_to_mlx_chw_float32(np.asarray(resized))
    return tensor, orig_height, orig_width


def _load_rgb_image_and_tensor(
    path: Path,
    image_size: int,
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
) -> tuple[Image.Image, Any]:
    frame = _load_rgb_image(path)
    return frame, _pil_to_mlx_image(
        frame,
        image_size=image_size,
        img_mean=img_mean,
        img_std=img_std,
    )


def _pil_to_mlx_image(
    image: Image.Image,
    image_size: int,
    img_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    img_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
):
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    resized = rgb_image.resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    return _uint8_hwc_to_official_normalized_mlx_chw_float32(
        np.asarray(resized),
        img_mean,
        img_std,
    )


def _mean_std_arrays(
    img_mean: tuple[float, float, float],
    img_std: tuple[float, float, float],
):
    mean_np = np.asarray(img_mean, dtype=np.float32)
    std_np = np.asarray(img_std, dtype=np.float32)
    if mean_np.shape != (3,) or std_np.shape != (3,):
        raise ValueError("img_mean and img_std must each contain three RGB values.")
    if not np.isfinite(mean_np).all() or not np.isfinite(std_np).all():
        raise ValueError("img_mean and img_std must contain only finite values.")
    if np.any(std_np == 0):
        raise ValueError("img_std values must be non-zero.")
    return (
        mx.array(mean_np, dtype=mx.float32).reshape(3, 1, 1),
        mx.array(std_np, dtype=mx.float32).reshape(3, 1, 1),
    )


def _validate_image_loading_args(
    image_size: int,
    offload_video_to_cpu: bool,
) -> None:
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    if offload_video_to_cpu not in (False, True):
        raise TypeError("offload_video_to_cpu must be a bool.")


def _sort_frame_paths(frame_paths: list[Path]) -> list[Path]:
    try:
        return sorted(frame_paths, key=lambda path: int(path.stem))
    except ValueError:
        return sorted(frame_paths, key=lambda path: path.name)


def masks_to_boxes_xyxy(binary_masks: np.ndarray) -> np.ndarray:
    """Compute pixel-space ``xyxy`` boxes from ``N x H x W`` boolean masks."""
    masks = np.asarray(binary_masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"binary_masks must have shape (N, H, W), got {masks.shape}.")
    boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    for idx, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if xs.size == 0 or ys.size == 0:
            continue
        boxes[idx] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return boxes
