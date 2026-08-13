from functools import lru_cache, partial
from numbers import Integral
from collections.abc import Callable, Mapping
from typing import Any, NoReturn, Protocol, cast
from PIL import Image
import numpy as np
import numpy.typing as npt
import mlx.core as mx

from sam3_mlx._device import is_mlx_runtime_device
from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model import box_ops
from sam3_mlx.model.bounded_cache import BoundedLRUCache
from sam3_mlx.model.data_misc import (
    FindStage,
    IndexArray,
    ResizeWeights,
    WeightArray,
    interpolate,
    reshape_array,
    transpose_array,
)
from sam3_mlx.model.geometry_encoders import Prompt
from sam3_mlx.model.sam3_image import (
    Output,
    RawPrediction,
    raw_prediction_from_output,
)
from sam3_mlx.precision import cast_visual_input, model_precision, parse_precision
from sam3_mlx.resolutions import DEFAULT_IMAGE_RESOLUTION, PATCH_SIZE

SAM3_IMAGE_PATCH_SIZE = PATCH_SIZE
PREPROCESSED_LAYOUTS = ("nchw", "nhwc")
PREPROCESSED_VALUE_CONTRACT = "normalized-minus1-plus1"
DEFAULT_TEXT_CACHE_SIZE = 8


def _raise_processor_unsupported(
    feature: str,
    *,
    reason: str,
    detail: str,
    alternative: str | None = None,
) -> NoReturn:
    raise_unsupported(
        feature,
        reason=reason,
        detail=detail,
        alternative=alternative,
    )


def _score_keep_indices(scores: mx.array, threshold: float) -> mx.array:
    keep = scores > threshold
    # MLX has no boolean indexing/nonzero yet. Sync only the scalar count, then
    # build the ordered variable-length index vector on device.
    keep_count = int(mx.sum(keep).item())
    if keep_count == 0:
        return mx.array([], dtype=mx.int64)
    positions = mx.arange(keep.shape[0], dtype=mx.int64)
    sentinel = mx.array(keep.shape[0], dtype=mx.int64)
    return mx.sort(mx.where(keep, positions, sentinel))[:keep_count]


def _single_image_keep_indices(out_probs: mx.array, threshold: float) -> mx.array:
    return _score_keep_indices(out_probs[0], threshold)


def _presence_weighted_scores(outputs: Mapping[str, object]) -> mx.array:
    out_logits = cast(mx.array, outputs["pred_logits"]).astype(mx.float32)
    presence_logit = cast(mx.array, outputs["presence_logit_dec"]).astype(mx.float32)
    out_probs = mx.sigmoid(out_logits)
    presence_score = mx.sigmoid(presence_logit)[:, None]
    return (out_probs * presence_score).squeeze(-1)


def _filter_single_image_detections(
    out_probs: mx.array,
    out_masks: mx.array,
    out_bbox: mx.array,
    *,
    threshold: float,
) -> tuple[mx.array, mx.array, mx.array]:
    keep_indices = _single_image_keep_indices(out_probs, threshold)
    return (
        out_probs[0][keep_indices],
        out_masks[0][keep_indices],
        out_bbox[0][keep_indices],
    )


def _boxes_to_original_xyxy(
    out_bbox: mx.array, img_h: int, img_w: int
) -> mx.array:
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox.astype(mx.float32))
    scale_fct = mx.array([img_w, img_h, img_w, img_h], dtype=mx.float32)
    return boxes * scale_fct[None, :]


def _upsample_spatial(array: mx.array, img_h: int, img_w: int) -> mx.array:
    return interpolate(
        array,
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False,
    )


def _upsample_and_activate_masks(
    out_masks: mx.array, img_h: int, img_w: int
) -> mx.array:
    return mx.sigmoid(
        _upsample_spatial(out_masks.astype(mx.float32)[:, None], img_h, img_w)
    )


def _filter_and_convert_single_image(
    outputs: Mapping[str, object],
    *,
    threshold: float,
    img_h: int,
    img_w: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Score-filter and convert boxes; does not upsample masks."""

    out_probs, out_masks, out_bbox = _filter_single_image_detections(
        _presence_weighted_scores(outputs),
        cast(mx.array, outputs["pred_masks"]),
        cast(mx.array, outputs["pred_boxes"]),
        threshold=threshold,
    )
    return out_probs, out_masks, _boxes_to_original_xyxy(out_bbox, img_h, img_w)


def _validate_processor_resolution(resolution: object) -> int:
    if isinstance(resolution, bool) or not isinstance(resolution, Integral):
        raise ValueError(
            "Processor resolution must be a positive integer multiple of "
            f"{SAM3_IMAGE_PATCH_SIZE}, got {resolution!r}."
        )
    resolution = int(resolution)
    if resolution <= 0 or resolution % SAM3_IMAGE_PATCH_SIZE != 0:
        raise ValueError(
            "Processor resolution must be a positive integer multiple of "
            f"{SAM3_IMAGE_PATCH_SIZE}, got {resolution}."
        )
    return resolution


def coerce_preprocessed_image(
    tensor: object,
    *,
    resolution: int,
    layout: str = "nchw",
    value_contract: str = PREPROCESSED_VALUE_CONTRACT,
) -> mx.array:
    """Accept a caller-normalized FP32 image and return batched NCHW."""

    if layout not in PREPROCESSED_LAYOUTS:
        raise ValueError(
            "layout must be one of "
            f"{PREPROCESSED_LAYOUTS}; got {layout!r}."
        )
    if value_contract != PREPROCESSED_VALUE_CONTRACT:
        raise ValueError(
            "value_contract must be "
            f"{PREPROCESSED_VALUE_CONTRACT!r}; got {value_contract!r}."
        )
    if isinstance(tensor, mx.array):
        array = tensor
    elif isinstance(tensor, np.ndarray):
        array = mx.array(tensor)
    else:
        raise TypeError("Preprocessed image must be an MLX or NumPy array.")
    if array.dtype != mx.float32:
        raise ValueError(
            "Preprocessed image must be float32; normalize in FP32, then cast "
            "once at the visual-model boundary."
        )
    if not bool(mx.all(mx.isfinite(array)).item()):
        raise ValueError("Preprocessed image must contain only finite values.")
    if array.size and (
        float(mx.min(array).item()) < -1.5 or float(mx.max(array).item()) > 1.5
    ):
        raise ValueError(
            "Preprocessed image must follow the "
            f"{PREPROCESSED_VALUE_CONTRACT} contract."
        )
    if layout == "nchw":
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4 or int(array.shape[1]) != 3:
            raise ValueError(
                "nchw preprocessed images must have shape (3, H, W) or "
                f"(N, 3, H, W); got {tuple(int(dim) for dim in array.shape)}."
            )
    else:
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4 or int(array.shape[-1]) != 3:
            raise ValueError(
                "nhwc preprocessed images must have shape (H, W, 3) or "
                f"(N, H, W, 3); got {tuple(int(dim) for dim in array.shape)}."
            )
        array = transpose_array(array, 0, 3, 1, 2)
    if int(array.shape[0]) != 1:
        raise ValueError(
            "set_preprocessed_image accepts a single image; got batch "
            f"{int(array.shape[0])}."
        )
    if int(array.shape[2]) != resolution or int(array.shape[3]) != resolution:
        raise ValueError(
            "Preprocessed spatial size must match processor resolution "
            f"{resolution}; got {int(array.shape[2])}x{int(array.shape[3])}."
        )
    return array


def _model_raw_prediction(
    model: object,
    *,
    backbone_out: Mapping[str, object],
    find_input: object,
    find_target: object | None,
    geometric_prompt: Prompt,
) -> RawPrediction:
    predict_raw = getattr(model, "predict_raw", None)
    if callable(predict_raw):
        return cast(RawPrediction, predict_raw(
            backbone_out,
            find_input,
            find_target,
            geometric_prompt,
        ))
    forward = getattr(model, "forward_grounding", None)
    if not callable(forward):
        raise TypeError("Processor model must define predict_raw or forward_grounding.")
    return raw_prediction_from_output(
        forward(
            backbone_out,
            find_input,
            find_target,
            geometric_prompt,
        )
    )


def _readonly_array(value: npt.NDArray[Any]) -> npt.NDArray[Any]:
    value.setflags(write=False)
    return value


@lru_cache(maxsize=128)
def _resize_weights_1d(in_size: int, out_size: int) -> ResizeWeights:
    scale = in_size / out_size
    weights_by_output: list[tuple[IndexArray, WeightArray]] = []
    if out_size < in_size:
        support = scale
        for out_index in range(out_size):
            center = (out_index + 0.5) * scale
            start = max(int(np.floor(center - support + 0.5)), 0)
            stop = min(int(np.floor(center + support + 0.5)), in_size)
            indices = np.arange(start, stop, dtype=np.int64)
            weights = 1.0 - np.abs((indices + 0.5 - center) / scale)
            weights = np.maximum(weights, 0.0).astype(np.float32)
            weights /= weights.sum(dtype=np.float32)
            weights_by_output.append(
                (_readonly_array(indices), _readonly_array(weights))
            )
        return tuple(weights_by_output)

    for out_index in range(out_size):
        source = (out_index + 0.5) * scale - 0.5
        left_raw = int(np.floor(source))
        right_raw = left_raw + 1
        weight_right = np.float32(source - left_raw)
        indices = np.array(
            [
                np.clip(left_raw, 0, in_size - 1),
                np.clip(right_raw, 0, in_size - 1),
            ],
            dtype=np.int64,
        )
        weights = np.array([1.0 - weight_right, weight_right], dtype=np.float32)
        weights_by_output.append((_readonly_array(indices), _readonly_array(weights)))
    return tuple(weights_by_output)


def _fused_multiply_add_float32(
    multiplicand: object, multiplier: object, addend: object
) -> npt.NDArray[np.float32]:
    return np.asarray(
        np.asarray(multiplicand, dtype=np.float64)
        * np.asarray(multiplier, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _resize_uint8_bilinear_like_torchvision(
    image: npt.NDArray[np.uint8],
    size: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Resize HWC uint8 RGB data like Torchvision tensor Resize(bilinear)."""

    out_h, out_w = size
    in_h, in_w, channels = image.shape
    if channels != 3:
        raise ValueError("Torchvision-style resize expects an RGB image array.")
    if (in_h, in_w) == (out_h, out_w):
        return image.copy()

    image_f = image.astype(np.float32)
    if out_h >= in_h and out_w >= in_w:
        scale_y = np.float32(in_h / out_h)
        scale_x = np.float32(in_w / out_w)
        x_source = (
            np.arange(out_w, dtype=np.float32) + np.float32(0.5)
        ) * scale_x - np.float32(0.5)
        x0_raw = np.floor(x_source).astype(np.int64)
        x_weight = (x_source - x0_raw.astype(np.float32)).astype(np.float32)
        x0 = np.clip(x0_raw, 0, in_w - 1)
        x1 = np.clip(x0_raw + 1, 0, in_w - 1)
        inv_x_weight = np.float32(1.0) - x_weight

        resized = np.empty((out_h, out_w, channels), dtype=np.float32)
        for out_y in range(out_h):
            y_source = (np.float32(out_y) + np.float32(0.5)) * scale_y - np.float32(0.5)
            y0_raw = int(np.floor(y_source))
            y_weight = np.float32(y_source - np.float32(y0_raw))
            y0 = np.clip(y0_raw, 0, in_h - 1)
            y1 = np.clip(y0_raw + 1, 0, in_h - 1)
            top = _fused_multiply_add_float32(
                image_f[y0, x1],
                x_weight[:, None],
                np.float32(image_f[y0, x0] * inv_x_weight[:, None]),
            )
            bottom = _fused_multiply_add_float32(
                image_f[y1, x1],
                x_weight[:, None],
                np.float32(image_f[y1, x0] * inv_x_weight[:, None]),
            )
            # PyTorch's float bilinear kernel is compiled with fused
            # multiply-add; emulating that keeps .5 ties on the same side
            # before torch.round()/np.rint() converts back to uint8.
            resized[out_y] = _fused_multiply_add_float32(
                bottom,
                y_weight,
                np.float32(top * (np.float32(1.0) - y_weight)),
            )
        return np.rint(resized).clip(0, 255).astype(np.uint8)

    y_weights = _resize_weights_1d(in_h, out_h)
    x_weights = _resize_weights_1d(in_w, out_w)

    tmp = np.empty((out_h, in_w, channels), dtype=np.float32)
    for out_y, (indices, weights) in enumerate(y_weights):
        tmp[out_y] = np.tensordot(weights, image_f[indices], axes=(0, 0))

    resized = np.empty((out_h, out_w, channels), dtype=np.float32)
    for out_x, (indices, weights) in enumerate(x_weights):
        resized[:, out_x] = np.tensordot(tmp[:, indices], weights, axes=([1], [0]))

    return np.rint(resized).clip(0, 255).astype(np.uint8)


def transform(
    image_path_or_pil: str | Image.Image | npt.NDArray[Any],
    resolution: object,
) -> mx.array:
    resolution = _validate_processor_resolution(resolution)
    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = _as_pil_rgb_image(image_path_or_pil)

    img_np = np.asarray(img, dtype=np.uint8)
    img_np = _resize_uint8_bilinear_like_torchvision(
        img_np,
        (resolution, resolution),
    )
    img_mx = mx.array(img_np, dtype=mx.float32) / 255.0  # [H, W, C]
    img_mx = (img_mx - 0.5) / 0.5
    return transpose_array(img_mx, 2, 0, 1)  # [H, W, C] -> [C, H, W]


def _as_pil_rgb_image(
    image: object,
) -> Image.Image:
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = np.asarray(cast(npt.NDArray[Any], image))
        if array.ndim not in (2, 3):
            raise ValueError("Image NumPy arrays must have shape HxW or HxWxC.")
        if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
            raise ValueError("Image NumPy arrays must have 1, 3, or 4 channels.")
        if not np.isfinite(array).all():
            raise ValueError("Image NumPy arrays must contain only finite values.")
        if np.issubdtype(array.dtype, np.floating):
            if array.size and array.min() >= 0.0 and array.max() <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        return Image.fromarray(array).convert("RGB")
    raise ValueError("Image must be a PIL image or a NumPy array")


_IMAGE_RESULT_KEYS = (
    "geometric_prompt",
    "boxes",
    "masks",
    "masks_logits",
    "mask_logits",
    "semantic_seg",
    "scores",
)
_IMAGE_SIZE_KEYS = (
    "original_height",
    "original_width",
    "original_heights",
    "original_widths",
)
_IMAGE_LANGUAGE_KEYS = (
    "language_features",
    "language_mask",
    "language_embeds",
)


ProcessorState = dict[str, object]
OriginalSizes = list[tuple[int, int]]


def _validate_original_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"original_{name} must be an integer.")
    dimension = int(value)
    if dimension < 1:
        raise ValueError(f"original_{name} must be positive.")
    return dimension


def _require_int_list(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of integers.")
    result: list[int] = []
    for item in cast(list[object], value):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"{name} must be a list of integers.")
        result.append(int(item))
    return result


def _require_backbone_out(state: ProcessorState) -> dict[str, object]:
    value = state.get("backbone_out")
    if not isinstance(value, dict):
        raise TypeError("processor state backbone_out must be a dictionary.")
    mapping = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise TypeError("processor state backbone_out must use string keys.")
    return cast(dict[str, object], mapping)


def _require_geometric_prompt(state: ProcessorState) -> Prompt:
    value = state.get("geometric_prompt")
    if not isinstance(value, Prompt):
        raise TypeError("processor state geometric_prompt must be a Prompt.")
    return value


def _batch_original_sizes(state: ProcessorState) -> OriginalSizes | None:
    has_heights = "original_heights" in state
    has_widths = "original_widths" in state
    if has_heights != has_widths:
        raise ValueError(
            "Batch state must contain both original_heights and original_widths."
        )
    if not has_heights:
        return None
    heights = _require_int_list(
        state["original_heights"],
        name="original_heights",
    )
    widths = _require_int_list(
        state["original_widths"],
        name="original_widths",
    )
    if len(heights) != len(widths):
        raise ValueError(
            "original_heights and original_widths must have the same length."
        )
    if len(heights) == 0:
        raise ValueError("Batch state must contain at least one original image size.")
    return list(zip(heights, widths))


def _clear_prompt_and_result_keys(state: ProcessorState) -> None:
    """Remove prompts and prediction outputs from a mutable processor state."""
    if "backbone_out" in state:
        backbone_out = _require_backbone_out(state)
        for key in _IMAGE_LANGUAGE_KEYS:
            backbone_out.pop(key, None)

    for key in _IMAGE_RESULT_KEYS:
        state.pop(key, None)


def _clear_image_dependent_state(state: ProcessorState) -> None:
    """Invalidate prompts, outputs, and size metadata before replacing an image.

    Reused states must not retain geometric prompts, opposite batch size keys, or
    stale outputs from a previous single-image or batch call.
    """
    _clear_prompt_and_result_keys(state)
    for key in _IMAGE_SIZE_KEYS:
        state.pop(key, None)


def _normalize_processor_device(device: object) -> str:
    if is_mlx_runtime_device(device):
        return "mlx"
    _raise_processor_unsupported(
        f"sam3_mlx.model.sam3_image_processor.Sam3Processor(device={device!r})",
        reason="unsupported-device",
        detail=(
            "sam3_mlx only runs on the explicit MLX runtime. Non-MLX "
            "device strings are not accepted as aliases."
        ),
        alternative="device='mlx'",
    )


class _ArrayLayer(Protocol):
    def __call__(self, value: mx.array) -> mx.array: ...


class _MaskDecoder(Protocol):
    conv_s0: _ArrayLayer
    conv_s1: _ArrayLayer


class _InteractiveModel(Protocol):
    sam_mask_decoder: _MaskDecoder


class _InteractivePredictor(Protocol):
    model: _InteractiveModel


class _ForwardImage(Protocol):
    def __call__(self, image: mx.array, /) -> object: ...


class _ForwardText(Protocol):
    def __call__(
        self,
        prompts: list[str],
        /,
        *,
        device: str | None = None,
    ) -> object: ...


class _ProcessorModel(Protocol):
    @property
    def backbone(self) -> object: ...

    @property
    def inst_interactive_predictor(self) -> object | None: ...

    def _get_dummy_prompt(self, num_prompts: int = 1) -> Prompt: ...

    def forward_grounding(
        self,
        backbone_out: Mapping[str, object],
        find_input: object,
        find_target: object | None,
        geometric_prompt: Prompt,
    ) -> Output: ...


def _forward_backbone_image(
    backbone: object,
    image: mx.array,
    *,
    precision: object = "fp32",
) -> dict[str, object]:
    forward_value = getattr(backbone, "forward_image", None)
    if not callable(forward_value):
        raise TypeError("Processor model backbone must define forward_image.")
    forward = cast(_ForwardImage, forward_value)
    output = forward(cast_visual_input(image, parse_precision(precision)))
    if not isinstance(output, Mapping):
        raise TypeError("Processor backbone forward_image must return a mapping.")
    string_output: dict[str, object] = {}
    for key, value in cast(Mapping[object, object], output).items():
        if not isinstance(key, str):
            raise TypeError("Processor backbone output keys must be strings.")
        string_output[key] = value
    return string_output


def _validate_text_outputs(value: object) -> dict[str, mx.array]:
    if not isinstance(value, Mapping):
        raise TypeError("Processor text output must be a mapping.")
    text_output: dict[str, mx.array] = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str) or not isinstance(item, mx.array):
            raise TypeError("Processor text output must map string keys to MLX arrays.")
        text_output[key] = item
    return text_output


def normalize_text_prompt_key(prompt: object) -> str:
    """Return the exact whitespace-normalized cache key for one prompt."""

    if not isinstance(prompt, str):
        raise TypeError("Text prompt must be a string.")
    return " ".join(prompt.split())


def _copy_text_outputs(value: Mapping[str, mx.array]) -> dict[str, mx.array]:
    return {key: item for key, item in value.items()}


def _validate_text_cache_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError("text_cache_size must be a non-negative integer.")
    return int(value)


def _model_is_training(model: object) -> bool:
    return bool(getattr(model, "training", False))


def _forward_backbone_text(
    backbone: object, prompts: list[str], *, device: str
) -> dict[str, mx.array]:
    forward_value = getattr(backbone, "forward_text", None)
    if not callable(forward_value):
        raise TypeError("Processor model backbone must define forward_text.")
    forward = cast(_ForwardText, forward_value)
    return _validate_text_outputs(forward(prompts, device=device))


def _evaluate_processor_state(state: ProcessorState) -> None:
    evaluate = cast(Callable[[object], None], getattr(mx, "eval"))
    evaluate(state)


def _model_dummy_prompt(
    model: _ProcessorModel,
    *,
    num_prompts: int = 1,
) -> Prompt:
    factory = getattr(model, "_get_dummy_prompt")
    return factory(num_prompts=num_prompts)


class Sam3Processor:
    def __init__(
        self,
        model: _ProcessorModel,
        resolution: object = DEFAULT_IMAGE_RESOLUTION,
        device: object = "mlx",
        confidence_threshold: float = 0.5,
        text_cache_size: object = DEFAULT_TEXT_CACHE_SIZE,
    ) -> None:
        runtime_device = _normalize_processor_device(device)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
        cache_size = _validate_text_cache_size(text_cache_size)
        self.model = model
        self.resolution = _validate_processor_resolution(resolution)
        self.device = runtime_device
        self.confidence_threshold = confidence_threshold
        self.text_cache_size = cache_size
        self._text_cache: BoundedLRUCache[str, dict[str, mx.array]] | None = (
            None if cache_size == 0 else BoundedLRUCache(cache_size)
        )
        self.transform = partial(transform, resolution=self.resolution)

        self.find_stage = FindStage(
            img_ids=mx.array([0], dtype=mx.int64),
            text_ids=mx.array([0], dtype=mx.int64),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def _find_stage_for_state(self, state: ProcessorState) -> FindStage:
        sizes = _batch_original_sizes(state)
        if sizes is None:
            return self.find_stage
        batch_size = len(sizes)
        return FindStage(
            img_ids=mx.arange(batch_size, dtype=mx.int64),
            text_ids=mx.zeros((batch_size,), dtype=mx.int64),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def clear_text_cache(self) -> None:
        """Drop every cached text-encoder output for this processor."""

        if self._text_cache is not None:
            self._text_cache.clear()

    def _text_outputs_for_prompt(self, prompt: str) -> dict[str, mx.array]:
        cache = self._text_cache
        if cache is None or _model_is_training(self.model):
            return _forward_backbone_text(
                self.model.backbone,
                [prompt],
                device=self.device,
            )
        key = normalize_text_prompt_key(prompt)
        cached = cache.get(key)
        if cached is not None:
            return _copy_text_outputs(cached)
        encoded = _forward_backbone_text(
            self.model.backbone,
            [key],
            device=self.device,
        )
        cache[key] = encoded
        return _copy_text_outputs(encoded)

    def _patch_interactive_backbone_features(
        self, backbone_out: Mapping[str, object]
    ) -> None:
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if not inst_interactivity_en or "sam2_backbone_out" not in backbone_out:
            return
        predictor = cast(_InteractivePredictor, self.model.inst_interactive_predictor)
        sam2_backbone_out = cast(dict[str, Any], backbone_out["sam2_backbone_out"])
        backbone_fpn = cast(list[mx.array], sam2_backbone_out["backbone_fpn"])
        backbone_fpn[0] = predictor.model.sam_mask_decoder.conv_s0(backbone_fpn[0])
        backbone_fpn[1] = predictor.model.sam_mask_decoder.conv_s1(backbone_fpn[1])

    def _apply_visual_tensor(
        self,
        image_tensor: mx.array,
        *,
        original_height: int,
        original_width: int,
        state: ProcessorState | None,
    ) -> ProcessorState:
        if state is None:
            state = {}
        else:
            _clear_image_dependent_state(state)
        state["original_height"] = original_height
        state["original_width"] = original_width
        backbone_out = _forward_backbone_image(
            self.model.backbone,
            image_tensor,
            precision=model_precision(self.model),
        )
        state["backbone_out"] = backbone_out
        _evaluate_processor_state(state)
        self._patch_interactive_backbone_features(backbone_out)
        return state

    def set_image(
        self,
        image: Image.Image | npt.NDArray[Any],
        state: ProcessorState | None = None,
    ) -> ProcessorState:
        pil_image = _as_pil_rgb_image(image)
        width, height = pil_image.size
        return self._apply_visual_tensor(
            self.transform(pil_image)[None],
            original_height=height,
            original_width=width,
            state=state,
        )

    def set_preprocessed_image(
        self,
        tensor: object,
        *,
        original_size: tuple[int, int] | list[int],
        layout: str = "nchw",
        value_contract: str = PREPROCESSED_VALUE_CONTRACT,
        state: ProcessorState | None = None,
    ) -> ProcessorState:
        if not isinstance(original_size, (tuple, list)) or len(original_size) != 2:
            raise ValueError("original_size must be (height, width).")
        height = _validate_original_dimension(original_size[0], "height")
        width = _validate_original_dimension(original_size[1], "width")
        image_tensor = coerce_preprocessed_image(
            tensor,
            resolution=self.resolution,
            layout=layout,
            value_contract=value_contract,
        )
        return self._apply_visual_tensor(
            image_tensor,
            original_height=height,
            original_width=width,
            state=state,
        )

    def set_image_batch(
        self,
        images: object,
        state: ProcessorState | None = None,
    ) -> ProcessorState:
        """Sets an image batch and computes batched backbone features."""

        if state is None:
            state = {}
        else:
            _clear_image_dependent_state(state)
        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or NumPy arrays.")
        typed_images = cast(list[Image.Image | npt.NDArray[Any]], images)
        if len(typed_images) == 0:
            raise ValueError("Images list must not be empty.")
        pil_images = [_as_pil_rgb_image(image) for image in typed_images]
        state["original_heights"] = [image.height for image in pil_images]
        state["original_widths"] = [image.width for image in pil_images]

        image_batch = mx.stack([self.transform(image) for image in pil_images], axis=0)
        backbone_out = _forward_backbone_image(
            self.model.backbone,
            image_batch,
            precision=model_precision(self.model),
        )
        state["backbone_out"] = backbone_out
        _evaluate_processor_state(state)
        self._patch_interactive_backbone_features(backbone_out)
        return state

    def set_text_prompt(
        self,
        prompt: str,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
        text_outputs: object | None = None,
    ) -> ProcessorState:
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        validated_text_outputs = (
            self._text_outputs_for_prompt(prompt)
            if text_outputs is None
            else _validate_text_outputs(text_outputs)
        )
        # will erase the previous text prompt if any
        backbone_out = _require_backbone_out(state)
        backbone_out.update(validated_text_outputs)
        if "geometric_prompt" not in state:
            sizes = _batch_original_sizes(state)
            num_prompts = len(sizes) if sizes is not None else 1
            state["geometric_prompt"] = _model_dummy_prompt(
                self.model,
                num_prompts=num_prompts,
            )
        return self._forward_grounding(state) if run_grounding else state

    def add_geometric_prompt(
        self,
        box: list[float],
        label: bool,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")
        if _batch_original_sizes(state) is not None:
            _raise_processor_unsupported(
                "sam3_mlx.model.sam3_image_processor.Sam3Processor.add_geometric_prompt(batch_state)",
                reason="image-interactivity",
                detail=(
                    "Batch geometric prompts are not supported in the MLX image "
                    "processor yet."
                ),
                alternative=(
                    "set_image for a single image or set_text_prompt for "
                    "text-only batches"
                ),
            )

        backbone_out = _require_backbone_out(state)
        if "language_features" not in backbone_out:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            backbone_out.update(self._text_outputs_for_prompt("visual"))

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _model_dummy_prompt(self.model)

        # adding a batch and sequence dimension
        boxes = reshape_array(mx.array(box, dtype=mx.float32), 1, 1, 4)
        labels = reshape_array(mx.array([label], dtype=mx.bool_), 1, 1)
        geometric_prompt = cast(Prompt, state["geometric_prompt"])
        geometric_prompt.append_boxes(boxes, labels)

        return self._forward_grounding(state) if run_grounding else state

    def add_point_prompt(
        self,
        point: list[float],
        label: bool,
        state: ProcessorState,
        *,
        run_grounding: bool = True,
    ) -> ProcessorState:
        """Adds a point prompt and run inference on the current image.

        The point is expected in normalized ``[x, y]`` image coordinates.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before add_point_prompt")
        if _batch_original_sizes(state) is not None:
            _raise_processor_unsupported(
                "sam3_mlx.model.sam3_image_processor.Sam3Processor.add_point_prompt(batch_state)",
                reason="image-interactivity",
                detail=(
                    "Batch point prompts are not supported in the MLX image "
                    "processor yet."
                ),
                alternative=(
                    "set_image for a single image or set_text_prompt for "
                    "text-only batches"
                ),
            )

        backbone_out = _require_backbone_out(state)
        if "language_features" not in backbone_out:
            backbone_out.update(self._text_outputs_for_prompt("visual"))

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = _model_dummy_prompt(self.model)

        points = reshape_array(mx.array(point, dtype=mx.float32), 1, 1, 2)
        labels = reshape_array(mx.array([label], dtype=mx.bool_), 1, 1)
        geometric_prompt = cast(Prompt, state["geometric_prompt"])
        geometric_prompt.append_points(points, labels)
        return self._forward_grounding(state) if run_grounding else state

    def run_grounding(self, state: ProcessorState) -> ProcessorState:
        """Execute one grounding pass after prompt mutation is complete."""
        return self.predict(state)

    def predict_raw(self, state: ProcessorState) -> RawPrediction:
        """Fixed-shape device arrays before filtering or original-res upsample."""

        return _model_raw_prediction(
            self.model,
            backbone_out=_require_backbone_out(state),
            find_input=self._find_stage_for_state(state),
            find_target=None,
            geometric_prompt=_require_geometric_prompt(state),
        )

    def predict(self, state: ProcessorState) -> ProcessorState:
        """Filter, upsample, and write host-facing boxes/masks/scores into state."""

        return self._materialize_prediction(state, self.predict_raw(state))

    def reset_all_prompts(self, state: ProcessorState) -> None:
        """Removes all the prompts and results"""
        _clear_prompt_and_result_keys(state)

    def set_confidence_threshold(
        self, threshold: float, state: ProcessorState | None = None
    ) -> ProcessorState | None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
        self.confidence_threshold = float(threshold)
        if state is not None and "boxes" in state:
            return self._forward_grounding(state)
        return state

    def _forward_grounding(self, state: ProcessorState) -> ProcessorState:
        return self.predict(state)

    def _materialize_prediction(
        self, state: ProcessorState, outputs: Mapping[str, object]
    ) -> ProcessorState:
        batch_sizes = _batch_original_sizes(state)

        out_bbox = cast(mx.array, outputs["pred_boxes"])
        out_masks = cast(mx.array, outputs["pred_masks"])
        out_probs = _presence_weighted_scores(outputs)

        if batch_sizes is not None:
            return self._forward_grounding_batch_outputs(
                state,
                outputs=outputs,
                out_bbox=out_bbox,
                out_masks=out_masks,
                out_probs=out_probs,
                original_sizes=batch_sizes,
            )

        if out_probs.shape[0] != 1:
            _raise_processor_unsupported(
                "sam3_mlx.model.sam3_image_processor.Sam3Processor._forward_grounding(batch_output)",
                reason="image-interactivity",
                detail=(
                    "Batch grounding is not supported in the MLX port yet; "
                    f"received batch size {out_probs.shape[0]}."
                ),
                alternative="set_image for a single image",
            )

        out_probs, out_masks, out_bbox = _filter_single_image_detections(
            out_probs,
            out_masks,
            out_bbox,
            threshold=self.confidence_threshold,
        )
        img_h = _validate_original_dimension(state.get("original_height"), "height")
        img_w = _validate_original_dimension(state.get("original_width"), "width")
        boxes = _boxes_to_original_xyxy(out_bbox, img_h, img_w)
        out_masks = _upsample_and_activate_masks(out_masks, img_h, img_w)
        seg_mask = cast(mx.array | None, outputs.get("semantic_seg"))
        if seg_mask is not None:
            state["semantic_seg"] = _upsample_spatial(seg_mask, img_h, img_w)
        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    def _forward_grounding_batch_outputs(
        self,
        state: ProcessorState,
        *,
        outputs: Output,
        out_bbox: mx.array,
        out_masks: mx.array,
        out_probs: mx.array,
        original_sizes: OriginalSizes,
    ) -> ProcessorState:
        batch_size = len(original_sizes)
        if out_probs.shape[0] != batch_size:
            raise ValueError(
                "Batch grounding output batch size must match original image sizes; "
                f"got outputs batch {out_probs.shape[0]} and {batch_size} sizes."
            )

        boxes_by_image: list[mx.array] = []
        masks_logits_by_image: list[mx.array] = []
        masks_by_image: list[mx.array] = []
        scores_by_image: list[mx.array] = []
        semantic_seg = cast(mx.array | None, outputs.get("semantic_seg"))
        semantic_seg_by_image: list[mx.array] | None = (
            [] if semantic_seg is not None else None
        )

        for batch_idx, (img_h, img_w) in enumerate(original_sizes):
            keep_indices = _score_keep_indices(
                out_probs[batch_idx], self.confidence_threshold
            )
            image_scores = out_probs[batch_idx][keep_indices]
            image_masks = out_masks[batch_idx][keep_indices]
            image_boxes = out_bbox[batch_idx][keep_indices]
            boxes = _boxes_to_original_xyxy(image_boxes, img_h, img_w)
            image_masks = _upsample_and_activate_masks(image_masks, img_h, img_w)

            boxes_by_image.append(boxes)
            masks_logits_by_image.append(image_masks)
            masks_by_image.append(image_masks > 0.5)
            scores_by_image.append(image_scores)

            if semantic_seg_by_image is not None:
                assert semantic_seg is not None
                semantic_seg_by_image.append(
                    _upsample_spatial(
                        semantic_seg[batch_idx : batch_idx + 1], img_h, img_w
                    )
                )

        if semantic_seg_by_image is not None:
            state["semantic_seg"] = semantic_seg_by_image
        state["masks_logits"] = masks_logits_by_image
        state["masks"] = masks_by_image
        state["boxes"] = boxes_by_image
        state["scores"] = scores_by_image
        return state
