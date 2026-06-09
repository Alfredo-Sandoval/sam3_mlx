from functools import lru_cache, partial
from numbers import Integral
from typing import Dict, List
import PIL
from PIL import Image
import numpy as np
import mlx.core as mx

from sam3_mlx._device import is_mlx_runtime_device
from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model import box_ops
from sam3_mlx.model.data_misc import FindStage, interpolate

SAM3_IMAGE_PATCH_SIZE = 14


def _raise_processor_unsupported(
    feature: str, *, reason: str, detail: str, alternative=None
):
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


def _validate_processor_resolution(resolution) -> int:
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


def _readonly_array(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


@lru_cache(maxsize=128)
def _resize_weights_1d(
    in_size: int, out_size: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    scale = in_size / out_size
    weights_by_output = []
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


def _fused_multiply_add_float32(multiplicand, multiplier, addend):
    return np.asarray(
        np.asarray(multiplicand, dtype=np.float64)
        * np.asarray(multiplier, dtype=np.float64)
        + np.asarray(addend, dtype=np.float64),
        dtype=np.float32,
    )


def _resize_uint8_bilinear_like_torchvision(
    image: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
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


def transform(image_path_or_pil, resolution):
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
    return img_mx.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]


def _as_pil_rgb_image(image):
    if isinstance(image, PIL.Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
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


def _batch_original_sizes(state: Dict):
    has_heights = "original_heights" in state
    has_widths = "original_widths" in state
    if has_heights != has_widths:
        raise ValueError(
            "Batch state must contain both original_heights and original_widths."
        )
    if not has_heights:
        return None
    heights = state["original_heights"]
    widths = state["original_widths"]
    if len(heights) != len(widths):
        raise ValueError(
            "original_heights and original_widths must have the same length."
        )
    if len(heights) == 0:
        raise ValueError("Batch state must contain at least one original image size.")
    return list(zip(heights, widths))


def _normalize_processor_device(device) -> str:
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


class Sam3Processor:
    def __init__(self, model, resolution=1008, device="mlx", confidence_threshold=0.5):
        runtime_device = _normalize_processor_device(device)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
        self.model = model
        self.resolution = _validate_processor_resolution(resolution)
        self.device = runtime_device
        self.confidence_threshold = confidence_threshold
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

    def _find_stage_for_state(self, state: Dict):
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

    def _patch_interactive_backbone_features(self, backbone_out):
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if not inst_interactivity_en or "sam2_backbone_out" not in backbone_out:
            return
        sam2_backbone_out = backbone_out["sam2_backbone_out"]
        sam2_backbone_out["backbone_fpn"][0] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
        )
        sam2_backbone_out["backbone_fpn"][1] = (
            self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        )

    def set_image(self, image, state=None):
        if state is None:
            state = {}

        image = _as_pil_rgb_image(image)
        width, height = image.size

        image = self.transform(image)[None]

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        mx.eval(state)
        self._patch_interactive_backbone_features(state["backbone_out"])
        return state

    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets an image batch and computes batched backbone features."""

        if state is None:
            state = {}
        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or NumPy arrays.")
        if len(images) == 0:
            raise ValueError("Images list must not be empty.")

        pil_images = [_as_pil_rgb_image(image) for image in images]
        state["original_heights"] = [image.height for image in pil_images]
        state["original_widths"] = [image.width for image in pil_images]

        image_batch = mx.stack([self.transform(image) for image in pil_images], axis=0)
        state["backbone_out"] = self.model.backbone.forward_image(image_batch)
        mx.eval(state)
        self._patch_interactive_backbone_features(state["backbone_out"])
        return state

    def set_text_prompt(self, prompt: str, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            sizes = _batch_original_sizes(state)
            num_prompts = len(sizes) if sizes is not None else 1
            state["geometric_prompt"] = self.model._get_dummy_prompt(
                num_prompts=num_prompts
            )
        return self._forward_grounding(state)

    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
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

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"], device=self.device
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = mx.array(box, dtype=mx.float32).reshape(1, 1, 4)
        labels = mx.array([label], dtype=mx.bool_).reshape(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def add_point_prompt(self, point: List, label: bool, state: Dict):
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

        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(["visual"])
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        points = mx.array(point, dtype=mx.float32).reshape(1, 1, 2)
        labels = mx.array([label], dtype=mx.bool_).reshape(1, 1)
        state["geometric_prompt"].append_points(points, labels)
        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = [
            "geometric_prompt",
            "boxes",
            "masks",
            "masks_logits",
            "mask_logits",
            "scores",
        ]
        for key in keys_to_del:
            if key in state:
                del state[key]

    def set_confidence_threshold(self, threshold: float, state=None):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0.")
        self.confidence_threshold = float(threshold)
        if state is not None and "boxes" in state:
            return self._forward_grounding(state)
        return state

    def _forward_grounding(self, state: Dict):
        batch_sizes = _batch_original_sizes(state)

        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self._find_stage_for_state(state),
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = mx.sigmoid(out_logits)
        presence_score = mx.sigmoid(outputs["presence_logit_dec"])[:, None]
        out_probs = (out_probs * presence_score).squeeze(-1)

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

        keep_indices = _single_image_keep_indices(out_probs, self.confidence_threshold)
        out_probs = out_probs[0][keep_indices]
        out_masks = out_masks[0][keep_indices]
        out_bbox = out_bbox[0][keep_indices]
        seg_mask = outputs.get("semantic_seg")

        # convert box to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = mx.array([img_w, img_h, img_w, img_h])
        boxes = boxes * scale_fct[None, :]

        interpolator = partial(
            interpolate,
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        )
        out_masks = interpolator(out_masks[:, None])
        out_masks = mx.sigmoid(out_masks)

        if seg_mask is not None:
            state["semantic_seg"] = interpolator(seg_mask)
        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    def _forward_grounding_batch_outputs(
        self,
        state: Dict,
        *,
        outputs: Dict,
        out_bbox: mx.array,
        out_masks: mx.array,
        out_probs: mx.array,
        original_sizes,
    ):
        batch_size = len(original_sizes)
        if out_probs.shape[0] != batch_size:
            raise ValueError(
                "Batch grounding output batch size must match original image sizes; "
                f"got outputs batch {out_probs.shape[0]} and {batch_size} sizes."
            )

        boxes_by_image = []
        masks_logits_by_image = []
        masks_by_image = []
        scores_by_image = []
        semantic_seg = outputs.get("semantic_seg")
        semantic_seg_by_image = [] if semantic_seg is not None else None

        for batch_idx, (img_h, img_w) in enumerate(original_sizes):
            keep_indices = _score_keep_indices(
                out_probs[batch_idx], self.confidence_threshold
            )
            image_scores = out_probs[batch_idx][keep_indices]
            image_masks = out_masks[batch_idx][keep_indices]
            image_boxes = out_bbox[batch_idx][keep_indices]

            boxes = box_ops.box_cxcywh_to_xyxy(image_boxes)
            scale_fct = mx.array([img_w, img_h, img_w, img_h])
            boxes = boxes * scale_fct[None, :]

            interpolator = partial(
                interpolate,
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            )
            image_masks = interpolator(image_masks[:, None])
            image_masks = mx.sigmoid(image_masks)

            boxes_by_image.append(boxes)
            masks_logits_by_image.append(image_masks)
            masks_by_image.append(image_masks > 0.5)
            scores_by_image.append(image_scores)

            if semantic_seg_by_image is not None:
                semantic_seg_by_image.append(
                    interpolator(semantic_seg[batch_idx : batch_idx + 1])
                )

        if semantic_seg_by_image is not None:
            state["semantic_seg"] = semantic_seg_by_image
        state["masks_logits"] = masks_logits_by_image
        state["masks"] = masks_by_image
        state["boxes"] = boxes_by_image
        state["scores"] = scores_by_image
        return state
