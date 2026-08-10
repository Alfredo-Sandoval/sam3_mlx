# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""MLX loss functions ported from official SAM3 training utilities.

The official module mixes reusable image losses with Torch-specific
matcher, distributed, and tracking losses. This MLX port implements the
image-safe callable subset and fails fast for the tracking-only surfaces.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Generic,
    NotRequired,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    cast,
)

import mlx.core as mx
from mlx import nn
import numpy as np

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model import box_ops, data_misc
import sam3_mlx.train.loss.mask_sampling as mask_sampling

# The package exports a function with this submodule's name, so import the
# module explicitly to retain access to both focal helpers.
_sigmoid_focal_loss_module = importlib.import_module(
    "sam3_mlx.train.loss.sigmoid_focal_loss"
)


class _Interpolate(Protocol):
    def __call__(
        self,
        input: mx.array,
        size: tuple[int, ...] | None = None,
        scale_factor: float | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        antialias: bool = False,
    ) -> mx.array: ...


class _PointSample(Protocol):
    def __call__(
        self, input: mx.array, point_coords: mx.array, **kwargs: object
    ) -> mx.array: ...


class _UncertainPoints(Protocol):
    def __call__(
        self,
        logits: mx.array,
        uncertainty_func: Callable[[mx.array], mx.array],
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ) -> mx.array: ...


class _FocalLoss(Protocol):
    def __call__(
        self,
        inputs: mx.array,
        targets: mx.array,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> mx.array: ...


_interpolate = cast(_Interpolate, getattr(data_misc, "interpolate"))
_calculate_uncertainty = cast(
    Callable[[mx.array], mx.array], getattr(mask_sampling, "calculate_uncertainty")
)
_get_uncertain_points = cast(
    _UncertainPoints,
    getattr(mask_sampling, "get_uncertain_point_coords_with_randomness"),
)
_point_sample = cast(_PointSample, getattr(mask_sampling, "point_sample"))
_elementwise_sigmoid_focal_loss = cast(
    _FocalLoss, getattr(_sigmoid_focal_loss_module, "sigmoid_focal_loss")
)
_reduced_sigmoid_focal_loss = cast(
    _FocalLoss, getattr(_sigmoid_focal_loss_module, "sigmoid_focal_loss_reduce")
)
_fast_diag_box_iou = cast(
    Callable[[mx.array, mx.array], mx.array], getattr(box_ops, "fast_diag_box_iou")
)
_fast_diag_generalized_box_iou = cast(
    Callable[[mx.array, mx.array], mx.array],
    getattr(box_ops, "fast_diag_generalized_box_iou"),
)

interpolate = _interpolate
calculate_uncertainty = _calculate_uncertainty
get_uncertain_point_coords_with_randomness = _get_uncertain_points
point_sample = _point_sample


MLX_LOSS_FNS_BASE_COMMIT = "4794409a19afd9e3faeac66a2f1c4373ddf10f5b"
CORE_LOSS_KEY = "core_loss"
TRAINING_LOSS_CPU_BOUNDARIES = {
    "instance_masks_to_semantic_masks": (
        "variable-length per-image instance mask reductions still materialize "
        "num_instances on the host until a segment-reduce style MLX port exists"
    ),
}


ArrayValue: TypeAlias = (
    mx.array | np.ndarray | bool | int | float | list[object] | tuple[object, ...]
)
MatchIndices: TypeAlias = (
    tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array | None]
)
LossDict: TypeAlias = dict[str, mx.array]
_LossArgs = ParamSpec("_LossArgs")


class BoxesOutputMap(TypedDict):
    pred_boxes: mx.array
    pred_boxes_xyxy: mx.array
    is_video_grounding_batch: NotRequired[bool]
    Q_det: NotRequired[int]


class BoxesTargetMap(TypedDict):
    boxes: mx.array
    boxes_xyxy: mx.array


class MasksOutputMap(TypedDict):
    pred_masks: mx.array
    is_video_grounding_batch: NotRequired[bool]
    Q_det: NotRequired[int]


class MasksTargetMap(TypedDict):
    masks: mx.array | None
    is_valid_mask: mx.array


class SemanticOutputMap(TypedDict):
    semantic_seg: mx.array
    presence_logit: NotRequired[mx.array]


class SemanticTargetMap(TypedDict):
    masks: mx.array
    num_boxes: ArrayValue
    semantic_masks: NotRequired[mx.array | None]


class IABCEOutputMap(TypedDict):
    pred_logits: mx.array
    pred_boxes_xyxy: mx.array
    is_video_grounding_batch: NotRequired[bool]
    Q_det: NotRequired[int]
    presence_logit_dec: NotRequired[mx.array]


class IABCETargetArraysOutputMap(TypedDict):
    pred_boxes_xyxy: mx.array


class IABCETargetArraysTargetMap(TypedDict):
    boxes_xyxy: mx.array


class IABCETargetMap(TypedDict):
    boxes_xyxy: mx.array
    is_exhaustive: NotRequired[mx.array]
    object_ids_padded: NotRequired[mx.array]
    boxes_padded: NotRequired[mx.array]


class _QDetMap(TypedDict):
    Q_det: int


class _PresenceTargetMap(TypedDict):
    object_ids_padded: mx.array
    boxes_padded: mx.array


_Weights: TypeAlias = Mapping[str, float]


_mx_eval = cast(Callable[[mx.array], None], getattr(mx, "eval"))


def _reshape(value: mx.array, shape: Sequence[int]) -> mx.array:
    return mx.reshape(value, shape)


def _transpose(value: mx.array, axes: Sequence[int]) -> mx.array:
    return mx.transpose(value, axes)


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int)


def _as_array(value: ArrayValue, dtype: mx.Dtype | None = None) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(dtype) if dtype is not None else value
    if dtype is None:
        return mx.array(value)
    return mx.array(value, dtype=dtype)


def _as_float_array(value: ArrayValue) -> mx.array:
    return _as_array(value, dtype=mx.float32)


def _as_bool_array(value: ArrayValue) -> mx.array:
    return _as_array(value).astype(mx.bool_)


def _to_numpy(value: ArrayValue) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, mx.array):
        _mx_eval(value)
    return np.asarray(value)


def _scalar_array(value: bool | int | float, dtype: mx.Dtype = mx.float32) -> mx.array:
    return mx.array(value, dtype=dtype)


def _num_boxes_array(num_boxes: ArrayValue) -> mx.array:
    return mx.maximum(_as_float_array(num_boxes), _scalar_array(1.0))


def _indices_target(indices: MatchIndices) -> mx.array | None:
    if len(indices) == 3:
        return indices[2]
    return None


def _gather_matched(values: mx.array, indices: MatchIndices) -> mx.array:
    return values[indices[0], indices[1]]


def _target_select(values: mx.array, indices: MatchIndices) -> mx.array:
    target_idx = _indices_target(indices)
    return values if target_idx is None else values[target_idx]


def _empty_scalar(dtype: mx.Dtype = mx.float32) -> mx.array:
    return mx.array(0.0, dtype=dtype)


def _bce_with_logits(inputs: ArrayValue, targets: ArrayValue) -> mx.array:
    inputs = _as_float_array(inputs)
    targets = _as_float_array(targets)
    max_val = mx.maximum(inputs, mx.zeros_like(inputs))
    return max_val - inputs * targets + mx.log(1 + mx.exp(-mx.abs(inputs)))


def _l1_loss(inputs: ArrayValue, targets: ArrayValue) -> mx.array:
    return mx.abs(_as_float_array(inputs) - _as_float_array(targets))


def _mse_loss(inputs: ArrayValue, targets: ArrayValue) -> mx.array:
    diff = _as_float_array(inputs) - _as_float_array(targets)
    return diff * diff


def instance_masks_to_semantic_masks(
    instance_masks: ArrayValue, num_instances: ArrayValue
) -> mx.array:
    """Convert collapsed instance masks into one semantic mask per image."""

    counts = _as_array(num_instances, dtype=mx.int64)
    # Named host boundary: per-image variable-length reductions are not claimed
    # as MLX-native training runtime yet.
    counts_np = _to_numpy(counts).reshape(-1).astype(np.int64)
    if counts_np.sum() == 0:
        return _reshape(counts, (counts.shape[0], 1, 1))

    masks = _as_bool_array(instance_masks)
    chunks: list[mx.array] = []
    start = 0
    for count in counts_np.tolist():
        stop = start + int(count)
        chunk = masks[start:stop]
        if int(count) == 0:
            chunks.append(mx.zeros(masks.shape[1:], dtype=mx.bool_))
        else:
            chunks.append(mx.any(chunk, axis=0))
        start = stop
    return mx.stack(chunks, axis=0)


def accuracy(
    output: ArrayValue, target: ArrayValue, topk: tuple[int, ...] = (1,)
) -> list[mx.array]:
    """Compute precision@k for the specified values of k."""

    output = _as_float_array(output)
    target = _as_array(target, dtype=mx.int64)
    if target.size == 0:
        return [mx.array(0.0, dtype=output.dtype)]
    maxk = max(topk)
    batch_size = target.shape[0]
    pred = _transpose(mx.argsort(output, axis=1)[:, -maxk:][:, ::-1], (1, 0))
    correct = mx.equal(pred, _reshape(target, (1, -1)))
    results: list[mx.array] = []
    for k in topk:
        correct_k = mx.sum(_reshape(correct[:k], (-1,)).astype(mx.float32))
        results.append(correct_k * (100.0 / batch_size))
    return results


def _dice_loss(
    inputs: ArrayValue,
    targets: ArrayValue,
    num_boxes: ArrayValue,
    loss_on_multimask: bool = False,
    reduce: bool = True,
) -> mx.array:
    inputs = mx.sigmoid(_as_float_array(inputs))
    targets = _as_float_array(targets)
    num_boxes = _num_boxes_array(num_boxes)
    if loss_on_multimask:
        if inputs.ndim != 4 or targets.ndim != 4:
            raise AssertionError("multimask DICE expects inputs/targets with rank 4.")
        inputs = _reshape(inputs, (*inputs.shape[:2], -1))
        targets = _reshape(targets, (*targets.shape[:2], -1))
        numerator = 2 * mx.sum(inputs * targets, axis=-1)
    else:
        inputs = _reshape(inputs, (inputs.shape[0], -1))
        targets = _reshape(targets, (targets.shape[0], -1))
        numerator = 2 * mx.sum(inputs * targets, axis=1)

    denominator = mx.sum(inputs, axis=-1) + mx.sum(targets, axis=-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_boxes
    if not reduce:
        return loss
    return mx.sum(loss) / num_boxes


def dice_loss(
    inputs: ArrayValue,
    targets: ArrayValue,
    num_boxes: ArrayValue,
    loss_on_multimask: bool = False,
    reduce: bool = True,
) -> mx.array:
    """DICE loss for binary masks, matching the official normalization."""

    return _dice_loss(inputs, targets, num_boxes, loss_on_multimask, reduce)


def sigmoid_focal_loss(
    inputs: ArrayValue,
    targets: ArrayValue,
    num_boxes: ArrayValue,
    alpha: float = 0.25,
    gamma: float = 2.0,
    loss_on_multimask: bool = False,
    reduce: bool = True,
    triton: bool = True,
) -> mx.array:
    """Official-shaped sigmoid focal loss wrapper over the MLX implementation."""

    del triton
    inputs = _as_float_array(inputs)
    targets = _as_float_array(targets)
    num_boxes = _num_boxes_array(num_boxes)
    if not 0 <= alpha <= 1:
        raise RuntimeError(f"Alpha should be in [0,1], got {alpha}")

    if reduce and not loss_on_multimask:
        return _reduced_sigmoid_focal_loss(inputs, targets, alpha, gamma) / (
            num_boxes * inputs.shape[1]
        )

    loss = _elementwise_sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma)
    if not reduce:
        return loss
    if loss_on_multimask:
        if loss.ndim != 4:
            raise AssertionError("multimask focal loss expects rank-4 loss.")
        return mx.mean(_reshape(loss, (*loss.shape[:2], -1)), axis=-1) / num_boxes
    return mx.sum(mx.mean(loss, axis=1)) / num_boxes


def iou_loss(
    inputs: ArrayValue,
    targets: ArrayValue,
    pred_ious: ArrayValue,
    num_boxes: ArrayValue,
    loss_on_multimask: bool = False,
    use_l1_loss: bool = False,
) -> mx.array:
    """MSE or L1 loss between predicted IoUs and thresholded mask IoUs."""

    inputs = _as_float_array(inputs)
    targets = _as_float_array(targets)
    pred_ious = _as_float_array(pred_ious)
    if inputs.ndim != 4 or targets.ndim != 4:
        raise AssertionError("iou_loss expects rank-4 inputs and targets.")

    pred_mask = _reshape(inputs, (*inputs.shape[:2], -1)) > 0
    gt_mask = _reshape(targets, (*targets.shape[:2], -1)) > 0
    area_i = mx.sum((pred_mask & gt_mask).astype(mx.float32), axis=-1)
    area_u = mx.sum((pred_mask | gt_mask).astype(mx.float32), axis=-1)
    actual_ious = area_i / mx.maximum(area_u, _scalar_array(1.0))

    loss = (
        _l1_loss(pred_ious, actual_ious)
        if use_l1_loss
        else _mse_loss(pred_ious, actual_ious)
    )
    if loss_on_multimask:
        return loss / _num_boxes_array(num_boxes)
    return mx.sum(loss) / _num_boxes_array(num_boxes)


def segment_miou(source: ArrayValue, target: ArrayValue) -> mx.array:
    """Compute mean IoU between paired semantic masks."""

    source = _as_bool_array(source)
    target = _as_bool_array(target)
    if source.shape != target.shape:
        raise AssertionError("The two masks must have the same shape.")
    if source.ndim != 3:
        raise AssertionError("The masks must be 3D.")

    flat_target = _reshape(target, (target.shape[0], -1))
    valid_mask = mx.any(flat_target, axis=1)
    valid_targets = mx.sum(valid_mask.astype(mx.float32))
    flat_source = _reshape(source, (source.shape[0], -1))
    intersection = mx.sum((flat_source & flat_target).astype(mx.float32), axis=1)
    union = mx.sum((flat_source | flat_target).astype(mx.float32), axis=1)
    iou = intersection / (union + 1e-8)
    iou = mx.where(valid_mask, iou, mx.zeros_like(iou))
    return mx.where(
        valid_targets > 0,
        mx.sum(iou) / mx.maximum(valid_targets, _scalar_array(1.0)),
        _scalar_array(1.0),
    )


def _keep_only_trk_queries_in_match_inds(
    inds: MatchIndices, Q_det: int
) -> MatchIndices:
    batch_idx, src_idx, *rest = inds
    keep = src_idx >= Q_det
    keep_count = int(mx.sum(keep.astype(mx.int64)).item())
    sorted_indices = mx.argsort(keep.astype(mx.int64))
    keep_indices = sorted_indices[-keep_count:] if keep_count else sorted_indices[:0]
    if len(inds) == 2:
        return batch_idx[keep_indices], src_idx[keep_indices] - Q_det
    target_idx = rest[0]
    return (
        batch_idx[keep_indices],
        src_idx[keep_indices] - Q_det,
        None if target_idx is None else target_idx[keep_indices],
    )


class LossWithWeights(nn.Module, Generic[_LossArgs]):
    def __init__(
        self,
        weight_dict: _Weights | None,
        compute_aux: bool,
        supports_o2m_loss: bool = True,
    ) -> None:
        super().__init__()
        self.weight_dict: _Weights = weight_dict if weight_dict is not None else {}
        self.compute_aux = compute_aux
        self.supports_o2m_loss = supports_o2m_loss
        self.target_keys: list[str] = []

    def _call(self, *args: object, is_aux: bool = False, **kwargs: object) -> LossDict:
        if is_aux and not self.compute_aux:
            return {CORE_LOSS_KEY: _empty_scalar()}
        get_loss = cast(Callable[..., LossDict], self.get_loss)
        losses = get_loss(*args, **kwargs)
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    # Loss subclasses accept different keyword contracts; keep the runtime
    # dispatcher typed without forcing one subclass signature onto the others.
    __call__: Callable[..., LossDict] = _call

    def get_loss(self, *_args: _LossArgs.args, **_kwargs: _LossArgs.kwargs) -> LossDict:
        raise NotImplementedError

    def reduce_loss(self, losses: LossDict) -> mx.array:
        reduced_loss = _empty_scalar()
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss = reduced_loss + losses[loss_key] * weight
        return reduced_loss


class Boxes(
    LossWithWeights[[BoxesOutputMap, BoxesTargetMap, MatchIndices, ArrayValue]]
):
    def __init__(
        self,
        weight_dict: _Weights | None = None,
        compute_aux: bool = True,
        apply_loss_to_det_queries_in_video_grounding: bool = True,
    ) -> None:
        super().__init__(weight_dict, compute_aux)
        self.apply_loss_to_det_queries_in_video_grounding = (
            apply_loss_to_det_queries_in_video_grounding
        )
        self.target_keys.extend(["boxes", "boxes_xyxy"])

    def get_loss(
        self,
        outputs: BoxesOutputMap,
        targets: BoxesTargetMap,
        indices: MatchIndices,
        num_boxes: ArrayValue,
    ) -> LossDict:
        if (
            outputs.get("is_video_grounding_batch", False)
            and not self.apply_loss_to_det_queries_in_video_grounding
        ):
            indices = _keep_only_trk_queries_in_match_inds(
                indices, Q_det=cast(_QDetMap, outputs)["Q_det"]
            )

        if "pred_boxes" not in outputs:
            raise AssertionError("Boxes loss requires outputs['pred_boxes'].")
        src_boxes = _gather_matched(outputs["pred_boxes"], indices)
        src_boxes_xyxy = _gather_matched(outputs["pred_boxes_xyxy"], indices)
        target_boxes = _target_select(targets["boxes"], indices)
        target_boxes_giou = _target_select(targets["boxes_xyxy"], indices)

        loss_bbox = _l1_loss(src_boxes, target_boxes)
        loss_giou = 1 - _fast_diag_generalized_box_iou(
            src_boxes_xyxy, target_boxes_giou
        )
        num_boxes = _num_boxes_array(num_boxes)
        return {
            "loss_bbox": mx.sum(loss_bbox) / num_boxes,
            "loss_giou": mx.sum(loss_giou) / num_boxes,
        }


class Masks(
    LossWithWeights[[MasksOutputMap, MasksTargetMap, MatchIndices, ArrayValue]]
):
    def __init__(
        self,
        weight_dict: _Weights | None = None,
        compute_aux: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        num_sample_points: int | None = None,
        oversample_ratio: int | None = None,
        importance_sample_ratio: float | None = None,
        apply_loss_to_det_queries_in_video_grounding: bool = True,
    ) -> None:
        super().__init__(weight_dict, compute_aux)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_sample_points = num_sample_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.apply_loss_to_det_queries_in_video_grounding = (
            apply_loss_to_det_queries_in_video_grounding
        )
        self.target_keys.extend(["masks", "is_valid_mask"])

    def _sampled_loss(
        self, src_masks: mx.array, target_masks: mx.array, num_boxes: ArrayValue
    ) -> LossDict:
        if src_masks.ndim != 3 or target_masks.ndim != 3:
            raise AssertionError("sampled mask loss expects rank-3 masks.")
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        point_coords = _get_uncertain_points(
            src_masks,
            _calculate_uncertainty,
            cast(int, self.num_sample_points),
            cast(int, self.oversample_ratio),
            cast(float, self.importance_sample_ratio),
        )
        sampled_target_masks = _point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        sampled_src_masks = _point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        return {
            "loss_mask": sigmoid_focal_loss(
                sampled_src_masks,
                sampled_target_masks,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            ),
            "loss_dice": dice_loss(sampled_src_masks, sampled_target_masks, num_boxes),
        }

    def get_loss(
        self,
        outputs: MasksOutputMap,
        targets: MasksTargetMap,
        indices: MatchIndices,
        num_boxes: ArrayValue,
    ) -> LossDict:
        if "pred_masks" not in outputs:
            raise AssertionError("Masks loss requires outputs['pred_masks'].")
        if "is_valid_mask" not in targets:
            raise AssertionError("Masks loss requires targets['is_valid_mask'].")
        if (
            outputs.get("is_video_grounding_batch", False)
            and not self.apply_loss_to_det_queries_in_video_grounding
        ):
            indices = _keep_only_trk_queries_in_match_inds(
                indices, Q_det=cast(_QDetMap, outputs)["Q_det"]
            )

        src_masks = outputs["pred_masks"]
        if targets["masks"] is None:
            zero = _empty_scalar(dtype=src_masks.dtype)
            return {"loss_mask": zero, "loss_dice": zero}

        target_masks = _target_select(targets["masks"], indices).astype(src_masks.dtype)
        keep = _target_select(targets["is_valid_mask"], indices).astype(mx.bool_)
        src_masks = _gather_matched(src_masks, indices)
        src_masks = src_masks[keep]
        target_masks = target_masks[keep]

        if self.num_sample_points is not None:
            return self._sampled_loss(src_masks, target_masks, num_boxes)

        if target_masks.shape[0] == 0 and src_masks.shape[0] == 0:
            src_masks = _reshape(src_masks, (src_masks.shape[0], -1))
            target_masks = _reshape(target_masks, src_masks.shape)
        else:
            if src_masks.ndim == 3:
                src_masks = src_masks[:, None]
            src_masks = _interpolate(
                src_masks.astype(mx.float32),
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            src_masks = _reshape(src_masks[:, 0], (src_masks.shape[0], -1))
            target_masks = _reshape(target_masks, (target_masks.shape[0], -1))

        return {
            "loss_mask": sigmoid_focal_loss(
                src_masks,
                target_masks,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            ),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }


class SemanticSegCriterion(LossWithWeights[[SemanticOutputMap, SemanticTargetMap]]):
    def __init__(
        self,
        weight_dict: _Weights,
        focal: bool = False,
        focal_alpha: float = 0.6,
        focal_gamma: float = 1.6,
        downsample: bool = True,
        presence_head: bool = False,
        presence_loss: bool = True,
    ) -> None:
        super().__init__(weight_dict, False)
        self.focal = focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.downsample = downsample
        self.presence_head = presence_head
        self.presence_loss = presence_loss

    def get_loss(
        self, out_dict: SemanticOutputMap, targets: SemanticTargetMap
    ) -> LossDict:
        outputs = _as_float_array(out_dict["semantic_seg"])
        presence_logit = out_dict.get("presence_logit")
        if (
            "semantic_masks" in targets
            and targets["semantic_masks"] is not None
            and targets["semantic_masks"].shape[0] > 0
        ):
            semantic_targets = _as_bool_array(targets["semantic_masks"])
            if self.downsample:
                semantic_targets = (
                    _interpolate(
                        semantic_targets.astype(mx.float32)[:, None],
                        size=outputs.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                    > 0.5
                )
        else:
            if self.downsample:
                segments = (
                    _interpolate(
                        _as_float_array(targets["masks"])[:, None],
                        size=outputs.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                    > 0.5
                )
            else:
                segments = _as_bool_array(targets["masks"])
            semantic_targets = instance_masks_to_semantic_masks(
                segments, targets["num_boxes"]
            )

        if not self.downsample:
            outputs = _interpolate(
                outputs.astype(mx.float32),
                size=semantic_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        logits = outputs[:, 0]
        semantic_float = semantic_targets.astype(mx.float32)
        if self.focal:
            loss = sigmoid_focal_loss(
                _reshape(logits, (logits.shape[0], -1)),
                _reshape(semantic_float, (semantic_float.shape[0], -1)),
                num_boxes=semantic_float.shape[0],
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduce=not self.presence_head,
            )
            if self.presence_head:
                loss = mx.mean(loss, axis=1)
        else:
            loss = _bce_with_logits(logits, semantic_float)
            loss = (
                mx.mean(_reshape(loss, (loss.shape[0], -1)), axis=1)
                if self.presence_head
                else mx.mean(loss)
            )

        loss_dice = dice_loss(
            _reshape(logits, (logits.shape[0], -1)),
            _reshape(semantic_float, (semantic_float.shape[0], -1)),
            semantic_float.shape[0],
            reduce=not self.presence_head,
        )
        miou = segment_miou(mx.sigmoid(logits) > 0.5, semantic_targets)
        loss_dict: LossDict = {}

        if self.presence_head:
            if presence_logit is None:
                raise ValueError(
                    "presence_head=True requires out_dict['presence_logit']."
                )
            presence_target = mx.any(
                _reshape(semantic_targets, (semantic_targets.shape[0], -1)),
                axis=-1,
            )
            if self.presence_loss:
                presence_loss = mx.mean(
                    _bce_with_logits(
                        _reshape(_as_float_array(presence_logit), (-1,)),
                        presence_target.astype(mx.float32),
                    )
                )
                presence_acc = mx.mean(
                    (
                        mx.equal(
                            mx.sigmoid(_reshape(_as_float_array(presence_logit), (-1,)))
                            > 0.5,
                            presence_target,
                        )
                    ).astype(mx.float32)
                )
            else:
                presence_loss = _empty_scalar(dtype=loss.dtype)
                presence_acc = _empty_scalar(dtype=loss.dtype)

            valid_count = mx.sum(presence_target.astype(mx.float32)) + 1e-6
            loss = mx.sum(loss * presence_target.astype(loss.dtype)) / valid_count
            loss_dice = (
                mx.sum(loss_dice * presence_target.astype(loss_dice.dtype))
                / valid_count
            )
            loss_dict["loss_semantic_presence"] = presence_loss
            loss_dict["presence_acc"] = presence_acc

        loss_dict.update(
            {
                "loss_semantic_seg": loss,
                "loss_semantic_dice": loss_dice,
                "miou_semantic_seg": miou,
            }
        )
        return loss_dict


class IABCEMdetr(
    LossWithWeights[[IABCEOutputMap, IABCETargetMap, MatchIndices, ArrayValue]]
):
    def __init__(
        self,
        pos_weight: float,
        weight_dict: _Weights | None = None,
        compute_aux: bool = True,
        gamma: float = 0.0,
        weak_loss: bool = True,
        alpha: float = 0.25,
        pad_n_queries: int | None = None,
        pad_scale_pos: float = 1.0,
        use_separate_loss_for_det_and_trk: bool = False,
        num_det_queries: int | None = None,
        det_exhaustive_loss_scale_pos: float = 1.0,
        det_exhaustive_loss_scale_neg: float = 1.0,
        det_non_exhaustive_loss_scale_pos: float = 1.0,
        det_non_exhaustive_loss_scale_neg: float = 1.0,
        trk_loss_scale_pos: float = 1.0,
        trk_loss_scale_neg: float = 1.0,
        no_loss_for_fp_propagation: bool = False,
        apply_loss_to_det_queries_in_video_grounding: bool = True,
        use_presence: bool = False,
        use_presence_semgseg: bool = False,
        presence_alpha: float = 0.5,
        presence_gamma: float = 0.0,
        pos_focal: bool = False,
    ) -> None:
        del (
            num_det_queries,
            det_exhaustive_loss_scale_pos,
            det_exhaustive_loss_scale_neg,
            det_non_exhaustive_loss_scale_pos,
            det_non_exhaustive_loss_scale_neg,
            trk_loss_scale_pos,
            trk_loss_scale_neg,
            no_loss_for_fp_propagation,
        )
        if use_separate_loss_for_det_and_trk:
            raise_unsupported(
                "sam3_mlx.train.loss.loss_fns.IABCEMdetr(use_separate_loss_for_det_and_trk=True)",
                reason="training-loop",
                detail=(
                    "Separate detection/tracking loss scaling is a video/tracking "
                    "path; this MLX port is image-safe only."
                ),
            )
        super().__init__(weight_dict, compute_aux)
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.weak_loss = weak_loss
        self.alpha = alpha
        self.target_keys.append("boxes_xyxy")
        if self.weak_loss:
            self.target_keys.append("is_exhaustive")
        self.pad_n_queries = pad_n_queries
        self.pad_scale_pos = pad_scale_pos
        if self.pad_scale_pos != 1.0 and self.pad_n_queries is None:
            raise AssertionError("pad_scale_pos requires pad_n_queries.")
        self.apply_loss_to_det_queries_in_video_grounding = (
            apply_loss_to_det_queries_in_video_grounding
        )
        self.use_presence = use_presence
        self.use_presence_semgseg = use_presence_semgseg
        if self.use_presence_semgseg and not self.use_presence:
            raise AssertionError("use_presence_semgseg requires use_presence.")
        if self.use_presence:
            self.target_keys.extend(["object_ids_padded", "boxes_padded"])
        self.presence_alpha = presence_alpha
        self.presence_gamma = presence_gamma
        self.pos_focal = pos_focal

    def _target_arrays(
        self,
        src_logits: mx.array,
        outputs: IABCETargetArraysOutputMap,
        targets: IABCETargetArraysTargetMap,
        indices: MatchIndices,
    ) -> tuple[mx.array, mx.array]:
        batch_idx = _reshape(_as_array(indices[0], dtype=mx.int64), (-1,))
        src_idx = _reshape(_as_array(indices[1], dtype=mx.int64), (-1,))
        linear_idx = batch_idx * src_logits.shape[1] + src_idx
        target_classes = mx.zeros(src_logits.shape, dtype=mx.float32)
        target_classes = mx.put_along_axis(
            target_classes,
            linear_idx,
            mx.ones(linear_idx.shape, dtype=mx.float32),
            axis=None,
        )

        positive_targets = target_classes
        if linear_idx.size > 0:
            src_boxes_xyxy = _gather_matched(outputs["pred_boxes_xyxy"], indices)
            target_boxes_giou = _target_select(targets["boxes_xyxy"], indices)
            iou = _fast_diag_box_iou(src_boxes_xyxy, target_boxes_giou)
            matched_prob = mx.sigmoid(src_logits)[indices[0], indices[1]]
            soft_targets = (matched_prob**self.alpha) * (iou ** (1 - self.alpha))
            soft_targets = mx.maximum(soft_targets, _scalar_array(0.01))
            positive_targets = mx.put_along_axis(
                positive_targets,
                linear_idx,
                soft_targets.astype(mx.float32),
                axis=None,
            )
        return target_classes, positive_targets

    def _binary_f1(self, probabilities: mx.array, target_classes: mx.array) -> mx.array:
        pred = probabilities > 0.5
        target = target_classes > 0.5
        tp = mx.sum((pred & target).astype(mx.float32))
        fp = mx.sum((pred & ~target).astype(mx.float32))
        fn = mx.sum((~pred & target).astype(mx.float32))
        denom = 2 * tp + fp + fn
        return mx.where(denom > 0, (2 * tp) / denom, _empty_scalar())

    def _presence_loss(
        self, outputs: IABCEOutputMap, targets: IABCETargetMap, loss_bce: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        presence_targets = cast(_PresenceTargetMap, targets)
        gt_ids = _as_array(presence_targets["object_ids_padded"], dtype=mx.int64)
        gt_boxes = _as_float_array(presence_targets["boxes_padded"])
        visible = (gt_ids >= 0) & (gt_boxes[..., 2] > 0) & (gt_boxes[..., 3] > 0)
        keep_loss = mx.any(visible, axis=-1).astype(mx.float32)[:, None]
        loss_bce = loss_bce * keep_loss

        if self.use_presence_semgseg or "presence_logit_dec" not in outputs:
            return loss_bce, _empty_scalar(), _empty_scalar()

        presence_logits = _reshape(
            _as_float_array(outputs["presence_logit_dec"]), keep_loss.shape
        )
        presence_loss = sigmoid_focal_loss(
            presence_logits,
            keep_loss,
            num_boxes=presence_logits.shape[0],
            alpha=self.presence_alpha,
            gamma=self.presence_gamma,
        )
        pred = (mx.sigmoid(presence_logits) > 0.5).astype(mx.float32)
        presence_acc = mx.mean(mx.equal(pred, keep_loss).astype(mx.float32))
        return loss_bce, presence_loss, presence_acc

    def get_loss(
        self,
        outputs: IABCEOutputMap,
        targets: IABCETargetMap,
        indices: MatchIndices,
        num_boxes: ArrayValue,
    ) -> LossDict:
        if len(outputs["pred_logits"].shape) <= 2:
            raise AssertionError("Incorrect predicted logits shape")
        if outputs["pred_logits"].shape[-1] != 1:
            raise AssertionError("Incorrect predicted logits shape")
        if "pred_boxes_xyxy" not in outputs:
            raise AssertionError("IABCEMdetr requires outputs['pred_boxes_xyxy'].")

        src_logits = _as_float_array(outputs["pred_logits"]).squeeze(-1)
        prob = mx.sigmoid(src_logits)
        target_classes, positive_target_classes = self._target_arrays(
            src_logits, outputs, targets, indices
        )

        if self.pos_focal:
            loss_bce = sigmoid_focal_loss(
                src_logits,
                positive_target_classes,
                num_boxes=1,
                alpha=0.5,
                gamma=self.gamma,
                reduce=False,
            )
        else:
            loss_bce = _bce_with_logits(src_logits, positive_target_classes)
        loss_bce = loss_bce * target_classes * self.pos_weight

        if _is_int(self.pad_n_queries) and loss_bce.shape[1] < self.pad_n_queries:
            loss_bce = loss_bce * self.pad_scale_pos

        loss_bce = loss_bce + (
            _bce_with_logits(src_logits, target_classes)
            * (1 - target_classes)
            * (prob**self.gamma)
        )

        if (
            outputs.get("is_video_grounding_batch", False)
            and not self.apply_loss_to_det_queries_in_video_grounding
        ):
            q_det = int(cast(_QDetMap, outputs)["Q_det"])
            keep_cols = mx.arange(loss_bce.shape[1], dtype=mx.int64) >= q_det
            loss_bce = loss_bce * keep_cols.astype(loss_bce.dtype)[None, :]

        presence_loss = _empty_scalar(dtype=src_logits.dtype)
        presence_dec_acc = _empty_scalar(dtype=src_logits.dtype)
        if self.use_presence:
            loss_bce, presence_loss, presence_dec_acc = self._presence_loss(
                outputs, targets, loss_bce
            )

        if self.weak_loss:
            if "is_exhaustive" not in targets:
                raise AssertionError("weak IABCEMdetr loss requires is_exhaustive.")
            is_exhaustive = _as_array(targets["is_exhaustive"]).astype(mx.bool_)
            if loss_bce.shape[0] != is_exhaustive.shape[0]:
                raise AssertionError("is_exhaustive batch dimension mismatch.")
            if is_exhaustive.ndim != 1:
                raise AssertionError("is_exhaustive must be rank 1.")
            loss_mask = mx.broadcast_to(~is_exhaustive[:, None], loss_bce.shape)
            loss_mask = loss_mask & (target_classes < 0.5)
            loss_mask = ~loss_mask
            loss_bce = loss_bce * loss_mask.astype(loss_bce.dtype)
            loss_bce = mx.sum(loss_bce) / (mx.sum(loss_mask.astype(mx.float32)) + 1e-6)
        elif self.pad_n_queries is None or loss_bce.shape[1] >= self.pad_n_queries:
            loss_bce = mx.mean(loss_bce)
        else:
            if not _is_int(self.pad_n_queries):
                raise AssertionError("pad_n_queries must be an int.")
            if loss_bce.shape[1] >= self.pad_n_queries:
                raise AssertionError(
                    "The number of predictions is more than the expected total "
                    f"after padding. Got {loss_bce.shape[1]} predictions."
                )
            loss_bce = mx.sum(loss_bce) / (self.pad_n_queries * loss_bce.shape[0])

        return {
            "loss_ce": loss_bce,
            "ce_f1": self._binary_f1(prob, target_classes),
            "presence_loss": presence_loss,
            "presence_dec_acc": presence_dec_acc,
        }


class Det2TrkAssoc(LossWithWeights[[]]):
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise_unsupported(
            "sam3_mlx.train.loss.loss_fns.Det2TrkAssoc",
            reason="training-loop",
            detail=(
                "Det2TrkAssoc is a video/tracking association loss; this MLX "
                "port is image-safe only."
            ),
        )


class TrackingByDetectionAssoc(LossWithWeights[[]]):
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise_unsupported(
            "sam3_mlx.train.loss.loss_fns.TrackingByDetectionAssoc",
            reason="training-loop",
            detail=(
                "TrackingByDetectionAssoc is a video/tracking association loss; "
                "this MLX port is image-safe only."
            ),
        )


__all__ = [
    "CORE_LOSS_KEY",
    "Det2TrkAssoc",
    "Boxes",
    "IABCEMdetr",
    "LossWithWeights",
    "MLX_LOSS_FNS_BASE_COMMIT",
    "Masks",
    "SemanticSegCriterion",
    "TrackingByDetectionAssoc",
    "TRAINING_LOSS_CPU_BOUNDARIES",
    "accuracy",
    "dice_loss",
    "instance_masks_to_semantic_masks",
    "iou_loss",
    "segment_miou",
    "sigmoid_focal_loss",
]
