"""Mask overlap removal implemented with NumPy RLE decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from sam3_mlx.agent.contracts import AgentOutput
from sam3_mlx.agent.helpers.rle import rle_decode, rle_encode


type BoolArray = NDArray[np.bool_]
type NumericArray = NDArray[np.number]


def _bool_array(value: object) -> BoolArray:
    return cast(BoolArray, np.asarray(value, dtype=bool))


def mask_intersection(masks1: NumericArray, masks2: NumericArray) -> NumericArray:
    masks1 = np.asarray(masks1, dtype=bool).reshape(len(masks1), -1)
    masks2 = np.asarray(masks2, dtype=bool).reshape(len(masks2), -1)
    return masks1.astype(np.uint8) @ masks2.astype(np.uint8).T


def mask_iom(masks1: NumericArray, masks2: NumericArray) -> NumericArray:
    inter = mask_intersection(masks1, masks2).astype(np.float32)
    areas2 = np.asarray(masks2, dtype=bool).reshape(len(masks2), -1).sum(axis=1)
    return np.where(areas2 > 0, inter / areas2[None, :], 0.0)


def _decode_single_mask(mask_repr: object, h: int, w: int) -> BoolArray:
    if isinstance(mask_repr, Mapping):
        payload = cast(Mapping[str, object], mask_repr)
        rle = payload if "size" in payload else {**payload, "size": [h, w]}
        return rle_decode(rle)
    if isinstance(mask_repr, str):
        return rle_decode({"counts": mask_repr, "size": [h, w]})
    mask = _bool_array(mask_repr)
    if mask.shape != (h, w):
        raise ValueError(f"Expected mask shape {(h, w)}, got {mask.shape}")
    return mask


def _decode_masks_to_torch_bool(
    pred_masks: Sequence[object], h: int, w: int
) -> BoolArray:
    """Compatibility name returning a NumPy bool array, not a Torch tensor."""
    if not pred_masks:
        return np.empty((0, h, w), dtype=bool)
    return np.stack([_decode_single_mask(mask, h, w) for mask in pred_masks], axis=0)


def remove_overlapping_masks(
    sample: AgentOutput, iom_thresh: float = 0.3
) -> AgentOutput:
    """Remove later masks that substantially overlap earlier masks."""
    h, w = int(sample["orig_img_h"]), int(sample["orig_img_w"])
    masks = _decode_masks_to_torch_bool(sample.get("pred_masks", []), h, w)
    if len(masks) <= 1:
        return sample

    keep: list[int] = []
    occupied = np.zeros((h, w), dtype=bool)
    for idx, mask in enumerate(masks):
        area = int(mask.sum())
        overlap = int((mask & occupied).sum())
        if area == 0 or overlap / area <= iom_thresh:
            keep.append(idx)
            occupied |= mask

    filtered = sample.copy()
    filtered["pred_boxes"] = [sample["pred_boxes"][index] for index in keep]
    filtered["pred_scores"] = [sample["pred_scores"][index] for index in keep]
    filtered["pred_masks"] = [rle["counts"] for rle in rle_encode(masks[keep])]
    return filtered
