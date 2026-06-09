"""COCO-style RLE helpers implemented with NumPy only."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _as_bool_masks(orig_mask: Any) -> np.ndarray:
    masks = np.asarray(orig_mask)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise AssertionError("Mask must be of shape (N, H, W)")
    return masks.astype(bool, copy=False)


def _compressed_counts_to_list(counts: str | bytes) -> list[int]:
    """Decode COCO's compact RLE counts string.

    This follows the same variable-length delta encoding used by pycocotools.
    """
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    result: list[int] = []
    ptr = 0
    while ptr < len(counts):
        shift = 0
        value = 0
        more = True
        while more:
            c = ord(counts[ptr]) - 48
            ptr += 1
            value |= (c & 0x1F) << shift
            more = bool(c & 0x20)
            shift += 5
            if not more and (c & 0x10):
                value |= -1 << shift
        if len(result) > 2:
            value += result[-2]
        result.append(int(value))
    return result


def _normalize_counts(counts: Any) -> list[int]:
    if isinstance(counts, str | bytes):
        stripped = counts.decode("ascii") if isinstance(counts, bytes) else counts
        if stripped and all(
            ch.isdigit() or ch.isspace() or ch in ",[]" for ch in stripped
        ):
            return [
                int(part)
                for part in stripped.replace("[", " ")
                .replace("]", " ")
                .replace(",", " ")
                .split()
            ]
        return _compressed_counts_to_list(counts)
    if isinstance(counts, np.ndarray):
        counts = counts.tolist()
    if not isinstance(counts, Iterable):
        raise TypeError(f"Unsupported RLE counts type: {type(counts)!r}")
    return [int(count) for count in counts]


def rle_encode(orig_mask, return_areas=False):
    """Encode masks of shape ``(N, H, W)`` into uncompressed COCO RLE dicts."""
    masks = _as_bool_masks(orig_mask)
    if masks.size == 0:
        return []

    batch_rles = []
    for mask in masks:
        pixels = mask.reshape(-1, order="F").astype(np.uint8)
        counts: list[int] = []
        last = 0
        run = 0
        for value in pixels:
            value_i = int(value)
            if value_i == last:
                run += 1
            else:
                counts.append(run)
                run = 1
                last = value_i
        counts.append(run)
        rle = {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}
        if return_areas:
            rle["area"] = int(mask.sum())
        batch_rles.append(rle)
    return batch_rles


def robust_rle_encode(masks):
    """Encode masks into uncompressed COCO RLE dicts."""
    return rle_encode(masks)


def rle_decode(rle: dict[str, Any]) -> np.ndarray:
    """Decode an uncompressed or compressed COCO RLE dict to a bool mask."""
    h, w = (int(v) for v in rle["size"])
    counts = _normalize_counts(rle["counts"])
    values = np.empty(sum(counts), dtype=np.uint8)
    index = 0
    value = 0
    for run in counts:
        if run:
            values[index : index + run] = value
        index += run
        value = 1 - value
    expected = h * w
    if values.size < expected:
        values = np.pad(values, (0, expected - values.size), constant_values=0)
    elif values.size > expected:
        values = values[:expected]
    return values.reshape((h, w), order="F").astype(bool)


def rle_area(rle: dict[str, Any]) -> int:
    """Return the foreground area for a COCO RLE dict."""
    counts = _normalize_counts(rle["counts"])
    return int(sum(counts[1::2]))


def rle_to_bbox(rle: dict[str, Any]) -> list[float]:
    """Return ``[x, y, w, h]`` for a COCO RLE dict."""
    mask = rle_decode(rle)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def ann_to_rle(segm, im_info):
    """Convert an annotation segmentation to an RLE when possible."""
    h, w = int(im_info["height"]), int(im_info["width"])
    if isinstance(segm, dict):
        if "size" not in segm:
            segm = {**segm, "size": [h, w]}
        return segm
    if isinstance(segm, list):
        from .masks import polygons_to_bitmask

        mask = polygons_to_bitmask([np.asarray(poly) for poly in segm], h, w)
        return rle_encode(mask[None, :, :])[0]
    raise TypeError(f"Unsupported segmentation type: {type(segm)!r}")
