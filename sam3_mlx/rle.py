"""COCO-style RLE helpers used by packaged runtime surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw


def _is_mlx_array(value) -> bool:
    return type(value).__module__.startswith("mlx.")


def _to_numpy(value, dtype=None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        if _is_mlx_array(value):
            import mlx.core as mx

            mx.eval(value)
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _decode_compressed_rle_counts(counts: str | bytes) -> list[int]:
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")

    decoded: list[int] = []
    index = 0
    while index < len(counts):
        shift = 0
        value = 0
        while True:
            if index >= len(counts):
                raise ValueError("Compressed COCO RLE ended in the middle of a count.")
            char_value = ord(counts[index]) - 48
            index += 1
            value |= (char_value & 0x1F) << shift
            shift += 5
            if (char_value & 0x20) == 0:
                break
        if char_value & 0x10:
            value |= -1 << shift
        if len(decoded) > 2:
            value += decoded[-2]
        decoded.append(int(value))
    return decoded


def _normalize_rle_counts(counts: Any) -> list[int]:
    if isinstance(counts, (str, bytes)):
        return _decode_compressed_rle_counts(counts)
    if isinstance(counts, np.ndarray):
        counts = counts.tolist()
    if not isinstance(counts, Iterable):
        raise TypeError("COCO RLE counts must be a sequence, str, or bytes.")
    return [int(count) for count in counts]


def _encode_compressed_rle_counts(counts: Sequence[int]) -> str:
    """Encode uncompressed COCO RLE counts using pycocotools' string format."""

    chars: list[str] = []
    counts_list = [int(count) for count in counts]
    for index, count in enumerate(counts_list):
        if count < 0:
            raise ValueError("COCO RLE counts must be non-negative.")
        value = count
        if index > 2:
            value -= counts_list[index - 2]

        more = True
        while more:
            char_value = value & 0x1F
            value >>= 5
            more = value != -1 if char_value & 0x10 else value != 0
            if more:
                char_value |= 0x20
            chars.append(chr(char_value + 48))
    return "".join(chars)


def _encode_counts_for_size(counts: Sequence[int], height: int, width: int) -> str:
    counts_list = [int(count) for count in counts]
    if any(count < 0 for count in counts_list):
        raise ValueError("COCO RLE counts must be non-negative.")
    expected_total = int(height) * int(width)
    actual_total = sum(counts_list)
    if actual_total != expected_total:
        raise ValueError(
            f"COCO RLE counts cover {actual_total} pixels, expected "
            f"{expected_total} for size {(height, width)}."
        )
    return _encode_compressed_rle_counts(counts_list)


def _mask_to_uncompressed_counts(mask: np.ndarray) -> list[int]:
    pixels = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    counts: list[int] = []
    last = False
    run = 0
    for value in pixels:
        current = bool(value)
        if current == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = current
    counts.append(run)
    return counts


def rle_encode(orig_mask, return_areas=False):
    """Encode boolean ``(N, H, W)`` masks as local COCO RLE dictionaries."""

    masks = _to_numpy(orig_mask)
    if masks.ndim != 3:
        raise ValueError("Mask must have shape (N, H, W).")
    if masks.dtype != np.bool_:
        raise TypeError("Mask must have boolean dtype.")
    if masks.size == 0:
        return []

    encoded = []
    for mask in masks:
        counts = _mask_to_uncompressed_counts(mask)
        item = {
            "size": list(mask.shape),
            "counts": _encode_compressed_rle_counts(counts),
        }
        if return_areas:
            item["area"] = int(mask.sum())
        encoded.append(item)
    return encoded


def robust_rle_encode(masks):
    """Encode a collection of boolean masks as local COCO RLE dictionaries."""

    masks_np = _to_numpy(masks)
    if masks_np.ndim != 3:
        raise ValueError("Mask must have shape (N, H, W).")
    if masks_np.dtype != np.bool_:
        raise TypeError("Mask must have boolean dtype.")
    return rle_encode(masks_np)


def rle_decode(rle: dict[str, Any]) -> np.ndarray:
    """Decode a compressed or uncompressed COCO RLE dictionary."""

    if not isinstance(rle, dict):
        raise TypeError("COCO RLE must be a dict with 'size' and 'counts'.")
    if "size" not in rle or "counts" not in rle:
        raise ValueError("COCO RLE must contain 'size' and 'counts'.")

    height, width = [int(v) for v in rle["size"]]
    if height < 0 or width < 0:
        raise ValueError("COCO RLE size values must be non-negative.")

    counts = _normalize_rle_counts(rle["counts"])
    total = height * width
    flat = np.zeros((total,), dtype=bool)
    offset = 0
    value = False
    for count in counts:
        if count < 0:
            raise ValueError("COCO RLE counts must be non-negative.")
        next_offset = offset + count
        if next_offset > total:
            raise ValueError(
                f"COCO RLE decoded past {total} pixels for size {(height, width)}."
            )
        if value and next_offset > offset:
            flat[offset:next_offset] = True
        offset = next_offset
        value = not value
    if offset != total:
        raise ValueError(
            f"COCO RLE decoded to {offset} pixels, expected {total} "
            f"for size {(height, width)}."
        )
    return flat.reshape((width, height)).T


def rle_area(rle: dict[str, Any]) -> int:
    """Return foreground area for a COCO RLE dictionary."""

    return int(rle_decode(rle).sum())


def rle_to_bbox(rle: dict[str, Any]) -> list[float]:
    """Return ``[x, y, w, h]`` for a COCO RLE dictionary."""

    mask = rle_decode(rle)
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def ann_to_rle(segm, im_info):
    """Convert COCO polygons or RLE annotations to a local COCO RLE dict."""

    try:
        height = int(im_info["height"])
        width = int(im_info["width"])
    except KeyError as exc:
        raise ValueError("im_info must contain 'height' and 'width'.") from exc

    if isinstance(segm, list):
        mask = PILImage.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in segm:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                draw.polygon([tuple(point) for point in points], outline=1, fill=1)
        mask_array = np.asarray(mask, dtype=bool)
        return rle_encode(mask_array[None, :, :])[0]

    if not isinstance(segm, dict):
        raise TypeError("COCO segmentation must be polygons or an RLE dict.")
    if "counts" not in segm:
        raise ValueError("COCO RLE segmentation must contain 'counts'.")

    counts = segm["counts"]
    if isinstance(counts, np.ndarray):
        return {
            "size": [height, width],
            "counts": _encode_counts_for_size(counts.tolist(), height, width),
        }
    if isinstance(counts, Sequence) and not isinstance(counts, (str, bytes)):
        return {
            "size": [height, width],
            "counts": _encode_counts_for_size(counts, height, width),
        }
    return {"size": [height, width], "counts": counts}
