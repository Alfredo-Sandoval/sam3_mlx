from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from sam3_mlx.perflib.masks_ops import mask_iom as _mask_iom

MLX_TRAIN_MASKS_OPS_BASE_COMMIT = "c5c10874844917434cff889be1d64d008a79035d"


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


def _from_numpy(value: np.ndarray, like):
    if _is_mlx_array(like):
        import mlx.core as mx

        return mx.array(value)
    return value


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
    if not isinstance(counts, Sequence):
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


def _decode_coco_rle(rle: dict[str, Any]) -> np.ndarray:
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


def _rle_area(rle: dict[str, Any]) -> int:
    return int(_decode_coco_rle(rle).sum())


def _rle_intersection_area(rle1: dict[str, Any], rle2: dict[str, Any]) -> int:
    mask1 = _decode_coco_rle(rle1)
    mask2 = _decode_coco_rle(rle2)
    if mask1.shape != mask2.shape:
        raise ValueError("COCO RLE masks must have matching sizes.")
    return int(np.logical_and(mask1, mask2).sum())


def _polygons_to_mask(polygons, height: int, width: int) -> np.ndarray:
    mask = PILImage.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        draw.polygon([tuple(point) for point in points], outline=1, fill=1)
    return np.asarray(mask, dtype=bool)


def instance_masks_to_semantic_masks(instance_masks, num_instances):
    masks = np.asarray(instance_masks).astype(bool, copy=False)
    counts = np.asarray(num_instances, dtype=np.int64).reshape(-1)
    if masks.ndim != 3:
        raise ValueError("instance_masks must have shape (N, H, W).")
    outputs = []
    start = 0
    for count in counts:
        chunk = masks[start : start + count]
        if chunk.size == 0:
            outputs.append(np.zeros(masks.shape[1:], dtype=bool))
        else:
            outputs.append(np.any(chunk, axis=0))
        start += int(count)
    return _from_numpy(np.stack(outputs, axis=0), instance_masks)


def mask_intersection_vectorized(masks1, masks2):
    m1 = np.asarray(masks1).astype(bool, copy=False)
    m2 = np.asarray(masks2).astype(bool, copy=False)
    if m1.shape[1:] != m2.shape[1:]:
        raise ValueError("masks must have matching spatial shapes.")
    out = (
        m1.reshape(m1.shape[0], -1).astype(np.int64)
        @ m2.reshape(m2.shape[0], -1).astype(np.int64).T
    )
    return _from_numpy(out, masks1)


def mask_intersection(masks1, masks2, block_size=16):
    del block_size
    return mask_intersection_vectorized(masks1, masks2)


def mask_iom(masks1, masks2):
    return _mask_iom(masks1, masks2)


def compute_boundary(seg):
    seg_np = np.asarray(seg).astype(bool, copy=False)
    boundary = np.zeros_like(seg_np, dtype=bool)
    boundary[..., :, :-1] |= seg_np[..., :, :-1] ^ seg_np[..., :, 1:]
    boundary[..., :-1, :] |= seg_np[..., :-1, :] ^ seg_np[..., 1:, :]
    boundary[..., :-1, :-1] |= seg_np[..., :-1, :-1] ^ seg_np[..., 1:, 1:]
    return _from_numpy(boundary, seg)


def dilation(mask, kernel_size):
    mask_np = np.asarray(mask).astype(bool, copy=False)
    kernel_size = int(kernel_size)
    if kernel_size % 2 != 1:
        raise ValueError("dilation expects an odd kernel size.")
    pad = kernel_size // 2
    padded = np.pad(mask_np, [(0, 0), (pad, pad), (pad, pad)], mode="constant")
    out = np.zeros_like(mask_np, dtype=bool)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            out |= padded[:, dy : dy + mask_np.shape[1], dx : dx + mask_np.shape[2]]
    return _from_numpy(out, mask)


def compute_F_measure(
    gt_boundary_rle, gt_dilated_boundary_rle, dt_boundary_rle, dt_dilated_boundary_rle
):
    """Compute the boundary F-measure from precomputed COCO RLE boundaries."""

    gt_match = _rle_intersection_area(gt_boundary_rle, dt_dilated_boundary_rle)
    dt_match = _rle_intersection_area(dt_boundary_rle, gt_dilated_boundary_rle)

    n_dt = _rle_area(dt_boundary_rle)
    n_gt = _rle_area(gt_boundary_rle)
    if n_dt == 0 and n_gt > 0:
        precision = 1.0
        recall = 0.0
    elif n_dt > 0 and n_gt == 0:
        precision = 0.0
        recall = 1.0
    elif n_dt == 0 and n_gt == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = dt_match / float(n_dt)
        recall = gt_match / float(n_gt)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _mask_to_uncompressed_counts(mask: np.ndarray) -> list[int]:
    flat = mask.T.reshape(-1)
    counts: list[int] = []
    last = False
    run = 0
    for value in flat:
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
    """Encode a collection of boolean masks as local COCO RLE dicts."""

    masks_np = _to_numpy(masks)
    if masks_np.ndim != 3:
        raise ValueError("Mask must have shape (N, H, W).")
    if masks_np.dtype != np.bool_:
        raise TypeError("Mask must have boolean dtype.")
    return rle_encode(masks_np)


def ann_to_rle(segm, im_info):
    """Convert COCO polygons or RLE annotations to a local COCO RLE dict."""

    try:
        height = int(im_info["height"])
        width = int(im_info["width"])
    except KeyError as exc:
        raise ValueError("im_info must contain 'height' and 'width'.") from exc

    if isinstance(segm, list):
        mask = _polygons_to_mask(segm, height=height, width=width)
        return rle_encode(mask[None, :, :])[0]

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
    if isinstance(counts, (str, bytes)):
        if "size" not in segm:
            return {**segm, "size": [height, width]}
        return segm
    raise TypeError("COCO RLE counts must be a sequence, str, or bytes.")
