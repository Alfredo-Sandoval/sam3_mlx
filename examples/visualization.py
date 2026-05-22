"""Visualization helpers for the example notebooks."""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from PIL import ImageDraw


def draw_box_on_image(image, box: Sequence[float], color=(0, 255, 0)):
    """Return a copy of `image` with a rectangle drawn in XYWH coordinates."""
    result = image.convert("RGB").copy()
    x, y, w, h = [int(v) for v in box]
    draw = ImageDraw.Draw(result)
    draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
    return result


def normalize_bbox(bbox_xywh, img_w: int, img_h: int):
    """Normalize XYWH boxes by image width and height."""
    scale = [img_w, img_h, img_w, img_h]
    if isinstance(bbox_xywh, list):
        if len(bbox_xywh) != 4:
            raise ValueError("bbox_xywh list must have 4 elements.")
        return [value / divisor for value, divisor in zip(bbox_xywh, scale)]

    if not isinstance(bbox_xywh, mx.array):
        raise TypeError("bbox_xywh must be a list or MLX array.")
    if bbox_xywh.shape[-1] != 4:
        raise ValueError("bbox_xywh must have last dimension size 4.")
    return bbox_xywh / mx.array(scale)


def plot_bbox(
    img_height,
    img_width,
    box,
    box_format="XYXY",
    relative_coords=True,
    color="r",
    linestyle="solid",
    text=None,
    ax=None,
):
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise ValueError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
        )
    )
    if text is not None:
        ax.text(
            x,
            y - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": "w", "alpha": 0.75, "pad": 2},
        )


def plot_mask(mask, color="r", ax=None):
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)


def plot_results(img, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    boxes = np.asarray(results["boxes"])
    masks = np.asarray(results["masks"])
    scores = np.asarray(results["scores"])
    colors = plt.get_cmap("tab20").colors
    for i, score in enumerate(scores):
        color = colors[i % len(colors)]
        plot_mask(masks[i].squeeze(0), color=color)
        width, height = img.size
        plot_bbox(
            height,
            width,
            boxes[i],
            text=f"(id={i}, prob={float(score.item()):.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )
