"""Small visualization/color helpers for the agent surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def rgb_to_hex(rgb_color):
    return "#" + "".join([hex(int(c))[2:].zfill(2) for c in rgb_color])


DEFAULT_COLOR_HEX_TO_NAME = {
    rgb_to_hex((255, 255, 0)): "yellow",
    rgb_to_hex((0, 255, 0)): "lime",
    rgb_to_hex((0, 255, 255)): "cyan",
    rgb_to_hex((255, 0, 255)): "magenta",
    rgb_to_hex((255, 0, 0)): "red",
    rgb_to_hex((255, 127, 0)): "orange",
    rgb_to_hex((127, 255, 0)): "chartreuse",
    rgb_to_hex((0, 255, 127)): "spring green",
    rgb_to_hex((255, 0, 127)): "rose",
    rgb_to_hex((127, 0, 255)): "violet",
    rgb_to_hex((192, 255, 0)): "electric lime",
    rgb_to_hex((255, 192, 0)): "vivid orange",
    rgb_to_hex((0, 255, 192)): "turquoise",
    rgb_to_hex((192, 0, 255)): "bright violet",
    rgb_to_hex((255, 0, 192)): "bright pink",
    rgb_to_hex((255, 64, 0)): "fiery orange",
    rgb_to_hex((64, 255, 0)): "bright chartreuse",
    rgb_to_hex((0, 255, 64)): "malachite",
    rgb_to_hex((64, 0, 255)): "deep violet",
    rgb_to_hex((255, 0, 64)): "hot pink",
}
DEFAULT_COLOR_PALETTE = list(DEFAULT_COLOR_HEX_TO_NAME.keys())


def _validate_color_hex(color_hex: str):
    color_hex = color_hex.lstrip("#")
    if not all(c in "0123456789abcdefABCDEF" for c in color_hex):
        raise ValueError("Invalid characters in color hash")
    if len(color_hex) not in (3, 6):
        raise ValueError("Invalid length of color hash")


@dataclass
class Color:
    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, color_hex: str):
        _validate_color_hex(color_hex)
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)

    @classmethod
    def to_hex(cls, color):
        return rgb_to_hex((color.r, color.g, color.b))

    def as_rgb(self) -> Tuple[int, int, int]:
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

    @classmethod
    def white(cls):
        return Color.from_hex("#ffffff")

    @classmethod
    def black(cls):
        return Color.from_hex("#000000")

    @classmethod
    def red(cls):
        return Color.from_hex("#ff0000")

    @classmethod
    def green(cls):
        return Color.from_hex("#00ff00")

    @classmethod
    def blue(cls):
        return Color.from_hex("#0000ff")


@dataclass
class ColorPalette:
    colors: List[Color]

    @classmethod
    def default(cls):
        return ColorPalette.from_hex(DEFAULT_COLOR_PALETTE)

    @classmethod
    def from_hex(cls, color_hex_list: List[str]):
        return cls([Color.from_hex(color_hex) for color_hex in color_hex_list])

    def by_idx(self, idx: int) -> Color:
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        return self.colors[idx % len(self.colors)]

    def find_farthest_color(self, img_array):
        pixels = np.asarray(img_array, dtype=np.float32).reshape(-1, 3)
        if pixels.size == 0:
            return self.by_idx(0), 0.0
        mean_rgb = pixels.mean(axis=0)
        palette = np.asarray(
            [color.as_rgb() for color in self.colors], dtype=np.float32
        )
        distances = np.linalg.norm(palette - mean_rgb[None, :], axis=1)
        idx = int(np.argmax(distances))
        return self.colors[idx], float(distances[idx])


def draw_box(ax, box_coord, alpha=0.8, edge_color="g", line_style="-", linewidth=2.0):
    import matplotlib.patches as patches

    x, y, w, h = box_coord
    ax.add_patch(
        patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=linewidth,
            edgecolor=edge_color,
            facecolor="none",
            linestyle=line_style,
            alpha=alpha,
        )
    )


def draw_text(
    ax,
    text,
    position,
    *,
    font_size=10,
    color="g",
    horizontal_alignment="left",
    rotation=0,
):
    ax.text(
        position[0],
        position[1],
        text,
        size=font_size,
        family="sans-serif",
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.7, "edgecolor": "none"},
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        zorder=10,
        rotation=rotation,
    )


def draw_mask(ax, mask, color=None, show_holes=True, alpha=0.5):
    mask = np.asarray(mask, dtype=bool)
    if color is None:
        color = np.array([0.0, 1.0, 0.0])
    color = np.asarray(color, dtype=float)
    if color.max() > 1:
        color = color / 255.0
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = color[:3]
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgba)


def _change_color_brightness(color, brightness_factor):
    color = np.asarray(color, dtype=float)
    return np.clip(color * brightness_factor, 0, 1)
