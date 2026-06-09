"""Color map helpers ported to NumPy."""

from __future__ import annotations

import random

import numpy as np


__all__ = ["colormap", "random_color", "random_colors"]

_COLORS = (
    np.array(
        [
            1.000,
            1.000,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            1.000,
            0.498,
            0.000,
            0.498,
            1.000,
            0.000,
            0.000,
            1.000,
            0.498,
            1.000,
            0.000,
            0.498,
            0.498,
            0.000,
            1.000,
            0.753,
            1.000,
            0.000,
            1.000,
            0.753,
            0.000,
            0.000,
            1.000,
            0.753,
            0.753,
            0.000,
            1.000,
            1.000,
            0.000,
            0.753,
            1.000,
            0.251,
            0.000,
            0.251,
            1.000,
            0.000,
            0.000,
            1.000,
            0.251,
            0.251,
            0.000,
            1.000,
            1.000,
            0.000,
            0.251,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def colormap(rgb=False, maximum=255):
    """Return the official bright color table as an ``Nx3`` NumPy array."""
    if maximum not in (255, 1):
        raise AssertionError(maximum)
    colors = _COLORS * maximum
    if not rgb:
        colors = colors[:, ::-1]
    return colors


def random_color(rgb=False, maximum=255):
    """Return one random color from the color table."""
    if maximum not in (255, 1):
        raise AssertionError(maximum)
    ret = _COLORS[np.random.randint(0, len(_COLORS))] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def random_colors(N, rgb=False, maximum=255):
    """Return ``N`` distinct random colors from the color table."""
    if maximum not in (255, 1):
        raise AssertionError(maximum)
    if N > len(_COLORS):
        raise ValueError(f"Cannot sample {N} unique colors from {len(_COLORS)} colors")
    ret = [_COLORS[i] * maximum for i in random.sample(range(len(_COLORS)), N)]
    if not rgb:
        ret = [color[::-1] for color in ret]
    return ret
