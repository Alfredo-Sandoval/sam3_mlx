"""Color map helpers ported to NumPy."""

from __future__ import annotations

import random

import numpy as np
from numpy.typing import NDArray


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


def _validate_maximum(maximum: int) -> None:
    if isinstance(maximum, bool) or maximum not in (255, 1):
        raise AssertionError(maximum)


def colormap(rgb: bool = False, maximum: int = 255) -> NDArray[np.float32]:
    """Return the official bright color table as an ``Nx3`` NumPy array."""
    _validate_maximum(maximum)
    colors = _COLORS * maximum
    if not rgb:
        colors = colors[:, ::-1]
    return colors


def random_color(rgb: bool = False, maximum: int = 255) -> NDArray[np.float32]:
    """Return one random color from the color table."""
    _validate_maximum(maximum)
    ret = _COLORS[np.random.randint(0, len(_COLORS))] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def random_colors(
    N: int, rgb: bool = False, maximum: int = 255
) -> list[NDArray[np.float32]]:
    """Return ``N`` distinct random colors from the color table."""
    if isinstance(N, bool):
        raise TypeError("N must be an integer, not bool.")
    _validate_maximum(maximum)
    if N > len(_COLORS):
        raise ValueError(f"Cannot sample {N} unique colors from {len(_COLORS)} colors")
    ret = [_COLORS[i] * maximum for i in random.sample(range(len(_COLORS)), N)]
    if not rgb:
        ret = [color[::-1] for color in ret]
    return ret
