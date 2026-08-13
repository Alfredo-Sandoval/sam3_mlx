"""ViT window-aligned image resolutions."""

from __future__ import annotations

PATCH_SIZE = 14
WINDOW_SIZE = 24
ALIGNED_WINDOW_STRIDE = PATCH_SIZE * WINDOW_SIZE
CANONICAL_ALIGNED_RESOLUTIONS = (336, 672, 1008)
DEFAULT_IMAGE_RESOLUTION = 1008
FAST_TIER_RESOLUTION = 336


def is_patch_aligned(resolution: int) -> bool:
    return resolution > 0 and resolution % PATCH_SIZE == 0


def is_window_aligned(resolution: int) -> bool:
    return resolution > 0 and resolution % ALIGNED_WINDOW_STRIDE == 0


def window_layout_is_exact(
    height: int,
    width: int,
    window_size: int = WINDOW_SIZE,
) -> bool:
    return (
        height > 0
        and width > 0
        and height % window_size == 0
        and width % window_size == 0
    )
