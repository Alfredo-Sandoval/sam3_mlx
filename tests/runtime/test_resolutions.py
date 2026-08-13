from sam3_mlx.release_contract import RELEASE_RESOLUTIONS
from sam3_mlx.resolutions import (
    ALIGNED_WINDOW_STRIDE,
    CANONICAL_ALIGNED_RESOLUTIONS,
    FAST_TIER_RESOLUTION,
    PATCH_SIZE,
    WINDOW_SIZE,
    is_patch_aligned,
    is_window_aligned,
    window_layout_is_exact,
)


def test_canonical_resolutions_are_exact_window_grids() -> None:
    assert PATCH_SIZE == 14
    assert WINDOW_SIZE == 24
    assert ALIGNED_WINDOW_STRIDE == 336
    assert FAST_TIER_RESOLUTION == 336
    assert CANONICAL_ALIGNED_RESOLUTIONS == (336, 672, 1008)
    assert all(is_window_aligned(value) for value in CANONICAL_ALIGNED_RESOLUTIONS)
    assert window_layout_is_exact(24, 24)
    assert window_layout_is_exact(48, 48)
    assert window_layout_is_exact(72, 72)


def test_504_is_patch_aligned_but_not_window_aligned() -> None:
    assert is_patch_aligned(504)
    assert not is_window_aligned(504)
    assert not window_layout_is_exact(36, 36)


def test_fast_tier_is_not_a_release_resolution() -> None:
    assert RELEASE_RESOLUTIONS == (1008, 672, 504)
    assert FAST_TIER_RESOLUTION not in RELEASE_RESOLUTIONS
    assert FAST_TIER_RESOLUTION == 336
