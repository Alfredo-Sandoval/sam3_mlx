"""Fail-fast connected-components Triton-kernel module for SAM3 MLX.

Mirrors the public surface of `sam3.perflib.triton.connected_components` at
upstream commit `2814fa619404a722d03e9a012e083e4f293a4e53` so callers can keep
their import paths, but raises `NotImplementedError` on invocation.
Triton kernels are not part of the SAM3 MLX runtime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from sam3_mlx._unsupported import unsupported_function


_UNSUPPORTED_DETAIL = "The official SAM3 implementation requires Triton kernels."
F = TypeVar("F", bound=Callable[..., object])


def _unsupported_kernel(name: str) -> Callable[[F], F]:
    return unsupported_function(
        f"sam3_mlx.perflib.triton.connected_components.{name}",
        reason="triton-kernel",
        alternative="sam3_mlx.perflib.connected_components",
        detail=_UNSUPPORTED_DETAIL,
    )


@_unsupported_kernel("_any_combine")
def _any_combine(a: object, b: object) -> None:
    return None


@_unsupported_kernel("tl_any")
def tl_any(a: object, dim: int = 0) -> None:
    return None


@_unsupported_kernel("_init_labels_kernel")
def _init_labels_kernel(
    input_ptr: object, labels_ptr: object, numel: object, BLOCK_SIZE: object
) -> None:
    return None


@_unsupported_kernel("find")
def find(labels_ptr: object, indices: object, mask: object) -> None:
    return None


@_unsupported_kernel("union")
def union(labels_ptr: object, a: object, b: object, process_mask: object) -> None:
    return None


@_unsupported_kernel("_merge_helper")
def _merge_helper(
    input_ptr: object,
    labels_ptr: object,
    base_offset: object,
    offsets_h: object,
    offsets_w: object,
    mask_2d: object,
    valid_current: object,
    current_values: object,
    current_labels: object,
    H: object,
    W: object,
    dx: object,
    dy: object,
) -> None:
    return None


@_unsupported_kernel("_local_prop_kernel")
def _local_prop_kernel(
    labels_ptr: object,
    input_ptr: object,
    H: object,
    W: object,
    BLOCK_SIZE_H: object,
    BLOCK_SIZE_W: object,
) -> None:
    return None


@_unsupported_kernel("_pointer_jump_kernel")
def _pointer_jump_kernel(
    labels_in_ptr: object,
    labels_out_ptr: object,
    numel: object,
    BLOCK_SIZE: object,
) -> None:
    return None


@_unsupported_kernel("_count_labels_kernel")
def _count_labels_kernel(
    labels_ptr: object, sizes_ptr: object, numel: object, BLOCK_SIZE: object
) -> None:
    return None


@_unsupported_kernel("_broadcast_sizes_kernel")
def _broadcast_sizes_kernel(
    labels_ptr: object,
    sizes_ptr: object,
    out_ptr: object,
    numel: object,
    BLOCK_SIZE: object,
) -> None:
    return None


@unsupported_function(
    "sam3_mlx.perflib.triton.connected_components.connected_components_triton",
    reason="triton-kernel",
    alternative="sam3_mlx.perflib.connected_components",
    detail=_UNSUPPORTED_DETAIL,
)
def connected_components_triton(input_tensor: object) -> None:
    return None


__all__ = [
    "_any_combine",
    "_broadcast_sizes_kernel",
    "_count_labels_kernel",
    "_init_labels_kernel",
    "_local_prop_kernel",
    "_merge_helper",
    "_pointer_jump_kernel",
    "connected_components_triton",
    "find",
    "tl_any",
    "union",
]
