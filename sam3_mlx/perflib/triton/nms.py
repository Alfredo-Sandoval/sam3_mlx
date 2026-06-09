"""Fail-fast NMS Triton-kernel module for SAM3 MLX.

Mirrors the public surface of `sam3.perflib.triton.nms` at upstream commit
`2814fa619404a722d03e9a012e083e4f293a4e53` so callers can keep their import
paths, but raises `NotImplementedError` on invocation. Triton kernels are not
part of the SAM3 MLX runtime.
"""

from __future__ import annotations

from sam3_mlx._unsupported import unsupported_function


_UNSUPPORTED_DETAIL = "The official SAM3 implementation requires Triton kernels."


@unsupported_function(
    "sam3_mlx.perflib.triton.nms._nms_suppression_kernel",
    reason="triton-kernel",
    alternative="sam3_mlx.perflib.nms",
    detail=_UNSUPPORTED_DETAIL,
)
def _nms_suppression_kernel(
    iou_mask_ptr,
    keep_mask_ptr,
    num_boxes,
    iou_mask_stride,
    cxpr_block_size,
):
    return None


@unsupported_function(
    "sam3_mlx.perflib.triton.nms.nms_triton",
    reason="triton-kernel",
    alternative="sam3_mlx.perflib.nms",
    detail=_UNSUPPORTED_DETAIL,
)
def nms_triton(ious, scores, iou_threshold):
    return None


__all__ = [
    "_nms_suppression_kernel",
    "nms_triton",
]
