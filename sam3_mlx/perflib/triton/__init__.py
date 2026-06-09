"""Import-compatible fail-fast modules for upstream Triton kernels."""

from sam3_mlx.perflib.triton.connected_components import connected_components_triton
from sam3_mlx.perflib.triton.nms import nms_triton

__all__ = ["connected_components_triton", "nms_triton"]
