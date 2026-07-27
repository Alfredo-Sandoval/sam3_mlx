"""Experimental APIs that are not part of the stable 0.1.x contract.

SAM 3.1 multiplex / temporal tracking builders live here until real multi-object
video parity is demonstrated (target: 0.2.0). Prefer the selected-frame SAM 3
surface for supported 0.1.x workflows.
"""

from __future__ import annotations

from sam3_mlx.model_builder import (
    build_sam3_multiplex_video_model,
    build_sam3_multiplex_video_predictor,
)

__all__ = [
    "build_sam3_multiplex_video_model",
    "build_sam3_multiplex_video_predictor",
]
