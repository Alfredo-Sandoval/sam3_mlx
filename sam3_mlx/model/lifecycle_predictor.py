"""Compatibility name for the lifecycle-safe base predictor."""

from __future__ import annotations

from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor


class LifecycleSafeSam3BasePredictor(Sam3BasePredictor):
    """Compatibility subclass; lifecycle hardening is enforced by the base class."""
