"""Shared fail-fast helpers for unported train surfaces."""

from __future__ import annotations

from sam3_mlx._unsupported import (
    Sam3MlxUnsupportedError,
    unsupported as _unsupported,
)


LANE_E_MESSAGE = (
    "Lane E currently ports only the official-shaped image Datapoint, "
    "BatchedDatapoint collator, and minimal image input transforms. "
    "Full training datasets, downloads, distributed loading, video sampling, "
    "and Torch/Torchvision transforms are not implemented in the MLX port yet."
)


def unsupported(feature: str) -> Sam3MlxUnsupportedError:
    return _unsupported(
        feature,
        reason="training-loop",
        detail=LANE_E_MESSAGE,
    )


def raise_unsupported(feature: str):
    raise unsupported(feature)
