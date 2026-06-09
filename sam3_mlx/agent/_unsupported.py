"""Shared fail-fast helpers for unported agent surfaces."""

from __future__ import annotations

from sam3_mlx._unsupported import (
    Sam3MlxUnsupportedError,
    unsupported as _unsupported,
)


AGENT_UNSUPPORTED_MESSAGE = (
    "The official SAM3 agent stack depends on external LLM services and "
    "PyTorch-oriented SAM3 runtime pieces that are not part of this MLX image "
    "port. This module is present for import compatibility only."
)


def unsupported(feature: str) -> Sam3MlxUnsupportedError:
    return _unsupported(
        feature,
        reason="agent-llm",
        detail=AGENT_UNSUPPORTED_MESSAGE,
    )


def raise_unsupported(feature: str):
    raise unsupported(feature)
