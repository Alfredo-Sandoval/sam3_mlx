"""Shared fail-fast helpers for unported eval surfaces."""

from __future__ import annotations

from sam3_mlx._unsupported import (
    Sam3MlxUnsupportedError,
    unsupported as _unsupported,
)


EVAL_UNSUPPORTED_MESSAGE = (
    "The official SAM3 evaluation surface depends on pycocotools, distributed "
    "training utilities, and PyTorch tensors. This MLX fork exposes the names "
    "for compatibility, but this evaluator has not been ported."
)


def unsupported(feature: str) -> Sam3MlxUnsupportedError:
    return _unsupported(
        feature,
        reason="eval-stack",
        detail=EVAL_UNSUPPORTED_MESSAGE,
    )


def raise_unsupported(feature: str):
    raise unsupported(feature)


class FailFastEvaluator:
    """Base class for official evaluators that are not part of the MLX port."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name):
        def _missing(*args, **kwargs):
            raise_unsupported(f"{self.__class__.__name__}.{name}")

        return _missing
