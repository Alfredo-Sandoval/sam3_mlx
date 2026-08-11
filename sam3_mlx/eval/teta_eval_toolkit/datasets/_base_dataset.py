"""Fail-fast base dataset for TETA compatibility."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import FailFastEvaluator


class _BaseDataset(FailFastEvaluator):  # pyright: ignore[reportUnusedClass]
    pass
