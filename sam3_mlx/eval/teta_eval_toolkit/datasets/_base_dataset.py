"""Fail-fast base dataset for TETA compatibility."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class _BaseDataset:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, name):
        def _missing(*args, **kwargs):
            raise_unsupported(f"{self.__class__.__name__}.{name}")

        return _missing
