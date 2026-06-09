"""TETA metric stub."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class TETA:
    fields = []

    @staticmethod
    def get_name():
        return "TETA"

    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.teta_eval_toolkit.metrics.TETA")
