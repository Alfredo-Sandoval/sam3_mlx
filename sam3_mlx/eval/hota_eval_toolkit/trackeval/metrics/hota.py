"""HOTA metric stub."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class HOTA:
    fields = []

    @staticmethod
    def get_name():
        return "HOTA"

    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.hota_eval_toolkit.trackeval.metrics.HOTA")
