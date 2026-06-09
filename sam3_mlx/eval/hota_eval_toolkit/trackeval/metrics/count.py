"""Count metric stub."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class Count:
    fields = []

    @staticmethod
    def get_name():
        return "Count"

    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.hota_eval_toolkit.trackeval.metrics.Count")
