"""Base metric stub for TETA compatibility."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class _BaseMetric:
    fields: list[str] = []

    @staticmethod
    def get_name():
        return "_BaseMetric"

    def eval_sequence(self, data):
        raise_unsupported("eval.teta_eval_toolkit.metrics._BaseMetric.eval_sequence")
