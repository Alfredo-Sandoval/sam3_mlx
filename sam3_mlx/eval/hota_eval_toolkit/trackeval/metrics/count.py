"""Count metric stub."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class Count:
    fields: list[str] = []

    @staticmethod
    def get_name() -> str:
        return "Count"

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_unsupported("eval.hota_eval_toolkit.trackeval.metrics.Count")
