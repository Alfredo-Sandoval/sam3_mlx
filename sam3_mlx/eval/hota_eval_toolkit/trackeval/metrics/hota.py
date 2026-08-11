"""HOTA metric stub."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class HOTA:
    fields: list[str] = []

    @staticmethod
    def get_name() -> str:
        return "HOTA"

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_unsupported("eval.hota_eval_toolkit.trackeval.metrics.HOTA")
