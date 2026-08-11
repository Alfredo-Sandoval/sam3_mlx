"""TETA metric stub."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class TETA:
    fields: list[str] = []

    @staticmethod
    def get_name() -> str:
        return "TETA"

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_unsupported("eval.teta_eval_toolkit.metrics.TETA")
