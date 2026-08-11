"""TAO dataset stub for TETA compatibility."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class TAO:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_unsupported("eval.teta_eval_toolkit.datasets.TAO")
