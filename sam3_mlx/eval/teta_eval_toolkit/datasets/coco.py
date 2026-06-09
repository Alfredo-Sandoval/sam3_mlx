"""COCO dataset stub for TETA compatibility."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class COCO:
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.teta_eval_toolkit.datasets.COCO")
