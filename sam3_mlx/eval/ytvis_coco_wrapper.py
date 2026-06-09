"""YTVIS COCO wrapper compatibility surface."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported


class YTVIS(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.ytvis_coco_wrapper.YTVIS")
