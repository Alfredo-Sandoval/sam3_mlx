"""Demo eval compatibility surface."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported


class DemoEval(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.demo_eval.DemoEval")


class DemoEvaluator(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.demo_eval.DemoEvaluator")
