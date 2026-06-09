"""Fail-fast CGF1 evaluator compatibility surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported


@dataclass
class Metric:
    name: str
    image_level: bool
    iou_threshold: Union[float, None]


CGF1_METRICS = [
    Metric(name="cgF1", image_level=False, iou_threshold=None),
    Metric(name="precision", image_level=False, iou_threshold=None),
    Metric(name="recall", image_level=False, iou_threshold=None),
    Metric(name="F1", image_level=False, iou_threshold=None),
    Metric(name="positive_macro_F1", image_level=False, iou_threshold=None),
    Metric(name="positive_micro_F1", image_level=False, iou_threshold=None),
    Metric(name="positive_micro_precision", image_level=False, iou_threshold=None),
    Metric(name="IL_precision", image_level=True, iou_threshold=None),
    Metric(name="IL_recall", image_level=True, iou_threshold=None),
    Metric(name="IL_F1", image_level=True, iou_threshold=None),
    Metric(name="IL_FPR", image_level=True, iou_threshold=None),
    Metric(name="IL_MCC", image_level=True, iou_threshold=None),
    Metric(name="cgF1", image_level=False, iou_threshold=0.5),
    Metric(name="precision", image_level=False, iou_threshold=0.5),
    Metric(name="recall", image_level=False, iou_threshold=0.5),
    Metric(name="F1", image_level=False, iou_threshold=0.5),
    Metric(name="positive_macro_F1", image_level=False, iou_threshold=0.5),
    Metric(name="positive_micro_F1", image_level=False, iou_threshold=0.5),
    Metric(name="positive_micro_precision", image_level=False, iou_threshold=0.5),
    Metric(name="cgF1", image_level=False, iou_threshold=0.75),
    Metric(name="precision", image_level=False, iou_threshold=0.75),
    Metric(name="recall", image_level=False, iou_threshold=0.75),
    Metric(name="F1", image_level=False, iou_threshold=0.75),
    Metric(name="positive_macro_F1", image_level=False, iou_threshold=0.75),
    Metric(name="positive_micro_F1", image_level=False, iou_threshold=0.75),
    Metric(name="positive_micro_precision", image_level=False, iou_threshold=0.75),
]


class COCOCustom(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.cgf1_eval.COCOCustom")


class CGF1Eval(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.cgf1_eval.CGF1Eval")


def _evaluate(self):
    raise_unsupported("eval.cgf1_eval._evaluate")


class CGF1Evaluator(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.cgf1_eval.CGF1Evaluator")
