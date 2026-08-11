"""COCO evaluator compatibility surface."""

from __future__ import annotations

from typing import Never

from sam3_mlx.eval._geometry import convert_to_xywh as convert_to_xywh
from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported


class CocoEvaluator(FailFastEvaluator):
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_unsupported("eval.coco_eval.CocoEvaluator")


def merge(
    img_ids: object, eval_imgs: object, gather_pred_via_filesys: bool = False
) -> Never:
    del img_ids, eval_imgs, gather_pred_via_filesys
    raise_unsupported("eval.coco_eval.merge")


def create_common_coco_eval(
    coco_eval: object,
    img_ids: object,
    eval_imgs: object,
    gather_pred_via_filesys: bool = False,
) -> Never:
    del coco_eval, img_ids, eval_imgs, gather_pred_via_filesys
    raise_unsupported("eval.coco_eval.create_common_coco_eval")


def segmentation_prepare(self: object) -> Never:
    del self
    raise_unsupported("eval.coco_eval.segmentation_prepare")


def evaluate(self: object, use_self_evaluate: bool) -> Never:
    del self, use_self_evaluate
    raise_unsupported("eval.coco_eval.evaluate")


def loadRes(self: object, resFile: object) -> Never:
    del self, resFile
    raise_unsupported("eval.coco_eval.loadRes")


def summarize(self: object) -> Never:
    del self
    raise_unsupported("eval.coco_eval.summarize")


def accumulate(self: object, use_self_eval: bool = False) -> Never:
    del self, use_self_eval
    raise_unsupported("eval.coco_eval.accumulate")
