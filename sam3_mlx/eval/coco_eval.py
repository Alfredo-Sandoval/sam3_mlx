"""COCO evaluator compatibility surface."""

from __future__ import annotations

import numpy as np

from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported


class CocoEvaluator(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.coco_eval.CocoEvaluator")


def convert_to_xywh(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    xmin, ymin, xmax, ymax = np.moveaxis(boxes, -1, 0)
    return np.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=-1)


def merge(img_ids, eval_imgs, gather_pred_via_filesys=False):
    raise_unsupported("eval.coco_eval.merge")


def create_common_coco_eval(
    coco_eval,
    img_ids,
    eval_imgs,
    gather_pred_via_filesys=False,
):
    raise_unsupported("eval.coco_eval.create_common_coco_eval")


def segmentation_prepare(self):
    raise_unsupported("eval.coco_eval.segmentation_prepare")


def evaluate(self, use_self_evaluate):
    raise_unsupported("eval.coco_eval.evaluate")


def loadRes(self, resFile):
    raise_unsupported("eval.coco_eval.loadRes")


def summarize(self):
    raise_unsupported("eval.coco_eval.summarize")


def accumulate(self, use_self_eval=False):
    raise_unsupported("eval.coco_eval.accumulate")
