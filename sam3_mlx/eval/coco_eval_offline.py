"""Offline COCO eval compatibility helpers."""

from __future__ import annotations

import heapq

import numpy as np

from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported


def convert_to_xywh(boxes):
    """Convert ``XYXY`` boxes to ``XYWH`` using NumPy-compatible arrays."""
    boxes = np.asarray(boxes, dtype=np.float32)
    xmin, ymin, xmax, ymax = np.moveaxis(boxes, -1, 0)
    return np.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=-1)


class HeapElement:
    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


class COCOevalCustom(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.coco_eval_offline.COCOevalCustom")


class CocoEvaluatorOfflineWithPredFileEvaluators(FailFastEvaluator):
    def __init__(self, *args, **kwargs):
        raise_unsupported(
            "eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators"
        )


def _topk_by_image(predictions, maxdets):
    by_image = {}
    for pred in predictions:
        heap = by_image.setdefault(pred["image_id"], [])
        item = HeapElement(pred)
        if len(heap) < maxdets:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)
    return [item.val for heap in by_image.values() for item in heap]
