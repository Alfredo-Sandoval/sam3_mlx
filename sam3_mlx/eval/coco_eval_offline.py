"""Offline COCO eval compatibility helpers."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import SupportsFloat, cast

from sam3_mlx.eval._geometry import convert_to_xywh as convert_to_xywh
from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported
from sam3_mlx.eval.coco_writer import CocoResult


def _score(result: CocoResult) -> float:
    return float(cast(SupportsFloat, result["score"]))


class HeapElement:
    def __init__(self, val: CocoResult) -> None:
        self.val = val

    def __lt__(self, other: HeapElement) -> bool:
        return _score(self.val) < _score(other.val)


class COCOevalCustom(FailFastEvaluator):
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_unsupported("eval.coco_eval_offline.COCOevalCustom")


class CocoEvaluatorOfflineWithPredFileEvaluators(FailFastEvaluator):
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise_unsupported(
            "eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators"
        )


def _topk_by_image(  # pyright: ignore[reportUnusedFunction]
    predictions: Sequence[CocoResult], maxdets: int
) -> list[CocoResult]:
    by_image: dict[object, list[HeapElement]] = {}
    for pred in predictions:
        heap = by_image.setdefault(pred["image_id"], [])
        item = HeapElement(pred)
        if len(heap) < maxdets:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)
    return [item.val for heap in by_image.values() for item in heap]
