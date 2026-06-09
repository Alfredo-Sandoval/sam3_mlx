"""COCO prediction dumper with local JSON writing only."""

from __future__ import annotations

import copy
import heapq
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sam3_mlx.agent.helpers.rle import rle_area, rle_encode
from sam3_mlx.eval.coco_eval_offline import convert_to_xywh


class HeapElement:
    """Utility class to make a heap with a custom comparator based on score."""

    def __init__(self, val):
        self.val = val

    def __lt__(self, other):
        return self.val["score"] < other.val["score"]


def _tolist(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


class PredictionDumper:
    """Collect and dump COCO-format predictions in a single-process MLX-safe path."""

    def __init__(
        self,
        dump_dir: str,
        postprocessor,
        maxdets: int,
        iou_type: str,
        gather_pred_via_filesys: bool = False,
        merge_predictions: bool = False,
        pred_file_evaluators: Optional[Any] = None,
    ):
        self.iou_type = iou_type
        self.maxdets = maxdets
        self.dump_dir = dump_dir
        self.postprocessor = postprocessor
        self.gather_pred_via_filesys = gather_pred_via_filesys
        self.merge_predictions = merge_predictions
        self.pred_file_evaluators = pred_file_evaluators
        if self.pred_file_evaluators is not None and not merge_predictions:
            raise AssertionError(
                "merge_predictions must be True if pred_file_evaluators are provided"
            )
        if self.dump_dir is None:
            raise AssertionError("dump_dir must be provided")
        os.makedirs(self.dump_dir, exist_ok=True)
        self.reset()

    def update(self, *args, **kwargs):
        predictions = self.postprocessor.process_results(*args, **kwargs)
        results = self.prepare(predictions, self.iou_type)
        self._dump(results)

    def _dump(self, results):
        dumped_results = copy.deepcopy(results)
        for result in dumped_results:
            if "bbox" in result:
                result["bbox"] = [round(float(coord), 5) for coord in result["bbox"]]
            if "score" in result:
                result["score"] = round(float(result["score"]), 5)
        self.dump.extend(dumped_results)

    def synchronize_between_processes(self):
        logging.info("Prediction Dumper: writing local predictions")
        if self.merge_predictions:
            self.dump = self.gather_and_merge_predictions()
            dumped_file = Path(self.dump_dir) / f"coco_predictions_{self.iou_type}.json"
        else:
            dumped_file = (
                Path(self.dump_dir) / f"coco_predictions_{self.iou_type}_0.json"
            )
        with dumped_file.open("w", encoding="utf-8") as f:
            json.dump(self.dump, f)
        self.reset()
        return dumped_file

    def gather_and_merge_predictions(self):
        preds_by_image = defaultdict(list)
        seen_img_cat = set()
        for pred in self.dump:
            key = (pred["image_id"], pred["category_id"])
            if key in seen_img_cat:
                continue
            seen_img_cat.add(key)
            heap = preds_by_image[pred["image_id"]]
            item = HeapElement(pred)
            if len(heap) < self.maxdets:
                heapq.heappush(heap, item)
            else:
                heapq.heappushpop(heap, item)
        return [
            heap_item.val
            for cur_preds in preds_by_image.values()
            for heap_item in cur_preds
        ]

    def compute_synced(self):
        dumped_file = self.synchronize_between_processes()
        meters = {}
        if self.pred_file_evaluators is not None:
            for evaluator in self.pred_file_evaluators:
                meters.update(evaluator.evaluate(dumped_file))
        return meters or {"": 0.0}

    def compute(self):
        return {"": 0.0}

    def reset(self):
        self.dump = []

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        raise ValueError(f"Unknown iou type: {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = convert_to_xywh(prediction["boxes"]).tolist()
            scores = _tolist(prediction["scores"])
            labels = _tolist(prediction["labels"])
            coco_results.extend(
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            scores = _tolist(prediction["scores"])
            labels = _tolist(prediction["labels"])
            boxes = None
            if "boxes" in prediction:
                boxes = convert_to_xywh(prediction["boxes"]).tolist()
                if len(boxes) != len(scores):
                    raise AssertionError("boxes and scores length mismatch")

            if "masks_rle" in prediction:
                rles = prediction["masks_rle"]
                areas = []
                for rle in rles:
                    h, w = rle["size"]
                    areas.append(rle_area(rle) / (h * w))
            else:
                masks = np.asarray(prediction["masks"]) > 0.5
                if masks.ndim == 4 and masks.shape[1] == 1:
                    masks = masks[:, 0]
                if masks.ndim != 3:
                    raise ValueError(
                        f"Expected masks with shape (N,H,W), got {masks.shape}"
                    )
                h, w = masks.shape[-2:]
                areas = (
                    masks.reshape(masks.shape[0], -1).sum(axis=1) / (h * w)
                ).tolist()
                rles = rle_encode(masks)

            if not (len(areas) == len(rles) == len(scores)):
                raise AssertionError("areas, RLEs, and scores length mismatch")

            for k, rle in enumerate(rles):
                payload = {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                    "area": areas[k],
                }
                if boxes is not None:
                    payload["bbox"] = boxes[k]
                coco_results.append(payload)
        return coco_results
