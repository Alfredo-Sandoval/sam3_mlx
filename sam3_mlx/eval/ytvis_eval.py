"""YT-VIS eval compatibility surface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

from sam3_mlx.eval._unsupported import FailFastEvaluator, raise_unsupported


class YTVISevalMixin:
    def _prepare(self):
        raise_unsupported("eval.ytvis_eval.YTVISevalMixin._prepare")

    def computeIoU(self, imgId, catId):
        raise_unsupported("eval.ytvis_eval.YTVISevalMixin.computeIoU")


class YTVISeval(YTVISevalMixin, FailFastEvaluator):
    sort_inds_by_scores_in_iou = True

    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.ytvis_eval.YTVISeval")


class VideoDemoF1Eval(YTVISevalMixin, FailFastEvaluator):
    sort_inds_by_scores_in_iou = False

    def __init__(self, *args, **kwargs):
        raise_unsupported("eval.ytvis_eval.VideoDemoF1Eval")


class YTVISResultsWriter:
    """Minimal local JSON writer for video predictions."""

    def __init__(
        self,
        dump_file: str,
        postprocessor,
        gather_pred_via_filesys=False,
        pred_file_evaluators: Optional[List] = None,
        save_per_frame_scores: bool = False,
        write_eval_metrics_file: bool = True,
        eval_metrics_file_suffix: str = ".sam3_eval_metrics",
    ):
        self.dump_file = dump_file
        self.dump = []
        self.postprocessor = postprocessor
        self.gather_pred_via_filesys = gather_pred_via_filesys
        self.pred_file_evaluators = pred_file_evaluators or []
        self.save_per_frame_scores = save_per_frame_scores
        self.write_eval_metrics_file = write_eval_metrics_file
        self.eval_metrics_file_suffix = eval_metrics_file_suffix
        dirname = os.path.dirname(self.dump_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def _dump_vid_preds(self, results):
        self.dump.extend(results if isinstance(results, list) else [results])

    def prepare(self, predictions):
        return predictions

    def set_sync_device(self, device):
        self.sync_device = device

    def update(self, *args, **kwargs):
        predictions = self.postprocessor.process_results(*args, **kwargs)
        self._dump_preds(self.prepare(predictions))

    def _dump_preds(self, results):
        self._dump_vid_preds(results)

    def synchronize_between_processes(self):
        with open(self.dump_file, "w", encoding="utf-8") as f:
            json.dump(self.dump, f)
        return self.dump_file

    def _dedup_pre_gather(self):
        return self.dump

    def _dedup_post_gather(self):
        return self.dump

    def compute_synced(self):
        pred_file = self.synchronize_between_processes()
        meters = {}
        for evaluator in self.pred_file_evaluators:
            meters.update(evaluator.evaluate(pred_file))
        if self.write_eval_metrics_file:
            metrics_file = Path(str(pred_file) + self.eval_metrics_file_suffix)
            with metrics_file.open("w", encoding="utf-8") as f:
                json.dump(meters, f)
        return meters or {"": 0.0}

    def compute(self):
        return {"": 0.0}

    def reset(self):
        self.dump = []
