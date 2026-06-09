"""SACO prediction-file evaluator compatibility surface."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from sam3_mlx.eval._unsupported import raise_unsupported
from sam3_mlx.eval.cgf1_eval import CGF1_METRICS


def _get_metric_index(metric_name: str, iou_threshold: Optional[float] = None) -> int:
    for idx, metric in enumerate(CGF1_METRICS):
        if metric.name == metric_name and metric.iou_threshold == iou_threshold:
            return idx
    raise ValueError(
        f"Metric {metric_name!r} with IoU threshold {iou_threshold} not found"
    )


class BasePredFileEvaluator:
    pass


class YTVISPredFileEvaluator(BasePredFileEvaluator):
    def __init__(
        self,
        gt_ann_file: str,
        dataset_name: str = "video",
        iou_types: Optional[Sequence[str]] = None,
    ):
        self.gt_ann_file = gt_ann_file
        self.dataset_name = dataset_name
        self.iou_types = list(iou_types) if iou_types is not None else ["bbox", "segm"]
        if not all(iou_type in ["bbox", "segm"] for iou_type in self.iou_types):
            raise AssertionError("iou_types must be bbox or segm")

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        raise_unsupported("eval.saco_veval_evaluators.YTVISPredFileEvaluator.evaluate")


class VideoPhraseApEvaluator(BasePredFileEvaluator):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        raise_unsupported("eval.saco_veval_evaluators.VideoPhraseApEvaluator.evaluate")


class VideoCGF1Evaluator(BasePredFileEvaluator):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        raise_unsupported("eval.saco_veval_evaluators.VideoCGF1Evaluator.evaluate")

    def extract_video_np_level_results(self, *args, **kwargs):
        raise_unsupported(
            "eval.saco_veval_evaluators.VideoCGF1Evaluator.extract_video_np_level_results"
        )


class VideoTetaEvaluator(BasePredFileEvaluator):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def process_predictions(self, *args, **kwargs):
        raise_unsupported(
            "eval.saco_veval_evaluators.VideoTetaEvaluator.process_predictions"
        )

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        raise_unsupported("eval.saco_veval_evaluators.VideoTetaEvaluator.evaluate")


class VideoPhraseHotaEvaluator(BasePredFileEvaluator):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def evaluate(self, pred_file: str) -> Dict[str, float]:
        raise_unsupported(
            "eval.saco_veval_evaluators.VideoPhraseHotaEvaluator.evaluate"
        )

    def _remap_gt_dt(self, *args, **kwargs):
        raise_unsupported(
            "eval.saco_veval_evaluators.VideoPhraseHotaEvaluator._remap_gt_dt"
        )

    def extract_video_np_level_results(self, *args, **kwargs):
        raise_unsupported(
            "eval.saco_veval_evaluators.VideoPhraseHotaEvaluator.extract_video_np_level_results"
        )


class VideoClassBasedHotaEvaluator(VideoPhraseHotaEvaluator):
    def _remap_gt_dt(self, *args, **kwargs):
        raise_unsupported(
            "eval.saco_veval_evaluators.VideoClassBasedHotaEvaluator._remap_gt_dt"
        )

    def extract_video_np_level_results(self, *args, **kwargs):
        raise_unsupported(
            "eval.saco_veval_evaluators.VideoClassBasedHotaEvaluator.extract_video_np_level_results"
        )


def _compress_rle(rle):
    return rle


def remap_video_category_pairs_to_unique_video_ids(
    gt,
    dt,
    categories_to_keep: Optional[Sequence[int]] = None,
):
    raise_unsupported(
        "eval.saco_veval_evaluators.remap_video_category_pairs_to_unique_video_ids"
    )


def remap_gt_dt_class_agnostic(gt, dt):
    raise_unsupported("eval.saco_veval_evaluators.remap_gt_dt_class_agnostic")


def _fill_in_ann_height_width(gt_json):
    return gt_json
