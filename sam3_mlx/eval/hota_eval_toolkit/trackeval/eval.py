"""Fail-fast HOTA track evaluator."""

from __future__ import annotations

from typing import Never

from sam3_mlx.eval._unsupported import raise_unsupported


class Evaluator:
    def __init__(self, config: object | None = None) -> None:
        self.config = config

    def evaluate(
        self,
        dataset_list: object,
        metrics_list: object,
        show_progressbar: bool = False,
    ) -> Never:
        del dataset_list, metrics_list, show_progressbar
        raise_unsupported("eval.hota_eval_toolkit.trackeval.Evaluator.evaluate")

    def evaluate_tracker(
        self,
        dataset: object,
        tracker: object,
        class_list: object,
        metrics_list: object,
        metric_names: object,
    ) -> Never:
        del dataset, tracker, class_list, metrics_list, metric_names
        raise_unsupported("eval.hota_eval_toolkit.trackeval.Evaluator.evaluate_tracker")


def eval_sequence(
    seq: object,
    dataset: object,
    tracker: object,
    class_list: object,
    metrics_list: object,
    metric_names: object,
) -> Never:
    del seq, dataset, tracker, class_list, metrics_list, metric_names
    raise_unsupported("eval.hota_eval_toolkit.trackeval.eval_sequence")
