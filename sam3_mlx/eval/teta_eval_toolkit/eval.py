"""Fail-fast TETA evaluator."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


class Evaluator:
    def __init__(self, config=None):
        self.config = config

    def evaluate(self, dataset_list, metrics_list, show_progressbar=False):
        raise_unsupported("eval.teta_eval_toolkit.eval.Evaluator.evaluate")

    def evaluate_tracker(
        self, dataset, tracker, class_list, metrics_list, metric_names
    ):
        raise_unsupported("eval.teta_eval_toolkit.eval.Evaluator.evaluate_tracker")


def eval_sequence(seq, dataset, tracker, class_list, metrics_list, metric_names):
    raise_unsupported("eval.teta_eval_toolkit.eval.eval_sequence")
