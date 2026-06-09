"""Fail-fast base metric shim for HOTA TrackEval compatibility."""

from __future__ import annotations

from sam3_mlx._unsupported import raise_unsupported


_DETAIL = (
    "The official SAM3 behavior depends on the TrackEval/HOTA metric computation area."
)


def _raise(method: str):
    raise_unsupported(
        f"sam3_mlx.eval.hota_eval_toolkit.trackeval.metrics._BaseMetric.{method}",
        reason="eval-stack",
        detail=_DETAIL,
    )


class _BaseMetric:
    fields: list[str] = []
    summary_fields: list[str] = []
    integer_fields: list[str] = []
    float_fields: list[str] = []
    array_labels: list[float] = []
    integer_array_fields: list[str] = []
    float_array_fields: list[str] = []
    plottable = False

    def __init__(self, *args, **kwargs):
        _raise("__init__")

    def eval_sequence(self, data):
        _raise("eval_sequence")

    def combine_sequences(self, all_res):
        _raise("combine_sequences")

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        _raise("combine_classes_class_averaged")

    def combine_classes_det_averaged(self, all_res):
        _raise("combine_classes_det_averaged")

    def plot_single_tracker_results(self, all_res, tracker, output_folder, cls):
        _raise("plot_single_tracker_results")

    @classmethod
    def get_name(cls):
        return cls.__name__

    @staticmethod
    def _combine_sum(all_res, field):
        _raise("_combine_sum")

    @staticmethod
    def _combine_weighted_av(all_res, field, comb_res, weight_field):
        _raise("_combine_weighted_av")

    def print_table(
        self, table_res, tracker, cls, res_field="COMBINED_SEQ", output_lable="COMBINED"
    ):
        _raise("print_table")

    def _summary_row(self, results_):
        _raise("_summary_row")

    @staticmethod
    def _row_print(*argv):
        _raise("_row_print")

    def summary_results(self, table_res):
        _raise("summary_results")

    def detailed_results(self, table_res):
        _raise("detailed_results")

    def _detailed_row(self, res):
        _raise("_detailed_row")
