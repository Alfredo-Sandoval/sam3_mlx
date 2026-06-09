"""Fail-fast base dataset shim for HOTA TrackEval compatibility."""

from __future__ import annotations

from sam3_mlx._unsupported import raise_unsupported


_DETAIL = (
    "The official SAM3 behavior depends on the TrackEval/HOTA dataset loading "
    "and preprocessing area."
)


def _raise(method: str):
    raise_unsupported(
        f"sam3_mlx.eval.hota_eval_toolkit.trackeval.datasets._BaseDataset.{method}",
        reason="eval-stack",
        detail=_DETAIL,
    )


class _BaseDataset:
    def __init__(self, *args, **kwargs):
        _raise("__init__")

    @staticmethod
    def get_default_dataset_config():
        _raise("get_default_dataset_config")

    def _load_raw_file(self, tracker, seq, is_gt):
        _raise("_load_raw_file")

    def get_preprocessed_seq_data(self, raw_data, cls):
        _raise("get_preprocessed_seq_data")

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        _raise("_calculate_similarities")

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def get_name(self):
        return self.get_class_name()

    def get_output_fol(self, tracker):
        _raise("get_output_fol")

    def get_display_name(self, tracker):
        return tracker

    def get_eval_info(self):
        _raise("get_eval_info")

    def get_raw_seq_data(self, tracker, seq):
        _raise("get_raw_seq_data")

    @staticmethod
    def _load_simple_text_file(
        file,
        time_col=0,
        id_col=None,
        remove_negative_ids=False,
        valid_filter=None,
        crowd_ignore_filter=None,
        convert_filter=None,
        is_zipped=False,
        zip_file=None,
        force_delimiters=None,
    ):
        _raise("_load_simple_text_file")

    @staticmethod
    def _calculate_mask_ious(masks1, masks2, is_encoded=False, do_ioa=False):
        _raise("_calculate_mask_ious")
