"""Fail-fast base dataset shim for HOTA TrackEval compatibility."""

from __future__ import annotations

from typing import Never

from sam3_mlx._unsupported import raise_unsupported


_DETAIL = (
    "The official SAM3 behavior depends on the TrackEval/HOTA dataset loading "
    "and preprocessing area."
)


def _raise(method: str) -> Never:
    raise_unsupported(
        f"sam3_mlx.eval.hota_eval_toolkit.trackeval.datasets._BaseDataset.{method}",
        reason="eval-stack",
        detail=_DETAIL,
    )


class _BaseDataset:  # pyright: ignore[reportUnusedClass]
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        _raise("__init__")

    @staticmethod
    def get_default_dataset_config() -> Never:
        _raise("get_default_dataset_config")

    def _load_raw_file(self, tracker: object, seq: object, is_gt: bool) -> Never:
        del tracker, seq, is_gt
        _raise("_load_raw_file")

    def get_preprocessed_seq_data(self, raw_data: object, cls: object) -> Never:
        del raw_data, cls
        _raise("get_preprocessed_seq_data")

    def _calculate_similarities(
        self, gt_dets_t: object, tracker_dets_t: object
    ) -> Never:
        del gt_dets_t, tracker_dets_t
        _raise("_calculate_similarities")

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    def get_name(self) -> str:
        return self.get_class_name()

    def get_output_fol(self, tracker: object) -> Never:
        del tracker
        _raise("get_output_fol")

    def get_display_name[T](self, tracker: T) -> T:
        return tracker

    def get_eval_info(self) -> Never:
        _raise("get_eval_info")

    def get_raw_seq_data(self, tracker: object, seq: object) -> Never:
        del tracker, seq
        _raise("get_raw_seq_data")

    @staticmethod
    def _load_simple_text_file(
        file: object,
        time_col: int = 0,
        id_col: int | None = None,
        remove_negative_ids: bool = False,
        valid_filter: object | None = None,
        crowd_ignore_filter: object | None = None,
        convert_filter: object | None = None,
        is_zipped: bool = False,
        zip_file: object | None = None,
        force_delimiters: object | None = None,
    ) -> Never:
        del (
            file,
            time_col,
            id_col,
            remove_negative_ids,
            valid_filter,
            crowd_ignore_filter,
            convert_filter,
            is_zipped,
            zip_file,
            force_delimiters,
        )
        _raise("_load_simple_text_file")

    @staticmethod
    def _calculate_mask_ious(
        masks1: object,
        masks2: object,
        is_encoded: bool = False,
        do_ioa: bool = False,
    ) -> Never:
        del masks1, masks2, is_encoded, do_ioa
        _raise("_calculate_mask_ious")
