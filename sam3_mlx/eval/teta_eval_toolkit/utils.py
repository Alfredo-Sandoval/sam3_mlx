"""Small TETA toolkit utility helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class _Metric(Protocol):
    @property
    def fields(self) -> Sequence[str]: ...

    def get_name(self) -> str: ...


def validate_metrics_list(metrics_list: Sequence[_Metric]) -> list[str]:
    metric_names = [metric.get_name() for metric in metrics_list]
    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException(
            "Code being run with multiple metrics of the same name"
        )
    fields: list[str] = []
    for metric in metrics_list:
        fields += metric.fields
    if len(fields) != len(set(fields)):
        raise TrackEvalException(
            "Code being run with multiple metrics with fields of the same name"
        )
    return metric_names


def get_track_id_str(ann: Mapping[str, object]) -> str:
    if "track_id" in ann:
        return "track_id"
    if "instance_id" in ann:
        return "instance_id"
    if "scalabel_id" in ann:
        return "scalabel_id"
    raise AssertionError("No track/instance ID.")


class TrackEvalException(Exception):
    """Custom exception for expected toolkit errors."""
