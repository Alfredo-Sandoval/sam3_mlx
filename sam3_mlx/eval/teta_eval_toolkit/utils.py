"""Small TETA toolkit utility helpers."""

from __future__ import annotations


def validate_metrics_list(metrics_list):
    metric_names = [metric.get_name() for metric in metrics_list]
    if len(metric_names) != len(set(metric_names)):
        raise TrackEvalException(
            "Code being run with multiple metrics of the same name"
        )
    fields = []
    for metric in metrics_list:
        fields += metric.fields
    if len(fields) != len(set(fields)):
        raise TrackEvalException(
            "Code being run with multiple metrics with fields of the same name"
        )
    return metric_names


def get_track_id_str(ann):
    if "track_id" in ann:
        return "track_id"
    if "instance_id" in ann:
        return "instance_id"
    if "scalabel_id" in ann:
        return "scalabel_id"
    raise AssertionError("No track/instance ID.")


class TrackEvalException(Exception):
    """Custom exception for expected toolkit errors."""
