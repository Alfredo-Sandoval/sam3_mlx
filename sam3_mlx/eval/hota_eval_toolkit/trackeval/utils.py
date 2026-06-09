"""Fail-fast HOTA TrackEval utility shims."""

from __future__ import annotations

from sam3_mlx._unsupported import unsupported_function


_DETAIL = (
    "The official SAM3 behavior depends on the TrackEval/HOTA evaluator "
    "configuration and result I/O area."
)


@unsupported_function(
    "sam3_mlx.eval.hota_eval_toolkit.trackeval.utils.init_config",
    reason="eval-stack",
    detail=_DETAIL,
)
def init_config(config, default_config, name=None):
    return None


@unsupported_function(
    "sam3_mlx.eval.hota_eval_toolkit.trackeval.utils.update_config",
    reason="eval-stack",
    detail=_DETAIL,
)
def update_config(config):
    return None


@unsupported_function(
    "sam3_mlx.eval.hota_eval_toolkit.trackeval.utils.get_code_path",
    reason="eval-stack",
    detail=_DETAIL,
)
def get_code_path():
    return None


@unsupported_function(
    "sam3_mlx.eval.hota_eval_toolkit.trackeval.utils.validate_metrics_list",
    reason="eval-stack",
    detail=_DETAIL,
)
def validate_metrics_list(metrics_list):
    return None


@unsupported_function(
    "sam3_mlx.eval.hota_eval_toolkit.trackeval.utils.write_summary_results",
    reason="eval-stack",
    detail=_DETAIL,
)
def write_summary_results(summaries, cls, output_folder):
    return None


@unsupported_function(
    "sam3_mlx.eval.hota_eval_toolkit.trackeval.utils.write_detailed_results",
    reason="eval-stack",
    detail=_DETAIL,
)
def write_detailed_results(details, cls, output_folder):
    return None


@unsupported_function(
    "sam3_mlx.eval.hota_eval_toolkit.trackeval.utils.load_detail",
    reason="eval-stack",
    detail=_DETAIL,
)
def load_detail(file):
    return None


class TrackEvalException(Exception):
    """Custom exception class kept for official TrackEval API compatibility."""

    pass
