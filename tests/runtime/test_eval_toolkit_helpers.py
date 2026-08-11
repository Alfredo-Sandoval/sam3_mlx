from __future__ import annotations

import sys

import pytest

import sam3_mlx.eval.hota_eval_toolkit.trackeval._timing as hota_timing
import sam3_mlx.eval.teta_eval_toolkit._timing as teta_timing
from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.eval.teta_eval_toolkit import config
from sam3_mlx.eval.teta_eval_toolkit.utils import (
    TrackEvalException,
    get_track_id_str,
    validate_metrics_list,
)


class _Metric:
    def __init__(self, name: str, fields: list[str]) -> None:
        self.name = name
        self.fields = fields

    def get_name(self) -> str:
        return self.name


def test_hota_timing_shim_remains_fail_fast() -> None:
    @hota_timing.time
    def evaluate(value: int) -> int:
        return value

    with pytest.raises(Sam3MlxUnsupportedError, match="timing and evaluator") as exc:
        evaluate(3)

    assert exc.value.reason == "eval-stack"


def test_teta_timing_decorator_preserves_result_and_records_elapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @teta_timing.time
    def add(left: int, right: int) -> int:
        return left + right

    assert add(2, 3) == 5

    teta_timing.timer_dict.clear()
    monkeypatch.setattr(teta_timing, "DO_TIMING", True)
    assert add(4, 5) == 9
    assert teta_timing.timer_dict["add"] >= 0


def test_teta_parse_configs_preserves_scalar_and_list_cli_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "teta",
            "--USE_PARALLEL",
            "False",
            "--NUM_PARALLEL_CORES",
            "3",
            "--METRICS",
            "TETA",
            "HOTA",
        ],
    )

    eval_config, dataset_config, metrics_config = config.parse_configs()

    assert eval_config["USE_PARALLEL"] is False
    assert eval_config["NUM_PARALLEL_CORES"] == 3
    assert metrics_config["METRICS"] == ["TETA", "HOTA"]
    assert dataset_config["SPLIT_TO_EVAL"] == "training"


def test_teta_config_fill_and_boolean_error_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defaults: config.Config = {"PRINT_CONFIG": False, "COUNT": 8}
    merged = config.init_config({"COUNT": 2}, defaults)
    assert merged == {"COUNT": 2, "PRINT_CONFIG": False}

    monkeypatch.setattr(sys, "argv", ["teta", "--PRINT_CONFIG", "yes"])
    with pytest.raises(Exception, match="PRINT_CONFIGmust be True or False"):
        config.update_config(merged)


def test_teta_metric_and_track_id_validation_contracts() -> None:
    metrics = [_Metric("TETA", ["LocA", "AssocA"]), _Metric("Count", ["Dets"])]

    assert validate_metrics_list(metrics) == ["TETA", "Count"]
    assert get_track_id_str({"instance_id": 4}) == "instance_id"

    with pytest.raises(TrackEvalException, match="same name"):
        validate_metrics_list([_Metric("TETA", []), _Metric("TETA", [])])
    with pytest.raises(AssertionError, match="No track/instance ID"):
        get_track_id_str({"id": 4})
