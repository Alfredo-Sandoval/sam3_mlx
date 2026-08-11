from __future__ import annotations

import sys

import pytest

import sam3_mlx.eval.hota_eval_toolkit.trackeval._timing as hota_timing
import sam3_mlx.eval.teta_eval_toolkit._timing as teta_timing
from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.eval import demo_eval, ytvis_coco_wrapper
from sam3_mlx.eval.hota_eval_toolkit import run_ytvis_eval
from sam3_mlx.eval.hota_eval_toolkit import trackeval as hota_trackeval
from sam3_mlx.eval.hota_eval_toolkit.trackeval.datasets._base_dataset import (
    _BaseDataset as HotaBaseDataset,  # pyright: ignore[reportPrivateUsage]
)
from sam3_mlx.eval.hota_eval_toolkit.trackeval.metrics._base_metric import (
    _BaseMetric as HotaBaseMetric,  # pyright: ignore[reportPrivateUsage]
)
from sam3_mlx.eval import teta_eval_toolkit
from sam3_mlx.eval.teta_eval_toolkit import config
from sam3_mlx.eval.teta_eval_toolkit.datasets._base_dataset import (
    _BaseDataset as TetaBaseDataset,  # pyright: ignore[reportPrivateUsage]
)
from sam3_mlx.eval.teta_eval_toolkit.metrics._base_metric import (
    _BaseMetric as TetaBaseMetric,  # pyright: ignore[reportPrivateUsage]
)
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


def test_hota_base_classes_keep_identity_helpers_and_fail_fast() -> None:
    assert HotaBaseDataset.get_class_name() == "_BaseDataset"
    assert HotaBaseMetric.get_name() == "_BaseMetric"

    with pytest.raises(Sam3MlxUnsupportedError, match="dataset loading") as dataset_exc:
        HotaBaseDataset()
    assert dataset_exc.value.reason == "eval-stack"

    with pytest.raises(
        Sam3MlxUnsupportedError, match="metric computation"
    ) as metric_exc:
        HotaBaseMetric()
    assert metric_exc.value.reason == "eval-stack"


def test_teta_base_classes_keep_shared_fail_fast_contract() -> None:
    dataset = TetaBaseDataset("arg", option=3)

    assert dataset.args == ("arg",)
    assert dataset.kwargs == {"option": 3}
    with pytest.raises(Sam3MlxUnsupportedError, match="evaluation surface"):
        dataset.missing_method()

    assert TetaBaseMetric.get_name() == "_BaseMetric"
    with pytest.raises(Sam3MlxUnsupportedError, match="evaluation surface"):
        TetaBaseMetric().eval_sequence({})


@pytest.mark.parametrize(
    ("factory", "feature"),
    [
        (demo_eval.DemoEval, "eval.demo_eval.DemoEval"),
        (demo_eval.DemoEvaluator, "eval.demo_eval.DemoEvaluator"),
        (ytvis_coco_wrapper.YTVIS, "eval.ytvis_coco_wrapper.YTVIS"),
        (
            hota_trackeval.datasets.TAO_OW,
            "eval.hota_eval_toolkit.trackeval.datasets.TAO_OW",
        ),
        (
            hota_trackeval.datasets.YouTubeVIS,
            "eval.hota_eval_toolkit.trackeval.datasets.YouTubeVIS",
        ),
        (
            hota_trackeval.metrics.Count,
            "eval.hota_eval_toolkit.trackeval.metrics.Count",
        ),
        (
            hota_trackeval.metrics.HOTA,
            "eval.hota_eval_toolkit.trackeval.metrics.HOTA",
        ),
        (teta_eval_toolkit.datasets.COCO, "eval.teta_eval_toolkit.datasets.COCO"),
        (teta_eval_toolkit.datasets.TAO, "eval.teta_eval_toolkit.datasets.TAO"),
        (teta_eval_toolkit.metrics.TETA, "eval.teta_eval_toolkit.metrics.TETA"),
    ],
)
def test_eval_leaf_constructors_keep_canonical_fail_fast_features(
    factory: type[object], feature: str
) -> None:
    with pytest.raises(Sam3MlxUnsupportedError) as exc:
        factory()

    assert exc.value.feature == feature
    assert exc.value.reason == "eval-stack"


def test_eval_toolkit_orchestrators_keep_config_and_fail_fast() -> None:
    hota_evaluator = hota_trackeval.Evaluator({"PRINT_RESULTS": False})
    teta_evaluator = teta_eval_toolkit.Evaluator({"PRINT_RESULTS": False})

    assert hota_evaluator.config == {"PRINT_RESULTS": False}
    assert teta_evaluator.config == {"PRINT_RESULTS": False}
    with pytest.raises(Sam3MlxUnsupportedError, match="evaluation surface"):
        hota_evaluator.evaluate([], [])
    with pytest.raises(Sam3MlxUnsupportedError, match="evaluation surface"):
        teta_evaluator.evaluate([], [])
    with pytest.raises(Sam3MlxUnsupportedError, match="evaluation surface"):
        run_ytvis_eval.run_ytvis_eval()


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
