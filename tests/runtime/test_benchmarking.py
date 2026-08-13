from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sam3_mlx.benchmarking import (
    BenchmarkOperation,
    IMAGE_BENCHMARK_SCHEMA,
    IMAGE_BENCHMARK_SUBSTAGE_SCHEMA,
    InterleavedTimingProtocol,
    SUBSTAGE_TIMING_MODE,
    RegressionThreshold,
    TimingProtocol,
    aggregate_stage_group,
    build_substage_artifact,
    compare_benchmark_artifacts,
    eager_isolated_substage_fields,
    percentile,
    profile_operations,
    profile_operations_interleaved,
    should_escalate_resolution,
    summarize_samples,
    synchronized_samples,
    validate_substage_artifact,
)
from tests._paths import REPO_ROOT


def _artifact(*, median: float, memory: int = 100) -> dict[str, object]:
    return {
        "schema_version": IMAGE_BENCHMARK_SCHEMA,
        "provenance": {"git_commit": "a" * 40, "dirty": False},
        "environment": {"chip": "fixture"},
        "runtime": {"backend": "mlx-metal"},
        "workload": {"resolution": 504, "repetitions": 5},
        "metrics": {"full": {"median_s": median}},
        "peak_active_memory_bytes": memory,
    }


def test_percentile_and_summary_use_observed_nearest_rank_samples() -> None:
    samples = [5.0, 1.0, 3.0, 2.0, 4.0]

    assert percentile(samples, 0.95) == 5.0
    assert summarize_samples(samples) == {
        "samples_s": samples,
        "p50_s": 3.0,
        "median_s": 3.0,
        "p95_s": 5.0,
        "min_s": 1.0,
        "max_s": 5.0,
    }


def test_synchronized_samples_executes_every_warmup_and_measured_boundary() -> None:
    runs: list[int] = []
    synchronized: list[int] = []

    def run() -> int:
        runs.append(len(runs))
        return runs[-1]

    samples = synchronized_samples(
        BenchmarkOperation(run=run, synchronize=synchronized.append),
        protocol=TimingProtocol(warmup_runs=2, repetitions=5),
    )

    assert runs == list(range(7))
    assert synchronized == runs
    assert len(samples) == 5
    assert all(sample >= 0.0 for sample in samples)


def test_timing_protocol_rejects_underpowered_measurements() -> None:
    with pytest.raises(ValueError, match="warmup_runs must be at least 1"):
        TimingProtocol(warmup_runs=0, repetitions=5)
    with pytest.raises(ValueError, match="repetitions must be at least 5"):
        TimingProtocol(warmup_runs=1, repetitions=4)


@pytest.mark.parametrize(
    ("current", "expected_status", "metric_status"),
    [
        (0.80, "pass", "improved"),
        (1.02, "pass", "flat-noisy"),
        (1.11, "fail", "regressed"),
    ],
)
def test_comparison_classifies_lower_is_better_regressions(
    current: float,
    expected_status: str,
    metric_status: str,
) -> None:
    result = compare_benchmark_artifacts(
        _artifact(median=1.0),
        _artifact(median=current),
        metric_paths=["metrics.full.median_s"],
        threshold=RegressionThreshold(max_regression_pct=10.0, noise_pct=3.0),
    )

    assert result["status"] == expected_status
    assert result["metrics"] == [
        {
            "metric": "metrics.full.median_s",
            "direction": "lower-is-better",
            "baseline": 1.0,
            "current": current,
            "delta": pytest.approx(current - 1.0),
            "percent_change": pytest.approx((current - 1.0) * 100.0),
            "noise_pct": 3.0,
            "max_regression_pct": 10.0,
            "status": metric_status,
        }
    ]


def test_should_escalate_resolution_uses_empty_or_margin_criterion() -> None:
    assert should_escalate_resolution(
        [],
        confidence_threshold=0.5,
        min_score_margin=0.1,
    )
    assert should_escalate_resolution(
        [0.55],
        confidence_threshold=0.5,
        min_score_margin=0.1,
    )
    assert not should_escalate_resolution(
        [0.61, 0.4],
        confidence_threshold=0.5,
        min_score_margin=0.1,
    )
    with pytest.raises(ValueError, match="min_score_margin"):
        should_escalate_resolution(
            [0.9], confidence_threshold=0.5, min_score_margin=-0.1
        )


def test_substage_fields_label_eager_isolated_not_compiled_graph() -> None:
    fields = eager_isolated_substage_fields(model_compile_policy="mlx-compiled-visual")

    assert fields == {
        "model_compile_policy": "mlx-compiled-visual",
        "timing_mode": SUBSTAGE_TIMING_MODE,
    }
    assert IMAGE_BENCHMARK_SUBSTAGE_SCHEMA == "sam3_mlx.image_runtime_substages.v2"
    assert SUBSTAGE_TIMING_MODE == "eager-isolated-modules"
    assert "compile_policy" not in fields


def test_interleaved_protocol_rejects_underpowered_substage_measurements() -> None:
    with pytest.raises(ValueError, match="warmup_runs must be at least 5"):
        InterleavedTimingProtocol(warmup_runs=4, measurements_per_round=10, rounds=3)
    with pytest.raises(ValueError, match="rounds must be at least 3"):
        InterleavedTimingProtocol(warmup_runs=5, measurements_per_round=10, rounds=2)
    with pytest.raises(ValueError, match="total measurements must be at least 30"):
        InterleavedTimingProtocol(warmup_runs=5, measurements_per_round=5, rounds=3)


def test_profile_operations_interleaved_rotates_stages_each_measurement() -> None:
    order: list[str] = []

    def make_op(name: str) -> BenchmarkOperation[int]:
        def run() -> int:
            order.append(name)
            return len(order)

        return BenchmarkOperation(run=run, synchronize=lambda _value: None)

    profiled = profile_operations_interleaved(
        {"a": make_op("a"), "b": make_op("b")},
        protocol=InterleavedTimingProtocol(),
    )

    assert order[:10] == (["a"] * 5) + (["b"] * 5)
    assert order[10:] == ["a", "b"] * 30
    assert len(profiled["a"]["samples_s"]) == 30
    assert len(profiled["b"]["samples_s"]) == 30


def test_profile_operations_and_group_aggregates_reuse_timing_protocol() -> None:
    values = iter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    def run_a() -> int:
        return next(values)

    def run_b() -> int:
        return next(values)

    profiled = profile_operations(
        {
            "a": BenchmarkOperation(run=run_a, synchronize=lambda _value: None),
            "b": BenchmarkOperation(run=run_b, synchronize=lambda _value: None),
        },
        protocol=TimingProtocol(warmup_runs=1, repetitions=5),
    )
    assert profiled["a"]["samples_s"]
    assert profiled["b"]["samples_s"]
    grouped = aggregate_stage_group(
        profiled,
        category="window",
        member_names=("a", "b"),
    )
    assert grouped["category"] == "window"
    assert grouped["members"] == ["a", "b"]
    assert grouped["sum_of_isolated_stage_medians_s"] == (
        profiled["a"]["median_s"] + profiled["b"]["median_s"]
    )
    assert "p50_sum_s" not in grouped
    assert "p95_sum_s" not in grouped
    assert "median_sum_s" not in grouped


def _dummy_summary(offset: float) -> dict[str, object]:
    samples = [offset + float(index) for index in range(1, 31)]
    return summarize_samples(samples)


def test_generated_substage_artifact_uses_v2_contract(tmp_path: Path) -> None:
    protocol = InterleavedTimingProtocol()
    isolated_names = (
        "preprocessing",
        "token_preparation",
        "vit_block_0",
        "vit_block_1",
        "neck_head_0",
        "neck_position_encoding_0",
        "model_grounding_core",
        "filtering_and_postprocess",
        "mask_upsample",
    )
    order: list[str] = []

    def make_op(name: str) -> BenchmarkOperation[str]:
        def run() -> str:
            order.append(name)
            return name

        return BenchmarkOperation(run=run, synchronize=lambda _value: None)

    timed = profile_operations_interleaved(
        {name: make_op(name) for name in isolated_names},
        protocol=protocol,
    )
    timed["text_encoding"] = _dummy_summary(100.0)
    timed["text_encoding_repeated"] = _dummy_summary(200.0)
    complete_samples = [float(index) for index in range(1, 31)]
    timed["complete_path"] = summarize_samples(complete_samples)
    timed["vit_block_0"] = dict(timed["vit_block_0"])
    timed["vit_block_1"] = dict(timed["vit_block_1"])

    artifact = build_substage_artifact(
        resolution=1008,
        model_compile_policy="mlx-compiled-visual",
        protocol=protocol,
        timed=timed,
        vit_blocks=[
            {**timed["vit_block_0"], "index": 0, "category": "window"},
            {**timed["vit_block_1"], "index": 1, "category": "global"},
        ],
        neck_heads=[{**timed["neck_head_0"], "index": 0, "scale": 4.0, "retained": True}],
        neck_position_encodings=[
            {
                **timed["neck_position_encoding_0"],
                "index": 0,
                "scale": 4.0,
                "retained": True,
            }
        ],
        window_names=("vit_block_0",),
        global_names=("vit_block_1",),
        model_path_names=(
            "token_preparation",
            "vit_block_0",
            "vit_block_1",
            "neck_head_0",
            "neck_position_encoding_0",
            "model_grounding_core",
            "filtering_and_postprocess",
            "mask_upsample",
        ),
        active_memory_bytes=1,
        peak_active_memory_bytes=2,
    )
    path = tmp_path / "substages.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    loaded = json.loads(path.read_text())

    validate_substage_artifact(loaded)
    assert loaded["schema_version"] == "sam3_mlx.image_runtime_substages.v2"
    assert "patch_embedding" not in loaded
    assert "grounding" not in loaded
    assert loaded["token_preparation"]["samples_s"]
    assert loaded["model_grounding_core"]["samples_s"]
    assert loaded["filtering_and_postprocess"]["samples_s"]
    assert loaded["mask_upsample"]["samples_s"]
    assert loaded["neck_position_encodings"][0]["index"] == 0
    assert loaded["complete_path"]["p50_s"] == summarize_samples(complete_samples)["p50_s"]
    assert loaded["complete_path"]["p95_s"] == summarize_samples(complete_samples)["p95_s"]
    isolated_p95_sum = sum(
        float(timed[name]["p95_s"])
        for name in isolated_names
    )
    assert loaded["complete_path"]["p95_s"] != isolated_p95_sum
    assert "p95_sum_s" not in loaded["vit_block_groups"]["window"]
    assert order[len(isolated_names) * 5 :: len(isolated_names)][:3] == [
        isolated_names[0],
        isolated_names[0],
        isolated_names[0],
    ]
    assert "p50_sum_s" not in json.dumps(loaded)
    assert "compile_policy" not in loaded


def test_validate_substage_artifact_rejects_summed_p95_and_v1_names() -> None:
    protocol = InterleavedTimingProtocol()
    timed = {
        name: _dummy_summary(float(index))
        for index, name in enumerate(
            (
                "preprocessing",
                "token_preparation",
                "text_encoding",
                "text_encoding_repeated",
                "model_grounding_core",
                "filtering_and_postprocess",
                "mask_upsample",
                "complete_path",
                "vit_block_0",
            ),
            start=1,
        )
    }
    artifact = build_substage_artifact(
        resolution=672,
        model_compile_policy="eager",
        protocol=protocol,
        timed=timed,
        vit_blocks=[{**timed["vit_block_0"], "index": 0, "category": "window"}],
        neck_heads=[],
        neck_position_encodings=[],
        window_names=("vit_block_0",),
        global_names=("vit_block_0",),
        model_path_names=("token_preparation", "vit_block_0", "model_grounding_core"),
        active_memory_bytes=0,
        peak_active_memory_bytes=0,
    )
    artifact["patch_embedding"] = artifact["token_preparation"]
    with pytest.raises(ValueError, match="forbidden in substage v2"):
        validate_substage_artifact(artifact)

    del artifact["patch_embedding"]
    artifact["complete_path"] = dict(artifact["complete_path"])
    artifact["complete_path"]["p95_s"] = 999.0
    with pytest.raises(ValueError, match="computed from complete_path samples"):
        validate_substage_artifact(artifact)


def test_comparison_rejects_different_runtime_or_workload_contracts() -> None:
    baseline = _artifact(median=1.0)
    current = _artifact(median=1.0)
    current["runtime"] = {"backend": "mlx-metal", "compile": True}

    with pytest.raises(ValueError, match="runtime differs"):
        compare_benchmark_artifacts(
            baseline,
            current,
            metric_paths=["metrics.full.median_s"],
            threshold=RegressionThreshold(),
        )


def test_image_benchmark_defaults_use_aligned_windows_not_504() -> None:
    spec = importlib.util.spec_from_file_location(
        "benchmark_image_runtime",
        REPO_ROOT / "scripts" / "benchmark_image_runtime.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DEFAULT_METRICS == (
        "resolutions.336.full_image_text.median_s",
        "resolutions.336.full_cached_text.median_s",
        "resolutions.336.set_image.median_s",
        "resolutions.672.full_image_text.median_s",
        "resolutions.672.full_cached_text.median_s",
        "resolutions.672.set_image.median_s",
        "resolutions.1008.full_image_text.median_s",
        "resolutions.1008.full_cached_text.median_s",
        "resolutions.1008.set_image.median_s",
        "peak_active_memory_bytes",
    )
    assert all("504" not in metric for metric in module.DEFAULT_METRICS)
    assert "preprocessed_core" not in "".join(module.DEFAULT_METRICS)
