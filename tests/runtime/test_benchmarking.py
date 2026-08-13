from __future__ import annotations

import pytest

from sam3_mlx.benchmarking import (
    BenchmarkOperation,
    IMAGE_BENCHMARK_SCHEMA,
    SUBSTAGE_TIMING_MODE,
    RegressionThreshold,
    TimingProtocol,
    aggregate_stage_group,
    compare_benchmark_artifacts,
    eager_isolated_substage_fields,
    percentile,
    profile_operations,
    should_escalate_resolution,
    summarize_samples,
    synchronized_samples,
)


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
    assert SUBSTAGE_TIMING_MODE == "eager-isolated-modules"
    assert "compile_policy" not in fields


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
    assert grouped["p50_sum_s"] == profiled["a"]["p50_s"] + profiled["b"]["p50_s"]


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
