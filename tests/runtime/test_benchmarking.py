from __future__ import annotations

import pytest

from sam3_mlx.benchmarking import (
    BenchmarkOperation,
    IMAGE_BENCHMARK_SCHEMA,
    RegressionThreshold,
    TimingProtocol,
    compare_benchmark_artifacts,
    percentile,
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
