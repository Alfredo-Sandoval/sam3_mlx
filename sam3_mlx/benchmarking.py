"""Durable synchronized benchmark and regression contracts for MLX inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import statistics
import time
from typing import Generic, TypeVar, cast

from sam3_mlx.release_contract import JsonObject

T = TypeVar("T")

IMAGE_BENCHMARK_SCHEMA = "sam3_mlx.image_runtime_benchmark.v1"
IMAGE_BENCHMARK_COMPARISON_SCHEMA = "sam3_mlx.image_runtime_comparison.v1"
IMAGE_BENCHMARK_SUBSTAGE_SCHEMA = "sam3_mlx.image_runtime_substages.v2"
SUBSTAGE_TIMING_MODE = "eager-isolated-modules"
COMPLETE_PATH_TIMING_MODE = "synchronized-end-to-end"
SUBSTAGE_MIN_WARMUP_RUNS = 5
SUBSTAGE_MIN_ROUNDS = 3
SUBSTAGE_MIN_MEASUREMENTS = 30
SUBSTAGE_REQUIRED_STAGES = (
    "preprocessing",
    "token_preparation",
    "text_encoding",
    "text_encoding_repeated",
    "model_grounding_core",
    "filtering_and_postprocess",
    "mask_upsample",
    "complete_path",
)
_FORBIDDEN_SUBSTAGE_KEYS = frozenset(
    {
        "compile_policy",
        "p50_sum_s",
        "p95_sum_s",
        "median_sum_s",
        "patch_embedding",
        "grounding",
    }
)


def eager_isolated_substage_fields(*, model_compile_policy: str) -> JsonObject:
    """Label isolated module timings; they are not a compiled-graph breakdown."""

    return {
        "model_compile_policy": model_compile_policy,
        "timing_mode": SUBSTAGE_TIMING_MODE,
    }


@dataclass(frozen=True)
class TimingProtocol:
    """Warmup and repetition counts for one synchronized operation."""

    warmup_runs: int = 1
    repetitions: int = 5

    def __post_init__(self) -> None:
        if self.warmup_runs < 1:
            raise ValueError("warmup_runs must be at least 1")
        if self.repetitions < 5:
            raise ValueError("repetitions must be at least 5")


@dataclass(frozen=True)
class InterleavedTimingProtocol:
    """Isolated-stage protocol: warmup, then interleaved measurement rounds."""

    warmup_runs: int = SUBSTAGE_MIN_WARMUP_RUNS
    measurements_per_round: int = 10
    rounds: int = SUBSTAGE_MIN_ROUNDS

    @property
    def repetitions(self) -> int:
        return self.measurements_per_round * self.rounds

    def sequential(self) -> TimingProtocol:
        return TimingProtocol(
            warmup_runs=self.warmup_runs, repetitions=self.repetitions
        )

    def __post_init__(self) -> None:
        if self.warmup_runs < SUBSTAGE_MIN_WARMUP_RUNS:
            raise ValueError(
                f"warmup_runs must be at least {SUBSTAGE_MIN_WARMUP_RUNS}"
            )
        if self.measurements_per_round < 1:
            raise ValueError("measurements_per_round must be at least 1")
        if self.rounds < SUBSTAGE_MIN_ROUNDS:
            raise ValueError(f"rounds must be at least {SUBSTAGE_MIN_ROUNDS}")
        if self.repetitions < SUBSTAGE_MIN_MEASUREMENTS:
            raise ValueError(
                "total measurements must be at least "
                f"{SUBSTAGE_MIN_MEASUREMENTS}"
            )


@dataclass(frozen=True)
class RegressionThreshold:
    """Allowed regression and expected noise for a lower-is-better metric."""

    max_regression_pct: float = 10.0
    noise_pct: float = 3.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_regression_pct) or self.max_regression_pct < 0:
            raise ValueError("max_regression_pct must be finite and non-negative")
        if not math.isfinite(self.noise_pct) or self.noise_pct < 0:
            raise ValueError("noise_pct must be finite and non-negative")
        if self.noise_pct > self.max_regression_pct:
            raise ValueError("noise_pct cannot exceed max_regression_pct")


@dataclass(frozen=True)
class BenchmarkOperation(Generic[T]):
    """One operation plus the boundary that proves its work completed."""

    run: Callable[[], T]
    synchronize: Callable[[T], None]


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(float(sample) for sample in samples)
    if not all(math.isfinite(sample) and sample >= 0.0 for sample in ordered):
        raise ValueError("samples must be finite and non-negative")
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize_samples(samples: Sequence[float]) -> JsonObject:
    values = [float(sample) for sample in samples]
    if not values:
        raise ValueError("samples must not be empty")
    return {
        "samples_s": values,
        "p50_s": statistics.median(values),
        "median_s": statistics.median(values),
        "p95_s": percentile(values, 0.95),
        "min_s": min(values),
        "max_s": max(values),
    }


def should_escalate_resolution(
    scores: Sequence[object],
    *,
    confidence_threshold: float,
    min_score_margin: float,
) -> bool:
    """Return whether a 504-first result should retry at a higher resolution."""

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")
    if not math.isfinite(min_score_margin) or min_score_margin < 0.0:
        raise ValueError("min_score_margin must be finite and non-negative")
    values: list[float] = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError("scores must be finite numbers")
        number = float(score)
        if not math.isfinite(number):
            raise ValueError("scores must be finite numbers")
        values.append(number)
    if not values:
        return True
    return max(values) < (confidence_threshold + min_score_margin)


def profile_operations(
    operations: Mapping[str, BenchmarkOperation[object]],
    *,
    protocol: TimingProtocol,
) -> dict[str, JsonObject]:
    """Time named operations with the same synchronized protocol as the image bench."""

    if not operations:
        raise ValueError("operations must not be empty")
    return {
        name: summarize_samples(synchronized_samples(operation, protocol=protocol))
        for name, operation in operations.items()
    }


def profile_operations_interleaved(
    operations: Mapping[str, BenchmarkOperation[object]],
    *,
    protocol: InterleavedTimingProtocol,
) -> dict[str, JsonObject]:
    """Warm each stage, then rotate through stages each measurement round."""

    if not operations:
        raise ValueError("operations must not be empty")
    items = tuple(operations.items())
    for _name, operation in items:
        for _ in range(protocol.warmup_runs):
            operation.synchronize(operation.run())

    samples: dict[str, list[float]] = {name: [] for name, _operation in items}
    for _round in range(protocol.rounds):
        for _measurement in range(protocol.measurements_per_round):
            for name, operation in items:
                started = time.perf_counter()
                value = operation.run()
                operation.synchronize(value)
                samples[name].append(time.perf_counter() - started)
    return {name: summarize_samples(values) for name, values in samples.items()}


def aggregate_stage_group(
    stages: Mapping[str, Mapping[str, object]],
    *,
    category: str,
    member_names: Sequence[str],
) -> JsonObject:
    """Sum isolated stage medians; this is not an end-to-end percentile."""

    if not member_names:
        raise ValueError("member_names must not be empty")
    summaries = [_require_object(stages[name], field=name) for name in member_names]
    return {
        "category": category,
        "members": list(member_names),
        "sum_of_isolated_stage_medians_s": sum(
            _require_finite_number(item["median_s"], field="median_s")
            for item in summaries
        ),
    }


def build_substage_artifact(
    *,
    resolution: int,
    model_compile_policy: str,
    protocol: InterleavedTimingProtocol,
    timed: Mapping[str, Mapping[str, object]],
    vit_blocks: Sequence[Mapping[str, object]],
    neck_heads: Sequence[Mapping[str, object]],
    neck_position_encodings: Sequence[Mapping[str, object]],
    window_names: Sequence[str],
    global_names: Sequence[str],
    model_path_names: Sequence[str],
    active_memory_bytes: int,
    peak_active_memory_bytes: int,
) -> JsonObject:
    """Assemble a v2 isolated-substage artifact from already-timed summaries."""

    missing = [name for name in SUBSTAGE_REQUIRED_STAGES if name not in timed]
    if missing:
        raise ValueError(f"substage artifact missing required stages: {missing}")
    artifact: JsonObject = {
        "schema_version": IMAGE_BENCHMARK_SUBSTAGE_SCHEMA,
        "resolution": resolution,
        **eager_isolated_substage_fields(model_compile_policy=model_compile_policy),
        "complete_path_timing_mode": COMPLETE_PATH_TIMING_MODE,
        "warmup_runs": protocol.warmup_runs,
        "repetitions": protocol.repetitions,
        "rounds": protocol.rounds,
        "measurements_per_round": protocol.measurements_per_round,
        "preprocessing": dict(timed["preprocessing"]),
        "token_preparation": dict(timed["token_preparation"]),
        "vit_blocks": [dict(block) for block in vit_blocks],
        "vit_block_groups": {
            "window": aggregate_stage_group(
                timed, category="window", member_names=window_names
            ),
            "global": aggregate_stage_group(
                timed, category="global", member_names=global_names
            ),
        },
        "neck_heads": [dict(head) for head in neck_heads],
        "neck_position_encodings": [
            dict(item) for item in neck_position_encodings
        ],
        "text_encoding": dict(timed["text_encoding"]),
        "text_encoding_repeated": dict(timed["text_encoding_repeated"]),
        "model_grounding_core": dict(timed["model_grounding_core"]),
        "filtering_and_postprocess": dict(timed["filtering_and_postprocess"]),
        "mask_upsample": dict(timed["mask_upsample"]),
        "complete_path": dict(timed["complete_path"]),
        "isolated_stage_groups": {
            "model_path": aggregate_stage_group(
                timed, category="model_path", member_names=model_path_names
            ),
        },
        "active_memory_bytes": active_memory_bytes,
        "peak_active_memory_bytes": peak_active_memory_bytes,
    }
    validate_substage_artifact(artifact)
    return artifact


def validate_substage_artifact(artifact: Mapping[str, object]) -> None:
    """Reject v1 names, summed percentiles, and complete-path quantile sums."""

    if artifact.get("schema_version") != IMAGE_BENCHMARK_SUBSTAGE_SCHEMA:
        raise ValueError("substage artifact schema must be v2")
    _reject_forbidden_substage_keys(artifact, path="substages")
    for name in SUBSTAGE_REQUIRED_STAGES:
        if name not in artifact:
            raise ValueError(f"substage artifact missing required stage {name}")
    warmup = _require_int(artifact.get("warmup_runs"), field="warmup_runs")
    repetitions = _require_int(artifact.get("repetitions"), field="repetitions")
    rounds = _require_int(artifact.get("rounds"), field="rounds")
    if warmup < SUBSTAGE_MIN_WARMUP_RUNS:
        raise ValueError(
            f"substage warmup_runs must be at least {SUBSTAGE_MIN_WARMUP_RUNS}"
        )
    if rounds < SUBSTAGE_MIN_ROUNDS:
        raise ValueError(f"substage rounds must be at least {SUBSTAGE_MIN_ROUNDS}")
    if repetitions < SUBSTAGE_MIN_MEASUREMENTS:
        raise ValueError(
            "substage repetitions must be at least "
            f"{SUBSTAGE_MIN_MEASUREMENTS}"
        )
    complete = _require_object(artifact.get("complete_path"), field="complete_path")
    samples = complete.get("samples_s")
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("complete_path.samples_s must be a sequence of timings")
    values = [
        _require_finite_number(sample, field="complete_path.samples_s")
        for sample in samples
    ]
    if len(values) < SUBSTAGE_MIN_MEASUREMENTS:
        raise ValueError("complete_path must include at least 30 samples")
    expected = summarize_samples(values)
    for field in ("p50_s", "median_s", "p95_s"):
        observed = _require_finite_number(complete.get(field), field=field)
        if observed != expected[field]:
            raise ValueError(
                f"complete_path.{field} must be computed from complete_path samples"
            )
    if complete.get("timing_mode") not in (None, COMPLETE_PATH_TIMING_MODE):
        raise ValueError("complete_path must not reuse isolated-module timing_mode")
    groups = _require_object(
        artifact.get("vit_block_groups"), field="vit_block_groups"
    )
    for name, group in groups.items():
        _validate_isolated_group(group, field=f"vit_block_groups.{name}")
    isolated_groups = _require_object(
        artifact.get("isolated_stage_groups"), field="isolated_stage_groups"
    )
    for name, group in isolated_groups.items():
        _validate_isolated_group(group, field=f"isolated_stage_groups.{name}")


def synchronized_samples(
    operation: BenchmarkOperation[T],
    *,
    protocol: TimingProtocol,
) -> list[float]:
    """Measure completed work; queued MLX work never counts as completion."""

    for _ in range(protocol.warmup_runs):
        value = operation.run()
        operation.synchronize(value)

    samples: list[float] = []
    for _ in range(protocol.repetitions):
        started = time.perf_counter()
        value = operation.run()
        operation.synchronize(value)
        samples.append(time.perf_counter() - started)
    return samples


def _require_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _reject_forbidden_substage_keys(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            if key in _FORBIDDEN_SUBSTAGE_KEYS:
                raise ValueError(f"{path}.{key} is forbidden in substage v2")
            _reject_forbidden_substage_keys(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_forbidden_substage_keys(item, path=f"{path}[{index}]")


def _validate_isolated_group(value: object, *, field: str) -> None:
    group = _require_object(value, field=field)
    if "sum_of_isolated_stage_medians_s" not in group:
        raise ValueError(f"{field} must include sum_of_isolated_stage_medians_s")
    _require_finite_number(
        group["sum_of_isolated_stage_medians_s"],
        field=f"{field}.sum_of_isolated_stage_medians_s",
    )


def _require_object(value: object, *, field: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result: JsonObject = {}
    for key, item in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise ValueError(f"{field} keys must be strings")
        result[key] = item
    return result


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _metric_value(artifact: JsonObject, path: str) -> float:
    current: object = artifact
    for part in path.split("."):
        current = _require_object(current, field=path).get(part)
    return _require_finite_number(current, field=path)


def _comparison_status(
    percent_change: float,
    *,
    threshold: RegressionThreshold,
) -> str:
    if percent_change > threshold.max_regression_pct:
        return "regressed"
    if percent_change < -threshold.noise_pct:
        return "improved"
    return "flat-noisy"


def compare_benchmark_artifacts(
    baseline: JsonObject,
    current: JsonObject,
    *,
    metric_paths: Sequence[str],
    threshold: RegressionThreshold,
) -> JsonObject:
    """Compare like-for-like lower-is-better metrics with explicit thresholds."""

    for label, artifact in (("baseline", baseline), ("current", current)):
        if artifact.get("schema_version") != IMAGE_BENCHMARK_SCHEMA:
            raise ValueError(f"{label} benchmark schema is unsupported")
    for field in ("environment", "workload", "runtime"):
        if baseline.get(field) != current.get(field):
            raise ValueError(
                f"Benchmark {field} differs; comparison is not like-for-like"
            )
    for label, artifact in (("baseline", baseline), ("current", current)):
        provenance = _require_object(artifact.get("provenance"), field="provenance")
        if provenance.get("dirty") is not False:
            raise ValueError(f"{label} benchmark must come from a clean worktree")
    if not metric_paths:
        raise ValueError("metric_paths must not be empty")

    comparisons: list[JsonObject] = []
    for path in metric_paths:
        baseline_value = _metric_value(baseline, path)
        current_value = _metric_value(current, path)
        if baseline_value == 0.0:
            raise ValueError(f"Baseline metric {path} must be positive")
        delta = current_value - baseline_value
        percent_change = delta / baseline_value * 100.0
        comparisons.append(
            {
                "metric": path,
                "direction": "lower-is-better",
                "baseline": baseline_value,
                "current": current_value,
                "delta": delta,
                "percent_change": percent_change,
                "noise_pct": threshold.noise_pct,
                "max_regression_pct": threshold.max_regression_pct,
                "status": _comparison_status(percent_change, threshold=threshold),
            }
        )

    return {
        "schema_version": IMAGE_BENCHMARK_COMPARISON_SCHEMA,
        "status": (
            "fail"
            if any(item["status"] == "regressed" for item in comparisons)
            else "pass"
        ),
        "baseline_provenance": baseline.get("provenance"),
        "current_provenance": current.get("provenance"),
        "metrics": comparisons,
    }
