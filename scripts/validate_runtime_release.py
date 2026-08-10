#!/usr/bin/env python3
"""Semantic runtime release gate for sam3_mlx.

Layer 1 packaging validation remains ``scripts/validate_release.py``.
This script validates (or generates) a machine-readable parity receipt.

Usage
-----
Validate a committed receipt (CI / release tag)::

    uv run python scripts/validate_runtime_release.py \\
        --receipt parity/receipts/latest.json

Generate a new receipt on Apple Silicon with checkpoints available::

    SAM3_MLX_PARITY_CHECKPOINT=/path/to/model.safetensors \\
    uv run python scripts/validate_runtime_release.py --generate \\
        --receipt parity/receipts/latest.json

Ordinary developer pytest stays Torch-free. A full end-to-end upstream parity
run requires official SAM 3 weights and optional TorchVision/oracle deps and is
documented in PARITY.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.release_contract import (  # noqa: E402
    JsonObject,
    require_json_finite_number,
    require_json_list,
    require_json_nonnegative_int,
    require_json_object,
    require_json_string,
)

REQUIRED_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "package_version",
    "git_commit",
    "generated_at",
    "python_version",
    "mlx_version",
    "platform",
    "machine",
    "checkpoint",
    "tests",
    "parity",
    "performance",
}

REQUIRED_CHECKPOINT_FIELDS = {
    "official_repo",
    "official_revision",
    "official_sha256",
    "converted_sha256",
    "conversion_manifest_sha256",
}

REQUIRED_TESTS_FIELDS = {
    "command",
    "exit_code",
    "passed",
    "failed",
    "skipped",
    "deselected",
    "skip_details",
    "counts",
}
ATTESTATION_PATH_PREFIXES = ("parity/receipts/", "parity/manifests/")
RELEASE_THRESHOLDS = {
    "mask_iou_min": 0.95,
    "mask_iou_mean_min": 0.99,
    "box_l_inf_max": 2.0,
    "score_abs_max": 0.025,
}


class ReceiptError(ValueError):
    """Raised when a parity receipt fails schema or binding checks."""


ThresholdContract = dict[str, float]


class TestEnvironment(TypedDict):
    python_version: str
    mlx_version: str | None
    platform: str
    machine: str


class SkipDetail(TypedDict):
    nodeid: str
    reason: str
    owner: str
    disposition: str


class ReportStats(TypedDict):
    passed: int
    failed: int
    skipped: int
    skip_details: list[SkipDetail]


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _git_commit() -> str:
    result = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    return result.stdout.strip()


def _git_is_clean() -> bool:
    result = _run(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    return not result.stdout.strip()


def _validate_receipt_git_binding(receipt_commit: str) -> str:
    """Validate a direct HEAD binding or a receipt-only attestation commit."""

    head = _git_commit()
    if receipt_commit == head:
        return receipt_commit

    parents = _run(
        ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
        cwd=REPO_ROOT,
    ).stdout.split()
    if len(parents) != 2 or receipt_commit != parents[1]:
        raise ReceiptError(
            "Receipt git_commit must be HEAD or the sole parent of a "
            "receipt-only attestation commit."
        )
    changed_paths = [
        path
        for path in _run(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"],
            cwd=REPO_ROOT,
        ).stdout.splitlines()
        if path
    ]
    if not changed_paths or any(
        not path.startswith(ATTESTATION_PATH_PREFIXES) for path in changed_paths
    ):
        raise ReceiptError(
            "A parent-bound receipt requires the current commit to change only "
            f"attestation paths under {ATTESTATION_PATH_PREFIXES}; "
            f"changed={changed_paths}."
        )
    return receipt_commit


def _package_version() -> str:
    # Prefer installed metadata; fall back to importlib on editable installs.
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("sam3-mlx")
    except PackageNotFoundError:
        import sam3_mlx

        return sam3_mlx.__version__


def _require_object(value: object, *, field: str) -> JsonObject:
    return require_json_object(value, field=field, error_type=ReceiptError)


def _require_list(value: object, *, field: str) -> list[object]:
    return require_json_list(value, field=field, error_type=ReceiptError)


def _require_string(value: object, *, field: str) -> str:
    return require_json_string(value, field=field, error_type=ReceiptError)


def _load_json_object(path: Path, *, label: str) -> JsonObject:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReceiptError(f"Could not read {label} {path}: {exc}") from exc
    return _require_object(value, field=label)


def _require_nonnegative_int(value: object, *, field: str) -> int:
    return require_json_nonnegative_int(value, field=field, error_type=ReceiptError)


def _test_environment(python_command: list[str]) -> TestEnvironment:
    """Read release metadata from the interpreter that will execute pytest."""

    probe = (
        "import json, platform\n"
        "from importlib.metadata import PackageNotFoundError, version\n"
        "try:\n"
        "    mlx_version = version('mlx')\n"
        "except PackageNotFoundError:\n"
        "    mlx_version = None\n"
        "print(json.dumps({\n"
        "    'python_version': platform.python_version(),\n"
        "    'mlx_version': mlx_version,\n"
        "    'platform': platform.platform(),\n"
        "    'machine': platform.machine(),\n"
        "}, sort_keys=True))\n"
    )
    completed = _run([*python_command, "-c", probe], cwd=REPO_ROOT)
    metadata = _require_object(
        json.loads(completed.stdout),
        field="test interpreter environment metadata",
    )
    required = {"python_version", "mlx_version", "platform", "machine"}
    if set(metadata) != required:
        raise ReceiptError(
            f"Test interpreter returned invalid environment metadata: {metadata!r}."
        )
    python_version = _require_string(
        metadata.get("python_version"),
        field="test interpreter python_version",
    )
    platform_name = _require_string(
        metadata.get("platform"),
        field="test interpreter platform",
    )
    machine = _require_string(
        metadata.get("machine"),
        field="test interpreter machine",
    )
    mlx_version = metadata.get("mlx_version")
    if mlx_version is not None and not isinstance(mlx_version, str):
        raise ReceiptError("test interpreter mlx_version must be a string or null.")
    return {
        "python_version": python_version,
        "mlx_version": mlx_version,
        "platform": platform_name,
        "machine": machine,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, *, root: Path | None = None) -> str:
    evidence_root = REPO_ROOT if root is None else root
    return str(path.resolve().relative_to(evidence_root.resolve()))


def _redact_repo_path(text: str) -> str:
    return text.replace(str(REPO_ROOT), "<repo>")


def _evidence_path(value: object, *, root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ReceiptError("Evidence paths must be non-empty strings.")
    path = Path(value)
    if path.is_absolute():
        raise ReceiptError(f"Evidence path must be repository-relative: {value!r}.")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ReceiptError(
            f"Evidence path escapes the repository root: {value!r}."
        ) from exc
    if not resolved.is_file():
        raise ReceiptError(f"Evidence file does not exist: {value!r}.")
    return resolved


def _require_finite_number(value: object, *, field: str) -> float:
    return require_json_finite_number(value, field=field, error_type=ReceiptError)


def _validate_parity_case(
    case: JsonObject,
    *,
    thresholds: ThresholdContract,
) -> None:
    name_value = case.get("name")
    name = name_value if isinstance(name_value, str) and name_value else "<unnamed>"
    official_count = case.get("official_detection_count")
    mlx_count = case.get("mlx_detection_count")
    if (
        isinstance(official_count, bool)
        or not isinstance(official_count, int)
        or official_count < 0
        or isinstance(mlx_count, bool)
        or not isinstance(mlx_count, int)
        or mlx_count < 0
    ):
        raise ReceiptError(f"Parity case {name!r} has invalid detection counts.")
    if (
        case.get("status") != "passed"
        or case.get("detection_count_match") is not True
        or official_count != mlx_count
    ):
        raise ReceiptError(f"Parity case {name!r} does not satisfy count parity.")

    matches = _require_list(case.get("matches"), field=f"Parity case {name!r} matches")
    if len(matches) != official_count:
        raise ReceiptError(
            f"Parity case {name!r} match count does not equal detection count."
        )
    metric_fields = (
        "mask_iou_min",
        "mask_iou_mean",
        "box_l_inf_max",
        "score_abs_max",
    )
    if official_count == 0:
        if any(case.get(field) is not None for field in metric_fields):
            raise ReceiptError(f"Empty parity case {name!r} must have null metrics.")
        return

    official_indices: set[int] = set()
    mlx_indices: set[int] = set()
    ious: list[float] = []
    for raw_match in matches:
        match = _require_object(raw_match, field=f"Parity case {name!r} match")
        official_index = match.get("official_index")
        mlx_index = match.get("mlx_index")
        if (
            isinstance(official_index, bool)
            or not isinstance(official_index, int)
            or not 0 <= official_index < official_count
            or isinstance(mlx_index, bool)
            or not isinstance(mlx_index, int)
            or not 0 <= mlx_index < mlx_count
        ):
            raise ReceiptError(f"Parity case {name!r} has an invalid match index.")
        official_indices.add(official_index)
        mlx_indices.add(mlx_index)
        iou = _require_finite_number(
            match.get("mask_iou"),
            field=f"Parity case {name!r} match mask_iou",
        )
        if not 0.0 <= iou <= 1.0:
            raise ReceiptError(f"Parity case {name!r} mask IoU is outside [0, 1].")
        ious.append(iou)
    if len(official_indices) != official_count or len(mlx_indices) != mlx_count:
        raise ReceiptError(f"Parity case {name!r} matches are not one-to-one.")

    mask_iou_min = _require_finite_number(
        case.get("mask_iou_min"),
        field=f"Parity case {name!r} mask_iou_min",
    )
    mask_iou_mean = _require_finite_number(
        case.get("mask_iou_mean"),
        field=f"Parity case {name!r} mask_iou_mean",
    )
    box_l_inf_max = _require_finite_number(
        case.get("box_l_inf_max"),
        field=f"Parity case {name!r} box_l_inf_max",
    )
    score_abs_max = _require_finite_number(
        case.get("score_abs_max"),
        field=f"Parity case {name!r} score_abs_max",
    )
    if not math.isclose(mask_iou_min, min(ious), rel_tol=0.0, abs_tol=1e-15):
        raise ReceiptError(f"Parity case {name!r} mask_iou_min is not reproducible.")
    if not math.isclose(
        mask_iou_mean,
        statistics.mean(ious),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ReceiptError(f"Parity case {name!r} mask_iou_mean is not reproducible.")
    if (
        mask_iou_min < thresholds["mask_iou_min"]
        or mask_iou_mean < thresholds["mask_iou_mean_min"]
        or box_l_inf_max > thresholds["box_l_inf_max"]
        or score_abs_max > thresholds["score_abs_max"]
    ):
        raise ReceiptError(f"Parity case {name!r} violates the metric thresholds.")


def _nearest_rank_percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _validate_performance(performance: JsonObject, *, profile: str) -> None:
    if performance.get("status") != "passed":
        raise ReceiptError(f"Parity profile {profile!r} performance must pass.")
    repetitions = performance.get("repetitions")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ReceiptError(f"Parity profile {profile!r} repetitions are invalid.")
    if repetitions < 5 or performance.get("warmup_runs") != 1:
        raise ReceiptError(
            f"Parity profile {profile!r} requires five runs and one warmup."
        )
    _require_finite_number(
        performance.get("cold_load_s"),
        field=f"Parity profile {profile!r} cold_load_s",
    )
    peak_memory = performance.get("peak_active_memory_bytes")
    if (
        isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory <= 0
    ):
        raise ReceiptError(f"Parity profile {profile!r} peak memory is invalid.")
    latencies = _require_object(
        performance.get("latency_by_resolution_s"),
        field=f"Parity profile {profile!r} latency_by_resolution_s",
    )
    if set(latencies) != {"1008", "672", "504"}:
        raise ReceiptError(
            f"Parity profile {profile!r} must measure 1008, 672, and 504."
        )
    for resolution, raw_summary in latencies.items():
        summary = _require_object(
            raw_summary,
            field=f"Parity profile {profile!r} resolution {resolution}",
        )
        samples = _require_list(
            summary.get("samples"),
            field=f"Parity profile {profile!r} resolution {resolution} samples",
        )
        if len(samples) != repetitions:
            raise ReceiptError(
                f"Parity profile {profile!r} resolution {resolution} "
                "sample count is invalid."
            )
        measured = [
            _require_finite_number(
                sample,
                field=f"Parity profile {profile!r} latency sample",
            )
            for sample in samples
        ]
        if any(sample <= 0 for sample in measured):
            raise ReceiptError(
                f"Parity profile {profile!r} latency samples must be positive."
            )
        expected_metrics = {
            "median": statistics.median(measured),
            "p95": _nearest_rank_percentile(measured, 0.95),
        }
        for field, expected in expected_metrics.items():
            observed = _require_finite_number(
                summary.get(field),
                field=(f"Parity profile {profile!r} resolution {resolution} {field}"),
            )
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15):
                raise ReceiptError(
                    f"Parity profile {profile!r} resolution {resolution} "
                    f"{field} is not reproducible."
                )


def validate_receipt(
    receipt: JsonObject,
    *,
    expected_commit: str | None = None,
    require_passed: bool = True,
    evidence_root: Path | None = None,
) -> None:
    missing = sorted(REQUIRED_RECEIPT_FIELDS - set(receipt))
    if missing:
        raise ReceiptError(f"Receipt missing required fields: {missing}")

    if receipt.get("schema_version") != 1:
        raise ReceiptError(
            f"Unsupported receipt schema_version={receipt.get('schema_version')!r}"
        )

    checkpoint = _require_object(receipt.get("checkpoint"), field="Receipt checkpoint")
    missing_ckpt = sorted(REQUIRED_CHECKPOINT_FIELDS - set(checkpoint))
    if missing_ckpt:
        raise ReceiptError(f"Receipt checkpoint missing fields: {missing_ckpt}")
    empty_checkpoint_fields = sorted(
        field for field in REQUIRED_CHECKPOINT_FIELDS if not checkpoint.get(field)
    )
    if empty_checkpoint_fields:
        raise ReceiptError(
            f"Receipt checkpoint fields must be non-empty: {empty_checkpoint_fields}"
        )

    tests = _require_object(receipt.get("tests"), field="Receipt tests")
    missing_tests = sorted(REQUIRED_TESTS_FIELDS - set(tests))
    if missing_tests:
        raise ReceiptError(f"Receipt tests missing fields: {missing_tests}")
    skip_details = _require_list(
        tests.get("skip_details"),
        field="Receipt tests.skip_details",
    )
    for raw_item in skip_details:
        item = _require_object(raw_item, field="Receipt tests.skip_details entry")
        required_skip_fields = {"nodeid", "reason", "owner", "disposition"}
        if not required_skip_fields.issubset(item):
            raise ReceiptError(
                "Each skip_details entry must include nodeid, reason, owner, "
                "and disposition."
            )
    for field in ("skipped", "deselected"):
        value = tests.get(field)
        _require_nonnegative_int(value, field=f"Receipt tests.{field}")
    skipped_count = _require_nonnegative_int(
        tests.get("skipped"),
        field="Receipt tests.skipped",
    )
    deselected_count = _require_nonnegative_int(
        tests.get("deselected"),
        field="Receipt tests.deselected",
    )
    recorded_skips = skipped_count + deselected_count
    if recorded_skips != len(skip_details):
        raise ReceiptError(
            "Receipt skip/deselection counts must match skip_details entries."
        )
    counts = _require_object(tests.get("counts"), field="Receipt tests.counts")
    required_counts = {"call_passed", "call_failed", "call_skipped"}
    if set(counts) != required_counts:
        raise ReceiptError("Receipt tests.counts must contain non-negative integers.")
    call_passed = _require_nonnegative_int(
        counts.get("call_passed"),
        field="Receipt tests.counts.call_passed",
    )
    call_failed = _require_nonnegative_int(
        counts.get("call_failed"),
        field="Receipt tests.counts.call_failed",
    )
    call_skipped = _require_nonnegative_int(
        counts.get("call_skipped"),
        field="Receipt tests.counts.call_skipped",
    )
    exit_code = tests.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ReceiptError("Receipt tests.exit_code must be an integer.")
    if (
        tests.get("passed") is not (exit_code == 0)
        or tests.get("failed") is not (exit_code != 0)
        or (exit_code == 0 and call_failed != 0)
        or (exit_code != 0 and call_failed < 1)
        or call_skipped != skipped_count
    ):
        raise ReceiptError(
            "Receipt test booleans, exit code, and outcome counts are inconsistent."
        )

    parity = _require_object(receipt.get("parity"), field="Receipt parity")
    if "status" not in parity:
        raise ReceiptError("Receipt parity.status is required.")
    if parity["status"] not in {"passed", "failed", "not_run", "skipped"}:
        raise ReceiptError(f"Invalid parity.status={parity['status']!r}")

    if expected_commit is not None and receipt.get("git_commit") != expected_commit:
        raise ReceiptError(
            "Receipt git_commit does not match the current package commit: "
            f"receipt={receipt.get('git_commit')!r}, expected={expected_commit!r}."
        )

    if require_passed and receipt.get("status") != "passed":
        raise ReceiptError(
            f"Receipt status is {receipt.get('status')!r}, not 'passed'."
        )

    if require_passed and parity.get("status") != "passed":
        raise ReceiptError(
            f"Receipt parity.status is {parity.get('status')!r}; "
            "release requires measured upstream parity with status 'passed'."
        )
    if require_passed and not parity.get("cases"):
        raise ReceiptError("Release receipt parity.cases must be non-empty.")
    if require_passed:
        thresholds_object = _require_object(
            parity.get("thresholds"),
            field="Receipt parity thresholds",
        )
        required_thresholds = {
            "mask_iou_min",
            "mask_iou_mean_min",
            "box_l_inf_max",
            "score_abs_max",
        }
        if any(thresholds_object.get(field) is None for field in required_thresholds):
            raise ReceiptError("Release receipt parity thresholds must be measured.")
        performance = _require_object(
            receipt.get("performance"),
            field="Receipt performance",
        )
        if performance.get("status") != "passed":
            raise ReceiptError("Release receipt performance.status must be 'passed'.")
        if not receipt.get("mlx_version"):
            raise ReceiptError("Release receipt mlx_version must be recorded.")
        if call_passed < 1 or call_failed != 0:
            raise ReceiptError("Release receipt must record a passing test suite.")
        if evidence_root is None:
            raise ReceiptError(
                "Release receipt validation requires an evidence repository root."
            )
        reports = _require_list(parity.get("reports"), field="Receipt parity reports")
        if len(reports) != 2:
            raise ReceiptError(
                "Release receipt must reference exactly two parity reports."
            )
        report_paths: list[Path] = []
        for raw_report in reports:
            report = _require_object(
                raw_report, field="Receipt parity report reference"
            )
            if set(report) != {"path", "sha256"}:
                raise ReceiptError(
                    "Parity report references must contain path and sha256."
                )
            report_path = _evidence_path(report.get("path"), root=evidence_root)
            recorded_sha256 = _require_string(
                report.get("sha256"),
                field="Receipt parity report sha256",
            )
            if _sha256(report_path) != recorded_sha256:
                raise ReceiptError(
                    f"Parity report digest mismatch: {report.get('path')!r}."
                )
            report_paths.append(report_path)
        lineage_path = _evidence_path(
            checkpoint.get("lineage_report"),
            root=evidence_root,
        )
        if _sha256(lineage_path) != _require_string(
            checkpoint.get("lineage_report_sha256"),
            field="Receipt checkpoint lineage_report_sha256",
        ):
            raise ReceiptError("Checkpoint lineage report digest mismatch.")
        projection = _measured_evidence_projection(
            parity_report_paths=report_paths,
            lineage_report_path=lineage_path,
            evidence_root=evidence_root,
        )
        for section, expected in projection.items():
            if receipt.get(section) != expected:
                raise ReceiptError(
                    f"Receipt {section} does not match referenced evidence."
                )


def _parse_pytest_report_log(report_path: Path) -> ReportStats:
    """Parse a pytest --report-log JSONL file into skip/deselection details."""
    passed = failed = skipped = 0
    skip_details: list[SkipDetail] = []
    if not report_path.exists():
        return {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "skip_details": skip_details,
        }
    for line in report_path.read_text().splitlines():
        if not line.strip():
            continue
        event = _require_object(json.loads(line), field="pytest report log event")
        if event.get("$report_type") != "TestReport":
            # Newer pytest-reportlog uses "when"/"outcome" on TestReport only.
            if event.get("nodeid") and event.get("when") == "call":
                outcome = event.get("outcome")
            else:
                continue
        else:
            if event.get("when") != "call":
                continue
            outcome = event.get("outcome")
        nodeid = event.get("nodeid") or ""
        if outcome == "passed":
            passed += 1
        elif outcome == "failed":
            failed += 1
        elif outcome == "skipped":
            skipped += 1
            longrepr = event.get("longrepr")
            reason = ""
            if isinstance(longrepr, (list, tuple)):
                parts = list(cast(list[object] | tuple[object, ...], longrepr))
                if parts:
                    reason = str(parts[-1])
            elif longrepr is not None:
                reason = str(longrepr)
            nodeid_text = nodeid if isinstance(nodeid, str) else str(nodeid)
            skip_details.append(
                {
                    "nodeid": nodeid_text,
                    "reason": reason or "skipped",
                    "owner": "unassigned",
                    "disposition": "unreviewed",
                }
            )
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "skip_details": skip_details,
    }


def generate_stub_receipt(
    *,
    pytest_command: list[str],
    test_python_command: list[str],
    checkpoint_path: str | None,
) -> JsonObject:
    """Create a schema-complete receipt without claiming e2e parity.

    Full upstream comparison should replace parity.status with measured metrics
    once a capable host and licensed checkpoints are available.
    """
    from sam3_mlx.convert import DEFAULT_MLX_CHECKPOINT

    git_commit = _git_commit()
    package_version = _package_version()
    test_environment = _test_environment(test_python_command)

    report_path = REPO_ROOT / "parity" / "receipts" / ".pytest-report.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report_path.unlink()

    command = list(pytest_command)
    # Prefer structured skip capture when the plugin is available; fall back
    # cleanly if the report log plugin is absent.
    command_with_report = command + [f"--report-log={report_path}"]
    pytest_env = dict(os.environ)
    pytest_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        command_with_report,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=pytest_env,
    )
    if completed.returncode != 0 and "unrecognized arguments: --report-log" in (
        completed.stderr + completed.stdout
    ):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            env=pytest_env,
        )
        report_stats: ReportStats = {
            "passed": 0,
            "failed": 0 if completed.returncode == 0 else 1,
            "skipped": 0,
            "skip_details": [],
        }
    else:
        report_stats = _parse_pytest_report_log(report_path)
    report_path.unlink(missing_ok=True)

    status = "blocked" if completed.returncode == 0 else "failed"
    parity_status = "skipped"
    if checkpoint_path:
        parity_detail = (
            "Checkpoint path provided but upstream oracle comparison is not "
            "automated in this stub generator; run scripts/run_image_parity.py "
            "and merge measured cases before claiming release-grade parity."
        )
    else:
        parity_detail = (
            "No SAM3_MLX_PARITY_CHECKPOINT / --checkpoint provided; "
            "end-to-end parity was not executed. Do not claim release-grade "
            "output parity until a measured upstream comparison is recorded."
        )

    return {
        "schema_version": 1,
        "status": status,
        "package_version": package_version,
        "git_commit": git_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **test_environment,
        "checkpoint": {
            "official_repo": "facebook/sam3",
            "official_revision": "not_recorded",
            "official_sha256": "not_recorded",
            "converted_sha256": DEFAULT_MLX_CHECKPOINT.output_sha256,
            "conversion_manifest_sha256": "not_recorded",
            "artifact_repo_or_path": checkpoint_path
            or f"{DEFAULT_MLX_CHECKPOINT.repo}@{DEFAULT_MLX_CHECKPOINT.revision}",
            "artifact_revision": DEFAULT_MLX_CHECKPOINT.revision,
            "architecture": DEFAULT_MLX_CHECKPOINT.architecture,
        },
        "tests": {
            "command": " ".join(command),
            "exit_code": completed.returncode,
            "passed": completed.returncode == 0,
            "failed": completed.returncode != 0,
            "skipped": report_stats["skipped"],
            "deselected": 0,
            "skip_details": report_stats["skip_details"],
            "counts": {
                "call_passed": report_stats["passed"],
                "call_failed": report_stats["failed"],
                "call_skipped": report_stats["skipped"],
            },
            "stdout_tail": _redact_repo_path(completed.stdout[-4000:]),
            "stderr_tail": _redact_repo_path(completed.stderr[-4000:]),
        },
        "parity": {
            "status": parity_status,
            "detail": parity_detail,
            "resolutions": [1008, 672, 504],
            "cases": [],
            "thresholds": {
                "mask_iou_min": None,
                "box_l_inf_max": None,
                "score_abs_max": None,
            },
        },
        "performance": {
            "status": "not_run",
            "cold_load_s": None,
            "steady_state_image_latency_s": None,
            "peak_active_memory_bytes": None,
        },
    }


def _measured_evidence_projection(
    *,
    parity_report_paths: list[Path],
    lineage_report_path: Path,
    evidence_root: Path,
) -> dict[str, JsonObject]:
    """Load measured reports and reconstruct the canonical receipt projection."""
    reports = [
        _load_json_object(path, label="measured parity report")
        for path in parity_report_paths
    ]
    lineage = _load_json_object(lineage_report_path, label="checkpoint lineage report")
    if not reports or any(report.get("status") != "passed" for report in reports):
        raise ReceiptError("Every parity report must have status='passed'.")
    profiles = {report.get("case_profile") for report in reports}
    if profiles != {"example", "holdout"}:
        raise ReceiptError(
            "Release parity requires both example and independent holdout profiles."
        )
    comparison = _require_object(lineage.get("comparison"), field="lineage comparison")
    if (
        lineage.get("status") != "passed"
        or comparison.get("semantic_match") is not True
    ):
        raise ReceiptError("Checkpoint lineage report must be a semantic pass.")

    source = _require_object(lineage.get("source"), field="lineage source")
    published = _require_object(
        lineage.get("published_artifact"),
        field="lineage published_artifact",
    )
    official_code_revisions = {
        _require_string(
            _require_object(
                report.get("official_code"),
                field="parity report official_code",
            ).get("revision"),
            field="parity report official_code revision",
        )
        for report in reports
    }
    if len(official_code_revisions) != 1:
        raise ReceiptError("Parity reports disagree on official code revision.")
    threshold_objects = [
        _require_object(report.get("thresholds"), field="parity report thresholds")
        for report in reports
    ]
    threshold_contracts = {
        json.dumps(threshold_object, sort_keys=True)
        for threshold_object in threshold_objects
    }
    if len(threshold_contracts) != 1:
        raise ReceiptError("Parity reports disagree on threshold contract.")
    thresholds_object = threshold_objects[0]
    required_thresholds = set(RELEASE_THRESHOLDS)
    if set(thresholds_object) != required_thresholds:
        raise ReceiptError("Parity reports contain an invalid threshold contract.")
    thresholds: ThresholdContract = {
        field: _require_finite_number(
            thresholds_object.get(field),
            field=f"Parity threshold {field}",
        )
        for field in sorted(required_thresholds)
    }
    if thresholds != RELEASE_THRESHOLDS:
        raise ReceiptError("Parity reports do not use the fixed release thresholds.")

    for report in reports:
        official_checkpoint = _require_object(
            report.get("official_checkpoint"),
            field="parity report official_checkpoint",
        )
        converted_checkpoint = _require_object(
            report.get("converted_checkpoint"),
            field="parity report converted_checkpoint",
        )
        case_entries = _require_list(report.get("cases"), field="parity report cases")
        if official_checkpoint.get("revision") != source.get("revision"):
            raise ReceiptError("Parity and lineage official revisions disagree.")
        if official_checkpoint.get("sha256") != source.get("checkpoint_sha256"):
            raise ReceiptError(
                "Parity and lineage official checkpoint hashes disagree."
            )
        if converted_checkpoint.get("revision") != published.get(
            "revision"
        ) or converted_checkpoint.get("sha256") != published.get("checkpoint_sha256"):
            raise ReceiptError("Parity and lineage converted artifacts disagree.")
        if not case_entries:
            raise ReceiptError("Every parity report must contain parity cases.")
        case_objects = [
            _require_object(case, field="parity report case") for case in case_entries
        ]
        if {case.get("resolution") for case in case_objects} != {1008, 672, 504}:
            raise ReceiptError(
                "Every parity profile must cover 1008, 672, and 504 resolutions."
            )
        for case in case_objects:
            _validate_parity_case(case, thresholds=thresholds)
        _validate_performance(
            _require_object(
                report.get("performance"), field="parity report performance"
            ),
            profile=_require_string(
                report.get("case_profile"),
                field="parity report case_profile",
            ),
        )

    cases: list[JsonObject] = []
    for report, path in zip(reports, parity_report_paths, strict=True):
        report_cases = _require_list(report.get("cases"), field="parity report cases")
        case_profile = _require_string(
            report.get("case_profile"), field="parity report case_profile"
        )
        image = _require_object(report.get("image"), field="parity report image")
        for raw_case in report_cases:
            case = _require_object(raw_case, field="parity report case")
            cases.append(
                {
                    **case,
                    "profile": case_profile,
                    "image_sha256": _require_string(
                        image.get("sha256"),
                        field="parity report image sha256",
                    ),
                    "report": _repo_relative(path, root=evidence_root),
                }
            )
    performance_runs: list[JsonObject] = []
    for report in reports:
        performance = _require_object(
            report.get("performance"), field="parity report performance"
        )
        image = _require_object(report.get("image"), field="parity report image")
        performance_runs.append(
            {
                "profile": _require_string(
                    report.get("case_profile"),
                    field="parity report case_profile",
                ),
                "image_sha256": _require_string(
                    image.get("sha256"),
                    field="parity report image sha256",
                ),
                **performance,
            }
        )

    measurement_boundaries = {
        _require_string(
            run.get("measurement_boundary"),
            field="parity performance measurement_boundary",
        )
        for run in performance_runs
    }
    lineage_reproduction = _require_object(
        lineage.get("reproduction"),
        field="lineage reproduction",
    )
    peak_memory_values = [
        _require_nonnegative_int(
            run.get("peak_active_memory_bytes"),
            field="parity performance peak_active_memory_bytes",
        )
        for run in performance_runs
    ]
    if len(measurement_boundaries) != 1:
        raise ReceiptError("Parity reports disagree on the performance boundary.")
    return {
        "checkpoint": {
            "official_repo": source["repo"],
            "official_revision": source["revision"],
            "official_code_revision": official_code_revisions.pop(),
            "official_sha256": source["checkpoint_sha256"],
            "converted_repo": published["repo"],
            "artifact_revision": published["revision"],
            "converted_sha256": published["checkpoint_sha256"],
            "conversion_manifest_sha256": _require_string(
                lineage_reproduction.get("manifest_sha256"),
                field="lineage reproduction manifest_sha256",
            ),
            "lineage_report": _repo_relative(
                lineage_report_path,
                root=evidence_root,
            ),
            "lineage_report_sha256": _sha256(lineage_report_path),
            "architecture": "sam3-image",
        },
        "parity": {
            "status": "passed",
            "mode": "official-torch-vs-mlx",
            "calibration_profile": "example",
            "validation_profile": "holdout",
            "thresholds": thresholds,
            "cases": cases,
            "reports": [
                {
                    "path": _repo_relative(path, root=evidence_root),
                    "sha256": _sha256(path),
                }
                for path in parity_report_paths
            ],
        },
        "performance": {
            "status": "passed",
            "runs": performance_runs,
            "peak_active_memory_bytes": max(peak_memory_values),
            "measurement_boundary": measurement_boundaries.pop(),
        },
    }


def promote_measured_receipt(
    receipt: JsonObject,
    *,
    parity_report_paths: list[Path],
    lineage_report_path: Path,
) -> JsonObject:
    """Promote a passing test receipt using validated measured evidence."""

    projection = _measured_evidence_projection(
        parity_report_paths=parity_report_paths,
        lineage_report_path=lineage_report_path,
        evidence_root=REPO_ROOT,
    )
    tests = _require_object(receipt.get("tests"), field="Receipt tests")
    receipt["status"] = "passed" if tests.get("passed") is True else "failed"
    receipt.update(projection)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=REPO_ROOT / "parity" / "receipts" / "latest.json",
        help="Path to the parity receipt JSON.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate a receipt (runs pytest) instead of only validating.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional local converted checkpoint path for parity work.",
    )
    parser.add_argument(
        "--parity-report",
        action="append",
        type=Path,
        default=[],
        help="Measured passed parity report; provide example and holdout reports.",
    )
    parser.add_argument(
        "--lineage-report",
        type=Path,
        help="Passed checkpoint semantic-lineage report.",
    )
    parser.add_argument(
        "--pytest-python",
        type=Path,
        help="Python executable for the release test suite (for an isolated oracle env).",
    )
    parser.add_argument(
        "--allow-not-passed",
        action="store_true",
        help="Accept receipts whose top-level status is not 'passed'.",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        default=None,
        help="Optional pytest args when generating (e.g. tests/checkpoint).",
    )
    args = parser.parse_args()

    if args.generate:
        # Prefer uv-managed pytest so the generator works when the active
        # interpreter does not have dev dependencies installed.
        if args.pytest_python is None:
            pytest_command = ["uv", "run", "pytest"]
            test_python_command = ["uv", "run", "python"]
        else:
            pytest_command = [str(args.pytest_python), "-m", "pytest"]
            test_python_command = [str(args.pytest_python)]
        pytest_command.extend(
            [
                "-q",
                "-p",
                "pytest_reportlog",
                "--ignore=third_party",
            ]
        )
        if args.pytest_args:
            pytest_command.extend(list(args.pytest_args))
        else:
            pytest_command.append("tests")
        receipt = generate_stub_receipt(
            pytest_command=pytest_command,
            test_python_command=test_python_command,
            checkpoint_path=args.checkpoint,
        )
        if args.parity_report or args.lineage_report:
            if not args.parity_report or args.lineage_report is None:
                parser.error(
                    "--parity-report and --lineage-report must be provided together."
                )
            try:
                receipt = promote_measured_receipt(
                    receipt,
                    parity_report_paths=args.parity_report,
                    lineage_report_path=args.lineage_report,
                )
            except ReceiptError as exc:
                raise SystemExit(str(exc)) from exc
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"wrote": str(args.receipt), "status": receipt["status"]}))
        validate_receipt(
            receipt,
            expected_commit=_require_string(
                receipt.get("git_commit"),
                field="Receipt git_commit",
            ),
            require_passed=bool(args.parity_report),
            evidence_root=REPO_ROOT,
        )
        expected_status = "passed" if args.parity_report else "blocked"
        if receipt["status"] != expected_status:
            raise SystemExit(1)
        return

    if not args.receipt.exists():
        raise SystemExit(
            f"Receipt not found: {args.receipt}. Generate one with --generate "
            "on a capable host, or pass --receipt."
        )
    receipt = _load_json_object(args.receipt, label="runtime release receipt")
    if not args.allow_not_passed and not _git_is_clean():
        raise SystemExit(
            "Release receipt validation requires a clean Git worktree. Commit "
            "the candidate and regenerate its commit-bound receipt first."
        )
    receipt_commit = receipt.get("git_commit")
    if not isinstance(receipt_commit, str):
        raise SystemExit("Receipt git_commit must be a commit SHA string.")
    try:
        expected_commit = _validate_receipt_git_binding(receipt_commit)
        validate_receipt(
            receipt,
            expected_commit=expected_commit,
            require_passed=not args.allow_not_passed,
            evidence_root=REPO_ROOT,
        )
    except ReceiptError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "receipt": str(args.receipt),
                "status": receipt.get("status"),
                "parity_status": _require_object(
                    receipt.get("parity"),
                    field="Receipt parity",
                ).get("status"),
                "git_commit": receipt.get("git_commit"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
