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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

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
    try:
        from importlib.metadata import version

        return version("sam3-mlx")
    except Exception:
        import sam3_mlx

        return sam3_mlx.__version__


def _test_environment(python_command: list[str]) -> dict[str, str | None]:
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
    metadata = json.loads(completed.stdout)
    required = {"python_version", "mlx_version", "platform", "machine"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ReceiptError(
            f"Test interpreter returned invalid environment metadata: {metadata!r}."
        )
    return metadata


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


def _evidence_path(value: Any, *, root: Path) -> Path:
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


def _require_finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptError(f"{field} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ReceiptError(f"{field} must be a finite number.")
    return number


def _validate_parity_case(
    case: dict[str, Any],
    *,
    thresholds: dict[str, Any],
) -> None:
    name = case.get("name", "<unnamed>")
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

    matches = case.get("matches")
    if not isinstance(matches, list) or len(matches) != official_count:
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
    for match in matches:
        if not isinstance(match, dict):
            raise ReceiptError(f"Parity case {name!r} contains an invalid match.")
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


def _validate_performance(performance: dict[str, Any], *, profile: str) -> None:
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
    latencies = performance.get("latency_by_resolution_s")
    if not isinstance(latencies, dict) or set(latencies) != {"1008", "672", "504"}:
        raise ReceiptError(
            f"Parity profile {profile!r} must measure 1008, 672, and 504."
        )
    for resolution, summary in latencies.items():
        samples = summary.get("samples") if isinstance(summary, dict) else None
        if not isinstance(samples, list) or len(samples) != repetitions:
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
    receipt: dict[str, Any],
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

    checkpoint = receipt.get("checkpoint") or {}
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

    tests = receipt.get("tests") or {}
    missing_tests = sorted(REQUIRED_TESTS_FIELDS - set(tests))
    if missing_tests:
        raise ReceiptError(f"Receipt tests missing fields: {missing_tests}")
    if not isinstance(tests.get("skip_details"), list):
        raise ReceiptError("Receipt tests.skip_details must be a list.")
    for item in tests["skip_details"]:
        required_skip_fields = {"nodeid", "reason", "owner", "disposition"}
        if not isinstance(item, dict) or not required_skip_fields.issubset(item):
            raise ReceiptError(
                "Each skip_details entry must include nodeid, reason, owner, "
                "and disposition."
            )
    for field in ("skipped", "deselected"):
        value = tests.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReceiptError(f"Receipt tests.{field} must be a non-negative integer.")
    recorded_skips = tests["skipped"] + tests["deselected"]
    if recorded_skips != len(tests["skip_details"]):
        raise ReceiptError(
            "Receipt skip/deselection counts must match skip_details entries."
        )
    counts = tests.get("counts")
    if not isinstance(counts, dict):
        raise ReceiptError("Receipt tests.counts must be an object.")
    required_counts = {"call_passed", "call_failed", "call_skipped"}
    if set(counts) != required_counts or any(
        isinstance(counts[field], bool)
        or not isinstance(counts[field], int)
        or counts[field] < 0
        for field in required_counts
    ):
        raise ReceiptError("Receipt tests.counts must contain non-negative integers.")
    exit_code = tests.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise ReceiptError("Receipt tests.exit_code must be an integer.")
    if (
        tests.get("passed") is not (exit_code == 0)
        or tests.get("failed") is not (exit_code != 0)
        or (exit_code == 0 and counts["call_failed"] != 0)
        or (exit_code != 0 and counts["call_failed"] < 1)
        or counts["call_skipped"] != tests.get("skipped")
    ):
        raise ReceiptError(
            "Receipt test booleans, exit code, and outcome counts are inconsistent."
        )

    parity = receipt.get("parity") or {}
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
        thresholds = parity.get("thresholds") or {}
        required_thresholds = {
            "mask_iou_min",
            "mask_iou_mean_min",
            "box_l_inf_max",
            "score_abs_max",
        }
        if any(thresholds.get(field) is None for field in required_thresholds):
            raise ReceiptError("Release receipt parity thresholds must be measured.")
        performance = receipt.get("performance") or {}
        if performance.get("status") != "passed":
            raise ReceiptError("Release receipt performance.status must be 'passed'.")
        if not receipt.get("mlx_version"):
            raise ReceiptError("Release receipt mlx_version must be recorded.")
        if counts["call_passed"] < 1 or counts["call_failed"] != 0:
            raise ReceiptError("Release receipt must record a passing test suite.")
        if evidence_root is None:
            raise ReceiptError(
                "Release receipt validation requires an evidence repository root."
            )
        reports = parity.get("reports")
        if not isinstance(reports, list) or len(reports) != 2:
            raise ReceiptError(
                "Release receipt must reference exactly two parity reports."
            )
        report_paths = []
        for report in reports:
            if not isinstance(report, dict) or set(report) != {"path", "sha256"}:
                raise ReceiptError(
                    "Parity report references must contain path and sha256."
                )
            report_path = _evidence_path(report["path"], root=evidence_root)
            if _sha256(report_path) != report["sha256"]:
                raise ReceiptError(
                    f"Parity report digest mismatch: {report['path']!r}."
                )
            report_paths.append(report_path)
        lineage_path = _evidence_path(
            checkpoint.get("lineage_report"),
            root=evidence_root,
        )
        if _sha256(lineage_path) != checkpoint.get("lineage_report_sha256"):
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


def _parse_pytest_report_log(report_path: Path) -> dict[str, Any]:
    """Parse a pytest --report-log JSONL file into skip/deselection details."""
    passed = failed = skipped = 0
    skip_details: list[dict[str, str]] = []
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
        event = json.loads(line)
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
            if isinstance(longrepr, (list, tuple)) and longrepr:
                reason = str(longrepr[-1])
            elif longrepr is not None:
                reason = str(longrepr)
            skip_details.append(
                {
                    "nodeid": nodeid,
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
) -> dict[str, Any]:
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
        report_stats = {
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
) -> dict[str, dict[str, Any]]:
    """Load measured reports and reconstruct the canonical receipt projection."""
    reports = [json.loads(path.read_text()) for path in parity_report_paths]
    lineage = json.loads(lineage_report_path.read_text())
    if not reports or any(report.get("status") != "passed" for report in reports):
        raise ReceiptError("Every parity report must have status='passed'.")
    profiles = {report.get("case_profile") for report in reports}
    if profiles != {"example", "holdout"}:
        raise ReceiptError(
            "Release parity requires both example and independent holdout profiles."
        )
    if lineage.get("status") != "passed" or not (
        (lineage.get("comparison") or {}).get("semantic_match")
    ):
        raise ReceiptError("Checkpoint lineage report must be a semantic pass.")

    source = lineage["source"]
    published = lineage["published_artifact"]
    official_code_revisions = {
        report["official_code"]["revision"] for report in reports
    }
    if len(official_code_revisions) != 1:
        raise ReceiptError("Parity reports disagree on official code revision.")
    threshold_contracts = {
        json.dumps(report["thresholds"], sort_keys=True) for report in reports
    }
    if len(threshold_contracts) != 1:
        raise ReceiptError("Parity reports disagree on threshold contract.")
    thresholds = reports[0]["thresholds"]
    required_thresholds = set(RELEASE_THRESHOLDS)
    if set(thresholds) != required_thresholds:
        raise ReceiptError("Parity reports contain an invalid threshold contract.")
    thresholds = {
        field: _require_finite_number(
            thresholds[field],
            field=f"Parity threshold {field}",
        )
        for field in sorted(required_thresholds)
    }
    if thresholds != RELEASE_THRESHOLDS:
        raise ReceiptError("Parity reports do not use the fixed release thresholds.")

    for report in reports:
        if report["official_checkpoint"]["revision"] != source["revision"]:
            raise ReceiptError("Parity and lineage official revisions disagree.")
        if report["official_checkpoint"]["sha256"] != source["checkpoint_sha256"]:
            raise ReceiptError(
                "Parity and lineage official checkpoint hashes disagree."
            )
        if (
            report["converted_checkpoint"]["revision"] != published["revision"]
            or report["converted_checkpoint"]["sha256"]
            != published["checkpoint_sha256"]
        ):
            raise ReceiptError("Parity and lineage converted artifacts disagree.")
        cases = report.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ReceiptError("Every parity report must contain parity cases.")
        if {case.get("resolution") for case in cases} != {1008, 672, 504}:
            raise ReceiptError(
                "Every parity profile must cover 1008, 672, and 504 resolutions."
            )
        for case in cases:
            _validate_parity_case(case, thresholds=thresholds)
        _validate_performance(
            report.get("performance") or {},
            profile=report["case_profile"],
        )

    cases = []
    for report, path in zip(reports, parity_report_paths, strict=True):
        for case in report["cases"]:
            cases.append(
                {
                    **case,
                    "profile": report["case_profile"],
                    "image_sha256": report["image"]["sha256"],
                    "report": _repo_relative(path, root=evidence_root),
                }
            )
    performance_runs = [
        {
            "profile": report["case_profile"],
            "image_sha256": report["image"]["sha256"],
            **report["performance"],
        }
        for report in reports
    ]

    measurement_boundaries = {run["measurement_boundary"] for run in performance_runs}
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
            "conversion_manifest_sha256": lineage["reproduction"]["manifest_sha256"],
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
            "peak_active_memory_bytes": max(
                run["peak_active_memory_bytes"] for run in performance_runs
            ),
            "measurement_boundary": measurement_boundaries.pop(),
        },
    }


def promote_measured_receipt(
    receipt: dict[str, Any],
    *,
    parity_report_paths: list[Path],
    lineage_report_path: Path,
) -> dict[str, Any]:
    """Promote a passing test receipt using validated measured evidence."""

    projection = _measured_evidence_projection(
        parity_report_paths=parity_report_paths,
        lineage_report_path=lineage_report_path,
        evidence_root=REPO_ROOT,
    )
    receipt["status"] = "passed" if receipt["tests"]["passed"] else "failed"
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
            expected_commit=receipt["git_commit"],
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
    receipt = json.loads(args.receipt.read_text())
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
                "parity_status": (receipt.get("parity") or {}).get("status"),
                "git_commit": receipt.get("git_commit"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
