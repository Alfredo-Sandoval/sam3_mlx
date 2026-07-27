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
import os
import platform
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
    "passed",
    "failed",
    "skipped",
    "deselected",
    "skip_details",
}
ATTESTATION_PATH_PREFIXES = ("parity/receipts/", "parity/manifests/")


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


def _mlx_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("mlx")
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _redact_repo_path(text: str) -> str:
    return text.replace(str(REPO_ROOT), "<repo>")


def validate_receipt(
    receipt: dict[str, Any],
    *,
    expected_commit: str | None = None,
    require_passed: bool = True,
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
    recorded_skips = int(tests.get("skipped", 0)) + int(tests.get("deselected", 0))
    if recorded_skips != len(tests["skip_details"]):
        raise ReceiptError(
            "Receipt skip/deselection counts must match skip_details entries."
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
    checkpoint_path: str | None,
) -> dict[str, Any]:
    """Create a schema-complete receipt without claiming e2e parity.

    Full upstream comparison should replace parity.status with measured metrics
    once a capable host and licensed checkpoints are available.
    """
    from sam3_mlx.convert import DEFAULT_MLX_CHECKPOINT

    git_commit = _git_commit()
    package_version = _package_version()

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
        "python_version": platform.python_version(),
        "mlx_version": _mlx_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
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


def promote_measured_receipt(
    receipt: dict[str, Any],
    *,
    parity_report_paths: list[Path],
    lineage_report_path: Path,
) -> dict[str, Any]:
    """Promote a passing test receipt using validated measured evidence."""

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
        if not report.get("cases") or any(
            case.get("status") != "passed" for case in report["cases"]
        ):
            raise ReceiptError("Every recorded parity case must pass.")
        if (report.get("performance") or {}).get("status") != "passed":
            raise ReceiptError("Every parity report must contain passed performance.")

    cases = []
    for report, path in zip(reports, parity_report_paths, strict=True):
        for case in report["cases"]:
            cases.append(
                {
                    **case,
                    "profile": report["case_profile"],
                    "image_sha256": report["image"]["sha256"],
                    "report": _repo_relative(path),
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

    receipt["status"] = "passed" if receipt["tests"]["passed"] else "failed"
    receipt["checkpoint"] = {
        "official_repo": source["repo"],
        "official_revision": source["revision"],
        "official_code_revision": official_code_revisions.pop(),
        "official_sha256": source["checkpoint_sha256"],
        "converted_repo": published["repo"],
        "artifact_revision": published["revision"],
        "converted_sha256": published["checkpoint_sha256"],
        "conversion_manifest_sha256": lineage["reproduction"]["manifest_sha256"],
        "lineage_report": _repo_relative(lineage_report_path),
        "lineage_report_sha256": _sha256(lineage_report_path),
        "architecture": "sam3-image",
    }
    receipt["parity"] = {
        "status": "passed",
        "mode": "official-torch-vs-mlx",
        "calibration_profile": "example",
        "validation_profile": "holdout",
        "thresholds": thresholds,
        "cases": cases,
        "reports": [
            {
                "path": _repo_relative(path),
                "sha256": _sha256(path),
            }
            for path in parity_report_paths
        ],
    }
    receipt["performance"] = {
        "status": "passed",
        "runs": performance_runs,
        "peak_active_memory_bytes": max(
            run["peak_active_memory_bytes"] for run in performance_runs
        ),
        "measurement_boundary": performance_runs[0]["measurement_boundary"],
    }
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
        else:
            pytest_command = [str(args.pytest_python), "-m", "pytest"]
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
