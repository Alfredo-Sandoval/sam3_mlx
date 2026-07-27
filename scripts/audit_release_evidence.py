#!/usr/bin/env python3
"""Independently replay and hard-pin every checked-in 0.1.2 release claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.parity_evidence import compare_case, load_evidence_bundle  # noqa: E402
from sam3_mlx.release_contract import (  # noqa: E402
    CHECKPOINT_TENSOR_COUNT,
    COMPARISON_ALGORITHM,
    EXPECTED_CASE_NAMES,
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    MLX_CHECKPOINT_SHA256,
    OFFICIAL_CHECKPOINT_REPO,
    OFFICIAL_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_CODE_REPO,
    OFFICIAL_CODE_REVISION,
    ORACLE_CPU_ADAPTERS,
    ORACLE_PRECISION,
    PACKAGE_VERSION,
    RELEASE_CONFIDENCE_THRESHOLD,
    RELEASE_IMAGES,
    RELEASE_RESOLUTIONS,
    RELEASE_THRESHOLDS,
    REPORT_SCHEMA_VERSION,
    build_oracle_bindings,
    canonical_json_sha256,
    sha256_path,
    validate_exact_mapping,
)

HARDENED_ORACLE = REPO_ROOT / "scripts" / "run_upstream_image_oracle_hardened.py"
HARDENED_LINEAGE = REPO_ROOT / "scripts" / "validate_checkpoint_lineage_hardened.py"
CONVERT_MODULE = REPO_ROOT / "sam3_mlx" / "convert.py"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class EvidenceAuditError(ValueError):
    """Raised when checked-in evidence does not satisfy the frozen contract."""


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceAuditError(f"Could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceAuditError(f"{label} must be a JSON object: {path}.")
    return value


def _repo_file(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceAuditError(f"{label} path must be a non-empty string.")
    path = Path(value)
    if path.is_absolute():
        raise EvidenceAuditError(f"{label} path must be repository-relative: {value!r}.")
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise EvidenceAuditError(f"{label} path escapes the repository: {value!r}.") from exc
    if not resolved.is_file():
        raise EvidenceAuditError(f"{label} file does not exist: {value!r}.")
    return resolved


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise EvidenceAuditError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_exact(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise EvidenceAuditError(
            f"{label} does not match the release contract: "
            f"observed={observed!r}, expected={expected!r}."
        )


def _case_specs(cases: Any, *, profile: str) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        raise EvidenceAuditError(f"{profile} report cases must be a list.")
    names = tuple(case.get("name") for case in cases if isinstance(case, dict))
    _require_exact(names, EXPECTED_CASE_NAMES[profile], label=f"{profile} case order")
    specs = []
    for case in cases:
        if not isinstance(case, dict):
            raise EvidenceAuditError(f"{profile} report contains a non-object case.")
        specs.append(
            {
                "name": case.get("name"),
                "resolution": case.get("resolution"),
                "prompt": case.get("prompt"),
                "geometric_prompts": case.get("geometric_prompts"),
            }
        )
    return specs


def _case_spec_sha256(specs: list[dict[str, Any]]) -> str:
    payload = (json.dumps(specs, indent=2) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_case_semantics(cases: list[dict[str, Any]], *, profile: str) -> None:
    by_name = {case["name"]: case for case in cases}
    for case in cases:
        _require_exact(case.get("status"), "passed", label=f"case {case['name']} status")
        _require_exact(
            case.get("detection_count_match"),
            True,
            label=f"case {case['name']} count parity",
        )
        _require_exact(
            case.get("official_detection_count"),
            case.get("mlx_detection_count"),
            label=f"case {case['name']} detection counts",
        )

    nonsense = by_name["text_nonsense_1008"]
    _require_exact(
        nonsense.get("official_detection_count"),
        0,
        label=f"{profile} negative-control detection count",
    )
    _require_exact(nonsense.get("matches"), [], label=f"{profile} negative matches")

    if profile == "example":
        _require_exact(
            by_name["positive_box_1008"]["geometric_prompts"][0]["label"],
            True,
            label="positive box label",
        )
        labels = [
            prompt["label"]
            for prompt in by_name["positive_negative_box_1008"]["geometric_prompts"]
        ]
        _require_exact(labels, [True, False], label="positive/negative box labels")
    for case in cases:
        if case["name"].startswith("text_"):
            if not isinstance(case.get("prompt"), str) or not case["prompt"]:
                raise EvidenceAuditError(f"Text case {case['name']} has no prompt.")
            _require_exact(
                case.get("geometric_prompts"),
                [],
                label=f"text case {case['name']} geometric prompts",
            )


def _validate_oracle(
    oracle: Any,
    *,
    report: dict[str, Any],
    specs: list[dict[str, Any]],
) -> None:
    if not isinstance(oracle, dict):
        raise EvidenceAuditError("Report oracle metadata must be an object.")
    expected_bindings = build_oracle_bindings(
        image_sha256=report["image"]["sha256"],
        case_spec_sha256=_case_spec_sha256(specs),
        confidence_threshold=report["confidence_threshold"],
        oracle_runner_sha256=sha256_path(HARDENED_ORACLE),
    )
    bindings = oracle.get("bindings")
    if not isinstance(bindings, dict):
        raise EvidenceAuditError("Oracle metadata is missing complete bindings.")
    try:
        validate_exact_mapping(bindings, expected_bindings, label="oracle bindings")
    except ValueError as exc:
        raise EvidenceAuditError(str(exc)) from exc
    _require_exact(
        oracle.get("cache_key"),
        canonical_json_sha256(expected_bindings),
        label="oracle cache key",
    )
    duplicate_fields = {
        "official_code": expected_bindings["official_code"],
        "official_checkpoint": expected_bindings["official_checkpoint"],
        "image_sha256": expected_bindings["image_sha256"],
        "case_spec_sha256": expected_bindings["case_spec_sha256"],
        "confidence_threshold": expected_bindings["confidence_threshold"],
        "precision": ORACLE_PRECISION,
        "cpu_adapters": list(ORACLE_CPU_ADAPTERS),
        "oracle_runner_sha256": expected_bindings["oracle_runner_sha256"],
        "release_contract_sha256": expected_bindings["release_contract_sha256"],
    }
    for field, expected in duplicate_fields.items():
        _require_exact(oracle.get(field), expected, label=f"oracle {field}")

    oracle_cases = oracle.get("cases")
    if not isinstance(oracle_cases, list) or len(oracle_cases) != len(specs):
        raise EvidenceAuditError("Oracle case metadata count is invalid.")
    for oracle_case, spec, report_case in zip(
        oracle_cases, specs, report["cases"], strict=True
    ):
        _require_exact(oracle_case.get("name"), spec["name"], label="oracle case name")
        _require_exact(
            oracle_case.get("resolution"),
            spec["resolution"],
            label=f"oracle {spec['name']} resolution",
        )
        _require_exact(
            oracle_case.get("detection_count"),
            report_case["official_detection_count"],
            label=f"oracle {spec['name']} detection count",
        )
        for timing_field in ("image_latency_s", "prompt_latency_s"):
            timing = oracle_case.get(timing_field)
            if (
                isinstance(timing, bool)
                or not isinstance(timing, (int, float))
                or not math.isfinite(float(timing))
                or timing < 0
            ):
                raise EvidenceAuditError(
                    f"Oracle {spec['name']} {timing_field} is invalid."
                )
    environment = oracle.get("environment")
    if not isinstance(environment, dict):
        raise EvidenceAuditError("Oracle environment metadata is missing.")
    _require_exact(environment.get("machine"), "arm64", label="oracle machine")


def _validate_report(
    report_path: Path,
    *,
    expected_profile: str,
) -> dict[str, Any]:
    report = _load_json(report_path, label=f"{expected_profile} parity report")
    _require_exact(
        report.get("schema_version"),
        REPORT_SCHEMA_VERSION,
        label=f"{expected_profile} report schema",
    )
    _require_exact(report.get("status"), "passed", label=f"{expected_profile} status")
    _require_exact(
        report.get("comparison_algorithm"),
        COMPARISON_ALGORITHM,
        label=f"{expected_profile} comparison algorithm",
    )
    _require_exact(
        report.get("case_profile"),
        expected_profile,
        label=f"{expected_profile} profile",
    )
    _require_exact(
        report.get("official_code"),
        {"repo": OFFICIAL_CODE_REPO, "revision": OFFICIAL_CODE_REVISION},
        label=f"{expected_profile} official code",
    )
    _require_exact(
        report.get("official_checkpoint"),
        {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
            "sha256": OFFICIAL_CHECKPOINT_SHA256,
        },
        label=f"{expected_profile} official checkpoint",
    )
    _require_exact(
        report.get("converted_checkpoint"),
        {
            "repo": MLX_CHECKPOINT_REPO,
            "revision": MLX_CHECKPOINT_REVISION,
            "sha256": MLX_CHECKPOINT_SHA256,
        },
        label=f"{expected_profile} MLX checkpoint",
    )
    _require_exact(
        report.get("image"),
        RELEASE_IMAGES[expected_profile],
        label=f"{expected_profile} image",
    )
    _require_exact(
        report.get("confidence_threshold"),
        RELEASE_CONFIDENCE_THRESHOLD,
        label=f"{expected_profile} confidence threshold",
    )
    _require_exact(
        report.get("thresholds"),
        RELEASE_THRESHOLDS,
        label=f"{expected_profile} thresholds",
    )

    cases = report.get("cases")
    specs = _case_specs(cases, profile=expected_profile)
    _validate_case_semantics(cases, profile=expected_profile)
    _validate_oracle(report.get("oracle"), report=report, specs=specs)

    raw_evidence = report.get("raw_evidence")
    if not isinstance(raw_evidence, dict) or set(raw_evidence) != {"path", "sha256"}:
        raise EvidenceAuditError(
            f"{expected_profile} report must reference one raw evidence bundle."
        )
    evidence_path = _repo_file(
        raw_evidence["path"], label=f"{expected_profile} raw evidence"
    )
    expected_digest = _require_sha256(
        raw_evidence["sha256"], label=f"{expected_profile} raw evidence digest"
    )
    _require_exact(
        sha256_path(evidence_path),
        expected_digest,
        label=f"{expected_profile} raw evidence digest",
    )
    try:
        metadata, official_outputs, mlx_outputs = load_evidence_bundle(evidence_path)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise EvidenceAuditError(
            f"Could not replay {expected_profile} raw evidence: {exc}"
        ) from exc

    expected_metadata = {
        "profile": expected_profile,
        "report_path": _repo_relative(report_path),
        "image": report["image"],
        "case_specs": specs,
        "oracle_cache_key": report["oracle"]["cache_key"],
        "official_code": report["official_code"],
        "official_checkpoint": report["official_checkpoint"],
        "converted_checkpoint": report["converted_checkpoint"],
        "confidence_threshold": RELEASE_CONFIDENCE_THRESHOLD,
        "thresholds": RELEASE_THRESHOLDS,
    }
    for field, expected in expected_metadata.items():
        _require_exact(
            metadata.get(field),
            expected,
            label=f"{expected_profile} evidence metadata {field}",
        )
    _require_exact(
        metadata.get("case_count"),
        len(specs),
        label=f"{expected_profile} evidence case count",
    )
    _require_exact(
        metadata.get("comparison_algorithm"),
        COMPARISON_ALGORITHM,
        label=f"{expected_profile} evidence comparison algorithm",
    )
    if len(official_outputs) != len(specs) or len(mlx_outputs) != len(specs):
        raise EvidenceAuditError(f"{expected_profile} evidence array count is invalid.")

    replayed_cases = [
        compare_case(spec, official, mlx, thresholds=RELEASE_THRESHOLDS)
        for spec, official, mlx in zip(
            specs, official_outputs, mlx_outputs, strict=True
        )
    ]
    _require_exact(
        replayed_cases,
        cases,
        label=f"{expected_profile} replayed parity cases",
    )

    host = report.get("host")
    if not isinstance(host, dict):
        raise EvidenceAuditError(f"{expected_profile} host metadata is missing.")
    _require_exact(host.get("machine"), "arm64", label=f"{expected_profile} host")
    performance = report.get("performance")
    if not isinstance(performance, dict) or performance.get("status") != "passed":
        raise EvidenceAuditError(f"{expected_profile} performance did not pass.")
    resolutions = performance.get("latency_by_resolution_s")
    _require_exact(
        set(resolutions or {}),
        {str(value) for value in RELEASE_RESOLUTIONS},
        label=f"{expected_profile} performance resolutions",
    )
    return report


def _validate_lineage(receipt: dict[str, Any]) -> dict[str, Any]:
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise EvidenceAuditError("Receipt checkpoint section is missing.")
    lineage_path = _repo_file(
        checkpoint.get("lineage_report"), label="checkpoint lineage report"
    )
    recorded_digest = _require_sha256(
        checkpoint.get("lineage_report_sha256"), label="lineage report digest"
    )
    _require_exact(
        sha256_path(lineage_path), recorded_digest, label="lineage report digest"
    )
    lineage = _load_json(lineage_path, label="checkpoint lineage report")
    _require_exact(lineage.get("schema_version"), 2, label="lineage schema")
    _require_exact(lineage.get("status"), "passed", label="lineage status")
    _require_exact(
        lineage.get("lineage_runner_sha256"),
        sha256_path(HARDENED_LINEAGE),
        label="lineage runner digest",
    )
    _require_exact(
        lineage.get("converter_module_sha256"),
        sha256_path(CONVERT_MODULE),
        label="converter module digest",
    )
    _require_exact(
        lineage.get("source"),
        {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
            "checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        },
        label="lineage source",
    )
    _require_exact(
        lineage.get("published_artifact"),
        {
            "repo": MLX_CHECKPOINT_REPO,
            "revision": MLX_CHECKPOINT_REVISION,
            "checkpoint_sha256": MLX_CHECKPOINT_SHA256,
        },
        label="lineage published artifact",
    )

    reproduction = lineage.get("reproduction")
    if not isinstance(reproduction, dict):
        raise EvidenceAuditError("Lineage reproduction section is missing.")
    _require_exact(
        reproduction.get("converter_version"),
        PACKAGE_VERSION,
        label="lineage converter version",
    )
    manifest = reproduction.get("manifest")
    if not isinstance(manifest, dict):
        raise EvidenceAuditError("Lineage must embed the reproduction manifest.")
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    _require_exact(
        reproduction.get("manifest_sha256"),
        manifest_digest,
        label="embedded reproduction manifest digest",
    )
    _require_exact(
        checkpoint.get("conversion_manifest_sha256"),
        manifest_digest,
        label="receipt conversion manifest digest",
    )
    required_manifest = {
        "architecture": "sam3-image",
        "source_repo": OFFICIAL_CHECKPOINT_REPO,
        "source_revision": OFFICIAL_CHECKPOINT_REVISION,
        "source_checkpoint_sha256": OFFICIAL_CHECKPOINT_SHA256,
        "output_sha256": reproduction.get("checkpoint_sha256"),
        "mapped_count": CHECKPOINT_TENSOR_COUNT,
        "unmapped_keys": [],
        "converter_version": PACKAGE_VERSION,
    }
    for field, expected in required_manifest.items():
        _require_exact(manifest.get(field), expected, label=f"manifest {field}")

    comparison = lineage.get("comparison")
    if not isinstance(comparison, dict):
        raise EvidenceAuditError("Lineage comparison section is missing.")
    required_comparison = {
        "published_tensor_count": CHECKPOINT_TENSOR_COUNT,
        "reproduced_tensor_count": CHECKPOINT_TENSOR_COUNT,
        "exact_tensor_count": CHECKPOINT_TENSOR_COUNT,
        "missing_keys": [],
        "extra_keys": [],
        "shape_mismatches": [],
        "dtype_mismatches": [],
        "value_mismatches": [],
        "semantic_match": True,
        "comparison_layout": "canonical MLX image-runtime layout",
    }
    for field, expected in required_comparison.items():
        _require_exact(comparison.get(field), expected, label=f"lineage {field}")
    return lineage


def audit_release_evidence(receipt_path: Path) -> dict[str, Any]:
    receipt = _load_json(receipt_path, label="runtime release receipt")
    _require_exact(receipt.get("schema_version"), 1, label="receipt schema")
    _require_exact(receipt.get("status"), "passed", label="receipt status")
    _require_exact(receipt.get("package_version"), PACKAGE_VERSION, label="package version")
    git_commit = receipt.get("git_commit")
    if not isinstance(git_commit, str) or not _COMMIT_PATTERN.fullmatch(git_commit):
        raise EvidenceAuditError("Receipt git_commit must be a lowercase commit SHA.")
    _require_exact(receipt.get("machine"), "arm64", label="receipt machine")
    if not receipt.get("mlx_version"):
        raise EvidenceAuditError("Receipt must record the MLX version.")

    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise EvidenceAuditError("Receipt checkpoint section is missing.")
    expected_checkpoint = {
        "architecture": "sam3-image",
        "artifact_revision": MLX_CHECKPOINT_REVISION,
        "converted_repo": MLX_CHECKPOINT_REPO,
        "converted_sha256": MLX_CHECKPOINT_SHA256,
        "official_code_revision": OFFICIAL_CODE_REVISION,
        "official_repo": OFFICIAL_CHECKPOINT_REPO,
        "official_revision": OFFICIAL_CHECKPOINT_REVISION,
        "official_sha256": OFFICIAL_CHECKPOINT_SHA256,
    }
    for field, expected in expected_checkpoint.items():
        _require_exact(checkpoint.get(field), expected, label=f"receipt checkpoint {field}")

    tests = receipt.get("tests")
    if not isinstance(tests, dict):
        raise EvidenceAuditError("Receipt tests section is missing.")
    required_test_values = {
        "exit_code": 0,
        "passed": True,
        "failed": False,
        "skipped": 0,
        "deselected": 0,
        "skip_details": [],
    }
    for field, expected in required_test_values.items():
        _require_exact(tests.get(field), expected, label=f"receipt tests {field}")
    counts = tests.get("counts")
    if not isinstance(counts, dict) or counts.get("call_failed") != 0:
        raise EvidenceAuditError("Receipt test outcome counts are invalid.")
    if not isinstance(counts.get("call_passed"), int) or counts["call_passed"] < 1:
        raise EvidenceAuditError("Receipt must record at least one passing test.")

    parity = receipt.get("parity")
    if not isinstance(parity, dict):
        raise EvidenceAuditError("Receipt parity section is missing.")
    _require_exact(parity.get("status"), "passed", label="receipt parity status")
    _require_exact(parity.get("mode"), "official-torch-vs-mlx", label="parity mode")
    _require_exact(parity.get("thresholds"), RELEASE_THRESHOLDS, label="receipt thresholds")
    _require_exact(parity.get("calibration_profile"), "example", label="calibration profile")
    _require_exact(parity.get("validation_profile"), "holdout", label="validation profile")

    report_refs = parity.get("reports")
    if not isinstance(report_refs, list) or len(report_refs) != 2:
        raise EvidenceAuditError("Receipt must reference exactly two parity reports.")
    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, Path] = {}
    for reference in report_refs:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise EvidenceAuditError("Parity report references must contain path and sha256.")
        path = _repo_file(reference["path"], label="parity report")
        digest = _require_sha256(reference["sha256"], label="parity report digest")
        _require_exact(sha256_path(path), digest, label=f"parity report {path} digest")
        candidate = _load_json(path, label="parity report")
        profile = candidate.get("case_profile")
        if profile not in EXPECTED_CASE_NAMES or profile in reports:
            raise EvidenceAuditError(f"Invalid or duplicate parity profile: {profile!r}.")
        reports[profile] = _validate_report(path, expected_profile=profile)
        report_paths[profile] = path
    _require_exact(set(reports), set(EXPECTED_CASE_NAMES), label="parity report profiles")

    projected_cases = []
    for profile in ("example", "holdout"):
        report = reports[profile]
        for case in report["cases"]:
            projected_cases.append(
                {
                    **case,
                    "profile": profile,
                    "image_sha256": report["image"]["sha256"],
                    "report": _repo_relative(report_paths[profile]),
                }
            )
    _require_exact(parity.get("cases"), projected_cases, label="receipt parity case projection")

    performance_runs = [
        {
            "profile": profile,
            "image_sha256": reports[profile]["image"]["sha256"],
            **reports[profile]["performance"],
        }
        for profile in ("example", "holdout")
    ]
    measurement_boundaries = {
        run.get("measurement_boundary") for run in performance_runs
    }
    if len(measurement_boundaries) != 1:
        raise EvidenceAuditError("Parity reports disagree on performance boundary.")
    expected_performance = {
        "status": "passed",
        "runs": performance_runs,
        "peak_active_memory_bytes": max(
            run["peak_active_memory_bytes"] for run in performance_runs
        ),
        "measurement_boundary": measurement_boundaries.pop(),
    }
    _require_exact(
        receipt.get("performance"),
        expected_performance,
        label="receipt performance projection",
    )

    lineage = _validate_lineage(receipt)
    return {
        "status": "passed",
        "receipt": _repo_relative(receipt_path),
        "git_commit": git_commit,
        "test_count": counts["call_passed"],
        "profiles": sorted(reports),
        "case_count": sum(len(report["cases"]) for report in reports.values()),
        "lineage_tensor_count": lineage["comparison"]["exact_tensor_count"],
        "raw_evidence": {
            profile: reports[profile]["raw_evidence"]["sha256"]
            for profile in sorted(reports)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=REPO_ROOT / "parity" / "receipts" / "latest.json",
    )
    args = parser.parse_args()
    try:
        result = audit_release_evidence(args.receipt)
    except EvidenceAuditError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
