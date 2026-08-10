#!/usr/bin/env python3
"""Audit replayable evidence and enforce one source commit across all artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.parity_evidence import load_evidence_bundle  # noqa: E402
from sam3_mlx.release_contract import (  # noqa: E402
    JsonObject,
    require_json_list,
    require_json_object,
)
from scripts import audit_release_evidence as _audit  # noqa: E402

_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class SourceBindingError(ValueError):
    """Raised when release artifacts do not share the receipt source commit."""


def _require_object(value: object, *, field: str) -> JsonObject:
    return require_json_object(value, field=field, error_type=SourceBindingError)


def _require_list(value: object, *, field: str) -> list[object]:
    return require_json_list(value, field=field, error_type=SourceBindingError)


def _load_json(path: Path, *, label: str) -> JsonObject:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceBindingError(f"Could not read {label} {path}: {exc}") from exc
    return _require_object(value, field=label)


def _repo_file(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SourceBindingError(f"{label} path must be a non-empty string.")
    path = Path(value)
    if path.is_absolute():
        raise SourceBindingError(f"{label} path must be repository-relative.")
    resolved = (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise SourceBindingError(
            f"{label} path escapes the repository: {value!r}."
        ) from exc
    if not resolved.is_file():
        raise SourceBindingError(f"{label} file does not exist: {value!r}.")
    return resolved


def _require_commit(value: object, *, expected: str, label: str) -> None:
    if not isinstance(value, str) or not _COMMIT_PATTERN.fullmatch(value):
        raise SourceBindingError(f"{label} must contain a lowercase commit SHA.")
    if value != expected:
        raise SourceBindingError(
            f"{label} was generated from {value}, but the receipt binds {expected}."
        )


def audit_source_binding(receipt_path: Path) -> JsonObject:
    receipt = _load_json(receipt_path, label="runtime release receipt")
    receipt_commit = receipt.get("git_commit")
    if not isinstance(receipt_commit, str) or not _COMMIT_PATTERN.fullmatch(
        receipt_commit
    ):
        raise SourceBindingError("Receipt git_commit must be a lowercase commit SHA.")

    parity = _require_object(receipt.get("parity"), field="Receipt parity")
    references = _require_list(parity.get("reports"), field="Receipt parity reports")
    if len(references) != 2:
        raise SourceBindingError("Receipt must reference exactly two parity reports.")

    profiles: list[str] = []
    for raw_reference in references:
        reference = _require_object(raw_reference, field="Parity report reference")
        report_path = _repo_file(reference.get("path"), label="parity report")
        report = _load_json(report_path, label="parity report")
        profile = report.get("case_profile")
        if not isinstance(profile, str) or not profile:
            raise SourceBindingError(f"Parity report {report_path} has no profile.")
        _require_commit(
            report.get("source_commit"),
            expected=receipt_commit,
            label=f"{profile} parity report source_commit",
        )

        raw_evidence = _require_object(
            report.get("raw_evidence"),
            field=f"{profile} report raw_evidence",
        )
        evidence_path = _repo_file(
            raw_evidence.get("path"), label=f"{profile} raw evidence"
        )
        try:
            metadata, _official, _mlx = load_evidence_bundle(evidence_path)
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise SourceBindingError(
                f"Could not load {profile} raw evidence source binding: {exc}"
            ) from exc
        _require_commit(
            metadata.get("source_commit"),
            expected=receipt_commit,
            label=f"{profile} raw evidence source_commit",
        )
        profiles.append(profile)

    checkpoint = _require_object(receipt.get("checkpoint"), field="Receipt checkpoint")
    lineage_path = _repo_file(
        checkpoint.get("lineage_report"), label="checkpoint lineage report"
    )
    lineage = _load_json(lineage_path, label="checkpoint lineage report")
    _require_commit(
        lineage.get("source_commit"),
        expected=receipt_commit,
        label="checkpoint lineage source_commit",
    )
    return {
        "source_commit": receipt_commit,
        "profiles": sorted(profiles),
        "lineage": str(lineage_path.relative_to(REPO_ROOT)),
    }


def audit_release_candidate(receipt_path: Path) -> JsonObject:
    replay_result = _audit.audit_release_evidence(receipt_path)
    source_result = audit_source_binding(receipt_path)
    return {
        **replay_result,
        "source_binding": source_result,
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
        result = audit_release_candidate(args.receipt)
    except (_audit.EvidenceAuditError, SourceBindingError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
