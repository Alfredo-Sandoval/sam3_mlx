import json
from pathlib import Path

import pytest

from sam3_mlx.parity_evidence import write_evidence_bundle
from sam3_mlx.release_contract import JsonObject
from scripts import audit_release_candidate as candidate


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _candidate_tree(tmp_path: Path, *, commit: str = "a" * 40) -> Path:
    evidence_dir = tmp_path / "parity" / "evidence"
    reports_dir = tmp_path / "parity" / "receipts"
    manifests_dir = tmp_path / "parity" / "manifests"

    references: list[JsonObject] = []
    for profile in ("example", "holdout"):
        evidence_path = evidence_dir / f"{profile}.npz"
        write_evidence_bundle(
            evidence_path,
            metadata={"source_commit": commit, "profile": profile},
            official_outputs=[],
            mlx_outputs=[],
        )
        report_path = reports_dir / f"{profile}.json"
        _write_json(
            report_path,
            {
                "case_profile": profile,
                "source_commit": commit,
                "raw_evidence": {
                    "path": f"parity/evidence/{profile}.npz",
                    "sha256": "0" * 64,
                },
            },
        )
        references.append(
            {"path": f"parity/receipts/{profile}.json", "sha256": "0" * 64}
        )

    lineage_path = manifests_dir / "checkpoint-lineage.json"
    _write_json(lineage_path, {"source_commit": commit})
    receipt_path = reports_dir / "latest.json"
    _write_json(
        receipt_path,
        {
            "git_commit": commit,
            "parity": {"reports": references},
            "checkpoint": {
                "lineage_report": "parity/manifests/checkpoint-lineage.json"
            },
        },
    )
    return receipt_path


def test_candidate_source_binding_accepts_one_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt_path = _candidate_tree(tmp_path)
    monkeypatch.setattr(candidate, "REPO_ROOT", tmp_path)

    result = candidate.audit_source_binding(receipt_path)

    assert result["source_commit"] == "a" * 40
    assert result["profiles"] == ["example", "holdout"]


def test_candidate_source_binding_rejects_raw_evidence_from_another_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt_path = _candidate_tree(tmp_path)
    monkeypatch.setattr(candidate, "REPO_ROOT", tmp_path)
    evidence_path = tmp_path / "parity" / "evidence" / "holdout.npz"
    write_evidence_bundle(
        evidence_path,
        metadata={"source_commit": "b" * 40, "profile": "holdout"},
        official_outputs=[],
        mlx_outputs=[],
    )

    with pytest.raises(
        candidate.SourceBindingError, match="raw evidence source_commit"
    ):
        candidate.audit_source_binding(receipt_path)
