import copy
import hashlib
import importlib.util
import json
import platform
import sys

import pytest

from tests._paths import REPO_ROOT

_SPEC = importlib.util.spec_from_file_location(
    "validate_runtime_release",
    REPO_ROOT / "scripts" / "validate_runtime_release.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_validate_runtime_release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_validate_runtime_release)
ReceiptError = _validate_runtime_release.ReceiptError
validate_receipt = _validate_runtime_release.validate_receipt


def test_test_environment_comes_from_selected_interpreter():
    observed = _validate_runtime_release._test_environment([sys.executable])

    assert observed["python_version"] == platform.python_version()
    assert observed["machine"] == platform.machine()
    assert observed["platform"] == platform.platform()
    assert observed["mlx_version"]


def _valid_receipt(**overrides):
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "package_version": "0.1.2",
        "git_commit": "a" * 40,
        "generated_at": "2026-07-27T00:00:00+00:00",
        "python_version": "3.12.0",
        "mlx_version": "0.30.0",
        "platform": "macOS",
        "machine": "arm64",
        "checkpoint": {
            "official_repo": "facebook/sam3",
            "official_revision": "b" * 40,
            "official_sha256": "c" * 64,
            "converted_sha256": "d" * 64,
            "conversion_manifest_sha256": "e" * 64,
        },
        "tests": {
            "command": "pytest -q tests",
            "exit_code": 0,
            "passed": True,
            "failed": False,
            "skipped": 0,
            "deselected": 0,
            "skip_details": [],
            "counts": {
                "call_passed": 1,
                "call_failed": 0,
                "call_skipped": 0,
            },
        },
        "parity": {
            "status": "passed",
            "cases": [{"name": "text-positive-1008", "status": "passed"}],
            "thresholds": {
                "mask_iou_min": 0.99,
                "mask_iou_mean_min": 0.995,
                "box_l_inf_max": 0.001,
                "score_abs_max": 0.001,
            },
        },
        "performance": {
            "status": "passed",
        },
    }
    receipt.update(overrides)
    return receipt


def test_release_receipt_schema_accepts_complete_receipt():
    validate_receipt(
        _valid_receipt(),
        expected_commit="a" * 40,
        require_passed=False,
    )


def test_release_receipt_schema_and_commit_binding():
    with pytest.raises(ReceiptError, match="missing required fields"):
        validate_receipt({"schema_version": 1}, require_passed=False)

    with pytest.raises(ReceiptError, match="checkpoint fields"):
        bad = _valid_receipt()
        bad["checkpoint"]["official_sha256"] = ""
        validate_receipt(bad, require_passed=False)

    with pytest.raises(ReceiptError, match="git_commit does not match"):
        validate_receipt(
            _valid_receipt(),
            expected_commit="d" * 40,
            require_passed=False,
        )

    with pytest.raises(ReceiptError, match="skip_details"):
        bad = _valid_receipt()
        bad["tests"]["skip_details"] = [{"nodeid": "tests/x.py::t"}]
        validate_receipt(bad, require_passed=False)

    with pytest.raises(ReceiptError, match="parity.status"):
        bad = _valid_receipt()
        bad["parity"]["status"] = "skipped"
        validate_receipt(bad)

    with pytest.raises(ReceiptError, match="performance.status"):
        bad = _valid_receipt()
        bad["performance"]["status"] = "partial"
        validate_receipt(bad)

    with pytest.raises(ReceiptError, match="not 'passed'"):
        validate_receipt(
            _valid_receipt(status="failed"),
            expected_commit="a" * 40,
        )

    with pytest.raises(ReceiptError, match="inconsistent"):
        bad = _valid_receipt()
        bad["tests"]["failed"] = True
        validate_receipt(bad, require_passed=False)


def test_parent_bound_receipt_requires_receipt_only_attestation(monkeypatch):
    monkeypatch.setattr(_validate_runtime_release, "_git_commit", lambda: "b" * 40)

    def fake_run(command, *, cwd):
        del cwd
        if command[:2] == ["git", "rev-list"]:
            stdout = f"{'b' * 40} {'a' * 40}\n"
        else:
            stdout = "parity/receipts/latest.json\nparity/manifests/lineage.json\n"
        return type("Completed", (), {"stdout": stdout})()

    monkeypatch.setattr(_validate_runtime_release, "_run", fake_run)
    assert _validate_runtime_release._validate_receipt_git_binding("a" * 40) == "a" * 40

    def bad_run(command, *, cwd):
        completed = fake_run(command, cwd=cwd)
        if command[:2] == ["git", "diff-tree"]:
            completed.stdout += "sam3_mlx/model_builder.py\n"
        return completed

    monkeypatch.setattr(_validate_runtime_release, "_run", bad_run)
    with pytest.raises(ReceiptError, match="attestation paths"):
        _validate_runtime_release._validate_receipt_git_binding("a" * 40)


def test_promote_measured_receipt_requires_calibration_and_holdout(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_validate_runtime_release, "REPO_ROOT", tmp_path)
    source_revision = "b" * 40
    source_sha = "c" * 64
    artifact_revision = "d" * 40
    artifact_sha = "e" * 64
    thresholds = {
        "mask_iou_min": 0.95,
        "mask_iou_mean_min": 0.99,
        "box_l_inf_max": 2.0,
        "score_abs_max": 0.025,
    }

    report_paths = []
    for profile in ("example", "holdout"):
        path = tmp_path / f"{profile}.json"
        cases = [
            {
                "name": f"{profile}-{resolution}",
                "resolution": resolution,
                "status": "passed",
                "official_detection_count": 1,
                "mlx_detection_count": 1,
                "detection_count_match": True,
                "mask_iou_min": 1.0,
                "mask_iou_mean": 1.0,
                "box_l_inf_max": 0.0,
                "score_abs_max": 0.0,
                "matches": [
                    {
                        "official_index": 0,
                        "mlx_index": 0,
                        "mask_iou": 1.0,
                    }
                ],
            }
            for resolution in (1008, 672, 504)
        ]
        path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "case_profile": profile,
                    "official_code": {"revision": "f" * 40},
                    "official_checkpoint": {
                        "revision": source_revision,
                        "sha256": source_sha,
                    },
                    "converted_checkpoint": {
                        "revision": artifact_revision,
                        "sha256": artifact_sha,
                    },
                    "thresholds": thresholds,
                    "image": {"sha256": "1" * 64},
                    "cases": cases,
                    "performance": {
                        "status": "passed",
                        "cold_load_s": 0.5,
                        "warmup_runs": 1,
                        "repetitions": 5,
                        "latency_by_resolution_s": {
                            str(resolution): {
                                "samples": [1.0, 2.0, 3.0, 4.0, 5.0],
                                "median": 3.0,
                                "p95": 5.0,
                            }
                            for resolution in (1008, 672, 504)
                        },
                        "peak_active_memory_bytes": 123,
                        "measurement_boundary": "fixture",
                    },
                }
            )
        )
        report_paths.append(path)

    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "source": {
                    "repo": "facebook/sam3",
                    "revision": source_revision,
                    "checkpoint_sha256": source_sha,
                },
                "published_artifact": {
                    "repo": "mlx-community/sam3-image",
                    "revision": artifact_revision,
                    "checkpoint_sha256": artifact_sha,
                },
                "reproduction": {"manifest_sha256": "2" * 64},
                "comparison": {"semantic_match": True},
            }
        )
    )

    promoted = _validate_runtime_release.promote_measured_receipt(
        _valid_receipt(),
        parity_report_paths=report_paths,
        lineage_report_path=lineage_path,
    )
    assert promoted["status"] == "passed"
    assert promoted["parity"]["status"] == "passed"
    assert len(promoted["parity"]["cases"]) == 6
    assert promoted["checkpoint"]["official_revision"] == source_revision
    validate_receipt(
        promoted,
        expected_commit="a" * 40,
        evidence_root=tmp_path,
    )

    tampered_report = json.loads(report_paths[0].read_text())
    tampered_report["cases"][0]["name"] = "tampered"
    report_paths[0].write_text(json.dumps(tampered_report))
    with pytest.raises(ReceiptError, match="digest mismatch"):
        validate_receipt(promoted, evidence_root=tmp_path)

    promoted_with_new_digest = copy.deepcopy(promoted)
    promoted_with_new_digest["parity"]["reports"][0]["sha256"] = hashlib.sha256(
        report_paths[0].read_bytes()
    ).hexdigest()
    with pytest.raises(ReceiptError, match="does not match referenced evidence"):
        validate_receipt(promoted_with_new_digest, evidence_root=tmp_path)

    dishonest_report = json.loads(report_paths[0].read_text())
    dishonest_report["cases"][0]["mask_iou_min"] = 0.0
    report_paths[0].write_text(json.dumps(dishonest_report))
    promoted_with_new_digest["parity"]["reports"][0]["sha256"] = hashlib.sha256(
        report_paths[0].read_bytes()
    ).hexdigest()
    with pytest.raises(ReceiptError, match="not reproducible"):
        validate_receipt(promoted_with_new_digest, evidence_root=tmp_path)
