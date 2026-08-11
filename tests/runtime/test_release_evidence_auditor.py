from copy import deepcopy
from typing import cast

import pytest

from sam3_mlx.release_contract import (
    JsonObject,
    build_oracle_bindings,
    canonical_json_sha256,
    sha256_path,
)
from scripts.audit_release_evidence import (
    CaseSpec,
    EvidenceAuditError,
    HARDENED_ORACLE,
    _case_spec_sha256,  # pyright: ignore[reportPrivateUsage]
    _validate_oracle,  # pyright: ignore[reportPrivateUsage]
)


def _object(value: object) -> JsonObject:
    assert isinstance(value, dict)
    raw_value = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in raw_value)
    return cast(JsonObject, raw_value)


def _valid_oracle_fixture() -> tuple[list[CaseSpec], JsonObject, JsonObject]:
    specs: list[CaseSpec] = [
        {
            "name": "synthetic",
            "resolution": 14,
            "prompt": "object",
            "geometric_prompts": [],
        }
    ]
    bindings = build_oracle_bindings(
        image_sha256="a" * 64,
        case_spec_sha256=_case_spec_sha256(specs),
        confidence_threshold=0.5,
        oracle_runner_sha256=sha256_path(HARDENED_ORACLE),
    )
    oracle: JsonObject = {
        "bindings": bindings,
        "cache_key": canonical_json_sha256(bindings),
        "official_code": bindings["official_code"],
        "official_checkpoint": bindings["official_checkpoint"],
        "image_sha256": bindings["image_sha256"],
        "case_spec_sha256": bindings["case_spec_sha256"],
        "confidence_threshold": bindings["confidence_threshold"],
        "precision": bindings["precision"],
        "cpu_adapters": bindings["cpu_adapters"],
        "oracle_runner_sha256": bindings["oracle_runner_sha256"],
        "release_contract_sha256": bindings["release_contract_sha256"],
        "environment": {
            "machine": "arm64",
            "platform": "macOS-test",
            "python": "3.13",
            "torch": "test",
        },
        "cases": [
            {
                "name": "synthetic",
                "resolution": 14,
                "detection_count": 1,
                "image_latency_s": 1.0,
                "prompt_latency_s": 1.0,
            }
        ],
    }
    report: JsonObject = {
        "image": {"sha256": "a" * 64},
        "confidence_threshold": 0.5,
        "cases": [{"official_detection_count": 1}],
    }
    return specs, report, oracle


def test_oracle_auditor_accepts_complete_cache_binding():
    specs, report, oracle = _valid_oracle_fixture()

    _validate_oracle(oracle, report=report, specs=specs)


def test_oracle_auditor_rejects_stale_or_forged_checkpoint_binding():
    specs, report, oracle = _valid_oracle_fixture()
    tampered = deepcopy(oracle)
    bindings = _object(tampered["bindings"])
    _object(bindings["official_checkpoint"])["revision"] = "0" * 40
    tampered["cache_key"] = canonical_json_sha256(bindings)

    with pytest.raises(EvidenceAuditError, match="oracle bindings"):
        _validate_oracle(tampered, report=report, specs=specs)


def test_oracle_auditor_rejects_runner_source_drift():
    specs, report, oracle = _valid_oracle_fixture()
    tampered = deepcopy(oracle)
    bindings = _object(tampered["bindings"])
    bindings["oracle_runner_sha256"] = "0" * 64
    tampered["oracle_runner_sha256"] = "0" * 64
    tampered["cache_key"] = canonical_json_sha256(bindings)

    with pytest.raises(EvidenceAuditError, match="oracle bindings"):
        _validate_oracle(tampered, report=report, specs=specs)
