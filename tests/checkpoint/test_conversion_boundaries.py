from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from sam3_mlx.convert import (
    _json_value,  # pyright: ignore[reportPrivateUsage]
    _torch_weights,  # pyright: ignore[reportPrivateUsage]
    _validate_source_revision,  # pyright: ignore[reportPrivateUsage]
    convert,
    validate_hub_checkpoint_provenance,
)


def test_provenance_rejects_non_object_manifest(tmp_path: Path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    (tmp_path / "conversion-manifest.json").write_text("[]")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="example/repo",
            expected_revision="a" * 40,
            expected_output_sha256=hashlib.sha256(b"weights").hexdigest(),
        )


@pytest.mark.parametrize("payload", [None, [], (), "weights", 1, True])
def test_checkpoint_payload_must_be_a_mapping(payload: object):
    with pytest.raises(ValueError, match="checkpoint payload must be a mapping"):
        _torch_weights(payload)


def test_checkpoint_weights_require_string_keys_and_numpy_method():
    with pytest.raises(TypeError, match="weight keys must be strings"):
        _torch_weights({1: object()})
    with pytest.raises(TypeError, match="weight 'detector.weight' must expose numpy"):
        _torch_weights({"detector.weight": object()})


def test_json_value_rejects_non_string_object_keys():
    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        _json_value({1: "value"})


@pytest.mark.parametrize("revision", [None, 1, True, object()])
def test_source_revision_rejects_non_string_values(revision: object):
    with pytest.raises(TypeError, match="source_revision must be a string"):
        _validate_source_revision(revision)


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("expected_repo", 1, TypeError, "expected_repo must be a non-empty string"),
        (
            "expected_revision",
            True,
            TypeError,
            "source_revision must be a string",
        ),
        (
            "expected_output_sha256",
            None,
            TypeError,
            "expected_output_sha256 must be a string",
        ),
        (
            "expected_output_sha256",
            "bad",
            ValueError,
            "64-character hexadecimal SHA-256",
        ),
        (
            "expected_architecture",
            "",
            TypeError,
            "expected_architecture must be a non-empty string",
        ),
    ],
)
def test_provenance_rejects_malformed_expected_fields(
    tmp_path: Path,
    field: str,
    value: object,
    exception: type[Exception],
    message: str,
):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    arguments: dict[str, object] = {
        "expected_repo": "example/repo",
        "expected_revision": "a" * 40,
        "expected_output_sha256": hashlib.sha256(b"weights").hexdigest(),
        "expected_architecture": "sam3-image",
    }
    arguments[field] = value

    with pytest.raises(exception, match=message):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo=cast(str, arguments["expected_repo"]),
            expected_revision=cast(str, arguments["expected_revision"]),
            expected_output_sha256=cast(str, arguments["expected_output_sha256"]),
            expected_architecture=cast(str, arguments["expected_architecture"]),
        )


@pytest.mark.parametrize(
    ("manifest_field", "manifest_value"),
    [
        ("artifact_repo", 1),
        ("artifact_revision", False),
        ("architecture", ["sam3-image"]),
        ("output_sha256", None),
    ],
)
def test_provenance_rejects_malformed_manifest_fields(
    tmp_path: Path, manifest_field: str, manifest_value: object
):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights")
    expected_sha = hashlib.sha256(b"weights").hexdigest()
    manifest: dict[str, object] = {
        "artifact_repo": "example/repo",
        "artifact_revision": "a" * 40,
        "architecture": "sam3-image",
        "output_sha256": expected_sha,
    }
    manifest[manifest_field] = manifest_value
    (tmp_path / "conversion-manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=manifest_field):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="example/repo",
            expected_revision="a" * 40,
            expected_output_sha256=expected_sha,
        )


def test_convert_keeps_safe_cpu_torch_load_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    captured: dict[str, object] = {}

    class _FakeTorch:
        def load(self, path: str, *, map_location: str, weights_only: bool) -> object:
            captured.update(
                path=path,
                map_location=map_location,
                weights_only=weights_only,
            )
            return []

    def fake_import_module(name: str) -> object:
        assert name == "torch"
        return _FakeTorch()

    monkeypatch.setattr("sam3_mlx.convert.import_module", fake_import_module)

    with pytest.raises(ValueError, match="checkpoint payload must be a mapping"):
        convert(tmp_path)
    assert captured == {
        "path": str(tmp_path / "sam3.pt"),
        "map_location": "cpu",
        "weights_only": True,
    }
