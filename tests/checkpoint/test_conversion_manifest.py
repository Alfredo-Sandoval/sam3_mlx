import hashlib
import json

import mlx.core as mx
import pytest

from sam3_mlx.convert import (
    _validate_cached_conversion,
    _validate_source_revision,
    _write_conversion_manifest,
)


def test_conversion_manifest_records_pinned_source_and_content_hashes(tmp_path):
    source_checkpoint = tmp_path / "sam3.pt"
    source_checkpoint.write_bytes(b"official-checkpoint")
    output_checkpoint = tmp_path / "model.safetensors"
    weights = {
        "head.weight": mx.ones((2, 3), dtype=mx.float32),
        "head.bias": mx.zeros((2,), dtype=mx.float16),
    }
    mx.save_safetensors(str(output_checkpoint), weights)

    manifest_path = _write_conversion_manifest(
        tmp_path,
        source_repo="facebook/sam3",
        source_revision="a" * 40,
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        weights=weights,
        ignored_keys=("tracker.memory.weight",),
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["architecture"] == "sam3-image"
    assert manifest["source_repo"] == "facebook/sam3"
    assert manifest["source_revision"] == "a" * 40
    assert manifest["source_checkpoint_sha256"] == hashlib.sha256(
        b"official-checkpoint"
    ).hexdigest()
    assert manifest["output_sha256"] == hashlib.sha256(
        output_checkpoint.read_bytes()
    ).hexdigest()
    assert manifest["mapped_count"] == 2
    assert manifest["unmapped_keys"] == []
    assert manifest["ignored_keys"] == ["tracker.memory.weight"]
    assert manifest["dtype_counts"] == {"float16": 1, "float32": 1}
    assert manifest["converter_version"]


def test_cached_conversion_rejects_revision_or_content_drift(tmp_path):
    weights_file = tmp_path / "model.safetensors"
    weights_file.write_bytes(b"converted")
    manifest_file = tmp_path / "conversion-manifest.json"
    manifest_file.write_text(
        json.dumps(
            {
                "source_repo": "facebook/sam3",
                "source_revision": "a" * 40,
                "output_sha256": hashlib.sha256(b"converted").hexdigest(),
            }
        )
    )

    _validate_cached_conversion(
        weights_file,
        manifest_file,
        source_repo="facebook/sam3",
        source_revision="a" * 40,
    )

    with pytest.raises(ValueError, match="source_revision"):
        _validate_cached_conversion(
            weights_file,
            manifest_file,
            source_repo="facebook/sam3",
            source_revision="b" * 40,
        )

    weights_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output_sha256"):
        _validate_cached_conversion(
            weights_file,
            manifest_file,
            source_repo="facebook/sam3",
            source_revision="a" * 40,
        )


@pytest.mark.parametrize("revision", ["main", "a" * 39, "g" * 40])
def test_conversion_requires_full_immutable_commit_revision(revision):
    with pytest.raises(ValueError, match="40-character hexadecimal commit SHA"):
        _validate_source_revision(revision)
