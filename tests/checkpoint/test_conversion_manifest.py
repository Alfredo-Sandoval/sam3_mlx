import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray
import pytest

from sam3_mlx.convert import (
    DEFAULT_MLX_CHECKPOINT,
    _convert_checkpoint_weights,  # pyright: ignore[reportPrivateUsage]
    _validate_cached_conversion,  # pyright: ignore[reportPrivateUsage]
    _validate_source_revision,  # pyright: ignore[reportPrivateUsage]
    _write_conversion_manifest,  # pyright: ignore[reportPrivateUsage]
    load_from_hub,
    validate_hub_checkpoint_provenance,
)
from tests._mlx_runtime import save_safetensors


class _TorchLikeTensor:
    def __init__(self, value: object) -> None:
        self._value = np.asarray(value, dtype=np.float32)

    def numpy(self) -> NDArray[np.float32]:
        return self._value


def test_conversion_normalizes_conv_layout_once():
    source = np.arange(2 * 3 * 1 * 1, dtype=np.float32).reshape(2, 3, 1, 1)
    weights, ignored = _convert_checkpoint_weights(
        {
            "detector.geometry_encoder.boxes_pool_project.weight": (
                _TorchLikeTensor(source)
            ),
            "tracker.unused.weight": _TorchLikeTensor([1.0]),
        },
        source_label="fixture",
    )

    converted = np.asarray(weights["geometry_encoder.boxes_pool_project.weight"])
    np.testing.assert_array_equal(converted, source.transpose(0, 2, 3, 1))
    assert converted.shape == (2, 1, 1, 3)
    assert ignored == ("tracker.unused.weight",)


def test_conversion_manifest_records_pinned_source_and_content_hashes(
    tmp_path: Path,
) -> None:
    source_checkpoint = tmp_path / "sam3.pt"
    source_checkpoint.write_bytes(b"official-checkpoint")
    output_checkpoint = tmp_path / "model.safetensors"
    weights = {
        "head.weight": mx.ones((2, 3), dtype=mx.float32),
        "head.bias": mx.zeros((2,), dtype=mx.float16),
    }
    save_safetensors(output_checkpoint, weights)

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
    assert (
        manifest["source_checkpoint_sha256"]
        == hashlib.sha256(b"official-checkpoint").hexdigest()
    )
    assert (
        manifest["output_sha256"]
        == hashlib.sha256(output_checkpoint.read_bytes()).hexdigest()
    )
    assert manifest["mapped_count"] == 2
    assert manifest["unmapped_keys"] == []
    assert manifest["ignored_keys"] == ["tracker.memory.weight"]
    assert manifest["dtype_counts"] == {"float16": 1, "float32": 1}
    assert manifest["converter_version"]


def test_cached_conversion_rejects_revision_or_content_drift(tmp_path: Path) -> None:
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
def test_conversion_requires_full_immutable_commit_revision(revision: str) -> None:
    with pytest.raises(ValueError, match="40-character hexadecimal commit SHA"):
        _validate_source_revision(revision)


def test_default_mlx_checkpoint_pin_is_immutable():
    assert DEFAULT_MLX_CHECKPOINT.repo == "mlx-community/sam3-image"
    assert _validate_source_revision(DEFAULT_MLX_CHECKPOINT.revision) == (
        DEFAULT_MLX_CHECKPOINT.revision
    )
    assert len(DEFAULT_MLX_CHECKPOINT.output_sha256) == 64
    assert DEFAULT_MLX_CHECKPOINT.architecture == "sam3-image"


def test_validate_hub_checkpoint_provenance_rejects_hash_and_manifest_drift(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"pinned-weights")
    expected_sha = hashlib.sha256(b"pinned-weights").hexdigest()
    revision = "a" * 40

    report = validate_hub_checkpoint_provenance(
        tmp_path,
        expected_repo="mlx-community/sam3-image",
        expected_revision=revision,
        expected_output_sha256=expected_sha,
    )
    assert report["status"] == "package-pinned"
    assert report["output_sha256"] == expected_sha

    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content hash does not match"):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="mlx-community/sam3-image",
            expected_revision=revision,
            expected_output_sha256=expected_sha,
        )

    weights.write_bytes(b"pinned-weights")
    (tmp_path / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "architecture": "not-sam3-image",
                "artifact_repo": "mlx-community/sam3-image",
                "artifact_revision": revision,
                "output_sha256": expected_sha,
            }
        )
    )
    with pytest.raises(ValueError, match="architecture"):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="mlx-community/sam3-image",
            expected_revision=revision,
            expected_output_sha256=expected_sha,
        )

    (tmp_path / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "architecture": "sam3-image",
                "artifact_repo": "mlx-community/sam3-image",
                "artifact_revision": revision,
                "output_sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(ValueError, match="output_sha256"):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="mlx-community/sam3-image",
            expected_revision=revision,
            expected_output_sha256=expected_sha,
        )

    (tmp_path / "conversion-manifest.json").write_text(
        json.dumps(
            {
                "architecture": "sam3-image",
                "artifact_repo": "mlx-community/sam3-image",
                "artifact_revision": "b" * 40,
                "output_sha256": expected_sha,
            }
        )
    )
    with pytest.raises(ValueError, match="artifact_revision"):
        validate_hub_checkpoint_provenance(
            tmp_path,
            expected_repo="mlx-community/sam3-image",
            expected_revision=revision,
            expected_output_sha256=expected_sha,
        )


def test_default_hf_checkpoint_is_pinned_and_manifest_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"default-pinned")
    expected_sha = hashlib.sha256(b"default-pinned").hexdigest()
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        "sam3_mlx.convert.snapshot_download",
        fake_snapshot_download,
    )
    monkeypatch.setattr(
        "sam3_mlx.convert.DEFAULT_MLX_CHECKPOINT",
        type(DEFAULT_MLX_CHECKPOINT)(
            repo="mlx-community/sam3-image",
            revision="c" * 40,
            output_sha256=expected_sha,
            architecture="sam3-image",
        ),
    )
    # DEFAULT_MLX_CHECKPOINT is imported by name in load_from_hub default args;
    # patch the module attribute used for compare inside the function.
    import sam3_mlx.convert as convert_mod

    convert_mod.DEFAULT_MLX_CHECKPOINT = type(DEFAULT_MLX_CHECKPOINT)(
        repo="mlx-community/sam3-image",
        revision="c" * 40,
        output_sha256=expected_sha,
        architecture="sam3-image",
    )

    path = load_from_hub(
        hf_repo="mlx-community/sam3-image",
        revision="c" * 40,
        expected_output_sha256=expected_sha,
    )
    assert path == weights
    assert captured["revision"] == "c" * 40
    captured_revision = captured["revision"]
    assert isinstance(captured_revision, str)
    assert len(captured_revision) == 40

    with pytest.raises(ValueError, match="requires an immutable"):
        load_from_hub(hf_repo="mlx-community/sam3-image")

    # Bad revision format is rejected before download.
    with pytest.raises(ValueError, match="40-character"):
        load_from_hub(
            hf_repo="mlx-community/sam3-image",
            revision="main",
            expected_output_sha256=expected_sha,
        )

    # Hash drift fails before any model mutation (snapshot returns same dir).
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="content hash does not match"):
        load_from_hub(
            hf_repo="mlx-community/sam3-image",
            revision="c" * 40,
            expected_output_sha256=expected_sha,
        )
