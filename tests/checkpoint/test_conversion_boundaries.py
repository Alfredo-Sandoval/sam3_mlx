from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sam3_mlx.convert import (
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
