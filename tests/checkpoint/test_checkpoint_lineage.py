import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from tests._paths import REPO_ROOT


_SPEC = importlib.util.spec_from_file_location(
    "validate_checkpoint_lineage",
    REPO_ROOT / "scripts" / "validate_checkpoint_lineage.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_lineage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_lineage)


def test_compare_tensors_requires_exact_keys_shapes_dtypes_and_values():
    published = {
        "weight": mx.array([[1.0, 2.0]], dtype=mx.float32),
        "bias": mx.array([3.0], dtype=mx.float32),
    }
    exact = _lineage._compare_tensors(
        published,
        {
            "weight": mx.array([[1.0, 2.0]], dtype=mx.float32),
            "bias": mx.array([3.0], dtype=mx.float32),
        },
    )
    assert exact["semantic_match"] is True
    assert exact["exact_tensor_count"] == 2

    changed = _lineage._compare_tensors(
        published,
        {
            "weight": mx.array([[1.0, 2.5]], dtype=mx.float32),
            "extra": mx.array([3.0], dtype=mx.float32),
        },
    )
    assert changed["semantic_match"] is False
    assert changed["missing_keys"] == ["bias"]
    assert changed["extra_keys"] == ["extra"]
    assert changed["value_mismatches"] == ["weight"]


def test_load_checkpoint_tensors_rejects_non_mapping_payload(tmp_path: Path) -> None:
    checkpoint = tmp_path / "array.npy"
    np.save(checkpoint, np.array([1.0], dtype=np.float32))

    with pytest.raises(ValueError, match="must contain a tensor mapping"):
        _lineage.load_checkpoint_tensors(checkpoint)
