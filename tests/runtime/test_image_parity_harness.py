import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts._oracle_runtime import OracleCase
from tests._paths import REPO_ROOT


_SPEC = importlib.util.spec_from_file_location(
    "run_image_parity",
    REPO_ROOT / "scripts" / "run_image_parity.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_parity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_parity)


def _outputs(mask_count: int = 1) -> dict[str, np.ndarray]:
    return {
        "masks": np.ones((mask_count, 1, 4, 4), dtype=np.bool_),
        "boxes": np.ones((mask_count, 4), dtype=np.float32),
        "scores": np.full((mask_count,), 0.75, dtype=np.float32),
    }


def test_release_threshold_contract_is_precision_aware_and_fixed():
    assert _parity.MASK_IOU_MIN == 0.95
    assert _parity.MASK_IOU_MEAN_MIN == 0.99
    assert _parity.BOX_L_INF_MAX == 2.0
    assert _parity.SCORE_ABS_MAX == 0.025


def test_compare_case_requires_count_and_numeric_contract():
    spec: OracleCase = {
        "name": "fixture",
        "resolution": 1008,
        "prompt": "object",
        "geometric_prompts": [],
    }
    exact = _parity._compare_case(spec, _outputs(), _outputs())
    assert exact["status"] == "passed"
    assert exact["mask_iou_min"] == 1.0

    count_mismatch = _parity._compare_case(spec, _outputs(), _outputs(mask_count=0))
    assert count_mismatch["status"] == "failed"
    assert count_mismatch["detection_count_match"] is False


def test_evidence_path_uses_checkout_relative_identity():
    checkout = Path("/tmp/official")
    assert (
        _parity._evidence_path(
            checkout / "assets/images/test.jpg",
            official_checkout=checkout,
        )
        == "official-checkout/assets/images/test.jpg"
    )


def test_official_checkout_requires_exact_clean_commit(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    assert (
        _parity._validate_official_checkout(
            tmp_path,
            expected_revision=revision,
        )
        == revision
    )

    tracked.write_text("dirty\n")
    with pytest.raises(ValueError, match="must be clean"):
        _parity._validate_official_checkout(
            tmp_path,
            expected_revision=revision,
        )
