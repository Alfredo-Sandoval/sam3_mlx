import importlib.util
from pathlib import Path

import numpy as np

from tests._paths import REPO_ROOT


_SPEC = importlib.util.spec_from_file_location(
    "run_image_parity",
    REPO_ROOT / "scripts" / "run_image_parity.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_parity = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_parity)


def _outputs(mask_count=1):
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
    spec = {
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
