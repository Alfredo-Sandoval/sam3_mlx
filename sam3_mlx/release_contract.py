"""Immutable constants and hashing helpers for SAM3-MLX release evidence.

This module intentionally depends only on the Python standard library so both
MLX and isolated upstream-Torch environments can import the same contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

PACKAGE_VERSION = "0.1.2"
REPORT_SCHEMA_VERSION = 2
ORACLE_SCHEMA_VERSION = 2
EVIDENCE_SCHEMA_VERSION = 1
COMPARISON_ALGORITHM = "hungarian-max-mask-iou-v1"

OFFICIAL_CODE_REPO = "https://github.com/facebookresearch/sam3"
OFFICIAL_CODE_REVISION = "2814fa619404a722d03e9a012e083e4f293a4e53"
OFFICIAL_CHECKPOINT_REPO = "facebook/sam3"
OFFICIAL_CHECKPOINT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
OFFICIAL_CHECKPOINT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)
MLX_CHECKPOINT_REPO = "mlx-community/sam3-image"
MLX_CHECKPOINT_REVISION = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
MLX_CHECKPOINT_SHA256 = (
    "0ad4c3f42ecf706c4cda63cf58d621699491ed65012b3999284ea370984f7173"
)
CHECKPOINT_TENSOR_COUNT = 1_400

RELEASE_CONFIDENCE_THRESHOLD = 0.5
RELEASE_RESOLUTIONS = (1008, 672, 504)
RELEASE_THRESHOLDS = {
    "mask_iou_min": 0.95,
    "mask_iou_mean_min": 0.99,
    "box_l_inf_max": 2.0,
    "score_abs_max": 0.025,
}

EXPECTED_CASE_NAMES = {
    "example": (
        "text_shoe_1008",
        "text_nonsense_1008",
        "positive_box_1008",
        "positive_negative_box_1008",
        "text_shoe_672",
        "text_shoe_504",
    ),
    "holdout": (
        "text_paper_bag_1008",
        "text_paper_bag_672",
        "text_paper_bag_504",
        "text_car_1008",
        "text_nonsense_1008",
    ),
}

ORACLE_PRECISION = "torch.cpu.autocast.bfloat16"
ORACLE_CPU_ADAPTERS = (
    "sam3.model.edt replaced with fail-fast unused stub",
    "construction-time CUDA cache tensors redirected to CPU",
    "pin_memory disabled for CPU-only staging",
    (
        "global-attention RoPE frequencies recomputed with the official "
        "formula for non-1008 processor grids"
    ),
)


def sha256_path(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize evidence deterministically for content-addressed bindings."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def build_oracle_bindings(
    *,
    image_sha256: str,
    case_spec_sha256: str,
    confidence_threshold: float,
    oracle_runner_sha256: str,
) -> dict[str, Any]:
    """Build the complete cache identity for upstream oracle outputs."""

    if not math.isfinite(float(confidence_threshold)):
        raise ValueError("confidence_threshold must be finite")
    return {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "official_code": {
            "repo": OFFICIAL_CODE_REPO,
            "revision": OFFICIAL_CODE_REVISION,
        },
        "official_checkpoint": {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
            "sha256": OFFICIAL_CHECKPOINT_SHA256,
        },
        "image_sha256": image_sha256,
        "case_spec_sha256": case_spec_sha256,
        "confidence_threshold": float(confidence_threshold),
        "precision": ORACLE_PRECISION,
        "cpu_adapters": list(ORACLE_CPU_ADAPTERS),
        "oracle_runner_sha256": oracle_runner_sha256,
    }


def validate_exact_mapping(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Raise a compact error when a release mapping differs from its contract."""

    if dict(observed) == dict(expected):
        return
    keys = sorted(set(observed) | set(expected))
    drift = {
        key: {"observed": observed.get(key), "expected": expected.get(key)}
        for key in keys
        if observed.get(key) != expected.get(key)
    }
    raise ValueError(f"{label} does not match the frozen release contract: {drift}")
