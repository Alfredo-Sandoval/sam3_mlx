#!/usr/bin/env python3
"""Generate hard-pinned, source-bound semantic checkpoint-lineage evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import mlx.core as mx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.convert import normalize_sam3_image_weight_layout  # noqa: E402
from sam3_mlx.release_contract import (  # noqa: E402
    CHECKPOINT_TENSOR_COUNT,
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    MLX_CHECKPOINT_SHA256,
    OFFICIAL_CHECKPOINT_REPO,
    OFFICIAL_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_SHA256,
    PACKAGE_VERSION,
    sha256_path,
)
from sam3_mlx.source_binding import validate_attestation_only_worktree  # noqa: E402

LINEAGE_SCHEMA_VERSION = 2
CONVERT_MODULE = REPO_ROOT / "sam3_mlx" / "convert.py"


def _compare_tensors(
    published: dict[str, mx.array],
    reproduced: dict[str, mx.array],
) -> dict[str, Any]:
    published = {
        key: normalize_sam3_image_weight_layout(key, value)
        for key, value in published.items()
    }
    reproduced = {
        key: normalize_sam3_image_weight_layout(key, value)
        for key, value in reproduced.items()
    }
    published_keys = set(published)
    reproduced_keys = set(reproduced)
    missing = sorted(published_keys - reproduced_keys)
    extra = sorted(reproduced_keys - published_keys)
    shape_mismatches = []
    dtype_mismatches = []
    value_mismatches = []
    exact = 0
    for key in sorted(published_keys & reproduced_keys):
        left = published[key]
        right = reproduced[key]
        if left.shape != right.shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "published": list(left.shape),
                    "reproduced": list(right.shape),
                }
            )
            continue
        if left.dtype != right.dtype:
            dtype_mismatches.append(
                {
                    "key": key,
                    "published": str(left.dtype),
                    "reproduced": str(right.dtype),
                }
            )
            continue
        if bool(mx.all(left == right).item()):
            exact += 1
        else:
            value_mismatches.append(key)
    return {
        "published_tensor_count": len(published_keys),
        "reproduced_tensor_count": len(reproduced_keys),
        "exact_tensor_count": exact,
        "missing_keys": missing,
        "extra_keys": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": value_mismatches,
        "semantic_match": not any(
            (
                missing,
                extra,
                shape_mismatches,
                dtype_mismatches,
                value_mismatches,
            )
        ),
        "comparison_layout": "canonical MLX image-runtime layout",
    }


def _validate_reproduction_manifest(
    manifest: dict[str, Any],
    *,
    official_sha256: str,
    reproduced_sha256: str,
) -> None:
    expected = {
        "architecture": "sam3-image",
        "source_repo": OFFICIAL_CHECKPOINT_REPO,
        "source_revision": OFFICIAL_CHECKPOINT_REVISION,
        "source_checkpoint_sha256": official_sha256,
        "output_sha256": reproduced_sha256,
        "mapped_count": CHECKPOINT_TENSOR_COUNT,
        "unmapped_keys": [],
        "converter_version": PACKAGE_VERSION,
    }
    drift = {
        field: {"manifest": manifest.get(field), "expected": value}
        for field, value in expected.items()
        if manifest.get(field) != value
    }
    if drift:
        raise ValueError(f"Reproduction manifest does not match release contract: {drift}")

    ignored_keys = manifest.get("ignored_keys")
    if not isinstance(ignored_keys, list) or any(
        not isinstance(key, str) or not key.startswith("tracker.")
        for key in ignored_keys
    ):
        raise ValueError(
            "Reproduction manifest ignored_keys must contain only explicit tracker keys."
        )
    if ignored_keys != sorted(set(ignored_keys)):
        raise ValueError(
            "Reproduction manifest ignored_keys must be sorted and unique."
        )

    dtype_counts = manifest.get("dtype_counts")
    if not isinstance(dtype_counts, dict) or not dtype_counts:
        raise ValueError("Reproduction manifest dtype_counts must be a non-empty object.")
    if any(
        not isinstance(dtype_name, str)
        or not dtype_name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for dtype_name, count in dtype_counts.items()
    ):
        raise ValueError(
            "Reproduction manifest dtype_counts must map dtype names to "
            "non-negative integer counts."
        )
    if sum(dtype_counts.values()) != CHECKPOINT_TENSOR_COUNT:
        raise ValueError("Reproduction manifest dtype_counts do not cover all tensors.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--official-revision",
        default=OFFICIAL_CHECKPOINT_REVISION,
    )
    parser.add_argument("--published-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--published-revision",
        default=MLX_CHECKPOINT_REVISION,
    )
    parser.add_argument("--reproduced-checkpoint", type=Path, required=True)
    parser.add_argument("--reproduction-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        source_commit, _ = validate_attestation_only_worktree(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.out.suffix != ".json":
        raise SystemExit("--out must end in .json.")
    if args.official_revision != OFFICIAL_CHECKPOINT_REVISION:
        raise SystemExit(
            "Official checkpoint revision does not match the frozen release pin."
        )
    if args.published_revision != MLX_CHECKPOINT_REVISION:
        raise SystemExit(
            "Published checkpoint revision does not match the frozen release pin."
        )

    official_sha256 = sha256_path(args.official_checkpoint)
    published_sha256 = sha256_path(args.published_checkpoint)
    reproduced_sha256 = sha256_path(args.reproduced_checkpoint)
    if official_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise SystemExit(
            "Official checkpoint content does not match the frozen release pin."
        )
    if published_sha256 != MLX_CHECKPOINT_SHA256:
        raise SystemExit(
            "Published MLX checkpoint content does not match the frozen release pin."
        )

    try:
        reproduction_manifest = json.loads(args.reproduction_manifest.read_text())
        if not isinstance(reproduction_manifest, dict):
            raise ValueError("Reproduction manifest must be a JSON object.")
        _validate_reproduction_manifest(
            reproduction_manifest,
            official_sha256=official_sha256,
            reproduced_sha256=reproduced_sha256,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    comparison = _compare_tensors(
        mx.load(str(args.published_checkpoint)),
        mx.load(str(args.reproduced_checkpoint)),
    )
    counts_match_contract = (
        comparison["published_tensor_count"] == CHECKPOINT_TENSOR_COUNT
        and comparison["reproduced_tensor_count"] == CHECKPOINT_TENSOR_COUNT
        and comparison["exact_tensor_count"] == CHECKPOINT_TENSOR_COUNT
    )
    passed = comparison["semantic_match"] and counts_match_contract

    try:
        final_source_commit, _ = validate_attestation_only_worktree(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if final_source_commit != source_commit:
        raise SystemExit("sam3_mlx source commit changed during lineage execution.")

    report = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "lineage_runner_sha256": sha256_path(Path(__file__)),
        "converter_module_sha256": sha256_path(CONVERT_MODULE),
        "source": {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
            "checkpoint_sha256": official_sha256,
        },
        "published_artifact": {
            "repo": MLX_CHECKPOINT_REPO,
            "revision": MLX_CHECKPOINT_REVISION,
            "checkpoint_sha256": published_sha256,
        },
        "reproduction": {
            "checkpoint_sha256": reproduced_sha256,
            "manifest_sha256": sha256_path(args.reproduction_manifest),
            "converter_version": reproduction_manifest["converter_version"],
            "manifest": reproduction_manifest,
        },
        "comparison": comparison,
        "byte_identity_required": False,
        "byte_identity_note": (
            "Safetensors bytes may differ across serializer versions; release "
            "lineage requires identical tensor keys, shapes, dtypes, and values."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "source_commit": source_commit,
                "status": report["status"],
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
