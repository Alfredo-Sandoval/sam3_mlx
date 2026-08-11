#!/usr/bin/env python3
"""Generate hard-pinned, source-bound semantic checkpoint-lineage evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.release_contract import (  # noqa: E402
    CHECKPOINT_TENSOR_COUNT,
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    MLX_CHECKPOINT_SHA256,
    OFFICIAL_CHECKPOINT_REPO,
    OFFICIAL_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_SHA256,
    PACKAGE_VERSION,
    JsonObject,
    require_json_object,
    sha256_path,
)
from sam3_mlx.source_binding import validate_attestation_only_worktree  # noqa: E402
from scripts.validate_checkpoint_lineage import (  # noqa: E402
    TensorComparison,
    compare_tensors,
    load_checkpoint_tensors,
)

LINEAGE_SCHEMA_VERSION = 2
CONVERT_MODULE = REPO_ROOT / "sam3_mlx" / "convert.py"


def _validate_reproduction_manifest(
    manifest: JsonObject,
    *,
    official_sha256: str,
    reproduced_sha256: str,
) -> None:
    expected: JsonObject = {
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
        raise ValueError(
            f"Reproduction manifest does not match release contract: {drift}"
        )

    ignored_value = manifest.get("ignored_keys")
    if not isinstance(ignored_value, list):
        raise ValueError(
            "Reproduction manifest ignored_keys must contain only explicit tracker keys."
        )
    ignored_keys: list[str] = []
    for key_value in cast(list[object], ignored_value):
        if not isinstance(key_value, str) or not key_value.startswith("tracker."):
            raise ValueError(
                "Reproduction manifest ignored_keys must contain only explicit tracker keys."
            )
        ignored_keys.append(key_value)
    if ignored_keys != sorted(set(ignored_keys)):
        raise ValueError(
            "Reproduction manifest ignored_keys must be sorted and unique."
        )

    dtype_counts = require_json_object(
        manifest.get("dtype_counts"), field="Reproduction manifest dtype_counts"
    )
    if not dtype_counts:
        raise ValueError(
            "Reproduction manifest dtype_counts must be a non-empty object."
        )
    normalized_dtype_counts: dict[str, int] = {}
    for dtype_name, count in dtype_counts.items():
        if (
            not dtype_name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError(
                "Reproduction manifest dtype_counts must map dtype names to "
                "non-negative integer counts."
            )
        normalized_dtype_counts[dtype_name] = count
    if sum(normalized_dtype_counts.values()) != CHECKPOINT_TENSOR_COUNT:
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
        reproduction_manifest = require_json_object(
            json.loads(args.reproduction_manifest.read_text()),
            field="Reproduction manifest",
        )
        _validate_reproduction_manifest(
            reproduction_manifest,
            official_sha256=official_sha256,
            reproduced_sha256=reproduced_sha256,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    comparison: TensorComparison = compare_tensors(
        load_checkpoint_tensors(args.published_checkpoint),
        load_checkpoint_tensors(args.reproduced_checkpoint),
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

    report: JsonObject = {
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
