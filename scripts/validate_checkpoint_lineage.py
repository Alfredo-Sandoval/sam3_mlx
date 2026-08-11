#!/usr/bin/env python3
"""Attest semantic lineage from an official checkpoint conversion to an artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, TypedDict, cast

import mlx.core as mx

from sam3_mlx.convert import normalize_sam3_image_weight_layout
from sam3_mlx.release_contract import JsonObject, require_json_object


class ShapeMismatch(TypedDict):
    key: str
    published: list[int]
    reproduced: list[int]


class DtypeMismatch(TypedDict):
    key: str
    published: str
    reproduced: str


class TensorComparison(TypedDict):
    published_tensor_count: int
    reproduced_tensor_count: int
    exact_tensor_count: int
    missing_keys: list[str]
    extra_keys: list[str]
    shape_mismatches: list[ShapeMismatch]
    dtype_mismatches: list[DtypeMismatch]
    value_mismatches: list[str]
    semantic_match: bool
    comparison_layout: str


class _MlxLoad(Protocol):
    def __call__(self, file: str | Path, /) -> object: ...


_mlx_load = cast(_MlxLoad, getattr(mx, "load"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_tensors(
    published: dict[str, mx.array],
    reproduced: dict[str, mx.array],
) -> TensorComparison:
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
    shape_mismatches: list[ShapeMismatch] = []
    dtype_mismatches: list[DtypeMismatch] = []
    value_mismatches: list[str] = []
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
        if bool(mx.all(mx.equal(left, right)).item()):
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


def load_checkpoint_tensors(path: Path) -> dict[str, mx.array]:
    """Load and validate the mapping shape required by lineage comparison."""

    payload = _mlx_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {path} must contain a tensor mapping.")
    raw_payload = cast(dict[object, object], payload)
    tensors: dict[str, mx.array] = {}
    for key, value in raw_payload.items():
        if not isinstance(key, str) or not isinstance(value, mx.array):
            raise ValueError(f"Checkpoint {path} must map string keys to MLX arrays.")
        tensors[key] = value
    return tensors


_compare_tensors = compare_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-revision", required=True)
    parser.add_argument("--published-checkpoint", type=Path, required=True)
    parser.add_argument("--published-revision", required=True)
    parser.add_argument("--reproduced-checkpoint", type=Path, required=True)
    parser.add_argument("--reproduction-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reproduction_manifest = require_json_object(
        json.loads(args.reproduction_manifest.read_text()),
        field="Reproduction manifest",
    )
    official_sha256 = _sha256(args.official_checkpoint)
    reproduced_sha256 = _sha256(args.reproduced_checkpoint)
    expected_manifest = {
        "source_revision": args.official_revision,
        "source_checkpoint_sha256": official_sha256,
        "output_sha256": reproduced_sha256,
    }
    mismatches = {
        field: {
            "manifest": reproduction_manifest.get(field),
            "observed": expected,
        }
        for field, expected in expected_manifest.items()
        if reproduction_manifest.get(field) != expected
    }
    if mismatches:
        raise SystemExit(f"Reproduction manifest mismatch: {mismatches}")

    comparison = compare_tensors(
        load_checkpoint_tensors(args.published_checkpoint),
        load_checkpoint_tensors(args.reproduced_checkpoint),
    )
    report: JsonObject = {
        "schema_version": 1,
        "status": "passed" if comparison["semantic_match"] else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repo": "facebook/sam3",
            "revision": args.official_revision,
            "checkpoint_sha256": official_sha256,
        },
        "published_artifact": {
            "repo": "mlx-community/sam3-image",
            "revision": args.published_revision,
            "checkpoint_sha256": _sha256(args.published_checkpoint),
        },
        "reproduction": {
            "checkpoint_sha256": reproduced_sha256,
            "manifest_sha256": _sha256(args.reproduction_manifest),
            "converter_version": reproduction_manifest["converter_version"],
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
    print(json.dumps({"wrote": str(args.out), "status": report["status"]}, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
