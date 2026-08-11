#!/usr/bin/env python3
"""Generate replayable, source-bound official-vs-MLX SAM 3 image evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.parity_evidence import (  # noqa: E402
    compare_case,
    normalize_outputs,
    write_evidence_bundle,
)
from sam3_mlx.release_contract import (  # noqa: E402
    COMPARISON_ALGORITHM,
    EXPECTED_CASE_NAMES,
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    MLX_CHECKPOINT_SHA256,
    OFFICIAL_CHECKPOINT_REPO,
    OFFICIAL_CHECKPOINT_REVISION,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_CODE_REPO,
    OFFICIAL_CODE_REVISION,
    RELEASE_CONFIDENCE_THRESHOLD,
    RELEASE_IMAGES,
    RELEASE_THRESHOLDS,
    REPORT_SCHEMA_VERSION,
    JsonObject,
    OracleBindings,
    ReleaseImage,
    build_oracle_bindings,
    canonical_json_sha256,
    sha256_path,
    require_json_list,
    require_json_object,
    validate_exact_mapping,
)
from sam3_mlx.source_binding import validate_attestation_only_worktree  # noqa: E402
from scripts.run_image_parity import (  # noqa: E402
    case_specs,
    evidence_path,
    mlx_outputs,
    validate_official_checkout,
)
from scripts._oracle_runtime import OracleCase  # noqa: E402

HARDENED_ORACLE = REPO_ROOT / "scripts" / "run_upstream_image_oracle_hardened.py"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Release evidence must be written inside the repository: {resolved}."
        ) from exc


def _validate_profile_image(
    *,
    image_path: Path,
    image: Image.Image,
    profile: str,
    official_checkout: Path,
) -> ReleaseImage:
    observed: ReleaseImage = {
        "path": evidence_path(image_path, official_checkout=official_checkout),
        "sha256": sha256_path(image_path),
        "size": list(image.size),
    }
    validate_exact_mapping(
        observed,
        RELEASE_IMAGES[profile],
        label=f"{profile} image",
    )
    return observed


def _validate_oracle_archive(
    path: Path,
    *,
    expected_bindings: OracleBindings,
    expected_specs: list[OracleCase],
    image_size: tuple[int, int],
) -> tuple[JsonObject, list[dict[str, np.ndarray]]]:
    try:
        archive_context = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load oracle cache {path}: {exc}") from exc

    with archive_context as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("Oracle cache is missing metadata_json.")
        metadata = require_json_object(
            json.loads(str(archive["metadata_json"])), field="Oracle metadata"
        )
        bindings = require_json_object(
            metadata.get("bindings"), field="Oracle metadata bindings"
        )
        validate_exact_mapping(
            bindings, expected_bindings, label="oracle cache bindings"
        )
        expected_cache_key = canonical_json_sha256(expected_bindings)
        if metadata.get("cache_key") != expected_cache_key:
            raise ValueError(
                "Oracle cache key does not match its complete provenance bindings."
            )

        cases_metadata = require_json_list(
            metadata.get("cases"), field="Oracle metadata cases"
        )
        if len(cases_metadata) != len(expected_specs):
            raise ValueError("Oracle cache case metadata count is invalid.")
        expected_names = [spec["name"] for spec in expected_specs]
        observed_names = [
            require_json_object(case, field="Oracle case metadata").get("name")
            for case in cases_metadata
        ]
        if observed_names != expected_names:
            raise ValueError(
                "Oracle cache case profile mismatch: "
                f"observed={observed_names}, expected={expected_names}."
            )

        width, height = image_size
        outputs: list[dict[str, np.ndarray]] = []
        for index, (spec, case_metadata) in enumerate(
            zip(expected_specs, cases_metadata, strict=True)
        ):
            case_metadata = require_json_object(
                case_metadata, field=f"Oracle case {spec['name']!r} metadata"
            )
            if case_metadata.get("resolution") != spec["resolution"]:
                raise ValueError(
                    f"Oracle case {spec['name']!r} resolution metadata drifted."
                )
            try:
                raw = {
                    field: np.array(archive[f"case_{index}_{field}"], copy=True)
                    for field in ("masks", "boxes", "scores")
                }
            except KeyError as exc:
                raise ValueError(
                    f"Oracle cache is missing arrays for case {spec['name']!r}."
                ) from exc
            normalized = normalize_outputs(raw, label="official")
            if normalized["masks"].shape[1:] != (height, width):
                raise ValueError(
                    f"Oracle case {spec['name']!r} masks have spatial shape "
                    f"{normalized['masks'].shape[1:]}, expected {(height, width)}."
                )
            if case_metadata.get("detection_count") != len(normalized["scores"]):
                raise ValueError(
                    f"Oracle case {spec['name']!r} detection metadata does not "
                    "match its stored arrays."
                )
            outputs.append(normalized)
    return metadata, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--mlx-checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--oracle-cache", type=Path)
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(EXPECTED_CASE_NAMES)),
        required=True,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=RELEASE_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()

    if not math.isclose(
        args.confidence_threshold,
        RELEASE_CONFIDENCE_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise SystemExit(
            "Release evidence confidence threshold is frozen at "
            f"{RELEASE_CONFIDENCE_THRESHOLD}."
        )
    if isinstance(args.repetitions, bool) or args.repetitions < 5:
        raise SystemExit("--repetitions must be at least 5 for release evidence.")
    if args.out.suffix != ".json":
        raise SystemExit("--out must end in .json.")
    if args.evidence_out.suffix != ".npz":
        raise SystemExit("--evidence-out must end in .npz.")

    try:
        source_commit, _ = validate_attestation_only_worktree(REPO_ROOT)
        report_relative = _repo_relative(args.out)
        evidence_relative = _repo_relative(args.evidence_out)
        if report_relative == evidence_relative:
            raise ValueError("Report and raw evidence paths must be distinct.")
        revision = validate_official_checkout(
            args.official_checkout,
            expected_revision=OFFICIAL_CODE_REVISION,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    official_checkpoint_sha256 = sha256_path(args.official_checkpoint)
    if official_checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise SystemExit(
            "Official checkpoint SHA-256 does not match the frozen release pin: "
            f"{official_checkpoint_sha256}."
        )
    mlx_checkpoint_sha256 = sha256_path(args.mlx_checkpoint)
    if mlx_checkpoint_sha256 != MLX_CHECKPOINT_SHA256:
        raise SystemExit(
            "MLX checkpoint SHA-256 does not match the frozen release pin: "
            f"{mlx_checkpoint_sha256}."
        )

    with Image.open(args.image) as source_image:
        image = source_image.convert("RGB")
    try:
        image_record = _validate_profile_image(
            image_path=args.image,
            image=image,
            profile=args.profile,
            official_checkout=args.official_checkout,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    specs = case_specs(image, args.profile)
    observed_names = tuple(spec["name"] for spec in specs)
    if observed_names != EXPECTED_CASE_NAMES[args.profile]:
        raise SystemExit(f"Case matrix drifted for {args.profile}: {observed_names}.")

    oracle_runner_sha256 = sha256_path(HARDENED_ORACLE)
    with tempfile.TemporaryDirectory(prefix="sam3-mlx-parity-v2-") as temp_dir:
        temp = Path(temp_dir)
        cases_path = temp / "cases.json"
        cases_path.write_text(json.dumps(specs, indent=2) + "\n")
        oracle_path = args.oracle_cache or temp / "official.npz"
        expected_bindings = build_oracle_bindings(
            image_sha256=image_record["sha256"],
            case_spec_sha256=sha256_path(cases_path),
            confidence_threshold=args.confidence_threshold,
            oracle_runner_sha256=oracle_runner_sha256,
        )

        if not oracle_path.exists():
            env = dict(os.environ)
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = os.pathsep.join(
                value
                for value in (str(args.official_checkout), existing_pythonpath)
                if value
            )
            subprocess.run(
                [
                    str(args.official_python),
                    str(HARDENED_ORACLE),
                    "--official-checkout",
                    str(args.official_checkout),
                    "--checkpoint",
                    str(args.official_checkpoint),
                    "--image",
                    str(args.image),
                    "--cases",
                    str(cases_path),
                    "--out",
                    str(oracle_path),
                    "--confidence-threshold",
                    str(args.confidence_threshold),
                ],
                cwd=REPO_ROOT,
                env=env,
                check=True,
            )
        try:
            oracle_metadata, official_outputs = _validate_oracle_archive(
                oracle_path,
                expected_bindings=expected_bindings,
                expected_specs=specs,
                image_size=image.size,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc

    runtime_outputs, performance = mlx_outputs(
        checkpoint=args.mlx_checkpoint,
        image=image,
        specs=specs,
        confidence_threshold=args.confidence_threshold,
        repetitions=args.repetitions,
    )
    cases = [
        compare_case(
            spec,
            official,
            mlx,
            thresholds=RELEASE_THRESHOLDS,
        )
        for spec, official, mlx in zip(
            specs, official_outputs, runtime_outputs, strict=True
        )
    ]
    status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"

    evidence_metadata = {
        "source_commit": source_commit,
        "profile": args.profile,
        "report_path": report_relative,
        "image": image_record,
        "case_specs": specs,
        "oracle_cache_key": oracle_metadata["cache_key"],
        "official_code": {
            "repo": OFFICIAL_CODE_REPO,
            "revision": revision,
        },
        "official_checkpoint": {
            "repo": OFFICIAL_CHECKPOINT_REPO,
            "revision": OFFICIAL_CHECKPOINT_REVISION,
            "sha256": official_checkpoint_sha256,
        },
        "converted_checkpoint": {
            "repo": MLX_CHECKPOINT_REPO,
            "revision": MLX_CHECKPOINT_REVISION,
            "sha256": mlx_checkpoint_sha256,
        },
        "confidence_threshold": args.confidence_threshold,
        "thresholds": RELEASE_THRESHOLDS,
    }
    write_evidence_bundle(
        args.evidence_out,
        metadata=evidence_metadata,
        official_outputs=official_outputs,
        mlx_outputs=runtime_outputs,
    )

    try:
        final_source_commit, _ = validate_attestation_only_worktree(REPO_ROOT)
        final_revision = validate_official_checkout(
            args.official_checkout,
            expected_revision=OFFICIAL_CODE_REVISION,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if final_source_commit != source_commit:
        raise SystemExit("sam3_mlx source commit changed during parity execution.")
    if final_revision != revision:
        raise SystemExit("Official checkout revision changed during parity execution.")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "status": status,
        "comparison_algorithm": COMPARISON_ALGORITHM,
        "official_code": evidence_metadata["official_code"],
        "official_checkpoint": evidence_metadata["official_checkpoint"],
        "converted_checkpoint": evidence_metadata["converted_checkpoint"],
        "image": image_record,
        "confidence_threshold": args.confidence_threshold,
        "case_profile": args.profile,
        "thresholds": RELEASE_THRESHOLDS,
        "cases": cases,
        "oracle": oracle_metadata,
        "raw_evidence": {
            "path": evidence_relative,
            "sha256": sha256_path(args.evidence_out),
        },
        "performance": performance,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "evidence": str(args.evidence_out),
                "source_commit": source_commit,
                "status": status,
                "comparison_algorithm": COMPARISON_ALGORITHM,
            },
            indent=2,
        )
    )
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
