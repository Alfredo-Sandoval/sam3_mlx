#!/usr/bin/env python3
"""Run the pinned official SAM 3 CPU oracle with cache-complete provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.release_contract import (  # noqa: E402
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_CODE_REVISION,
    build_oracle_bindings,
    canonical_json_sha256,
    sha256_path,
)
from run_upstream_image_oracle import (  # noqa: E402
    _install_cpu_oracle_adapters,
    _restore_construction_adapters,
    _run_prompt,
    _set_official_global_rope_grid,
)


def _validate_official_checkout(checkout: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if revision != OFFICIAL_CODE_REVISION:
        raise ValueError(
            f"Official checkout must be {OFFICIAL_CODE_REVISION}, found {revision}."
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status.strip():
        raise ValueError(
            "Official checkout must be clean; git status reported:\n"
            f"{status.rstrip()}"
        )
    submodules = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    dirty_submodules = [line for line in submodules if line[:1] in {"+", "-", "U"}]
    if dirty_submodules:
        raise ValueError(
            "Official checkout submodules must match the pinned commit:\n"
            + "\n".join(dirty_submodules)
        )
    return revision


def _validate_case_specs(specs: Any) -> list[dict[str, Any]]:
    if not isinstance(specs, list) or not specs:
        raise ValueError("Oracle cases must be a non-empty JSON list.")
    names: set[str] = set()
    validated = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"Oracle case {index} must be an object.")
        required = {"name", "resolution", "prompt", "geometric_prompts"}
        if set(spec) != required:
            raise ValueError(
                f"Oracle case {index} fields must be exactly {sorted(required)}."
            )
        name = spec["name"]
        resolution = spec["resolution"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Oracle case name is invalid or duplicated: {name!r}.")
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            raise ValueError(f"Oracle case {name!r} resolution must be an integer.")
        if resolution <= 0 or resolution % 14 != 0:
            raise ValueError(
                f"Oracle case {name!r} resolution must be a positive multiple of 14."
            )
        if not isinstance(spec["geometric_prompts"], list):
            raise ValueError(f"Oracle case {name!r} geometric_prompts must be a list.")
        names.add(name)
        validated.append(spec)
    return validated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        revision = _validate_official_checkout(args.official_checkout)
        specs = _validate_case_specs(json.loads(args.cases.read_text()))
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    checkpoint_sha256 = sha256_path(args.checkpoint)
    if checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise SystemExit(
            "Official checkpoint SHA-256 does not match the frozen release pin: "
            f"{checkpoint_sha256}."
        )
    image_sha256 = sha256_path(args.image)
    case_spec_sha256 = sha256_path(args.cases)
    runner_sha256 = sha256_path(Path(__file__))
    bindings = build_oracle_bindings(
        image_sha256=image_sha256,
        case_spec_sha256=case_spec_sha256,
        confidence_threshold=args.confidence_threshold,
        oracle_runner_sha256=runner_sha256,
    )

    import torch
    from PIL import Image

    originals = _install_cpu_oracle_adapters(torch)
    import sam3.model_builder as model_builder
    from sam3.model.sam3_image_processor import Sam3Processor

    load_started = time.perf_counter()
    model = model_builder.build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device="cpu",
        enable_inst_interactivity=False,
    )
    _restore_construction_adapters(torch, originals)
    load_s = time.perf_counter() - load_started

    image = Image.open(args.image).convert("RGB")
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "schema_version": bindings["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings,
        "cache_key": canonical_json_sha256(bindings),
        "official_code": bindings["official_code"],
        "official_checkpoint": bindings["official_checkpoint"],
        "image_sha256": image_sha256,
        "case_spec_sha256": case_spec_sha256,
        "confidence_threshold": float(args.confidence_threshold),
        "precision": bindings["precision"],
        "cpu_adapters": bindings["cpu_adapters"],
        "oracle_runner_sha256": runner_sha256,
        "cold_load_s": load_s,
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "cases": [],
    }

    states: dict[int, dict[str, Any]] = {}
    processors: dict[int, Any] = {}
    torch.Tensor.pin_memory = lambda tensor, *_args, **_kwargs: tensor
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            for index, spec in enumerate(specs):
                resolution = int(spec["resolution"])
                if resolution not in states:
                    _set_official_global_rope_grid(model, resolution)
                    processor = Sam3Processor(
                        model,
                        device="cpu",
                        resolution=resolution,
                        confidence_threshold=args.confidence_threshold,
                    )
                    started = time.perf_counter()
                    states[resolution] = processor.set_image(image)
                    processors[resolution] = processor
                    image_latency_s = time.perf_counter() - started
                else:
                    processor = processors[resolution]
                    image_latency_s = 0.0

                started = time.perf_counter()
                state = _run_prompt(processor, states[resolution], spec)
                prompt_latency_s = time.perf_counter() - started
                prefix = f"case_{index}"
                arrays[f"{prefix}_masks"] = (
                    state["masks"].detach().cpu().numpy().astype(np.bool_)
                )
                arrays[f"{prefix}_boxes"] = (
                    state["boxes"].detach().float().cpu().numpy()
                )
                arrays[f"{prefix}_scores"] = (
                    state["scores"].detach().float().cpu().numpy()
                )
                metadata["cases"].append(
                    {
                        "name": spec["name"],
                        "resolution": resolution,
                        "detection_count": int(len(state["scores"])),
                        "image_latency_s": image_latency_s,
                        "prompt_latency_s": prompt_latency_s,
                    }
                )
                print(
                    json.dumps(
                        {
                            "case": spec["name"],
                            "detections": len(state["scores"]),
                            "elapsed_s": image_latency_s + prompt_latency_s,
                        }
                    ),
                    flush=True,
                )
    finally:
        torch.Tensor.pin_memory = originals["pin_memory"]

    # Recheck after inference so a checkout mutation during a long oracle run
    # cannot silently survive into release evidence.
    try:
        final_revision = _validate_official_checkout(args.official_checkout)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if final_revision != revision:
        raise SystemExit("Official checkout revision changed during oracle execution.")

    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(
        json.dumps(
            {
                "wrote": str(args.out),
                "cache_key": metadata["cache_key"],
                "case_count": len(specs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
