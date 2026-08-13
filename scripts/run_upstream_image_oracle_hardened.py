#!/usr/bin/env python3
"""Run the pinned official SAM 3 CPU oracle with cache-complete provenance."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import import_module
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_CONTRACT_PATH = REPO_ROOT / "sam3_mlx" / "release_contract.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.release_contract import (  # noqa: E402
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_CODE_REVISION,
    build_oracle_bindings,
    canonical_json_sha256,
    sha256_path,
)
from _oracle_runtime import (  # noqa: E402
    ConstructionAdapters,
    OracleCase,
    OracleModel,
    OracleModelBuilder,
    OracleProcessor,
    OracleProcessorFactory,
    OracleState,
    TorchRuntime,
    install_cpu_oracle_adapters,
    restore_construction_adapters,
    run_prompt,
    save_oracle_arrays,
    set_official_global_rope_grid,
    validate_case_specs,
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
            f"Official checkout must be clean; git status reported:\n{status.rstrip()}"
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


def _require_module_from_checkout(
    module: types.ModuleType, checkout: Path, *, label: str
) -> str:
    source = getattr(module, "__file__", None)
    if not source:
        raise ValueError(f"{label} has no import source path.")
    resolved = Path(source).resolve()
    try:
        relative = resolved.relative_to(checkout.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{label} was imported from {resolved}, outside {checkout.resolve()}."
        ) from exc
    return str(relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()
    official_checkout = cast(Path, args.official_checkout)
    checkpoint = cast(Path, args.checkpoint)
    image_path = cast(Path, args.image)
    cases_path = cast(Path, args.cases)
    output_path = cast(Path, args.out)
    confidence_threshold = cast(float, args.confidence_threshold)

    if output_path.suffix != ".npz":
        raise SystemExit("--out must end in .npz.")
    if not np.isfinite(confidence_threshold) or not (
        0.0 <= confidence_threshold <= 1.0
    ):
        raise SystemExit("--confidence-threshold must be finite and within [0, 1].")
    try:
        revision = _validate_official_checkout(official_checkout)
        specs: list[OracleCase] = validate_case_specs(
            json.loads(cases_path.read_text(encoding="utf-8"))
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    checkpoint_sha256 = sha256_path(checkpoint)
    if checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise SystemExit(
            "Official checkpoint SHA-256 does not match the frozen release pin: "
            f"{checkpoint_sha256}."
        )
    image_sha256 = sha256_path(image_path)
    case_spec_sha256 = sha256_path(cases_path)
    runner_sha256 = sha256_path(Path(__file__))
    release_contract_sha256 = sha256_path(RELEASE_CONTRACT_PATH)
    bindings = build_oracle_bindings(
        image_sha256=image_sha256,
        case_spec_sha256=case_spec_sha256,
        confidence_threshold=confidence_threshold,
        oracle_runner_sha256=runner_sha256,
        release_contract_sha256=release_contract_sha256,
    )

    from PIL import Image

    torch_runtime = cast(TorchRuntime, import_module("torch"))
    originals: ConstructionAdapters = install_cpu_oracle_adapters(torch_runtime)
    model_builder = import_module("sam3.model_builder")
    processor_class = getattr(
        import_module("sam3.model.sam3_image_processor"), "Sam3Processor"
    )
    processor_module_name = getattr(processor_class, "__module__", None)
    if not isinstance(processor_module_name, str):
        raise SystemExit("Sam3Processor has no module name.")
    builder = cast(OracleModelBuilder, model_builder)
    processor_factory = cast(OracleProcessorFactory, processor_class)

    try:
        model_builder_source = _require_module_from_checkout(
            model_builder,
            official_checkout,
            label="sam3.model_builder",
        )
        processor_module = sys.modules[processor_module_name]
        processor_source = _require_module_from_checkout(
            processor_module,
            official_checkout,
            label=processor_module_name,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    load_started = time.perf_counter()
    try:
        model: OracleModel = builder.build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device="cpu",
            enable_inst_interactivity=False,
        )
    finally:
        restore_construction_adapters(torch_runtime, originals)
    load_s = time.perf_counter() - load_started

    image = Image.open(image_path).convert("RGB")
    arrays: dict[str, object] = {}
    metadata: dict[str, object] = {
        "schema_version": bindings["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bindings": bindings,
        "cache_key": canonical_json_sha256(bindings),
        "official_code": bindings["official_code"],
        "official_checkpoint": bindings["official_checkpoint"],
        "image_sha256": image_sha256,
        "case_spec_sha256": case_spec_sha256,
        "confidence_threshold": confidence_threshold,
        "precision": bindings["precision"],
        "cpu_adapters": bindings["cpu_adapters"],
        "oracle_runner_sha256": runner_sha256,
        "release_contract_sha256": release_contract_sha256,
        "cold_load_s": load_s,
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch_runtime.__version__),
        },
        "official_imports": {
            "model_builder": model_builder_source,
            "image_processor": processor_source,
        },
        "cases": [],
    }

    states: dict[int, OracleState] = {}
    processors: dict[int, OracleProcessor] = {}
    try:
        with torch_runtime.autocast("cpu", dtype=torch_runtime.bfloat16):
            for index, spec in enumerate(specs):
                resolution = spec["resolution"]
                if resolution not in states:
                    set_official_global_rope_grid(model, resolution)
                    processor = processor_factory(
                        model,
                        device="cpu",
                        resolution=resolution,
                        confidence_threshold=confidence_threshold,
                    )
                    started = time.perf_counter()
                    states[resolution] = processor.set_image(image)
                    processors[resolution] = processor
                    image_latency_s = time.perf_counter() - started
                else:
                    processor = processors[resolution]
                    image_latency_s = 0.0

                started = time.perf_counter()
                state = run_prompt(processor, states[resolution], spec)
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
                cases_metadata = cast(list[object], metadata["cases"])
                cases_metadata.append(
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
        torch_runtime.Tensor.pin_memory = originals.pin_memory

    try:
        final_revision = _validate_official_checkout(official_checkout)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if final_revision != revision:
        raise SystemExit("Official checkout revision changed during oracle execution.")

    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_oracle_arrays(output_path, arrays)
    print(
        json.dumps(
            {
                "wrote": str(output_path),
                "cache_key": metadata["cache_key"],
                "case_count": len(specs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
