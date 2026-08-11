#!/usr/bin/env python3
"""Run the pinned official SAM 3 image oracle and write comparison arrays.

This runner is separate from ``run_image_parity.py`` because the official
Torch runtime and the MLX package require distinct dependency environments.
The pinned upstream commit assumes CUDA during import and cache construction;
on CPU-only macOS this runner redirects only those construction-time tensors
to CPU, disables pinned-memory staging, and makes the unused Triton EDT path
fail fast.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import time
from pathlib import Path
from typing import cast

import numpy as np

from _oracle_runtime import (
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()
    checkpoint = cast(Path, args.checkpoint)
    image_path = cast(Path, args.image)
    cases_path = cast(Path, args.cases)
    output_path = cast(Path, args.out)
    confidence_threshold = cast(float, args.confidence_threshold)

    import torch
    from PIL import Image

    torch_runtime = cast(TorchRuntime, torch)
    originals: ConstructionAdapters = install_cpu_oracle_adapters(torch_runtime)
    model_builder = import_module("sam3.model_builder")
    processor_class = getattr(
        import_module("sam3.model.sam3_image_processor"), "Sam3Processor"
    )

    builder = cast(OracleModelBuilder, model_builder)
    processor_factory = cast(OracleProcessorFactory, processor_class)
    load_started = time.perf_counter()
    model: OracleModel = builder.build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device="cpu",
        enable_inst_interactivity=False,
    )
    restore_construction_adapters(torch_runtime, originals)
    load_s = time.perf_counter() - load_started

    image = Image.open(image_path).convert("RGB")
    specs: list[OracleCase] = validate_case_specs(
        json.loads(cases_path.read_text(encoding="utf-8"))
    )
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "cold_load_s": load_s,
        "precision": "torch.cpu.autocast.bfloat16",
        "image_sha256": _sha256(image_path),
        "case_spec_sha256": _sha256(cases_path),
        "cases": [],
        "cpu_adapters": [
            "sam3.model.edt replaced with fail-fast unused stub",
            "construction-time CUDA cache tensors redirected to CPU",
            "pin_memory disabled for CPU-only staging",
            (
                "global-attention RoPE frequencies recomputed with the official "
                "formula for non-1008 processor grids"
            ),
        ],
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

    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_oracle_arrays(output_path, arrays)
    print(json.dumps({"wrote": str(output_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
