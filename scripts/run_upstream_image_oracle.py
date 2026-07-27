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
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_cpu_oracle_adapters(torch) -> dict[str, Any]:
    edt_module = types.ModuleType("sam3.model.edt")

    def unavailable_edt(*_args, **_kwargs):
        raise RuntimeError("Triton EDT is unavailable in the image-only CPU oracle.")

    edt_module.edt_triton = unavailable_edt
    sys.modules["sam3.model.edt"] = edt_module

    originals = {
        "zeros": torch.zeros,
        "arange": torch.arange,
        "pin_memory": torch.Tensor.pin_memory,
    }

    def _cpu_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        updated = dict(kwargs)
        if str(updated.get("device", "")).startswith("cuda"):
            updated["device"] = "cpu"
        return updated

    torch.zeros = lambda *args, **kwargs: originals["zeros"](
        *args, **_cpu_kwargs(kwargs)
    )
    torch.arange = lambda *args, **kwargs: originals["arange"](
        *args, **_cpu_kwargs(kwargs)
    )
    torch.Tensor.pin_memory = lambda tensor, *_args, **_kwargs: tensor
    return originals


def _restore_construction_adapters(torch, originals: dict[str, Any]) -> None:
    torch.zeros = originals["zeros"]
    torch.arange = originals["arange"]


def _run_prompt(processor, state: dict[str, Any], spec: dict[str, Any]):
    processor.reset_all_prompts(state)
    prompt = spec["prompt"]
    if prompt is not None:
        state = processor.set_text_prompt(prompt, state)
    for geometric_prompt in spec["geometric_prompts"]:
        state = processor.add_geometric_prompt(
            geometric_prompt["box"],
            geometric_prompt["label"],
            state,
        )
    return state


def _set_official_global_rope_grid(model, resolution: int) -> None:
    """Recompute the official global-attention RoPE grid for a processor size."""

    grid_size = resolution // 14
    trunk = model.backbone.vision_backbone.trunk
    for block in trunk.blocks:
        if block.window_size != 0 or not block.attn.use_rope:
            continue
        attention = block.attn
        scale_pos = 1.0
        if attention.rope_interp:
            scale_pos = attention.rope_pt_size[0] / grid_size
        attention.freqs_cis = attention.compute_cis(
            end_x=grid_size,
            end_y=grid_size,
            scale_pos=scale_pos,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    args = parser.parse_args()

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
    specs = json.loads(args.cases.read_text())
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {
        "cold_load_s": load_s,
        "precision": "torch.cpu.autocast.bfloat16",
        "image_sha256": _sha256(args.image),
        "case_spec_sha256": _sha256(args.cases),
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

    arrays["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(json.dumps({"wrote": str(args.out), **metadata}, indent=2))


if __name__ == "__main__":
    main()
