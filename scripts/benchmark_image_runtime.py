#!/usr/bin/env python3
"""Benchmark synchronized SAM 3 image inference and compare durable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Protocol, cast

import mlx.core as mx
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sam3_mlx  # noqa: E402
from sam3_mlx.benchmarking import (  # noqa: E402
    BenchmarkOperation,
    IMAGE_BENCHMARK_SCHEMA,
    IMAGE_BENCHMARK_SUBSTAGE_SCHEMA,
    RegressionThreshold,
    TimingProtocol,
    aggregate_stage_group,
    compare_benchmark_artifacts,
    profile_operations,
    summarize_samples,
    synchronized_samples,
)
from sam3_mlx.mlx_runtime import synchronize_completed  # noqa: E402
from sam3_mlx.model.necks import Sam3DualViTDetNeck, output_levels_for_scalp  # noqa: E402
from sam3_mlx.model.sam3_image import Sam3Image  # noqa: E402
from sam3_mlx.model.sam3_image_processor import (  # noqa: E402
    ProcessorState,
    Sam3Processor,
)
from sam3_mlx.model.vitdet import Block, ViT  # noqa: E402
from sam3_mlx.model.vl_combiner import SAM3VLBackbone  # noqa: E402
from sam3_mlx.release_contract import JsonObject  # noqa: E402
from sam3_mlx.source_binding import git_commit  # noqa: E402


class _Parameters(Protocol):
    def parameters(self) -> object: ...


DEFAULT_METRICS = (
    "resolutions.504.full_image_text.median_s",
    "resolutions.504.full_cached_text.median_s",
    "resolutions.504.set_image.median_s",
    "resolutions.672.full_image_text.median_s",
    "resolutions.672.full_cached_text.median_s",
    "resolutions.672.set_image.median_s",
    "resolutions.1008.full_image_text.median_s",
    "resolutions.1008.full_cached_text.median_s",
    "resolutions.1008.set_image.median_s",
    "peak_active_memory_bytes",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image(args: argparse.Namespace) -> tuple[Image.Image, JsonObject]:
    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        return image, {
            "kind": "file",
            "sha256": _sha256_file(args.image),
            "size": list(image.size),
        }
    rng = np.random.default_rng(args.synthetic_seed)
    array = rng.integers(
        0,
        256,
        size=(args.synthetic_height, args.synthetic_width, 3),
        dtype=np.uint8,
    )
    return Image.fromarray(array, "RGB"), {
        "kind": "synthetic-rgb-u8",
        "seed": args.synthetic_seed,
        "sha256": _sha256_bytes(array.tobytes()),
        "size": [args.synthetic_width, args.synthetic_height],
    }


def _sysctl(name: str) -> str:
    return subprocess.run(
        ["sysctl", "-n", name],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _environment() -> JsonObject:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": _sysctl("machdep.cpu.brand_string"),
        "memory_bytes": int(_sysctl("hw.memsize")),
        "python_version": platform.python_version(),
        "mlx_version": importlib.metadata.version("mlx"),
    }


def _sync_state(state: ProcessorState) -> None:
    synchronize_completed(state["masks"], state["boxes"], state["scores"])


def _sync_value(value: object) -> None:
    synchronize_completed(value)


def _require_dual_neck(
    model: Sam3Image,
) -> tuple[SAM3VLBackbone, Sam3DualViTDetNeck, ViT]:
    backbone = model.backbone
    if not isinstance(backbone, SAM3VLBackbone):
        raise TypeError("Image model backbone must be SAM3VLBackbone")
    neck = backbone.vision_backbone
    if not isinstance(neck, Sam3DualViTDetNeck):
        raise TypeError("Vision backbone must be Sam3DualViTDetNeck")
    trunk = neck.trunk
    if not isinstance(trunk, ViT):
        raise TypeError("Neck trunk must be ViT")
    return backbone, neck, trunk


def _block_category(block: Block) -> str:
    return "global" if block.window_size == 0 else "window"


def _profile_vision_substages(
    model: Sam3Image,
    image: Image.Image,
    *,
    resolution: int,
    prompt: str,
    confidence_threshold: float,
    protocol: TimingProtocol,
    compile_policy: str,
) -> JsonObject:
    backbone, neck, trunk = _require_dual_neck(model)
    processor = Sam3Processor(
        model, resolution=resolution, confidence_threshold=confidence_threshold
    )
    image_tensor = processor.transform(image)[None]
    synchronize_completed(image_tensor)
    tokens, _s, _nested, _mask = trunk._prepare_tokens(image_tensor)
    synchronize_completed(tokens)

    block_inputs: list[mx.array] = []
    current = tokens
    for block in trunk.blocks:
        if not isinstance(block, Block):
            raise TypeError("ViT.blocks must contain Block modules")
        block_inputs.append(current)
        current = block(current)
        synchronize_completed(current)

    trunk_features = trunk(image_tensor)[-1]
    if not isinstance(trunk_features, mx.array):
        raise TypeError("ViT must return a feature array for image tensors")
    neck_input = mx.transpose(trunk_features, axes=(0, 2, 3, 1))
    synchronize_completed(neck_input)
    kept_levels = output_levels_for_scalp(len(neck.convs), scalp=backbone.scalp)
    retained = len(neck.convs) if kept_levels is None else kept_levels

    operations: dict[str, BenchmarkOperation[object]] = {
        "preprocessing": BenchmarkOperation(
            run=lambda: processor.transform(image)[None],
            synchronize=_sync_value,
        ),
        "patch_embedding": BenchmarkOperation(
            run=lambda: trunk._prepare_tokens(image_tensor)[0],
            synchronize=_sync_value,
        ),
    }
    block_names: list[str] = []
    window_names: list[str] = []
    global_names: list[str] = []
    for index, block in enumerate(trunk.blocks):
        if not isinstance(block, Block):
            raise TypeError("ViT.blocks must contain Block modules")
        name = f"vit_block_{index}"
        block_input = block_inputs[index]
        operations[name] = BenchmarkOperation(
            run=lambda module=block, tokens=block_input: module(tokens),
            synchronize=_sync_value,
        )
        block_names.append(name)
        if _block_category(block) == "window":
            window_names.append(name)
        else:
            global_names.append(name)

    neck_names: list[str] = []
    for index, conv in enumerate(neck.convs):
        name = f"neck_head_{index}"
        operations[name] = BenchmarkOperation(
            run=lambda head=conv: head(neck_input),
            synchronize=_sync_value,
        )
        neck_names.append(name)

    text_state = processor.set_image(image)

    def encode_text() -> ProcessorState:
        processor.clear_text_cache()
        return processor.set_text_prompt(prompt, text_state, run_grounding=False)

    processor.set_text_prompt(prompt, text_state, run_grounding=False)
    operations["text_encoding"] = BenchmarkOperation(
        run=encode_text,
        synchronize=_sync_value,
    )
    operations["text_encoding_repeated"] = BenchmarkOperation(
        run=lambda: processor.set_text_prompt(prompt, text_state, run_grounding=False),
        synchronize=_sync_value,
    )
    operations["grounding"] = BenchmarkOperation(
        run=lambda: processor.run_grounding(text_state),
        synchronize=_sync_state,
    )

    mx.reset_peak_memory()
    active_before = int(mx.get_active_memory())
    timed = profile_operations(operations, protocol=protocol)
    blocks: list[JsonObject] = []
    for index, name in enumerate(block_names):
        block = trunk.blocks[index]
        if not isinstance(block, Block):
            raise TypeError("ViT.blocks must contain Block modules")
        summary = dict(timed[name])
        summary["index"] = index
        summary["category"] = _block_category(block)
        blocks.append(summary)
    neck_heads: list[JsonObject] = []
    for index, name in enumerate(neck_names):
        summary = dict(timed[name])
        summary["index"] = index
        summary["scale"] = float(neck.scale_factors[index])
        summary["retained"] = index < retained
        neck_heads.append(summary)

    return {
        "schema_version": IMAGE_BENCHMARK_SUBSTAGE_SCHEMA,
        "resolution": resolution,
        "compile_policy": compile_policy,
        "warmup_runs": protocol.warmup_runs,
        "repetitions": protocol.repetitions,
        "preprocessing": timed["preprocessing"],
        "patch_embedding": timed["patch_embedding"],
        "vit_blocks": blocks,
        "vit_block_groups": {
            "window": aggregate_stage_group(
                timed, category="window", member_names=window_names
            ),
            "global": aggregate_stage_group(
                timed, category="global", member_names=global_names
            ),
        },
        "neck_heads": neck_heads,
        "text_encoding": timed["text_encoding"],
        "text_encoding_repeated": timed["text_encoding_repeated"],
        "grounding": timed["grounding"],
        "active_memory_bytes": active_before,
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
    }


def _text_outputs(state: ProcessorState) -> dict[str, mx.array]:
    backbone_out = state.get("backbone_out")
    if not isinstance(backbone_out, dict):
        raise TypeError("Processor state backbone_out must be a dict")
    outputs: dict[str, mx.array] = {}
    for key, value in cast(dict[object, object], backbone_out).items():
        if isinstance(key, str) and key.startswith("language_"):
            if not isinstance(value, mx.array):
                raise TypeError("Processor language outputs must be MLX arrays")
            outputs[key] = value
    if not outputs:
        raise RuntimeError("Processor did not produce language outputs")
    return outputs


def _benchmark_resolution(
    model: Sam3Image,
    image: Image.Image,
    *,
    resolution: int,
    prompt: str,
    confidence_threshold: float,
    protocol: TimingProtocol,
) -> JsonObject:
    processor = Sam3Processor(
        model, resolution=resolution, confidence_threshold=confidence_threshold
    )

    preprocess = BenchmarkOperation(
        run=lambda: processor.transform(image)[None],
        synchronize=_sync_value,
    )
    set_image = BenchmarkOperation(
        run=lambda: processor.set_image(image),
        synchronize=lambda _state: mx.synchronize(),
    )

    seed_state = processor.set_image(image)
    seed_state = processor.set_text_prompt(prompt, seed_state, run_grounding=False)
    cached_text = _text_outputs(seed_state)
    synchronize_completed(cached_text)

    def full_first_text() -> ProcessorState:
        processor.clear_text_cache()
        state = processor.set_image(image)
        return processor.set_text_prompt(prompt, state)

    def full_image_text() -> ProcessorState:
        state = processor.set_image(image)
        return processor.set_text_prompt(prompt, state)

    def full_cached_text() -> ProcessorState:
        state = processor.set_image(image)
        return processor.set_text_prompt(prompt, state, text_outputs=cached_text)

    return {
        "preprocess": summarize_samples(
            synchronized_samples(preprocess, protocol=protocol)
        ),
        "set_image": summarize_samples(
            synchronized_samples(set_image, protocol=protocol)
        ),
        "full_image_text": summarize_samples(
            synchronized_samples(
                BenchmarkOperation(run=full_image_text, synchronize=_sync_state),
                protocol=protocol,
            )
        ),
        "full_cached_text": summarize_samples(
            synchronized_samples(
                BenchmarkOperation(run=full_cached_text, synchronize=_sync_state),
                protocol=protocol,
            )
        ),
        "full_first_text": summarize_samples(
            synchronized_samples(
                BenchmarkOperation(run=full_first_text, synchronize=_sync_state),
                protocol=protocol,
            )
        ),
    }


def _dirty_paths() -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return [line[3:] for line in output.splitlines() if line]


def generate_benchmark(args: argparse.Namespace) -> JsonObject:
    dirty_paths = _dirty_paths()
    if dirty_paths and not args.allow_dirty:
        raise ValueError(
            "Benchmark generation requires a clean worktree; dirty paths: "
            f"{dirty_paths}"
        )
    image, image_identity = _load_image(args)
    checkpoint = args.checkpoint.resolve()
    load_started = time.perf_counter()
    model = sam3_mlx.build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
        compile=args.compile,
    )
    synchronize_completed(cast(_Parameters, model).parameters())
    cold_load_s = time.perf_counter() - load_started

    protocol = TimingProtocol(
        warmup_runs=args.warmup_runs,
        repetitions=args.repetitions,
    )
    compile_policy = "mlx-compiled-visual" if args.compile else "eager"
    first_resolution = args.resolution[0]
    compile_processor = Sam3Processor(
        model,
        resolution=first_resolution,
        confidence_threshold=args.confidence_threshold,
    )
    compile_started = time.perf_counter()
    compile_processor.set_image(image)
    mx.synchronize()
    cold_compile_s = time.perf_counter() - compile_started

    mx.clear_cache()
    mx.reset_peak_memory()
    resolutions: JsonObject = {}
    for resolution in args.resolution:
        measured = _benchmark_resolution(
            model,
            image,
            resolution=resolution,
            prompt=args.prompt,
            confidence_threshold=args.confidence_threshold,
            protocol=protocol,
        )
        if args.profile_substages:
            measured["substages"] = _profile_vision_substages(
                model,
                image,
                resolution=resolution,
                prompt=args.prompt,
                confidence_threshold=args.confidence_threshold,
                protocol=protocol,
                compile_policy=compile_policy,
            )
        resolutions[str(resolution)] = measured

    return {
        "schema_version": IMAGE_BENCHMARK_SCHEMA,
        "provenance": {
            "git_commit": git_commit(REPO_ROOT),
            "dirty": bool(dirty_paths),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
        },
        "environment": _environment(),
        "runtime": {
            "backend": "mlx-metal",
            "dtype_policy": "checkpoint-defined",
            "compile_policy": compile_policy,
            "synchronization_boundary": "mx.eval outputs followed by mx.synchronize",
        },
        "workload": {
            "image": image_identity,
            "prompt": args.prompt,
            "confidence_threshold": args.confidence_threshold,
            "resolutions": args.resolution,
            "warmup_runs": protocol.warmup_runs,
            "repetitions": protocol.repetitions,
        },
        "cold_load_s": cold_load_s,
        "cold_compile_s": cold_compile_s,
        "resolutions": resolutions,
        "active_memory_bytes": int(mx.get_active_memory()),
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
    }


def _write_json(path: Path, value: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--synthetic-height", type=int, default=720)
    parser.add_argument("--synthetic-width", type=int, default=960)
    parser.add_argument("--synthetic-seed", type=int, default=20260812)
    parser.add_argument("--prompt", default="shoe")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, action="append", default=[])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--comparison-out", type=Path)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--noise-pct", type=float, default=3.0)
    parser.add_argument("--max-regression-pct", type=float, default=10.0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--profile-substages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure synchronized vision substages (default: disabled).",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the compiled MLX visual backbone (default: enabled).",
    )
    args = parser.parse_args()
    if not args.resolution:
        args.resolution = [504, 672, 1008]
    if any(value <= 0 or value % 14 != 0 for value in args.resolution):
        parser.error("--resolution values must be positive multiples of 14")
    if args.synthetic_height <= 0 or args.synthetic_width <= 0:
        parser.error("synthetic dimensions must be positive")
    if (args.baseline is None) != (args.comparison_out is None):
        parser.error("--baseline and --comparison-out must be provided together")
    return args


def main() -> None:
    args = _parse_args()
    artifact = generate_benchmark(args)
    _write_json(args.out, artifact)
    result: JsonObject = {"benchmark": str(args.out), "status": "passed"}
    if args.baseline is not None:
        baseline = cast(JsonObject, json.loads(args.baseline.read_text()))
        comparison = compare_benchmark_artifacts(
            baseline,
            artifact,
            metric_paths=args.metric or DEFAULT_METRICS,
            threshold=RegressionThreshold(
                max_regression_pct=args.max_regression_pct,
                noise_pct=args.noise_pct,
            ),
        )
        assert args.comparison_out is not None
        _write_json(args.comparison_out, comparison)
        result["comparison"] = str(args.comparison_out)
        result["status"] = comparison["status"]
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
