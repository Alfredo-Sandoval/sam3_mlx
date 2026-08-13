#!/usr/bin/env python3
"""Run release-grade official-vs-MLX SAM 3 image parity and performance.

The thresholds are declared independently of observed results and apply to
every non-empty matched case:

* exact detection count;
* minimum per-object mask IoU >= 0.95 and case mean >= 0.99;
* maximum box coordinate error <= 2 pixels;
* maximum score error <= 0.025.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_mlx.model.sam3_image_processor import (  # noqa: E402
    ProcessorState,
    Sam3Processor,
)
from sam3_mlx.benchmarking import (  # noqa: E402
    BenchmarkOperation,
    TimingProtocol,
    percentile,
    synchronized_samples,
)
from scripts._oracle_runtime import OracleCase  # noqa: E402


class _MlxEval(Protocol):
    def __call__(self, *values: object) -> None: ...


MASK_IOU_MIN = 0.95
MASK_IOU_MEAN_MIN = 0.99
BOX_L_INF_MAX = 2.0
SCORE_ABS_MAX = 0.025
OFFICIAL_REVISION = "2814fa619404a722d03e9a012e083e4f293a4e53"
OFFICIAL_CHECKPOINT_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
OFFICIAL_CHECKPOINT_SHA256 = (
    "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
)
MLX_CHECKPOINT_REVISION = "b72a14d8127e17e6f2a3d2e075bbbf4307ba146e"
MLX_CHECKPOINT_SHA256 = (
    "0ad4c3f42ecf706c4cda63cf58d621699491ed65012b3999284ea370984f7173"
)


def case_specs(image: Image.Image, profile: str) -> list[OracleCase]:
    if profile == "holdout":
        holdout_cases: list[OracleCase] = [
            {
                "name": f"text_paper_bag_{resolution}",
                "resolution": resolution,
                "prompt": "paper bag",
                "geometric_prompts": [],
            }
            for resolution in (1008, 672, 504)
        ]
        holdout_cases.extend(
            [
                {
                    "name": "text_car_1008",
                    "resolution": 1008,
                    "prompt": "car",
                    "geometric_prompts": [],
                },
                {
                    "name": "text_nonsense_1008",
                    "resolution": 1008,
                    "prompt": "zzzz_not_a_real_class_qqq",
                    "geometric_prompts": [],
                },
            ]
        )
        return holdout_cases
    if profile != "example":
        raise ValueError(f"Unknown case profile: {profile!r}")

    width, height = image.size

    def normalized_cxcywh(x: float, y: float, w: float, h: float) -> list[float]:
        return [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]

    positive = normalized_cxcywh(480, 290, 110, 360)
    negative = normalized_cxcywh(370, 280, 115, 375)
    example_cases: list[OracleCase] = [
        {
            "name": "text_shoe_1008",
            "resolution": 1008,
            "prompt": "shoe",
            "geometric_prompts": [],
        },
        {
            "name": "text_nonsense_1008",
            "resolution": 1008,
            "prompt": "zzzz_not_a_real_class_qqq",
            "geometric_prompts": [],
        },
        {
            "name": "positive_box_1008",
            "resolution": 1008,
            "prompt": None,
            "geometric_prompts": [{"box": positive, "label": True}],
        },
        {
            "name": "positive_negative_box_1008",
            "resolution": 1008,
            "prompt": None,
            "geometric_prompts": [
                {"box": positive, "label": True},
                {"box": negative, "label": False},
            ],
        },
        {
            "name": "text_shoe_672",
            "resolution": 672,
            "prompt": "shoe",
            "geometric_prompts": [],
        },
        {
            "name": "text_shoe_504",
            "resolution": 504,
            "prompt": "shoe",
            "geometric_prompts": [],
        },
    ]
    return example_cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_path(path: Path, *, official_checkout: Path) -> str:
    try:
        relative = path.resolve().relative_to(official_checkout.resolve())
    except ValueError:
        return str(path)
    return f"official-checkout/{relative}"


def validate_official_checkout(
    checkout: Path,
    *,
    expected_revision: str = OFFICIAL_REVISION,
) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise ValueError(
            f"Official checkout must be {expected_revision}, found {revision}."
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


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left, right).sum(dtype=np.int64)
    union = np.logical_or(left, right).sum(dtype=np.int64)
    return 1.0 if union == 0 else float(intersection / union)


def _match_objects(
    official_masks: np.ndarray,
    mlx_masks: np.ndarray,
) -> list[tuple[int, int, float]]:
    count = len(official_masks)
    candidates = sorted(
        (
            (_mask_iou(official_masks[i], mlx_masks[j]), i, j)
            for i, j in itertools.product(range(count), repeat=2)
        ),
        reverse=True,
    )
    matches: list[tuple[int, int, float]] = []
    official_used: set[int] = set()
    mlx_used: set[int] = set()
    for iou, official_index, mlx_index in candidates:
        if official_index in official_used or mlx_index in mlx_used:
            continue
        matches.append((official_index, mlx_index, iou))
        official_used.add(official_index)
        mlx_used.add(mlx_index)
        if len(matches) == count:
            break
    return sorted(matches)


def _run_prompt(
    processor: Sam3Processor, state: ProcessorState, spec: OracleCase
) -> ProcessorState:
    processor.reset_all_prompts(state)
    if spec["prompt"] is not None:
        state = processor.set_text_prompt(spec["prompt"], state)
    for geometric_prompt in spec["geometric_prompts"]:
        state = processor.add_geometric_prompt(
            geometric_prompt["box"],
            geometric_prompt["label"],
            state,
        )
    return state


def mlx_outputs(
    *,
    checkpoint: Path,
    image: Image.Image,
    specs: list[OracleCase],
    confidence_threshold: float,
    repetitions: int,
    precision: str = "fp32",
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    import mlx.core as mx
    import sam3_mlx
    from sam3_mlx.model.sam3_image_processor import Sam3Processor

    mlx_eval = cast(_MlxEval, getattr(mx, "eval"))

    load_started = time.perf_counter()
    model = sam3_mlx.build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_segmentation=True,
        precision=precision,
    )
    mlx_eval(cast(object, model.parameters()))
    cold_load_s = time.perf_counter() - load_started

    outputs: list[dict[str, np.ndarray]] = []
    states: dict[int, dict[str, Any]] = {}
    processors: dict[int, Sam3Processor] = {}
    latencies: dict[int, list[float]] = {}
    mx.clear_cache()
    mx.reset_peak_memory()

    for spec in specs:
        resolution = int(spec["resolution"])
        if resolution not in states:
            processor = Sam3Processor(
                model,
                resolution=resolution,
                confidence_threshold=confidence_threshold,
            )
            states[resolution] = processor.set_image(image)
            processors[resolution] = processor
        processor = processors[resolution]
        state = _run_prompt(processor, states[resolution], spec)
        mlx_eval(state["masks"], state["boxes"], state["scores"])
        outputs.append(
            {
                "masks": np.asarray(state["masks"], dtype=np.bool_),
                "boxes": np.asarray(state["boxes"], dtype=np.float32),
                "scores": np.asarray(state["scores"], dtype=np.float32),
            }
        )

    protocol = TimingProtocol(warmup_runs=1, repetitions=repetitions)
    for resolution in (1008, 672, 504):
        spec = next(
            item
            for item in specs
            if item["resolution"] == resolution and item["prompt"] is not None
        )
        processor = processors[resolution]

        def run() -> ProcessorState:
            return _run_prompt(processor, processor.set_image(image), spec)

        def synchronize(state: ProcessorState) -> None:
            mlx_eval(state["masks"], state["boxes"], state["scores"])
            mx.synchronize()

        latencies[resolution] = synchronized_samples(
            BenchmarkOperation(run=run, synchronize=synchronize),
            protocol=protocol,
        )

    return outputs, {
        "status": "passed",
        "cold_load_s": cold_load_s,
        "warmup_runs": 1,
        "repetitions": repetitions,
        "latency_by_resolution_s": {
            str(resolution): {
                "samples": samples,
                "median": statistics.median(samples),
                "p95": percentile(samples, 0.95),
            }
            for resolution, samples in latencies.items()
        },
        "peak_active_memory_bytes": int(mx.get_peak_memory()),
        "measurement_boundary": (
            "MLX active allocator peak after model load; synchronized full "
            "set_image plus text-grounding latency"
        ),
        "precision": precision,
    }


_case_specs = case_specs
_evidence_path = evidence_path
_mlx_outputs = mlx_outputs
_validate_official_checkout = validate_official_checkout


def _compare_case(
    spec: OracleCase,
    official: dict[str, np.ndarray],
    mlx: dict[str, np.ndarray],
) -> dict[str, Any]:
    official_count = len(official["scores"])
    mlx_count = len(mlx["scores"])
    count_match = official_count == mlx_count
    matches = (
        _match_objects(official["masks"], mlx["masks"])
        if count_match and official_count
        else []
    )
    mask_ious = [match[2] for match in matches]
    box_errors = [
        float(
            np.max(np.abs(official["boxes"][official_index] - mlx["boxes"][mlx_index]))
        )
        for official_index, mlx_index, _iou in matches
    ]
    score_errors = [
        float(abs(official["scores"][official_index] - mlx["scores"][mlx_index]))
        for official_index, mlx_index, _iou in matches
    ]
    mask_iou_min = min(mask_ious) if mask_ious else None
    box_l_inf_max = max(box_errors) if box_errors else None
    score_abs_max = max(score_errors) if score_errors else None
    passed = count_match and (
        official_count == 0
        or (
            mask_iou_min is not None
            and mask_iou_min >= MASK_IOU_MIN
            and statistics.mean(mask_ious) >= MASK_IOU_MEAN_MIN
            and box_l_inf_max is not None
            and box_l_inf_max <= BOX_L_INF_MAX
            and score_abs_max is not None
            and score_abs_max <= SCORE_ABS_MAX
        )
    )
    return {
        "name": spec["name"],
        "resolution": spec["resolution"],
        "prompt": spec["prompt"],
        "geometric_prompts": spec["geometric_prompts"],
        "status": "passed" if passed else "failed",
        "official_detection_count": official_count,
        "mlx_detection_count": mlx_count,
        "detection_count_match": count_match,
        "mask_iou_min": mask_iou_min,
        "mask_iou_mean": statistics.mean(mask_ious) if mask_ious else None,
        "box_l_inf_max": box_l_inf_max,
        "score_abs_max": score_abs_max,
        "matches": [
            {
                "official_index": official_index,
                "mlx_index": mlx_index,
                "mask_iou": iou,
            }
            for official_index, mlx_index, iou in matches
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--mlx-checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle-cache", type=Path)
    parser.add_argument(
        "--profile",
        choices=("example", "holdout"),
        default="example",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "mixed"),
        default="fp32",
        help="Image-runtime precision policy (default: fp32).",
    )
    args = parser.parse_args()

    try:
        revision = validate_official_checkout(args.official_checkout)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.repetitions < 5:
        raise SystemExit("--repetitions must be at least 5 for release evidence.")

    official_checkpoint_sha256 = _sha256(args.official_checkpoint)
    if official_checkpoint_sha256 != OFFICIAL_CHECKPOINT_SHA256:
        raise SystemExit(
            "Official checkpoint SHA-256 does not match the pinned Hugging Face "
            f"artifact: {official_checkpoint_sha256}."
        )
    mlx_checkpoint_sha256 = _sha256(args.mlx_checkpoint)
    if mlx_checkpoint_sha256 != MLX_CHECKPOINT_SHA256:
        raise SystemExit(
            "MLX checkpoint SHA-256 does not match the package pin: "
            f"{mlx_checkpoint_sha256}."
        )

    image = Image.open(args.image).convert("RGB")
    specs = case_specs(image, args.profile)
    with tempfile.TemporaryDirectory(prefix="sam3-mlx-parity-") as temp_dir:
        temp = Path(temp_dir)
        cases_path = temp / "cases.json"
        oracle_path = args.oracle_cache or temp / "official.npz"
        cases_path.write_text(json.dumps(specs, indent=2) + "\n")
        if not oracle_path.exists():
            env = dict(os.environ)
            env["PYTHONPATH"] = str(args.official_checkout)
            subprocess.run(
                [
                    str(args.official_python),
                    str(REPO_ROOT / "scripts" / "run_upstream_image_oracle.py"),
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
        with np.load(oracle_path, allow_pickle=False) as archive:
            oracle_metadata = json.loads(str(archive["metadata_json"]))
            observed_case_names = [case["name"] for case in oracle_metadata["cases"]]
            expected_case_names = [spec["name"] for spec in specs]
            if observed_case_names != expected_case_names:
                raise SystemExit(
                    "Oracle cache case profile mismatch: "
                    f"observed={observed_case_names}, expected={expected_case_names}."
                )
            expected_cache_bindings = {
                "image_sha256": _sha256(args.image),
                "case_spec_sha256": _sha256(cases_path),
            }
            for field, expected in expected_cache_bindings.items():
                if oracle_metadata.get(field) != expected:
                    raise SystemExit(
                        f"Oracle cache {field} mismatch: "
                        f"observed={oracle_metadata.get(field)!r}, expected={expected!r}."
                    )
            official_outputs = [
                {
                    "masks": archive[f"case_{index}_masks"],
                    "boxes": archive[f"case_{index}_boxes"],
                    "scores": archive[f"case_{index}_scores"],
                }
                for index in range(len(specs))
            ]

    runtime_outputs, performance = mlx_outputs(
        checkpoint=args.mlx_checkpoint,
        image=image,
        specs=specs,
        confidence_threshold=args.confidence_threshold,
        repetitions=args.repetitions,
        precision=args.precision,
    )
    cases = [
        _compare_case(spec, official, mlx)
        for spec, official, mlx in zip(
            specs, official_outputs, runtime_outputs, strict=True
        )
    ]
    status = "passed" if all(case["status"] == "passed" for case in cases) else "failed"
    report = {
        "schema_version": 1,
        "status": status,
        "official_code": {
            "repo": "https://github.com/facebookresearch/sam3",
            "revision": revision,
        },
        "official_checkpoint": {
            "repo": "facebook/sam3",
            "revision": OFFICIAL_CHECKPOINT_REVISION,
            "sha256": official_checkpoint_sha256,
        },
        "converted_checkpoint": {
            "repo": "mlx-community/sam3-image",
            "revision": MLX_CHECKPOINT_REVISION,
            "sha256": mlx_checkpoint_sha256,
        },
        "image": {
            "path": evidence_path(
                args.image,
                official_checkout=args.official_checkout,
            ),
            "sha256": _sha256(args.image),
            "size": list(image.size),
        },
        "confidence_threshold": args.confidence_threshold,
        "precision": args.precision,
        "case_profile": args.profile,
        "thresholds": {
            "mask_iou_min": MASK_IOU_MIN,
            "mask_iou_mean_min": MASK_IOU_MEAN_MIN,
            "box_l_inf_max": BOX_L_INF_MAX,
            "score_abs_max": SCORE_ABS_MAX,
        },
        "cases": cases,
        "oracle": oracle_metadata,
        "performance": performance,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wrote": str(args.out), "status": status}, indent=2))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
