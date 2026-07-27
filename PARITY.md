# Parity status

The `0.1.2` release boundary requires a **passed**, source-commit-bound receipt
under `parity/receipts/`. The claim is limited to the SAM 3 image runtime and
the measured prompt/resolution matrix below.

## Verified contracts

- Synthetic RGB preprocessing is compared with TorchVision over multiple aspect
  ratios with absolute and relative tolerance `1e-6`.
- Checkpoint shape and required-key coverage are validated before model
  mutation.
- Default hub downloads use a pinned revision and package-embedded output
  SHA-256 (`sam3_mlx.convert.DEFAULT_MLX_CHECKPOINT`).
- Checkpoint lineage is reproduced from pinned official checkpoint revision
  `3c879f39826c281e95690f02c7821c4de09afae7`; all 1,400 published tensors
  match exactly after canonical runtime layout normalization.
- Tracker and multiplex component fixtures record their upstream reference
  commit in the corresponding test modules.

The ordinary test suite may skip Torch/TorchVision comparisons when those
oracle dependencies are absent. A parity environment must preflight those
dependencies and treat their absence as a failure:

```bash
make preprocess-parity-check
```

## Release gates

| Gate | Command | Proves |
| --- | --- | --- |
| Packaging | `make artifact-check` | Wheel/sdist members, clean import, stable exports |
| Runtime receipt | `make runtime-release-check` | Passed upstream parity receipt bound to git commit |
| Release | `make release-check` | Both packaging and runtime gates |

Generate a schema-complete **blocked** receipt without measured evidence:

```bash
uv run python scripts/validate_runtime_release.py --generate \
  --receipt parity/receipts/latest.json
```

Generate a passed receipt after producing the example, independent holdout, and
checkpoint-lineage reports:

```bash
SAM3_OFFICIAL_CHECKOUT=/path/to/sam3-at-2814fa6 \
uv run python scripts/validate_runtime_release.py --generate \
  --pytest-python /path/to/python-with-torch-and-torchvision \
  --receipt parity/receipts/latest.json \
  --parity-report parity/receipts/example-image-parity.json \
  --parity-report parity/receipts/holdout-image-parity.json \
  --lineage-report parity/manifests/checkpoint-lineage.json
```

The receipt binds the source commit. A following receipt-only attestation commit
may add files under `parity/receipts/` and `parity/manifests/`; the runtime gate
verifies that no source file changed in that commit.

## Evidence required before an end-to-end parity claim

A checked-in receipt must identify:

- official repository commit and checkpoint revision;
- source and converted checkpoint SHA-256 values;
- converter version, reproduction manifest, and semantic lineage report;
- input corpus and positive, negative, and empty prompts;
- exact detection-count agreement, per-object and mean mask IoU, box error, and
  score error;
- MLX version, Apple device, dtype, and input resolution;
- results at `1008`, `672`, and `504`;
- steady-state latency and peak active memory;
- every skipped/deselected test node ID with rationale.

## Numerical contract

The first example-image run was used only for calibration. Its deliberately
strict initial limits (`mask_iou_min=0.98`, `score_abs_max=0.02`) rejected one
small 504px mask at `0.96675` IoU and one BF16-vs-FP32 score delta at `0.02010`.
Before running the independent holdout, the release contract was frozen at:

- exact detection count;
- per-object mask IoU at least `0.95` and case mean at least `0.99`;
- box coordinate L-infinity error at most `2.0` pixels;
- score absolute error at most `0.025`.

The independent `groceries.jpg` holdout passed all five cases. Its worst
measured values were mask IoU `0.99745`, box error `0.28699` pixels, and score
error `0.00214`. The example profile also passed the frozen contract and
includes positive, negative, and empty prompts.

Official inference uses CPU BF16 on the Apple-Silicon oracle host. The pinned
upstream commit assumes CUDA for cache construction and Triton EDT imports, so
the oracle records four scoped adaptations: unused EDT fails fast, construction
cache tensors use CPU, pinned-memory staging is disabled, and the official RoPE
formula is recomputed for 672/504 grids.

MLX performance uses one warmup and five synchronized full
`set_image + grounding` samples per resolution. Across the example and holdout
profiles, median latency was `1.225–1.264 s` at 1008, `0.552–0.569 s` at 672,
and `0.412–0.422 s` at 504. Peak active MLX memory was `8,888,103,879` bytes.

## Current receipt (`parity/receipts/latest.json`)

- Status: see the machine-readable `status` and `git_commit`; release requires
  `status=passed` and a clean source/attestation binding.
- Package version target: **0.1.2**
- Default hub pin: revision + output SHA-256 in
  `sam3_mlx.convert.DEFAULT_MLX_CHECKPOINT`
- The receipt must reference both measured parity profiles and the semantic
  checkpoint-lineage report.
- MLX self-determinism alone is never accepted as upstream parity.

## Intentionally deferred (not 0.1.2)

- Full SAM 3.1 multiplex/temporal tracking and multi-object state-transition
  parity (target **0.2.0**; builders live under `sam3_mlx.experimental`)
- Splitting `model_builder.py` into many modules
- Device-native preprocessing, shared RoPE across blocks, conversion
  concurrency / atomic multi-process writes
