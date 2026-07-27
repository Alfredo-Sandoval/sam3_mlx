# Parity and release evidence

## Scope

The 0.1.x parity claim is limited to the SAM 3 image runtime. It covers the
frozen image, prompt, resolution, checkpoint, precision, and threshold matrix in
`sam3_mlx.release_contract`.

It does not cover:

- SAM 3.1 multiplex or temporal tracking;
- persistent cross-frame identity;
- training or official evaluation tooling;
- unmeasured processor resolutions;
- broad task-level segmentation quality.

## Frozen numerical contract

Every non-empty case requires:

- exact detection count;
- per-object mask IoU at least `0.95`;
- case mean mask IoU at least `0.99`;
- box-coordinate L-infinity error at most `2.0` pixels;
- score absolute error at most `0.025`.

The confidence threshold is `0.5`. Measured resolutions are `1008`, `672`, and
`504`.

The calibration and holdout images, prompt cases, official source revision,
official checkpoint, published MLX checkpoint, CPU precision, and allowed CPU
adaptations are content-addressed in `sam3_mlx.release_contract`.

## Evidence layers

### 1. Preprocessing

Synthetic RGB inputs are compared against TorchVision preprocessing across
multiple aspect ratios. Release preprocessing tests must run with TorchVision
available:

```bash
make preprocess-parity-check
```

### 2. Checkpoint lineage

A fresh conversion from the pinned official checkpoint is compared with the
pinned published MLX artifact after canonical runtime-layout normalization. A
passing report requires exactly 1,400 matching tensor keys, shapes, dtypes, and
values.

The schema-v2 lineage report embeds the complete conversion manifest and records
the generating source commit plus lineage-runner and converter-module SHA-256
values.

### 3. Upstream oracle

The hardened official oracle verifies the official checkout before and after
inference and binds its cache to:

- official code revision and clean status;
- official checkpoint revision and SHA-256;
- image SHA-256;
- exact case-spec bytes;
- confidence threshold;
- CPU BF16 precision;
- the explicitly allowed CPU adaptations;
- oracle runner SHA-256;
- release-contract SHA-256;
- actual import paths inside the pinned official checkout.

A cache missing any binding is not release evidence.

### 4. Raw output parity

The hardened parity runner stores official and MLX masks, boxes, and scores for
every case in compressed NPZ bundles. Object pairing uses deterministic Hungarian
maximum-mask-IoU assignment.

Schema-v2 reports reference the raw bundle by SHA-256. Summary metrics are
recomputed by the release auditor rather than trusted directly. Both the report
and raw bundle record the exact `sam3_mlx` source commit that generated them.

### 5. Source-bound receipt

The runtime receipt binds:

- source commit;
- package and MLX versions;
- complete test outcomes and skip details;
- two parity reports and their SHA-256 values;
- checkpoint-lineage report and SHA-256;
- case and performance projections.

`audit_release_candidate.py` requires the receipt, both reports, both raw bundles,
and the lineage report to name the same source commit. This prevents internally
valid evidence generated from different source revisions from being combined.

A following attestation commit may contain only files under `parity/receipts/`,
`parity/manifests/`, and `parity/evidence/`.

## Generate hardened evidence

Commit all source changes first. Evidence generators reject non-evidence
worktree changes. Use a clean official checkout at the revision recorded in
`sam3_mlx.release_contract`.

### Calibration profile

```bash
uv run python scripts/run_hardened_image_parity.py \
  --official-checkout /path/to/pinned/sam3 \
  --official-checkpoint /path/to/sam3.pt \
  --official-python /path/to/oracle-python \
  --mlx-checkpoint /path/to/model.safetensors \
  --image /path/to/pinned/sam3/assets/images/test_image.jpg \
  --profile example \
  --out parity/receipts/example-image-parity.json \
  --evidence-out parity/evidence/example-image-parity.npz
```

### Independent holdout profile

```bash
uv run python scripts/run_hardened_image_parity.py \
  --official-checkout /path/to/pinned/sam3 \
  --official-checkpoint /path/to/sam3.pt \
  --official-python /path/to/oracle-python \
  --mlx-checkpoint /path/to/model.safetensors \
  --image /path/to/pinned/sam3/assets/images/groceries.jpg \
  --profile holdout \
  --out parity/receipts/holdout-image-parity.json \
  --evidence-out parity/evidence/holdout-image-parity.npz
```

### Checkpoint lineage

```bash
uv run python scripts/validate_checkpoint_lineage_hardened.py \
  --official-checkpoint /path/to/sam3.pt \
  --published-checkpoint /path/to/published/model.safetensors \
  --reproduced-checkpoint /path/to/reproduced/model.safetensors \
  --reproduction-manifest /path/to/reproduced/conversion-manifest.json \
  --out parity/manifests/checkpoint-lineage.json
```

### Source-bound receipt

```bash
uv run python scripts/validate_runtime_release_hardened.py --generate \
  --pytest-python /path/to/release-python \
  --receipt parity/receipts/latest.json \
  --parity-report parity/receipts/example-image-parity.json \
  --parity-report parity/receipts/holdout-image-parity.json \
  --lineage-report parity/manifests/checkpoint-lineage.json
```

The release suite must record zero failures, zero skips, and zero deselections.
Commit only the generated evidence paths in the attestation commit.

## Validate a candidate

```bash
make release-evidence-audit
make runtime-release-check
make artifact-check
make release-check
```

`release-evidence-audit` invokes `audit_release_candidate.py`. It loads raw NPZ
files with pickle disabled, replays every case, verifies all cache and lineage
bindings, reconstructs the receipt projections, and checks the cross-artifact
source-commit chain. `validate_runtime_release_hardened.py` then verifies the
source/attestation commit relationship, including raw evidence paths.
`validate_release.py` builds and inspects the wheel and source distribution.

## Current repository state

The checked-in schema-v2 reports, raw NPZ bundles, checkpoint-lineage manifest,
and runtime receipt are source-bound to the commit recorded in
`parity/receipts/latest.json`. The independent replay audit covers 11 cases
across the example and holdout profiles, and the recorded release suite contains
701 passing tests with zero failures, skips, or deselections. `make
release-check` passes from the clean receipt-only attestation commit.

## Current performance evidence

The source-bound Apple-Silicon run measured synchronized full `set_image` plus
text grounding with one warmup and five samples per resolution. Median ranges
across the example and holdout profiles were:

- `1008`: `1.15–1.19 s`;
- `672`: `0.50–0.55 s`;
- `504`: `0.37–0.42 s`;
- peak active MLX allocation: about `8.89 GB`.

These measurements describe one Apple-Silicon host and the source commit named
by the receipt. They are not performance guarantees for later commits.

## Intentionally deferred

- full SAM 3.1 multiplex and temporal parity;
- device-native preprocessing;
- shared text-feature caching for the general image API;
- sequential OpenCV decode optimization;
- atomic multi-process conversion output;
- decomposition of `model_builder.py`;
- external task-diverse validation beyond the pinned calibration and holdout.
