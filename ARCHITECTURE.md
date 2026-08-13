# sam3_mlx architecture and hardening map

## Current product boundary

`sam3_mlx` is release-ready only as a SAM 3 image runtime plus a selected-frame
request API. The stable surface includes image model construction, pinned
checkpoint loading, text and geometric prompting, and frame-local inference.

SAM 3.1 multiplex and temporal tracking remain experimental. They are not
covered by the image-runtime parity claim.

The previous schema-v1 release receipt was superseded by schema-v2 reports, raw
replayable NPZ evidence, hardened checkpoint lineage, one source commit across
every artifact, and a receipt-only attestation commit. Those artifacts are
checked in and pass the local release gate for
`fe8b1e168c73b712467befab095f64cd21cb77c0` only. They do not attest later
source. This worktree is not accuracy-attested and is not claimed as the
fastest SAM 3 MLX implementation.

## Package map

### Stable API

- `build_sam3_image_model`
- `build_sam3_predictor`
- `build_sam3_video_model`
- `build_sam3_video_predictor`
- `build_tracker`
- `download_ckpt_from_hf`
- `Sam3MlxUnsupportedError`

Top-level imports are lazy. Experimental multiplex builders live under
`sam3_mlx.experimental` and are excluded from stable `__all__`.

Repository paths such as `agent`, `eval`, `train`, and Triton-specific code are
source references or future-port material. Artifact validation verifies that
these paths do not enter the wheel.

## Runtime data flow

### Image inference

```text
PIL / NumPy image
  -> RGB conversion
  -> TorchVision-matched FP32 resize and normalization
  -> one visual-boundary cast (`fp32` / `fp16` / `bf16` / `mixed`)
  -> MLX vision backbone
  -> image feature state

Caller-normalized NCHW/NHWC tensor
  -> `set_preprocessed_image` (skips PIL)
  -> same visual-boundary cast
  -> MLX vision backbone

text prompt
  -> tokenizer and language backbone
  -> language features

box / point prompt
  -> normalized geometry
  -> geometry encoder

image + language + geometry
  -> `predict_raw` (fixed-shape logits, boxes, low-res masks, presence)
  -> `predict` convenience wrapper
  -> confidence filtering
  -> interpolation to original image size
```

`Sam3Processor` owns mutable per-image state. Processor resolutions may be any
positive multiple of 14. Exact window-aligned staging sizes are 336, 672, and
1008. Release parity is measured only at 1008, 672, and 504. 336 is an optional
fast tier, not an accuracy-equivalent default.

### Checkpoint path

```text
builder
  -> model component construction
  -> checkpoint normalization
  -> key / shape / coverage audit
  -> model mutation
  -> MLX evaluation mode
```

The default image checkpoint is pinned by immutable Hugging Face revision and
SHA-256. Official PyTorch conversion records source revision, source and output
hashes, key counts, ignored tracker keys, and dtype counts.

### Selected-frame video

```text
image / folder / PIL sequence / OpenCV video
  -> bounded frame provider
  -> session state
  -> frame-local image inference
  -> streamed detections
```

This is not temporal tracking. It does not maintain persistent identities or
tracker memory. `out_obj_ids` are frame-local. Cross-frame object removal is
rejected.

Image folders use a bounded on-demand RGB cache. Video files use a bounded
OpenCV cache. Encoded frame features and text features are bounded per session.
Full-resolution output history is not retained.

The legacy asynchronous folder loader eagerly retains every decoded frame. The
hardened public predictor rejects that configuration rather than violating the
bounded-memory claim.

## Concurrency and lifecycle

`Sam3BasePredictor` is the lifecycle authority for every public and future
subclass. It uses a global session-registry lock plus a reentrant lock per
session. Session state includes a version, propagation flag, cancellation event,
and closing/closed flags. `LifecycleSafeSam3BasePredictor` remains only as a
zero-logic compatibility subclass.

The hardened base enforces:

- `shutdown()` is terminal and prevents future sessions;
- an in-flight loader cannot publish after shutdown;
- `close_session(id)` cancels publication when that ID is still loading;
- a state that finishes loading after close or shutdown is disposed;
- reservation IDs cannot be cleared and reused to overwrite a live state;
- disposal clears state and marks the session closed even if a frame provider's
  `close()` raises;
- diagnostic session snapshots do not iterate a mutating registry unsafely.

Shared position-encoding, RoPE, and decoder-coordinate LRUs are synchronized.
This is required because concurrent sessions share one model.

## Experimental SAM 3.1 path

The source tree includes multiplex detector/tracker composition, memory encoding,
mask decoding, checkpoint-key normalization, and a predictor wrapper. It still
lacks sequence-level official-vs-MLX evidence, identity-transition parity,
long-video memory characterization, and complete multi-object validation.

A temporal release belongs to 0.2 and needs its own frozen contract.

## Release evidence architecture

### Frozen contract

`sam3_mlx.release_contract` pins:

- package version;
- official source commit;
- official and MLX checkpoint revisions and hashes;
- calibration and holdout image paths, sizes, and hashes;
- exact prompt matrix;
- confidence threshold and resolutions;
- metric thresholds;
- CPU oracle precision and allowed adaptations;
- schema and comparison algorithm versions.

Changing any item requires new evidence.

### Committed-source gate

`sam3_mlx.source_binding` requires all source changes to be committed before
evidence generation. Dirty paths are allowed only under:

- `parity/receipts/`
- `parity/manifests/`
- `parity/evidence/`

The parity reports, raw NPZ metadata, checkpoint-lineage report, and runtime
receipt each record the same source commit. `audit_release_candidate.py` rejects
cross-commit mixtures even when every individual digest is internally valid.

### Hardened oracle

The official CPU oracle verifies the checkout, submodules, checkpoint, image,
prompt matrix, threshold, script digest, release-contract digest, and actual
import source paths. The cache identity includes all of them. The checkout is
checked before and after the run.

### Replayable parity

The hardened parity runner stores raw official and MLX masks, boxes, and scores
in compressed NPZ files. Detections are paired using deterministic Hungarian
maximum-mask-IoU assignment rather than greedy pairing. Pairwise IoU computation
uses bounded scratch memory instead of materializing an `N x N x pixels` tensor.

The report references the raw bundle by SHA-256. Summary metrics are therefore
recomputable instead of merely asserted.

### Hardened lineage

The lineage gate hard-pins source and published checkpoints, embeds the complete
conversion manifest, records the source commit plus lineage-runner and converter
digests, and requires exact equality of 1,400 canonical tensor keys, shapes,
dtypes, and values.

### Independent candidate auditor

`audit_release_candidate.py` composes two independent checks:

1. `audit_release_evidence.py` verifies every hard pin and digest, loads NPZ
   evidence with pickle disabled, recomputes assignment and all count, IoU, box,
   and score metrics, reconstructs receipt case/performance projections, checks
   zero skipped or deselected release tests, and verifies the embedded manifest
   plus 1,400-tensor lineage.
2. The candidate source-binding layer proves that the receipt, both reports,
   both raw bundles, and lineage all name the same source commit.

`make runtime-release-check` runs the candidate audit before the hardened
receipt/attestation validator.

## Validation matrix

| Layer | Gate |
| --- | --- |
| Package | clean-wheel install, stable exports, excluded source paths |
| Preprocessing | TorchVision numerical comparison |
| Checkpoint | key, shape, and required-component coverage |
| Provenance | immutable revisions and SHA-256 values |
| Source binding | one committed source revision across all artifacts |
| Lineage | 1,400 exact canonical tensors |
| Image outputs | replayed official-vs-MLX evidence at 1008/672/504 |
| Controls | positive text, negative text, positive box, positive/negative box |
| Performance | synchronized latency and active-memory samples |
| Video API | resource, session, cancellation, and bounded-cache contracts |
| Concurrency | cache stress, close/loading, and shutdown/publication tests |
| Artifacts | wheel and sdist content inspection |

## Fixed on the hardening branch

- Unsynchronized shared LRUs.
- Session publication after shutdown.
- Session publication after close while loading.
- Duplicate-ID risk after reservation clearing.
- Incomplete cleanup after frame-provider close failure.
- Racy live-session diagnostics.
- Unbounded asynchronous folder loading in selected-frame mode.
- Oracle caches bound only to image and case-spec hashes.
- Oracle imports not proven to originate from the pinned checkout.
- Greedy object pairing.
- Unbounded pairwise-IoU scratch memory.
- Summary-only parity evidence.
- Caller-selected lineage revisions.
- Conversion manifests represented only by an external digest.
- Evidence files that were internally valid but generated from different source
  commits.

## Release gate status

The local `0.1.2` release gate is complete **for the receipt-bound commit
`fe8b1e1`**:

1. Both schema-v2 raw parity bundles and reports are checked in.
2. Checkpoint lineage passes for all 1,400 tensors.
3. The receipt records 701 passing tests with zero failures, skips, or
   deselections.
4. Every evidence artifact names `fe8b1e168c73b712467befab095f64cd21cb77c0`.
5. The evidence-only attestation commit has that source commit as its sole
   parent.
6. `make release-check` passes from a clean worktree of that attested SHA,
   including wheel and source distribution inspection and clean-install
   validation.

Later source, including `58a50c9` and this worktree, must be re-attested
before any accuracy or “fastest” claim.

Creating a version tag or publishing an artifact is a separate distribution
action and is not implied by the checked-in evidence.

## Remaining product risks

- Two pinned images prove implementation parity, not broad segmentation quality.
- No external task-diverse validation corpus is part of the release gate.
- Selected-frame inference can be mistaken for tracking if documentation drifts.
- Random-access OpenCV seeking may be slow on long videos.
- True SAM 3.1 temporal behavior is unvalidated.
- `model_builder.py` is a large maintainability hotspot.
- Conversion writes are not yet atomic across competing processes.
- Performance evidence represents one Apple Silicon host.

## Highest-leverage next work

1. Add a broader external image corpus without weakening the frozen holdout.
2. Profile device-native preprocessing and repeated-prompt text caching.
3. Add a sequential video decode path for forward propagation.
4. Split checkpoint normalization and component factories out of
   `model_builder.py` without changing the stable release contract.
5. Treat temporal tracking as a separate 0.2 assurance program.
