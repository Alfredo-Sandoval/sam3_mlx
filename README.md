# sam3_mlx

**An unofficial Apple MLX port of SAM 3 / SAM 3.1 image-segmentation components for Apple Silicon.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-000000?logo=apple&logoColor=white)](#requirements)
[![Backend](https://img.shields.io/badge/Backend-MLX-1C7C54)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/License-SAM-blue)](LICENSE)

`sam3_mlx` brings selected Segment Anything Model 3 image components to Apple
Silicon through Apple's [MLX](https://github.com/ml-explore/mlx) framework.

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Release assurance](#release-assurance)
- [Limitations](#limitations)
- [Attribution](#attribution)
- [License](#license)

## Features

- **SAM 3 image runtime** with text and geometric prompting on Apple Silicon.
- **Pinned default checkpoint** verified by immutable revision and SHA-256.
- **Selected-frame request API** with bounded frame and feature caches.
- **Fail-explicit unsupported surfaces** instead of silent Torch/CUDA fallbacks.
- **Replayable release evidence** with raw official and MLX output arrays.

> [!NOTE]
> SAM 3.1 Object Multiplex and temporal tracking are experimental. They are not
> covered by the stable image-runtime parity claim.

## Requirements

- macOS on Apple Silicon (M-series)
- Python ≥ 3.12
- [MLX](https://github.com/ml-explore/mlx) ≥ 0.30
- SAM 3 / SAM 3.1 checkpoints obtained separately

## Installation

Install from PyPI:

```bash
pip install sam3-mlx
```

Or add it with [uv](https://docs.astral.sh/uv/):

```bash
uv add sam3-mlx
```

For local development:

```bash
uv sync
```

Optional extras:

```bash
pip install "sam3-mlx[viz]"     # matplotlib plotting helpers
pip install "sam3-mlx[video]"   # OpenCV video-file decoding
```

Agent-facing image segmentation is registered and invoked through FIESTA,
which owns the prompt workflow and selects this package with
`backend = "sam3-mlx"`. This repository intentionally publishes no separate
tool adapter.

Default image weights download from a pinned `mlx-community/sam3-image`
revision and are verified against a package-embedded SHA-256 before model
mutation.

For stronger provenance, convert from the official PyTorch checkpoint with
`convert_from_pytorch=True` and an immutable source revision. Conversion writes
`conversion-manifest.json` with source and output hashes, mapped-key counts,
ignored tracker keys, and dtype counts. PyTorch is intentionally not installed
by a package extra; use a separate conversion environment.

Verify the install:

```bash
python -c "import sam3_mlx; print(sam3_mlx.__version__)"
```

From a checkout:

```bash
python -m compileall -q sam3_mlx tests scripts
```

## Quickstart

### Image segmentation

```python
from sam3_mlx import build_sam3_image_model
from sam3_mlx.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model, resolution=1008)

state = processor.set_image("image.jpg")
state = processor.set_text_prompt("shoe", state)

masks = state["masks"]
boxes = state["boxes"]
scores = state["scores"]
```

`Sam3Processor.resolution` is the square image size fed into the ViT backbone. It
must be a positive multiple of the patch size (`14`). Lower resolutions are
faster but lose detail. Exact local-attention window grids are multiples of
`336` (`14 × 24`): `336`, `672`, and `1008`. `336` is an optional fast tier, not
an accuracy-equivalent default. `504` is still in the frozen release matrix even
though it pads to the same `48×48` window topology as `672`.

The runtime accepts other multiples of 14, but the frozen official-vs-MLX release
matrix covers only `1008`, `672`, and `504`.

### Selected-frame video

```python
from sam3_mlx import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3")
```

`build_sam3_predictor()` defaults to the selected-frame SAM 3 path. Request
`version="sam3.1"` explicitly for the experimental multiplex predictor; that
path currently requires a locally converted checkpoint.

The selected-frame path accepts image paths, image folders, PIL image sequences,
and OpenCV-decodable video files. It performs independent image inference on
each frame:

- `out_obj_ids` are frame-local detection IDs;
- no persistent temporal identity or tracker memory is claimed;
- cross-frame `remove_object` is rejected;
- prompt mutation during propagation is rejected except cancellation;
- full-resolution output history is not retained;
- image folders and video files use bounded host caches;
- encoded frame and text features are bounded per session.

Asynchronous image-folder preloading is rejected in selected-frame mode because
the legacy loader retains every decoded frame and would violate the bounded
memory contract.

Experimental multiplex builders are available under `sam3_mlx.experimental` and
are not part of stable top-level `__all__`.

Checkpoint loading requires complete required-component coverage by default. For
controlled development experiments only, `build_sam3_image_model` accepts
`strict_checkpoint_loading=False`; the resulting audit is available as
`model.checkpoint_load_report`.

## Release assurance

The release contract pins source and checkpoint revisions, hashes, images,
prompts, resolutions, precision, thresholds, and comparison algorithms.

The hardened release process includes:

1. a self-contained pinned official CPU oracle;
2. raw official and MLX masks, boxes, and scores in compressed NPZ bundles;
3. deterministic Hungarian maximum-IoU object assignment;
4. exact 1,400-tensor checkpoint lineage with an embedded conversion manifest;
5. an independent auditor that replays every reported metric;
6. source-commit and receipt-only-attestation binding;
7. wheel and source-distribution content checks.

See [`PARITY.md`](PARITY.md) for commands and [`ARCHITECTURE.md`](ARCHITECTURE.md)
for the complete system and risk map.

> [!IMPORTANT]
> The checked-in schema-v2 parity bundles, checkpoint lineage, and runtime
> receipt satisfy the local release gate for commit
> `fe8b1e168c73b712467befab095f64cd21cb77c0` only. They do not attest later
> source, including `58a50c9` or an uncommitted worktree. This repository is
> not accuracy-attested at HEAD and does not claim to be the fastest SAM 3
> MLX implementation. Run `make release-check` from a clean worktree after
> regenerating a receipt bound to the final SHA before tagging or publishing.

## Limitations

Unsupported paths raise `Sam3MlxUnsupportedError`:

- **Apple Silicon / MLX only.** Non-MLX devices and `torch.compile` are not
  accepted as aliases.
- **Image interactivity is intentionally narrow.** Batch geometric prompts and
  unsupported official multi-mask behaviors fail explicitly.
- **Selected-frame video is not tracking.** It has no persistent object identity,
  memory propagation, or cross-frame object removal.
- **SAM 3.1 multiplex is experimental.** Automatic checkpoint conversion,
  temporal parity, multi-GPU execution, and TorchCodec decoding are unavailable.
- **Training is unsupported.** Training loops, autograd, distributed execution,
  and the official evaluation toolkit are excluded from the wheel.
- **Parity is scoped.** The release matrix measures pinned text and geometric
  image cases at 1008/672/504. It is not broad external model validation.

## Attribution

Portions of this repository are derived from, adapted from, or structured for
parity with the official SAM 3 implementation
([`facebookresearch/sam3`](https://github.com/facebookresearch/sam3)). Original
SAM materials are copyright Meta Platforms, Inc. and distributed under the SAM
License.

This repository does not ship official Meta checkpoint weights or converted SAM
checkpoint weights. Obtain official checkpoints from Hugging Face
([`facebook/sam3`](https://huggingface.co/facebook/sam3) and
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)) and comply with the
SAM License and applicable access terms.

## License

Distributed under the SAM License; see [`LICENSE`](LICENSE). Not affiliated with
or endorsed by Meta.
