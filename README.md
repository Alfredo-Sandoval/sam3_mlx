# sam3_mlx

**An unofficial Apple MLX port of SAM 3 / SAM 3.1 image-segmentation components for Apple Silicon.**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-000000?logo=apple&logoColor=white)](#requirements)
[![Backend](https://img.shields.io/badge/Backend-MLX-1C7C54)](https://github.com/ml-explore/mlx)
[![License](https://img.shields.io/badge/License-SAM-blue)](LICENSE)

`sam3_mlx` brings selected Segment Anything Model 3 (SAM 3 / SAM 3.1) image
components to Apple Silicon through Apple's [MLX](https://github.com/ml-explore/mlx)
framework.

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Limitations](#limitations)
- [Attribution](#attribution)
- [License](#license)

## Features

- **Image segmentation runtime** for SAM 3 on Apple Silicon, with exact
  synthetic preprocessing comparisons against TorchVision.
- **Selected-frame video API** backed by the image runtime.
- **Clear errors on unsupported paths.** Unported surfaces (training,
  evaluation, multiplex video, Triton) raise `Sam3MlxUnsupportedError`.

> [!NOTE]
> SAM 3.1 Object Multiplex / video tracking is experimental and incomplete.

## Requirements

- macOS on Apple Silicon (M-series)
- Python ≥ 3.12
- [MLX](https://github.com/ml-explore/mlx) ≥ 0.30
- SAM 3 / SAM 3.1 checkpoints, obtained separately (see
  [Attribution](#attribution))

## Installation

Install from PyPI:

```bash
pip install sam3-mlx
```

Or add it to a project with [uv](https://docs.astral.sh/uv/):

```bash
uv add sam3-mlx
```

For local development from a checkout:

```bash
uv sync
```

Optional extras:

```bash
pip install "sam3-mlx[viz]"     # matplotlib plotting helpers
pip install "sam3-mlx[video]"   # OpenCV video-file decoding
```

Or, from a local checkout:

```bash
uv sync --extra viz --extra video
```

Default image weights download from a **pinned** `mlx-community/sam3-image`
revision and are verified against a package-embedded SHA-256 before load. For
stronger provenance, convert from the official PyTorch checkpoint with
`convert_from_pytorch=True` and an immutable `--source-revision`; that path
writes `conversion-manifest.json` with source/output hashes, key counts, and
dtypes. The `0.1.2` release evidence also compares all 1,400 canonical tensors
in the pinned published artifact with a fresh conversion. PyTorch is not
installed by a `sam3-mlx` extra—use a separate environment for conversion.

Verify the install:

```bash
python -c "import sam3_mlx; print(sam3_mlx.__version__)"
```

From a local checkout, you can also run:

```bash
python -m compileall -q sam3_mlx tests
```

## Quickstart

### Image segmentation

```python
from sam3_mlx import build_sam3_image_model
from sam3_mlx.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model, resolution=1008)
```

`Sam3Processor.resolution` is the square image size fed into the ViT backbone. It
must be a positive multiple of `14` (the image patch size).

> [!TIP]
> Lower the resolution to speed up inference. Any multiple of `14` works. For
> example, `672` or `504` run faster than the default `1008`, at the cost of
> fine detail.

### Selected-frame video

```python
from sam3_mlx import build_sam3_predictor

predictor = build_sam3_predictor(version="sam3")
```

`build_sam3_predictor()` defaults to the working `version="sam3"` selected-frame
path. Request `version="sam3.1"` explicitly for the experimental multiplex
predictor; that path requires a locally converted checkpoint
(`checkpoint_path=..., load_from_HF=False`). Automatic SAM 3.1 checkpoint
download and conversion are not wired up yet.

The video slice accepts image paths, image folders, PIL image sequences, and
OpenCV-decodable video files (requires `pip install "sam3-mlx[video]"`). It
performs independent framewise inference: `out_obj_ids` are frame-local
detection IDs, not persistent identities. Cross-frame `remove_object` behavior
is rejected until temporal association is implemented. Mixed-resolution frame
collections and unsupported CPU-offload controls fail explicitly. Session
propagation cannot be mutated mid-stream (except cancellation). Selected-frame
mode does not retain full-resolution output history, and image-folder sessions
decode host RGB frames through a bounded on-demand cache rather than keeping
every frame resident.

Experimental SAM 3.1 multiplex builders live under `sam3_mlx.experimental` and
are not part of the stable 0.1.x top-level API.

Checkpoint loading requires complete model coverage by default. For controlled
development experiments only, `build_sam3_image_model` accepts
`strict_checkpoint_loading=False`; the returned model exposes the audit as
`model.checkpoint_load_report`.

## Limitations

Unsupported paths raise `Sam3MlxUnsupportedError`:

- **Apple Silicon / MLX only.** Requesting any non-MLX device is not
  supported, and neither is `torch.compile`.
- **Single-prompt image API.** Batch geometric prompts and multiple masks per
  prompt are not supported; single text or geometric prompts work.
- **SAM 3.1 multiplex needs local weights.** The multiplex video predictor runs
  only from a locally converted checkpoint; automatic download/conversion,
  multi-GPU video, and TorchCodec decoding are unavailable.
- **Training is currently not supported.** Training loops, autograd, distributed
  execution, and the official eval toolkit are not available yet.
- **Parity is scoped to the SAM 3 image runtime.** The `0.1.2` receipt covers
  pinned official-vs-MLX image outputs, including text and geometric prompts at
  1008/672/504. It does not extend to SAM 3.1 multiplex/temporal tracking,
  training, or unsupported APIs. See [`PARITY.md`](PARITY.md).

## Attribution

Portions of this repository are derived from, adapted from, or structured for
parity with the official SAM3 implementation
([`facebookresearch/sam3`](https://github.com/facebookresearch/sam3)). Original
SAM materials are copyright Meta Platforms, Inc. and are distributed under the
SAM License.

This repository does not ship official Meta checkpoint weights or converted SAM
checkpoint weights. Obtain the official checkpoints from Hugging Face
([`facebook/sam3`](https://huggingface.co/facebook/sam3) and
[`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1)), then comply with
the SAM License and any applicable access terms.

## License

Distributed under the SAM License; see [`LICENSE`](LICENSE). Not affiliated with
or endorsed by Meta.
