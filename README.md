# sam3_mlx

**An unofficial Apple MLX port of SAM 3 / SAM 3.1 image-segmentation components for Apple Silicon.**

[![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
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

- **Image segmentation runtime** for SAM 3 / SAM 3.1 on Apple Silicon, with
  output validated against the official SAM 3 image model.
- **Selected-frame video API** backed by the image runtime.
- **Clear errors on unsupported paths.** Unported surfaces (training,
  evaluation, multiplex video, Triton) raise `Sam3MlxUnsupportedError`.

> [!NOTE]
> SAM 3.1 Object Multiplex / video tracking is experimental and incomplete.

## Requirements

- macOS on Apple Silicon (M-series)
- Python ≥ 3.13
- [MLX](https://github.com/ml-explore/mlx) ≥ 0.30
- SAM 3 / SAM 3.1 checkpoints, obtained separately (see
  [Attribution](#attribution))

## Installation

Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Plotting dependencies are optional:

```bash
uv sync --extra viz
```

Verify the install:

```bash
python -m compileall -q sam3_mlx tests
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
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

`build_sam3_predictor()` defaults to `version="sam3.1"` to match the official
SAM3 API shape, but that path routes to the unported multiplex predictor and
raises `Sam3MlxUnsupportedError(reason="video-multiplex")`. Use `version="sam3"`
for the current MLX selected-frame video slice.

The video slice accepts image paths, image folders, PIL image sequences, and
OpenCV-decodable video files.

## Limitations

Unsupported paths raise `Sam3MlxUnsupportedError`:

- **Apple Silicon / MLX only.** Requesting any non-MLX device is not
  supported, and neither is `torch.compile`.
- **Single-prompt image API.** Batch geometric prompts and multiple masks per
  prompt are not supported; single text or geometric prompts work.
- **Selected-frame video only.** SAM 3.1 multiplex prediction, tracker memory,
  multi-GPU video, and TorchCodec decoding are unavailable.
- **Training is currently not supported.** Training loops, autograd, distributed
  execution, and the official eval toolkit are not available yet.

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
