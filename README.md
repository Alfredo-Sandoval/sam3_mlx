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
- [Reproducibility](#reproducibility)
- [Attribution](#attribution)
- [License](#license)

## Features

- **Checkpoint-loaded image segmentation runtime** for SAM 3 / SAM 3.1 on
  Apple Silicon, covered by an end-to-end real-weight inference workflow.
- **Selected-frame video API** backed by the image runtime.
- **Clear errors on unsupported runtime paths.** Packaged APIs fail with
  `Sam3MlxUnsupportedError` when a requested operation is not available on MLX.

> [!NOTE]
> SAM 3.1 Object Multiplex / video tracking is experimental and incomplete.

## Requirements

- macOS on Apple Silicon (M-series)
- Python ≥ 3.13
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
uv sync --locked --group dev
uv run ruff check sam3_mlx tests
uv run pyright
uv run pytest -q
```

Plotting dependencies are optional:

```bash
pip install "sam3-mlx[viz]"
```

Or, from a local checkout:

```bash
uv sync --extra viz
```

OpenCV video-file decoding is also optional:

```bash
pip install "sam3-mlx[video]"
```

Checkpoint conversion helpers are included for advanced use, but PyTorch is not
installed by a `sam3-mlx` extra in this release. Use a separate compatible
PyTorch environment before running `sam3_mlx.convert`.

Custom PyTorch repositories require an immutable revision, and converted caches
record their source repository and revision:

```bash
python -m sam3_mlx.convert --convert \
  --pytorch-repo owner/repository \
  --pytorch-revision <commit-sha> \
  --mlx-path sam3-mod-weights
```

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
from PIL import Image

from sam3_mlx import build_sam3_image_model
from sam3_mlx.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model, resolution=1008)

image = Image.open("image.jpg").convert("RGB")
state = processor.set_image(image)
output = processor.set_text_prompt(prompt="person", state=state)

masks = output["masks"]
boxes = output["boxes"]
scores = output["scores"]
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

predictor = build_sam3_predictor()

session = predictor.handle_request(
    {"type": "start_session", "resource_path": "frames/"}
)
result = predictor.handle_request(
    {
        "type": "add_prompt",
        "session_id": session["session_id"],
        "frame_index": 0,
        "text": "person",
    }
)
```

The MLX convenience builder defaults to `version="sam3"`, the implemented
selected-frame path. Pass `version="sam3.1"` explicitly for the experimental
SAM 3.1 multiplex predictor. That path can be constructed with a locally
converted checkpoint
(`checkpoint_path=..., load_from_HF=False`); automatic checkpoint download and
conversion are not wired up yet, so `load_from_HF=True` raises
`Sam3MlxUnsupportedError(reason="video-multiplex")`. Use `version="sam3"` for
the selected-frame video slice with automatic weights.

The video slice accepts image paths, image folders, PIL image sequences, and
OpenCV-decodable video files. Install `sam3-mlx[video]` when passing encoded
video files. Encoded videos and image folders are decoded on demand by the
selected-frame runtime instead of also materializing an unused full tensor
stack.

## Limitations

Unsupported paths raise `Sam3MlxUnsupportedError`:

- **Apple Silicon / MLX only.** Requesting any non-MLX device is not
  supported, and neither is `torch.compile`.
- **Single-prompt image API.** Batch geometric prompts and multiple masks per
  prompt are not supported; single text or geometric prompts work.
- **No video CPU-offload or background-loading modes.** The MLX selected-frame
  API rejects the upstream Torch offload and async-loading flags instead of
  silently ignoring them.
- **SAM 3.1 multiplex needs local weights.** The multiplex video predictor runs
  only from a locally converted checkpoint; automatic download/conversion,
  multi-GPU video, and TorchCodec decoding are unavailable.
- **Training is currently not supported.** Training loops, autograd, distributed
  execution, and the official eval toolkit are not available yet.
- **Source-only compatibility trees are not distributed.** The checkout keeps
  `agent`, `eval`, `train`, and Triton compatibility modules for porting and
  contract tests; release wheels intentionally exclude those unsupported trees.

## Reproducibility

Default checkpoint downloads are revision-pinned. The official source commit,
upstream API comparison, intentional MLX default differences, and exact model
revisions are recorded in [`UPSTREAM.md`](UPSTREAM.md).

Pull requests run the Apple Silicon unit, packaging, lint, and maintained-surface
type gates. [Weighted inference](.github/workflows/weighted-inference.yml) also
runs when model-loading code changes, on manual dispatch, and weekly against the
pinned real checkpoint.

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
