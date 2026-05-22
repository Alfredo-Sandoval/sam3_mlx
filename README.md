# SAM 3.1 MLX

Maintained MLX fork of the community SAM3 image port, with a tooling-safe
Python identity:

- distribution: `mlx-sam3p1`
- import package: `mlx_sam3p1`
- display name: `SAM 3.1 MLX`

This repository starts from `Deekshith-Dade/mlx_sam3` and is being prepared for
SAM 3.1-oriented image segmentation work on Apple Silicon. It deliberately does
not use the import name `sam3`; that name belongs to the official Meta package.

## Status

Early maintenance fork. The current code is not yet parity-certified against
official SAM3/SAM 3.1 behavior.

Known upstream baseline:

- MLX source: `Deekshith-Dade/mlx_sam3`
- MLX source commit: `d9a92badb6000a93135e01b89cd81a54e7ff9825`
- Official oracle repo: `facebookresearch/sam3`
- Official oracle commit used during audit:
  `2814fa619404a722d03e9a012e083e4f293a4e53`

Upstream provenance and the license-review hold are summarized in
[`UPSTREAM.md`](UPSTREAM.md).

## Setup

```bash
uv sync
```

Notebook and plotting dependencies are optional:

```bash
uv sync --extra notebooks --extra viz
```

The runtime target is macOS Apple Silicon with MLX. Linux can be useful for
static checks and packaging review, but inference is not expected to run there.

## Smoke Checks

```bash
python -m compileall -q mlx_sam3p1 examples
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

## Package Use

```python
from mlx_sam3p1 import build_sam3_image_model
from mlx_sam3p1.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)
```

## Maintenance Priorities

1. Replace residual PyTorch idioms with valid MLX operations.
2. Implement or remove public `pass` methods.
3. Restore processor state-key consistency for mask logits.
4. Add parity fixtures against official SAM3 image APIs.
5. Add synthetic tests for prompt, threshold, and no-mask behavior.
6. Review upstream licensing before publication or redistribution.

## Licensing

This is an internal maintenance fork until license review is complete. Upstream
rights and restrictions remain with their respective holders. Do not publish or
redistribute this repository until `LICENSE` and `NOTICE.md` are finalized for
the exact upstream materials retained here.
