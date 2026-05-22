# MLX Port Audit

## Reference Sources

- Official SAM 3 oracle: `https://github.com/facebookresearch/sam3`
- Local ignored checkout: `third_party/facebook-sam3/`
- Official audit commit: `2814fa619404a722d03e9a012e083e4f293a4e53`
- Community MLX import commit: `d9a92badb6000a93135e01b89cd81a54e7ff9825`

## Fixed In This Pass

- Removed the web demo app and demo-only entrypoint so the fork focuses on the
  reusable MLX package.
- Renamed the import package to `sam3_mlx` and moved runtime assets into the
  package.
- Removed broad PyTorch and visualization dependencies from the core install;
  conversion and notebooks now live behind extras.
- Replaced obvious PyTorch API leftovers in model paths, including `dim=`,
  `.float()`, `.view()`, `.permute()`, and `mx.cat` translations where they
  would fail or behave incorrectly under MLX.
- Fixed prompt state handling in `Sam3ImageProcessor.reset_all_prompts` and
  `set_confidence_threshold`.
- Fixed RoPE y-coordinate generation to use floor division like upstream SAM 3.
- Fixed absolute-position resize shape handling in the ViT backbone.
- Implemented relative-position attention concatenation for the fixed-size MLX
  path instead of silently skipping it.
- Replaced the activation-checkpoint `pass` path with normal block execution so
  training-mode calls do not skip ViT blocks.
- Normalized PyTorch-style and MLX-style multi-head attention call signatures
  at one wrapper boundary.
- Converted ignored PyTorch compile/checkpoint flags into explicit
  `NotImplementedError` failures.

## Remaining Explicit Boundaries

- `ViT.get_layer_id` is still unsupported.
- Relative-position interpolation is unsupported when the learned table size
  does not match the requested runtime size.
- Untied decoder box heads are unsupported.
- Batched image setup in `Sam3ImageProcessor` is unsupported.
- Neck scale factors outside `{4.0, 2.0, 1.0, 0.5}` are unsupported.
- MLX runtime import and inference still need to be validated on an Apple
  Silicon environment with `mlx` installed.

## Current Static Gates

```bash
ruff check .
python -m compileall -q sam3_mlx examples tests
python -c "import json, pathlib, tomllib; tomllib.load(open('pyproject.toml', 'rb')); [json.load(open(p)) for p in pathlib.Path('examples').glob('*.ipynb')]"
uv lock --check
uv run --with pytest python -m pytest -q
```

These gates passed after the package identity cutover. On this Linux host, the
MLX runtime-backed geometry tests are skipped because `mlx.core` cannot load
`libmlx.so`; static package/import contract tests still run.
