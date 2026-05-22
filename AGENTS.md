# AGENTS.md

Project rules for agents operating in this repo. If instructions conflict,
priority is: (1) explicit user request, (2) this file, (3) other docs.

## Repository

- Name / slug: `mlx-sam3p1` / local-only until a GitHub remote exists
- Display name: SAM 3.1 MLX
- Import package: `mlx_sam3p1`
- Preset: `science`
- Primary platform: macOS Apple Silicon
- Secondary platform: Linux for static analysis and packaging checks

The display name intentionally includes `SAM 3.1`; the package and repository
names use `sam3p1` to avoid dotted-name packaging friction and to avoid
shadowing Meta's official `sam3` package.

## Project Scope

This repo is a maintained MLX fork of the community SAM3 image port. The goal is
to make a reliable SAM 3.1-oriented Apple Silicon image segmentation runtime
with explicit parity checks against `facebookresearch/sam3`.

Owned here:

- MLX image-model construction and inference paths
- tokenizer/package assets needed by the MLX runtime
- parity notes, fixtures, and regression tests for the MLX port
- example notebooks and repo-local visualization helpers

Not owned here:

- official Meta PyTorch/CUDA SAM3 implementation
- video tracker/multiplex features until deliberately ported
- downstream application workflows

## License Policy

Internal maintenance fork while license review is incomplete. Do not publish,
redistribute, relicense, or add open-source license metadata without explicit
user instruction. Preserve upstream attribution and keep `LICENSE`, `NOTICE.md`,
`README.md`, and `pyproject.toml` aligned.

## Setup And Run

```bash
uv sync
python -m compileall -q mlx_sam3p1 examples
```

Run inference only on hosts with MLX support. Do not add CPU fallbacks or
silent backend switching.

## Coding Standards

- Fail fast on missing resources and unsupported backends.
- Keep MLX and PyTorch paths explicit; no hidden backend fallback.
- Avoid broad exception handling outside top-level entry points.
- Use repo-relative paths and `pathlib` for new path handling.
- Add parity tests for every behavioral fix.
- Do not use the import package name `sam3`.

## Porting Policy

Official `facebookresearch/sam3` is the oracle for API shape and behavior.
When changing runtime logic, record the official source commit and the MLX fork
commit used for comparison. Do not claim parity without executed evidence.

## Git Safety

- Never discard user changes without explicit instruction.
- Avoid destructive commands unless explicitly requested.
- Do not create a new feature branch by default.
- Inspect `git status` before committing.

## Skill Routing

- `repo-setup`: setup, dependencies, package identity, CODEOWNERS, license.
- `port`: upstream SAM3 parity work and behavior comparison.
- `mlx-runtime`: MLX lazy evaluation, array semantics, and backend behavior.
- `test-writing`: parity and regression coverage.
