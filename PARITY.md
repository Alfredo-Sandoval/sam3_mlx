# Parity status

`sam3_mlx` does not currently claim release-grade end-to-end numerical parity
with the official SAM 3 image runtime.

## Verified contracts

- Synthetic RGB preprocessing is compared with TorchVision over multiple aspect
  ratios with absolute and relative tolerance `1e-6`.
- Checkpoint shape and required-key coverage are validated before model
  mutation.
- Tracker and multiplex component fixtures record their upstream reference
  commit in the corresponding test modules.

The ordinary test suite may skip Torch/TorchVision comparisons when those
oracle dependencies are absent. A parity environment must preflight those
dependencies and treat their absence as a failure:

```bash
make preprocess-parity-check
```

## Evidence required before an end-to-end parity claim

A checked-in result must identify:

- official repository commit and checkpoint revision;
- source and converted checkpoint SHA-256 values;
- converter version and conversion manifest;
- input corpus and positive, negative, and empty prompts;
- detection-count agreement, per-object mask IoU, box error, and score/logit
  error;
- MLX version, Apple device, dtype, and input resolution;
- results at `1008`, `672`, and `504`;
- steady-state latency and peak active memory.

Until that artifact exists, preprocessing/component parity must not be described
as full model-output parity.
