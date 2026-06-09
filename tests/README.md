# Test Suite Layout

The suite is grouped by the kind of contract each test protects:

- `api/`: package identity, public exports, and user-facing builder contracts.
- `checkpoint/`: checkpoint key normalization, tensor layout, and loading audits.
- `image/`: image-model processors, predictors, and image-backbone contracts.
- `multiplex/`: multiplex model construction, detector helpers, and image-base logic.
- `perflib/`: MLX ports of performance-library helpers and upstream fixtures.
- `port/tracker/`: tracker and multiplex-video port parity, oracle fixtures, and helper-island regressions.
- `runtime/`: MLX runtime primitives, attention, geometry, RoPE, RLE, and array helpers.
- `training/`: training/loss helper contracts and unsupported training-runtime boundaries.
- `unsupported/`: canonical fail-fast unsupported-surface registry and documentation sync.
- `video/`: selected-frame video API and video helper contracts.

Shared path constants live in `tests/_paths.py`. Use those constants for repo
roots and fixture paths so tests can move between folders without breaking
parity fixtures.
