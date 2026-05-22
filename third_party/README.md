# Third-party Reference Checkouts

This directory is for local, ignored reference checkouts used while auditing
the sam3_mlx port against upstream implementations.

## Facebook SAM 3

Official upstream:
`https://github.com/facebookresearch/sam3.git`

Current local reference commit:
`2814fa6`

Local path:
`third_party/facebook-sam3/`

Refresh command:

```bash
rm -rf third_party/facebook-sam3
git clone --depth=1 https://github.com/facebookresearch/sam3.git third_party/facebook-sam3
git -C third_party/facebook-sam3 rev-parse --short HEAD
```

The checkout is intentionally gitignored. Keep only audit notes, parity tests,
and ported MLX implementation changes in this repository unless license review
explicitly approves vendoring upstream source into tracked history.
