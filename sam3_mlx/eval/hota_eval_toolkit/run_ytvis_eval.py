"""Fail-fast HOTA YTVIS runner."""

from __future__ import annotations

from sam3_mlx.eval._unsupported import raise_unsupported


def run_ytvis_eval(args=None, gt_json=None, dt_json=None):
    raise_unsupported("eval.hota_eval_toolkit.run_ytvis_eval")
