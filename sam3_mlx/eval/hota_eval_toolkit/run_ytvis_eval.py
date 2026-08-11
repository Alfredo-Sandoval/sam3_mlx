"""Fail-fast HOTA YTVIS runner."""

from __future__ import annotations

from typing import Never

from sam3_mlx.eval._unsupported import raise_unsupported


def run_ytvis_eval(
    args: object | None = None,
    gt_json: object | None = None,
    dt_json: object | None = None,
) -> Never:
    del args, gt_json, dt_json
    raise_unsupported("eval.hota_eval_toolkit.run_ytvis_eval")
