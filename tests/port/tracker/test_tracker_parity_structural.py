"""Structural parity of the ported Sam3TrackerBase against the official module.

The oracle fixture `tests/fixtures/port/tracker/official_tracker_base_state.json`
records `{name: shape}` for every parameter in the official torch
`Sam3TrackerBase.state_dict()`, captured on a CPU-shim oracle by
`scripts/tracker_parity.py` at upstream commit
2814fa619404a722d03e9a012e083e4f293a4e53.

This test reads that committed fixture (no torch needed) and asserts the MLX
parameter tree covers every official learned parameter with a compatible shape.
The only MLX-side extras allowed are deterministic precomputed buffers (RoPE
frequency tables and sine position-encoding caches) that torch keeps as
non-persistent buffers and therefore omits from its state_dict.
"""

import json

import pytest
from mlx.utils import tree_flatten

from sam3_mlx.model.sam3_tracker_base import Sam3TrackerBase
from sam3_mlx.model_builder import (
    _create_tracker_maskmem_backbone,
    _create_tracker_transformer,
)
from tests._paths import PORT_TRACKER_FIXTURE_ROOT

OFFICIAL_STATE_FIXTURE = PORT_TRACKER_FIXTURE_ROOT / "official_tracker_base_state.json"

# MLX stores these as array attributes; torch computes/keeps them as
# non-persistent buffers, so they never appear in the official state_dict.
_DERIVED_BUFFER_MARKERS = ("freqs_cis", "position_encoding.cache")


def _normalize_mlx_name(name: str) -> str:
    # NCHW conv wrappers (_NCHWConv2d / Conv2dNCHW) insert a ``.conv`` segment
    # that the official torch nn.Conv2d module does not have.
    return name.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")


def _layout_equivalent(a: tuple, b: tuple) -> bool:
    # conv NCHW (O,I,H,W) vs MLX NHWC (O,H,W,I): same dims, different order.
    return sorted(a) == sorted(b)


@pytest.fixture(scope="module")
def official_state():
    if not OFFICIAL_STATE_FIXTURE.exists():
        pytest.skip(
            "official tracker oracle fixture missing; regenerate with "
            "scripts/tracker_parity.py --refresh-oracle"
        )
    return {
        k: tuple(v) for k, v in json.loads(OFFICIAL_STATE_FIXTURE.read_text()).items()
    }


@pytest.fixture(scope="module")
def mlx_state():
    base = Sam3TrackerBase(
        backbone=None,
        transformer=_create_tracker_transformer(),
        maskmem_backbone=_create_tracker_maskmem_backbone(),
        image_size=1008,
        num_maskmem=7,
        backbone_stride=14,
        multimask_output_in_sam=True,
        forward_backbone_per_frame_for_eval=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        non_overlap_masks_for_mem_enc=False,
        max_cond_frames_in_attn=4,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )
    return {k: tuple(v.shape) for k, v in tree_flatten(base.parameters())}


def test_every_official_param_is_covered(official_state, mlx_state):
    mlx_norm = {_normalize_mlx_name(k): v for k, v in mlx_state.items()}
    missing = [name for name in official_state if name not in mlx_norm]
    assert missing == [], f"official params absent from MLX tree: {missing[:10]}"


def test_no_shape_mismatches(official_state, mlx_state):
    mlx_norm = {_normalize_mlx_name(k): v for k, v in mlx_state.items()}
    mismatches = []
    for name, oshape in official_state.items():
        mshape = mlx_norm[name]
        if tuple(mshape) != tuple(oshape) and not _layout_equivalent(
            tuple(mshape), tuple(oshape)
        ):
            mismatches.append((name, oshape, mshape))
    assert mismatches == [], f"shape mismatches: {mismatches[:10]}"


def test_mlx_only_params_are_derived_buffers(official_state, mlx_state):
    official_norm = set(official_state)
    extras = [
        name for name in mlx_state if _normalize_mlx_name(name) not in official_norm
    ]
    non_derived = [
        name
        for name in extras
        if not any(marker in name for marker in _DERIVED_BUFFER_MARKERS)
    ]
    assert non_derived == [], (
        "MLX tracker has non-derived parameters with no official counterpart: "
        f"{non_derived[:10]}"
    )


def test_official_oracle_param_count(official_state):
    # Guards against the fixture silently changing if the oracle is regenerated
    # against a different upstream revision.
    assert len(official_state) == 309
