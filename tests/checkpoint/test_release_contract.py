from pathlib import Path

from sam3_mlx.convert import DEFAULT_MLX_CHECKPOINT
from sam3_mlx.release_contract import (
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    MLX_CHECKPOINT_SHA256,
    RELEASE_IMAGES,
    build_oracle_bindings,
    sha256_path,
)


def test_release_contract_matches_default_hub_checkpoint_pin():
    assert DEFAULT_MLX_CHECKPOINT.repo == MLX_CHECKPOINT_REPO
    assert DEFAULT_MLX_CHECKPOINT.revision == MLX_CHECKPOINT_REVISION
    assert DEFAULT_MLX_CHECKPOINT.output_sha256 == MLX_CHECKPOINT_SHA256


def test_oracle_bindings_include_current_contract_digest():
    contract_path = (
        Path(__file__).resolve().parents[2] / "sam3_mlx" / "release_contract.py"
    )

    bindings = build_oracle_bindings(
        image_sha256="a" * 64,
        case_spec_sha256="b" * 64,
        confidence_threshold=0.5,
        oracle_runner_sha256="c" * 64,
    )

    assert bindings["release_contract_sha256"] == sha256_path(contract_path)


def test_release_images_are_distinct_and_content_addressed():
    assert set(RELEASE_IMAGES) == {"example", "holdout"}
    assert RELEASE_IMAGES["example"]["sha256"] != RELEASE_IMAGES["holdout"]["sha256"]
    assert RELEASE_IMAGES["example"]["path"] != RELEASE_IMAGES["holdout"]["path"]
    for image in RELEASE_IMAGES.values():
        assert image["path"].startswith("official-checkout/assets/images/")
        assert len(image["sha256"]) == 64
        assert len(image["size"]) == 2
