from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PACKAGE_PREFIXES = (
    "sam3_mlx/agent/",
    "sam3_mlx/eval/",
    "sam3_mlx/train/",
    "sam3_mlx/perflib/triton/",
)
REQUIRED_WHEEL_MEMBERS = {
    "sam3_mlx/__init__.py",
    "sam3_mlx/assets/bpe_simple_vocab_16e6.txt.gz",
    "sam3_mlx/model_builder.py",
    "sam3_mlx/model/lifecycle_predictor.py",
    "sam3_mlx/parity_evidence.py",
    "sam3_mlx/release_contract.py",
    "sam3_mlx/source_binding.py",
}
REQUIRED_SDIST_SUFFIXES = {
    "pyproject.toml",
    "README.md",
    "PARITY.md",
    "ARCHITECTURE.md",
    "Makefile",
    "sam3_mlx/__init__.py",
    "sam3_mlx/assets/bpe_simple_vocab_16e6.txt.gz",
    "sam3_mlx/parity_evidence.py",
    "sam3_mlx/release_contract.py",
    "sam3_mlx/source_binding.py",
    "scripts/validate_release.py",
    "scripts/validate_runtime_release.py",
    "scripts/validate_runtime_release_hardened.py",
    "scripts/audit_release_evidence.py",
    "scripts/audit_release_candidate.py",
    "scripts/run_image_parity.py",
    "scripts/run_hardened_image_parity.py",
    "scripts/run_upstream_image_oracle_hardened.py",
    "scripts/validate_checkpoint_lineage_hardened.py",
    "parity/receipts/latest.json",
    "parity/receipts/example-image-parity.json",
    "parity/receipts/holdout-image-parity.json",
    "parity/manifests/checkpoint-lineage.json",
    "parity/evidence/example-image-parity.npz",
    "parity/evidence/holdout-image-parity.npz",
}


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _inspect_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        metadata_name = next(
            name for name in members if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
    missing = sorted(REQUIRED_WHEEL_MEMBERS - members)
    excluded = sorted(
        member
        for member in members
        if member.startswith(EXCLUDED_PACKAGE_PREFIXES)
    )
    if missing or excluded:
        raise RuntimeError(
            f"Wheel content mismatch: missing={missing}, excluded_present={excluded}"
        )
    requires_dist = metadata.get_all("Requires-Dist", [])
    if not any(
        requirement.startswith("opencv-python-headless")
        and 'extra == "video"' in requirement
        for requirement in requires_dist
    ):
        raise RuntimeError("Wheel metadata is missing the [video] OpenCV extra.")


def _inspect_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        members = [member.name for member in archive.getmembers()]
    missing = sorted(
        suffix
        for suffix in REQUIRED_SDIST_SUFFIXES
        if not any(name.endswith(suffix) for name in members)
    )
    if missing:
        raise RuntimeError(f"Source distribution is missing required files: {missing}")


def _validate_installed_wheel(wheel: Path, python: str, workspace: Path) -> None:
    environment = workspace / f"venv-{Path(python).name}"
    _run(["uv", "venv", "--python", python, str(environment)], cwd=workspace)
    environment_python = environment / "bin" / "python"
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(environment_python),
            f"{wheel}[video]",
        ],
        cwd=workspace,
    )
    contract = """
import importlib.resources
import importlib.util
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
import sam3_mlx
from sam3_mlx.parity_evidence import optimal_assignment
from sam3_mlx.release_contract import (
    MLX_CHECKPOINT_REPO,
    MLX_CHECKPOINT_REVISION,
    MLX_CHECKPOINT_SHA256,
    PACKAGE_VERSION,
)
from sam3_mlx.convert import DEFAULT_MLX_CHECKPOINT
from sam3_mlx.model.lifecycle_predictor import LifecycleSafeSam3BasePredictor
from sam3_mlx.source_binding import ATTESTATION_PATH_PREFIXES

expected_exports = {
    "Sam3MlxUnsupportedError",
    "build_sam3_image_model",
    "build_sam3_predictor",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "build_tracker",
    "download_ckpt_from_hf",
}
assert set(sam3_mlx.__all__) == expected_exports
assert sam3_mlx.__version__ == PACKAGE_VERSION
assert DEFAULT_MLX_CHECKPOINT.repo == MLX_CHECKPOINT_REPO
assert DEFAULT_MLX_CHECKPOINT.revision == MLX_CHECKPOINT_REVISION
assert DEFAULT_MLX_CHECKPOINT.output_sha256 == MLX_CHECKPOINT_SHA256
assert LifecycleSafeSam3BasePredictor.__name__ == "LifecycleSafeSam3BasePredictor"
assert ATTESTATION_PATH_PREFIXES == (
    "parity/receipts/",
    "parity/manifests/",
    "parity/evidence/",
)
assert optimal_assignment(np.eye(2)) == [(0, 0), (1, 1)]
# Experimental multiplex builders remain importable but are not stable __all__.
import sam3_mlx.experimental as experimental

assert hasattr(experimental, "build_sam3_multiplex_video_model")
assert hasattr(experimental, "build_sam3_multiplex_video_predictor")
asset = importlib.resources.files("sam3_mlx").joinpath(
    "assets/bpe_simple_vocab_16e6.txt.gz"
)
assert asset.is_file()
assert asset.read_bytes()[:2] == b"\\x1f\\x8b"
model = sam3_mlx.build_sam3_image_model(load_from_HF=False)
assert model.device == "mlx"
assert model.training is False
for module in ("sam3_mlx.agent", "sam3_mlx.eval", "sam3_mlx.train"):
    assert importlib.util.find_spec(module) is None, module
# The built wheel's [video] extra must install a working decoder path.
from sam3_mlx.model.io_utils import load_video_frames_from_video_file
with tempfile.TemporaryDirectory() as temp_dir:
    video_path = Path(temp_dir) / "fixture.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (8, 6),
    )
    assert writer.isOpened()
    writer.write(np.zeros((6, 8, 3), dtype=np.uint8))
    writer.release()
    frames = load_video_frames_from_video_file(
        video_path,
        image_size=14,
        materialize_mlx_frames=False,
    )
    assert len(frames) == 1
    assert frames[0].size == (8, 6)
    frames.close()
print(json.dumps({"exports": sorted(expected_exports), "model": type(model).__name__}))
"""
    _run(
        [str(environment_python), "-I", "-c", contract],
        cwd=workspace,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate installed sam3-mlx release artifacts."
    )
    parser.add_argument(
        "--python",
        action="append",
        dest="pythons",
        help="Python executable/version for clean-wheel validation; repeat for a matrix.",
    )
    args = parser.parse_args()
    pythons = args.pythons or [sys.executable]

    with tempfile.TemporaryDirectory(prefix="sam3-mlx-release-") as temp_dir:
        workspace = Path(temp_dir)
        dist_dir = workspace / "dist"
        _run(["uv", "build", "--out-dir", str(dist_dir)], cwd=REPO_ROOT)
        wheels = sorted(dist_dir.glob("*.whl"))
        sdists = sorted(dist_dir.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise RuntimeError(
                f"Expected one wheel and one sdist, got {wheels=} and {sdists=}."
            )
        _inspect_wheel(wheels[0])
        _inspect_sdist(sdists[0])
        for python in pythons:
            _validate_installed_wheel(wheels[0], python, workspace)
        print(
            json.dumps(
                {
                    "wheel": wheels[0].name,
                    "sdist": sdists[0].name,
                    "python_matrix": pythons,
                    "status": "passed",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
