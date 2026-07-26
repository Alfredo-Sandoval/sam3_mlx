from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


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
}


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _inspect_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
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


def _inspect_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as archive:
        members = [member.name for member in archive.getmembers()]
    suffixes = {
        "pyproject.toml",
        "sam3_mlx/__init__.py",
        "sam3_mlx/assets/bpe_simple_vocab_16e6.txt.gz",
    }
    missing = sorted(
        suffix for suffix in suffixes if not any(name.endswith(suffix) for name in members)
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
            str(wheel),
        ],
        cwd=workspace,
    )
    contract = """
import importlib.resources
import importlib.util
import json
import sam3_mlx

expected_exports = {
    "Sam3MlxUnsupportedError",
    "build_sam3_image_model",
    "build_sam3_multiplex_video_model",
    "build_sam3_multiplex_video_predictor",
    "build_sam3_predictor",
    "build_sam3_video_model",
    "build_sam3_video_predictor",
    "build_tracker",
    "download_ckpt_from_hf",
}
assert set(sam3_mlx.__all__) == expected_exports
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
