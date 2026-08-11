import argparse
import hashlib
import json
import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import NotRequired, Protocol, Required, TypedDict, cast

import mlx.core as mx
import numpy as np
import numpy.typing as npt


MLX_COMMUNITY_REPO = "mlx-community/sam3-image"
PYTORCH_REPO = "facebook/sam3"
CONVERSION_MANIFEST = "conversion-manifest.json"
_COMMIT_REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


class _SnapshotDownload(Protocol):
    def __call__(
        self,
        *,
        repo_id: str,
        allow_patterns: Sequence[str],
        revision: str,
        local_dir: str | None = None,
    ) -> str: ...


class _TorchTensor(Protocol):
    def numpy(self) -> npt.NDArray[np.generic]: ...


class _TransposableArray(Protocol):
    def transpose(self, *axes: int) -> mx.array: ...


class _SaveSafetensors(Protocol):
    def __call__(self, path: str, arrays: dict[str, mx.array]) -> object: ...


class _TorchModule(Protocol):
    def load(
        self,
        path: str,
        *,
        map_location: str,
        weights_only: bool,
    ) -> object: ...


class _CliArgs(Protocol):
    mlx_repo: str
    source_revision: str | None
    pytorch_repo: str
    mlx_path: str | None
    convert: bool


class ConversionProvenance(TypedDict):
    status: Required[str]
    repo: Required[str]
    revision: Required[str]
    architecture: Required[str]
    output_sha256: Required[str]
    manifest_path: Required[str | None]
    manifest: NotRequired[dict[str, JsonValue]]


class _WeightIndex(TypedDict):
    metadata: dict[str, int]
    weight_map: dict[str, str]


snapshot_download = cast(
    _SnapshotDownload,
    getattr(import_module("huggingface_hub"), "snapshot_download"),
)


def _torch_weights(value: object) -> dict[str, _TorchTensor]:
    if not isinstance(value, Mapping):
        raise ValueError("SAM3 PyTorch checkpoint payload must be a mapping.")
    result: dict[str, _TorchTensor] = {}
    for key, tensor in cast(Mapping[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError("checkpoint weight keys must be strings")
        if not callable(getattr(tensor, "numpy", None)):
            raise TypeError(f"checkpoint weight {key!r} must expose numpy()")
        result[key] = cast(_TorchTensor, tensor)
    return result


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = _json_value(item)
        return result
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def _json_object(text: str, source: str) -> dict[str, JsonValue]:
    value = _json_value(json.loads(text))
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object.")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


@dataclass(frozen=True)
class DefaultMlxCheckpoint:
    """Pinned default preconverted MLX image checkpoint.

    The community snapshot does not currently ship a conversion-manifest.json, so
    the package pin (immutable revision + output SHA-256) is the release
    provenance source of truth. When a conversion-manifest.json is present it is
    also validated against this pin.
    """

    repo: str
    revision: str
    output_sha256: str
    architecture: str = "sam3-image"


# Pinned to the mlx-community/sam3-image revision that matches the audited
# local artifact hash. Bump both fields together when publishing a new default.
DEFAULT_MLX_CHECKPOINT = DefaultMlxCheckpoint(
    repo=MLX_COMMUNITY_REPO,
    revision="b72a14d8127e17e6f2a3d2e075bbbf4307ba146e",
    output_sha256=("0ad4c3f42ecf706c4cda63cf58d621699491ed65012b3999284ea370984f7173"),
)

SAM3_IMAGE_CONV_TRANSPOSE2D_WEIGHTS = frozenset(
    {
        "backbone.vision_backbone.convs.0.dconv_2x2_0.weight",
        "backbone.vision_backbone.convs.0.dconv_2x2_1.weight",
        "backbone.vision_backbone.convs.1.dconv_2x2.weight",
        "backbone.vision_backbone.sam2_convs.0.dconv_2x2_0.weight",
        "backbone.vision_backbone.sam2_convs.0.dconv_2x2_1.weight",
        "backbone.vision_backbone.sam2_convs.1.dconv_2x2.weight",
    }
)

SAM3_IMAGE_CONV2D_WEIGHTS = frozenset(
    {
        "backbone.vision_backbone.trunk.patch_embed.proj.weight",
        "backbone.vision_backbone.convs.0.conv_1x1.weight",
        "backbone.vision_backbone.convs.0.conv_3x3.weight",
        "backbone.vision_backbone.convs.1.conv_1x1.weight",
        "backbone.vision_backbone.convs.1.conv_3x3.weight",
        "backbone.vision_backbone.convs.2.conv_1x1.weight",
        "backbone.vision_backbone.convs.2.conv_3x3.weight",
        "backbone.vision_backbone.convs.3.conv_1x1.weight",
        "backbone.vision_backbone.convs.3.conv_3x3.weight",
        "backbone.vision_backbone.sam2_convs.0.conv_1x1.weight",
        "backbone.vision_backbone.sam2_convs.0.conv_3x3.weight",
        "backbone.vision_backbone.sam2_convs.1.conv_1x1.weight",
        "backbone.vision_backbone.sam2_convs.1.conv_3x3.weight",
        "backbone.vision_backbone.sam2_convs.2.conv_1x1.weight",
        "backbone.vision_backbone.sam2_convs.2.conv_3x3.weight",
        "backbone.vision_backbone.sam2_convs.3.conv_1x1.weight",
        "backbone.vision_backbone.sam2_convs.3.conv_3x3.weight",
        "geometry_encoder.boxes_pool_project.weight",
        "segmentation_head.pixel_decoder.conv_layers.0.weight",
        "segmentation_head.pixel_decoder.conv_layers.1.weight",
        "segmentation_head.pixel_decoder.conv_layers.2.weight",
        "segmentation_head.semantic_seg_head.weight",
        "segmentation_head.instance_seg_head.weight",
    }
)


def normalize_sam3_image_weight_layout(key: str, value: mx.array) -> mx.array:
    """Map known PyTorch conv kernels into MLX's channels-last kernel layout."""

    if (
        key in SAM3_IMAGE_CONV_TRANSPOSE2D_WEIGHTS
        and len(value.shape) == 4
        and value.shape[2] == 2
        and value.shape[3] == 2
    ):
        return cast(_TransposableArray, value).transpose(1, 2, 3, 0)

    if (
        key in SAM3_IMAGE_CONV2D_WEIGHTS
        and len(value.shape) == 4
        and value.shape[2] == value.shape[3]
        and value.shape[1] != value.shape[2]
    ):
        return cast(_TransposableArray, value).transpose(0, 2, 3, 1)

    return value


def load_from_hub(
    hf_repo: str = DEFAULT_MLX_CHECKPOINT.repo,
    local_dir: str | None = None,
    revision: str | None = None,
    *,
    expected_output_sha256: str | None = None,
    expected_architecture: str = "sam3-image",
    verify_provenance: bool = True,
) -> Path:
    if revision is None:
        raise ValueError(
            "load_from_hub requires an immutable Hugging Face revision "
            "(full 40-character commit SHA). Pass revision= explicitly or use "
            "DEFAULT_MLX_CHECKPOINT.revision."
        )
    revision = _validate_source_revision(revision)

    if local_dir:
        snapshot_path = snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
            revision=revision,
            local_dir=local_dir,
        )
    else:
        snapshot_path = snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.safetensors", "*.json"],
            revision=revision,
        )
    model_path = Path(snapshot_path)
    weights_file = model_path / "model.safetensors"

    if not weights_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {hf_repo}.")

    if verify_provenance:
        if expected_output_sha256 is None:
            if (
                hf_repo == DEFAULT_MLX_CHECKPOINT.repo
                and revision == DEFAULT_MLX_CHECKPOINT.revision
            ):
                expected_output_sha256 = DEFAULT_MLX_CHECKPOINT.output_sha256
                expected_architecture = DEFAULT_MLX_CHECKPOINT.architecture
            else:
                raise ValueError(
                    "load_from_hub provenance verification requires "
                    "expected_output_sha256 for non-default repository/revision "
                    "pairs. Pass expected_output_sha256=... or "
                    "verify_provenance=False for an unverified experimental load."
                )
        validate_hub_checkpoint_provenance(
            weights_file.parent,
            expected_repo=hf_repo,
            expected_revision=revision,
            expected_output_sha256=expected_output_sha256,
            expected_architecture=expected_architecture,
        )

    return weights_file


def validate_hub_checkpoint_provenance(
    checkpoint_dir: str | Path,
    *,
    expected_repo: str,
    expected_revision: str,
    expected_output_sha256: str,
    expected_architecture: str = "sam3-image",
) -> ConversionProvenance:
    """Validate a downloaded MLX checkpoint before model mutation.

    Always checks the weights digest against the package/caller pin. When a
    conversion-manifest.json is present, architecture and output_sha256 must
    match the pin as well.
    """
    expected_repo = _nonempty_string(expected_repo, "expected_repo")
    expected_architecture = _nonempty_string(
        expected_architecture, "expected_architecture"
    )
    expected_revision = _validate_source_revision(expected_revision)
    expected_output_sha256 = _validate_sha256(
        expected_output_sha256, "expected_output_sha256"
    )

    checkpoint_dir = Path(checkpoint_dir)
    weights_file = checkpoint_dir / "model.safetensors"
    if not weights_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {checkpoint_dir}.")

    actual_sha = _sha256(weights_file)
    if actual_sha != expected_output_sha256:
        raise ValueError(
            "Downloaded MLX checkpoint content hash does not match the pinned "
            f"expected digest: actual={actual_sha}, "
            f"expected={expected_output_sha256}, repo={expected_repo}, "
            f"revision={expected_revision}."
        )

    manifest_file = checkpoint_dir / CONVERSION_MANIFEST
    provenance: ConversionProvenance = {
        "status": "package-pinned",
        "repo": expected_repo,
        "revision": expected_revision,
        "architecture": expected_architecture,
        "output_sha256": actual_sha,
        "manifest_path": None,
    }
    if not manifest_file.exists():
        return provenance

    manifest = _json_object(manifest_file.read_text(), "conversion-manifest.json")
    expected = {
        "architecture": expected_architecture,
        "artifact_repo": expected_repo,
        "artifact_revision": expected_revision,
        "output_sha256": expected_output_sha256,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: manifest={cached!r}, expected={requested!r}"
            for key, (cached, requested) in sorted(mismatches.items())
        )
        raise ValueError(
            "conversion-manifest.json does not match the pinned default "
            f"checkpoint provenance. {details}"
        )
    provenance["status"] = "manifest-verified"
    provenance["manifest_path"] = str(manifest_file)
    provenance["manifest"] = manifest
    return provenance


def save_weights(save_path: str | Path, weights: dict[str, mx.array]) -> Path:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    total_size = sum(v.nbytes for v in weights.values())
    index_data: _WeightIndex = {
        "metadata": {"total_size": total_size},
        "weight_map": {},
    }

    model_path = save_path / "model.safetensors"
    save_safetensors = cast(_SaveSafetensors, getattr(mx, "save_safetensors"))
    save_safetensors(str(model_path), weights)

    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "model.safetensors"

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)
    return model_path


def download(hf_repo: str, *, revision: str) -> Path:
    return Path(
        snapshot_download(
            repo_id=hf_repo,
            revision=revision,
            allow_patterns=["*.pt", "*.json"],
        )
    )


def update_attn_keys(key: str, mlx_weights: MutableMapping[str, mx.array]) -> None:
    value = mlx_weights[key]
    del mlx_weights[key]

    if "in_proj_weight" in key:
        qkv, _ = value.shape[0], value.shape[1]
        qkv_dim = qkv // 3
        key_prefix = key.rsplit(".", 1)[0]
        new_dict = {
            f"{key_prefix}.query_proj.weight": value[0:qkv_dim, :],
            f"{key_prefix}.key_proj.weight": value[qkv_dim : 2 * qkv_dim, :],
            f"{key_prefix}.value_proj.weight": value[2 * qkv_dim :, :],
        }
        mlx_weights.update(new_dict)

    if "in_proj_bias" in key:
        qkv = value.shape[0]
        qkv_dim = qkv // 3
        key_prefix = key.rsplit(".", 1)[0]
        new_dict = {
            f"{key_prefix}.query_proj.bias": value[0:qkv_dim],
            f"{key_prefix}.key_proj.bias": value[qkv_dim : 2 * qkv_dim],
            f"{key_prefix}.value_proj.bias": value[2 * qkv_dim :],
        }
        mlx_weights.update(new_dict)


def _unwrap_checkpoint_payload(payload: object) -> object:
    if isinstance(payload, Mapping):
        mapping = cast(Mapping[object, object], payload)
        model = mapping.get("model")
        if isinstance(model, Mapping):
            return cast(Mapping[object, object], model)
    return cast(object, payload)


def _remap_official_checkpoint_keys[T](
    weights: Mapping[str, T],
) -> Mapping[str, T]:
    if not any(
        k.startswith("sam3_model.") or k.startswith("sam2_predictor.") for k in weights
    ):
        return weights

    remapped: dict[str, T] = {}
    for key, value in weights.items():
        if key.startswith("sam3_model."):
            key = "detector." + key[len("sam3_model.") :]
        elif key.startswith("sam2_predictor."):
            key = "tracker." + key[len("sam2_predictor.") :]
        remapped[key] = value
    return remapped


def _convert_checkpoint_weights(
    weights: Mapping[str, _TorchTensor], *, source_label: str
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    mlx_weights: dict[str, mx.array] = {}
    ignored_keys: list[str] = []
    unmapped_keys: list[str] = []
    for k, v in weights.items():
        source_key = k
        if k.startswith("tracker."):
            ignored_keys.append(source_key)
            continue
        # Vision Encoder
        if "detector" in k:
            k = k.replace("detector.", "")
            # vision and language backbone
            if k.startswith("backbone."):
                v = mx.array(v.numpy())
                v = normalize_sam3_image_weight_layout(k, v)
                mlx_weights[k] = v

            # transformer fusion encoder, detr decoder
            elif k.startswith("transformer."):
                v = mx.array(v.numpy())
                mlx_weights[k] = v

            # dot product scoring mlp layer
            elif k.startswith("dot_prod_scoring."):
                v = mx.array(v.numpy())
                mlx_weights[k] = v

            # segmentation_head
            elif k.startswith("segmentation_head."):
                v = mx.array(v.numpy())
                v = normalize_sam3_image_weight_layout(k, v)
                mlx_weights[k] = v

            # geometry encoder
            elif k.startswith("geometry_encoder."):
                v = mx.array(v.numpy())
                v = normalize_sam3_image_weight_layout(k, v)
                mlx_weights[k] = v
            else:
                unmapped_keys.append(source_key)
                continue

            if k.endswith("in_proj_weight") or k.endswith("in_proj_bias"):
                update_attn_keys(k, mlx_weights)
        else:
            unmapped_keys.append(source_key)

    if not mlx_weights:
        raise ValueError(
            f"No detector weights were converted from {source_label}. Expected "
            "official SAM3 keys with a detector. prefix."
        )
    if unmapped_keys:
        examples = ", ".join(sorted(unmapped_keys)[:5])
        raise ValueError(
            "SAM3 conversion encountered source keys without a reviewed mapping: "
            f"count={len(unmapped_keys)}; examples={examples}"
        )

    return mlx_weights, tuple(sorted(ignored_keys))


def convert(model_path: Path) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    torch = cast(_TorchModule, import_module("torch"))
    weight_file = str(model_path / "sam3.pt")
    weights = torch.load(weight_file, map_location="cpu", weights_only=True)
    weights = _unwrap_checkpoint_payload(weights)
    weights = _remap_official_checkpoint_keys(_torch_weights(weights))
    return _convert_checkpoint_weights(weights, source_label=weight_file)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_conversion_manifest(
    output_dir: Path,
    *,
    source_repo: str,
    source_revision: str,
    source_checkpoint: Path,
    output_checkpoint: Path,
    weights: Mapping[str, mx.array],
    ignored_keys: tuple[str, ...],
) -> Path:
    dtype_counts: dict[str, int] = {}
    for value in weights.values():
        dtype_name = str(value.dtype).rsplit(".", 1)[-1]
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
    manifest = {
        "architecture": "sam3-image",
        "source_repo": source_repo,
        "source_revision": source_revision,
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "converter_version": version("sam3-mlx"),
        "output_sha256": _sha256(output_checkpoint),
        "mapped_count": len(weights),
        "unmapped_keys": [],
        "ignored_keys": list(ignored_keys),
        "dtype_counts": dict(sorted(dtype_counts.items())),
    }
    manifest_path = output_dir / CONVERSION_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _validate_cached_conversion(
    weights_file: Path,
    manifest_file: Path,
    *,
    source_repo: str,
    source_revision: str,
) -> None:
    manifest = _json_object(manifest_file.read_text(), "conversion-manifest.json")
    expected = {
        "source_repo": source_repo,
        "source_revision": source_revision,
        "output_sha256": _sha256(weights_file),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: cached={cached!r}, requested={requested!r}"
            for key, (cached, requested) in sorted(mismatches.items())
        )
        raise ValueError(
            "Cached MLX conversion provenance does not match the requested "
            f"source; use force=True to regenerate. {details}"
        )


def _validate_source_revision(source_revision: object) -> str:
    if not isinstance(source_revision, str):
        raise TypeError("source_revision must be a string")
    if not _COMMIT_REVISION_PATTERN.fullmatch(source_revision):
        raise ValueError(
            "source_revision must be a full 40-character hexadecimal commit SHA."
        )
    return source_revision.lower()


def download_and_convert(
    hf_repo: str = PYTORCH_REPO,
    mlx_path: str | Path = "sam3-mod-weights",
    force: bool = False,
    *,
    source_revision: str,
) -> Path:
    source_revision = _validate_source_revision(source_revision)
    mlx_path = Path(mlx_path)
    weights_file = mlx_path / "model.safetensors"
    index_file = mlx_path / "model.safetensors.index.json"
    manifest_file = mlx_path / CONVERSION_MANIFEST

    if (
        weights_file.exists()
        and index_file.exists()
        and manifest_file.exists()
        and not force
    ):
        _validate_cached_conversion(
            weights_file,
            manifest_file,
            source_repo=hf_repo,
            source_revision=source_revision,
        )
        return weights_file
    if not force and any(
        path.exists() for path in (weights_file, index_file, manifest_file)
    ):
        raise ValueError(
            "Cached MLX conversion is incomplete and cannot be verified; use "
            "force=True to regenerate it."
        )

    print(f"Downloading and converting weights from {hf_repo}...")
    model_path = download(hf_repo, revision=source_revision)
    source_checkpoint = model_path / "sam3.pt"

    mlx_path.mkdir(parents=True, exist_ok=True)

    mlx_weights, ignored_keys = convert(model_path)
    output_checkpoint = save_weights(mlx_path, mlx_weights)
    _write_conversion_manifest(
        mlx_path,
        source_repo=hf_repo,
        source_revision=source_revision,
        source_checkpoint=source_checkpoint,
        output_checkpoint=output_checkpoint,
        weights=mlx_weights,
        ignored_keys=ignored_keys,
    )

    return weights_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download SAM-3 MLX weights or convert from PyTorch"
    )
    parser.add_argument(
        "--mlx-repo",
        default=MLX_COMMUNITY_REPO,
        type=str,
        help=f"MLX Community repo to download pre-converted weights (default: {MLX_COMMUNITY_REPO})",
    )
    parser.add_argument(
        "--source-revision",
        type=str,
        default=None,
        help="Immutable Hugging Face commit revision required for conversion.",
    )
    parser.add_argument(
        "--pytorch-repo",
        default=PYTORCH_REPO,
        type=str,
        help=f"PyTorch repo to download and convert weights (default: {PYTORCH_REPO})",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default=None,
        help="Local path to save/cache the MLX Model weights.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert from PyTorch weights instead of loading pre-converted MLX weights",
    )
    args = cast(_CliArgs, parser.parse_args())

    if args.convert:
        if not args.source_revision:
            parser.error("--source-revision is required with --convert")
        try:
            args.source_revision = _validate_source_revision(args.source_revision)
        except ValueError as exc:
            parser.error(str(exc))
        mlx_path = args.mlx_path or "sam3-mod-weights"
        print(f"Converting PyTorch weights from {args.pytorch_repo}...")
        model_path = download(
            args.pytorch_repo,
            revision=args.source_revision,
        )

        mlx_path = Path(mlx_path)
        mlx_path.mkdir(parents=True, exist_ok=True)

        source_checkpoint = model_path / "sam3.pt"
        mlx_weights, ignored_keys = convert(model_path)
        output_checkpoint = save_weights(mlx_path, mlx_weights)
        _write_conversion_manifest(
            mlx_path,
            source_repo=args.pytorch_repo,
            source_revision=args.source_revision,
            source_checkpoint=source_checkpoint,
            output_checkpoint=output_checkpoint,
            weights=mlx_weights,
            ignored_keys=ignored_keys,
        )
        print(f"Converted weights saved to {mlx_path}")
    else:
        print(f"Downloading MLX weights from {args.mlx_repo}...")
        if args.mlx_repo == DEFAULT_MLX_CHECKPOINT.repo and not args.source_revision:
            revision = DEFAULT_MLX_CHECKPOINT.revision
            expected_sha = DEFAULT_MLX_CHECKPOINT.output_sha256
        elif args.source_revision:
            revision = _validate_source_revision(args.source_revision)
            expected_sha = None
        else:
            parser.error(
                "--source-revision is required when downloading a non-default "
                "MLX repository (use a full 40-character commit SHA)."
            )
        weights_path = load_from_hub(
            args.mlx_repo,
            args.mlx_path,
            revision=revision,
            expected_output_sha256=expected_sha,
            verify_provenance=expected_sha is not None,
        )
        print(f"MLX weights available at: {weights_path}")
