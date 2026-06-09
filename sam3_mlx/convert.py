import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Union, Optional

import mlx.core as mx
from huggingface_hub import snapshot_download


MLX_COMMUNITY_REPO = "mlx-community/sam3-image"
PYTORCH_REPO = "facebook/sam3"

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


def normalize_sam3_image_weight_layout(key: str, value):
    """Map known PyTorch conv kernels into MLX's channels-last kernel layout."""

    if (
        key in SAM3_IMAGE_CONV_TRANSPOSE2D_WEIGHTS
        and len(value.shape) == 4
        and value.shape[2] == 2
        and value.shape[3] == 2
    ):
        return value.transpose(1, 2, 3, 0)

    if (
        key in SAM3_IMAGE_CONV2D_WEIGHTS
        and len(value.shape) == 4
        and value.shape[2] == value.shape[3]
        and value.shape[1] != value.shape[2]
    ):
        return value.transpose(0, 2, 3, 1)

    return value


def load_from_hub(
    hf_repo: str = MLX_COMMUNITY_REPO,
    local_dir: Optional[str] = None,
) -> Path:
    download_kwargs = {
        "repo_id": hf_repo,
        "allow_patterns": ["*.safetensors", "*.json"],
    }

    if local_dir:
        download_kwargs["local_dir"] = local_dir

    model_path = Path(snapshot_download(**download_kwargs))
    weights_file = model_path / "model.safetensors"

    if not weights_file.exists():
        raise FileNotFoundError(f"model.safetensors not found in {hf_repo}.")

    return weights_file


def save_weights(save_path: Union[str, Path], weights: Dict[str, mx.array]) -> None:
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    model_path = save_path / "model.safetensors"
    mx.save_safetensors(str(model_path), weights)

    for weight_name in weights.keys():
        index_data["weight_map"][weight_name] = "model.safetensors"

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=4)


def download(hf_repo):
    return Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.pt", "*.json"],
        )
    )


def update_attn_keys(key, mlx_weights):
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


def _unwrap_checkpoint_payload(payload):
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), Mapping):
        return payload["model"]
    return payload


def _remap_official_checkpoint_keys(weights):
    if not any(
        k.startswith("sam3_model.") or k.startswith("sam2_predictor.") for k in weights
    ):
        return weights

    remapped = {}
    for key, value in weights.items():
        if key.startswith("sam3_model."):
            key = "detector." + key[len("sam3_model.") :]
        elif key.startswith("sam2_predictor."):
            key = "tracker." + key[len("sam2_predictor.") :]
        remapped[key] = value
    return remapped


def convert(model_path):
    import torch

    weight_file = str(model_path / "sam3.pt")
    weights = torch.load(weight_file, map_location="cpu", weights_only=True)
    weights = _unwrap_checkpoint_payload(weights)
    if not isinstance(weights, Mapping):
        raise ValueError("SAM3 PyTorch checkpoint payload must be a mapping.")
    weights = _remap_official_checkpoint_keys(dict(weights))

    mlx_weights = dict()
    for k, v in weights.items():
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

                mlx_weights[k] = v

            if k.endswith("in_proj_weight") or k.endswith("in_proj_bias"):
                update_attn_keys(k, mlx_weights)

    if not mlx_weights:
        raise ValueError(
            f"No detector weights were converted from {weight_file}. Expected "
            "official SAM3 keys with a detector. prefix."
        )

    return mlx_weights


def download_and_convert(
    hf_repo: str = PYTORCH_REPO,
    mlx_path: Union[str, Path] = "sam3-mod-weights",
    force: bool = False,
) -> Path:
    mlx_path = Path(mlx_path)
    weights_file = mlx_path / "model.safetensors"
    index_file = mlx_path / "model.safetensors.index.json"

    if weights_file.exists() and index_file.exists() and not force:
        return weights_file

    print(f"Downloading and converting weights from {hf_repo}...")
    model_path = download(hf_repo)

    mlx_path.mkdir(parents=True, exist_ok=True)

    mlx_weights = convert(model_path)
    save_weights(mlx_path, mlx_weights)

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
    args = parser.parse_args()

    if args.convert:
        mlx_path = args.mlx_path or "sam3-mod-weights"
        print(f"Converting PyTorch weights from {args.pytorch_repo}...")
        model_path = download(args.pytorch_repo)

        mlx_path = Path(mlx_path)
        mlx_path.mkdir(parents=True, exist_ok=True)

        mlx_weights = convert(model_path)
        save_weights(mlx_path, mlx_weights)
        print(f"Converted weights saved to {mlx_path}")
    else:
        print(f"Downloading MLX weights from {args.mlx_repo}...")
        weights_path = load_from_hub(args.mlx_repo, args.mlx_path)
        print(f"MLX weights available at: {weights_path}")
