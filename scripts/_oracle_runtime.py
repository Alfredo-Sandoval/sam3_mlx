"""Typed shared boundaries for the official Torch image-oracle runners."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Never, Protocol, Self, TypedDict, cast

import numpy as np
from numpy.typing import NDArray


type DynamicCall = Callable[..., object]


class OracleTensor(Protocol):
    def detach(self) -> Self: ...

    def cpu(self) -> Self: ...

    def float(self) -> Self: ...

    def numpy(self) -> NDArray[np.generic]: ...

    def __len__(self) -> int: ...


class OracleState(TypedDict):
    masks: OracleTensor
    boxes: OracleTensor
    scores: OracleTensor


class GeometricPrompt(TypedDict):
    box: list[int | float]
    label: bool


class OracleCase(TypedDict):
    name: str
    resolution: int
    prompt: str | None
    geometric_prompts: list[GeometricPrompt]


class OracleProcessor(Protocol):
    def reset_all_prompts(self, state: OracleState) -> object: ...

    def set_text_prompt(self, prompt: str, state: OracleState) -> OracleState: ...

    def add_geometric_prompt(
        self, box: Sequence[int | float], label: bool, state: OracleState
    ) -> OracleState: ...

    def set_image(self, image: object) -> OracleState: ...


class TensorClass(Protocol):
    pin_memory: DynamicCall


class TorchRuntime(Protocol):
    zeros: DynamicCall
    arange: DynamicCall
    Tensor: TensorClass
    autocast: Callable[..., ContextManager[None]]
    bfloat16: object
    __version__: object


class OracleModelBuilder(Protocol):
    def build_sam3_image_model(
        self,
        *,
        checkpoint_path: str,
        load_from_HF: bool,
        device: str,
        enable_inst_interactivity: bool,
    ) -> OracleModel: ...


class OracleProcessorFactory(Protocol):
    def __call__(
        self,
        model: OracleModel,
        *,
        device: str,
        resolution: int,
        confidence_threshold: float,
    ) -> OracleProcessor: ...


class _SavezCompressed(Protocol):
    def __call__(self, file: str | Path, **arrays: object) -> None: ...


@dataclass(frozen=True)
class ConstructionAdapters:
    zeros: DynamicCall
    arange: DynamicCall
    pin_memory: DynamicCall


class _RopeAttention(Protocol):
    use_rope: bool
    rope_interp: bool
    rope_pt_size: Sequence[int]
    freqs_cis: object

    def compute_cis(self, *, end_x: int, end_y: int, scale_pos: float) -> object: ...


class _RopeBlock(Protocol):
    window_size: int
    attn: _RopeAttention


class _VisionTrunk(Protocol):
    blocks: Sequence[_RopeBlock]


class _VisionBackbone(Protocol):
    trunk: _VisionTrunk


class _Backbone(Protocol):
    vision_backbone: _VisionBackbone


class OracleModel(Protocol):
    backbone: _Backbone


def _cpu_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    updated = dict(kwargs)
    if str(updated.get("device", "")).startswith("cuda"):
        updated["device"] = "cpu"
    return updated


def install_cpu_oracle_adapters(torch: TorchRuntime) -> ConstructionAdapters:
    """Install only the construction-time CPU adapters required upstream."""

    edt_module = types.ModuleType("sam3.model.edt")

    def unavailable_edt(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        raise RuntimeError("Triton EDT is unavailable in the image-only CPU oracle.")

    setattr(edt_module, "edt_triton", unavailable_edt)
    sys.modules["sam3.model.edt"] = edt_module
    originals = ConstructionAdapters(
        zeros=torch.zeros,
        arange=torch.arange,
        pin_memory=torch.Tensor.pin_memory,
    )

    def zeros(*args: object, **kwargs: object) -> object:
        return originals.zeros(*args, **_cpu_kwargs(kwargs))

    def arange(*args: object, **kwargs: object) -> object:
        return originals.arange(*args, **_cpu_kwargs(kwargs))

    def no_pin_memory(tensor: object, *args: object, **kwargs: object) -> object:
        del args, kwargs
        return tensor

    torch.zeros = zeros
    torch.arange = arange
    torch.Tensor.pin_memory = no_pin_memory
    return originals


def restore_construction_adapters(
    torch: TorchRuntime, originals: ConstructionAdapters
) -> None:
    torch.zeros = originals.zeros
    torch.arange = originals.arange


def run_prompt(
    processor: OracleProcessor, state: OracleState, spec: OracleCase
) -> OracleState:
    processor.reset_all_prompts(state)
    prompt = spec["prompt"]
    if prompt is not None:
        state = processor.set_text_prompt(prompt, state)
    for geometric_prompt in spec["geometric_prompts"]:
        state = processor.add_geometric_prompt(
            geometric_prompt["box"], geometric_prompt["label"], state
        )
    return state


def set_official_global_rope_grid(model: OracleModel, resolution: int) -> None:
    """Recompute official global-attention RoPE for a processor resolution."""

    grid_size = resolution // 14
    for block in model.backbone.vision_backbone.trunk.blocks:
        if block.window_size != 0 or not block.attn.use_rope:
            continue
        attention = block.attn
        scale_pos = 1.0
        if attention.rope_interp:
            scale_pos = attention.rope_pt_size[0] / grid_size
        attention.freqs_cis = attention.compute_cis(
            end_x=grid_size,
            end_y=grid_size,
            scale_pos=scale_pos,
        )


def _string_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    mapping = cast(Mapping[object, object], value)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{name} must use string keys.")
    return cast(Mapping[str, object], mapping)


def validate_case_specs(value: object) -> list[OracleCase]:
    """Validate parsed case JSON and return its precise runtime contract."""

    if not isinstance(value, list) or not value:
        raise ValueError("Oracle cases must be a non-empty JSON list.")
    raw_specs = cast(list[object], value)
    names: set[str] = set()
    validated: list[OracleCase] = []
    required = {"name", "resolution", "prompt", "geometric_prompts"}
    for index, raw_spec in enumerate(raw_specs):
        spec = _string_mapping(raw_spec, f"Oracle case {index}")
        if set(spec) != required:
            raise ValueError(
                f"Oracle case {index} fields must be exactly {sorted(required)}."
            )
        name = spec["name"]
        resolution = spec["resolution"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Oracle case name is invalid or duplicated: {name!r}.")
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            raise ValueError(f"Oracle case {name!r} resolution must be an integer.")
        if resolution <= 0 or resolution % 14 != 0:
            raise ValueError(
                f"Oracle case {name!r} resolution must be a positive multiple of 14."
            )
        prompt = spec["prompt"]
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"Oracle case {name!r} prompt must be a string or null.")
        prompts_value = spec["geometric_prompts"]
        if not isinstance(prompts_value, list):
            raise ValueError(f"Oracle case {name!r} geometric_prompts must be a list.")
        geometric_prompts: list[GeometricPrompt] = []
        for prompt_index, raw_prompt in enumerate(cast(list[object], prompts_value)):
            geometric_prompt = _string_mapping(
                raw_prompt, f"Oracle case {name!r} geometric prompt {prompt_index}"
            )
            if set(geometric_prompt) != {"box", "label"}:
                raise ValueError(
                    f"Oracle case {name!r} geometric prompts require box and label."
                )
            box_value = geometric_prompt["box"]
            if not isinstance(box_value, list):
                raise ValueError(f"Oracle case {name!r} contains an invalid box.")
            box_values = cast(list[object], box_value)
            if len(box_values) != 4:
                raise ValueError(f"Oracle case {name!r} contains an invalid box.")
            box: list[int | float] = []
            for coordinate in box_values:
                if (
                    isinstance(coordinate, bool)
                    or not isinstance(coordinate, int | float)
                    or not np.isfinite(coordinate)
                ):
                    raise ValueError(f"Oracle case {name!r} contains an invalid box.")
                box.append(coordinate)
            label = geometric_prompt["label"]
            if not isinstance(label, bool):
                raise ValueError(f"Oracle case {name!r} contains an invalid box label.")
            geometric_prompts.append({"box": box, "label": label})
        names.add(name)
        validated.append(
            {
                "name": name,
                "resolution": resolution,
                "prompt": prompt,
                "geometric_prompts": geometric_prompts,
            }
        )
    return validated


def save_oracle_arrays(path: Path, arrays: Mapping[str, object]) -> None:
    """Write a dynamic named-array mapping through NumPy's archive boundary."""

    save_arrays = cast(_SavezCompressed, np.savez_compressed)
    save_arrays(path, **dict(arrays))
