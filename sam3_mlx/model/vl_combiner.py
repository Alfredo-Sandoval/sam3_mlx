from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from typing import NoReturn, Protocol, cast

import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import raise_unsupported
from sam3_mlx.model.act_ckpt_utils import activation_ckpt_wrapper
from sam3_mlx.model.necks import Sam3DualViTDetNeck, Sam3TriViTDetNeck
from sam3_mlx.model.text_encoder_ve import PreencodedText


def _raise_vl_unsupported(
    feature: str, *, detail: str, alternative: str | None = None
) -> NoReturn:
    raise_unsupported(
        feature,
        reason="torch-compile",
        detail=detail,
        alternative=alternative,
    )


def _feature_tensor(feature: object) -> object:
    return getattr(feature, "tensors", feature)


def _feature_mask(feature: object) -> object | None:
    return getattr(feature, "mask", None)


class _LanguageBackbone(Protocol):
    def __call__(
        self,
        text: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]: ...


class _TriVisionBackbone(Protocol):
    def forward(
        self,
        samples: mx.array,
        *,
        need_sam3_out: bool,
        need_interactive_out: bool,
        need_propagation_out: bool,
    ) -> tuple[
        list[object],
        list[mx.array],
        list[object],
        list[mx.array],
        list[object],
        list[mx.array],
    ]: ...


class _VisionBackbone(Protocol):
    def forward(self, samples: mx.array) -> tuple[list[object], list[mx.array]]: ...


def _forward_tri_vision(
    backbone: _TriVisionBackbone,
    samples: mx.array,
    *,
    scalp: int,
    need_sam3_out: bool,
    need_interactive_out: bool,
    need_propagation_out: bool,
) -> dict[str, object]:
    (
        sam3_features,
        sam3_pos,
        interactive_features,
        interactive_pos,
        propagation_features,
        propagation_pos,
    ) = backbone.forward(
        samples,
        need_sam3_out=need_sam3_out,
        need_interactive_out=need_interactive_out,
        need_propagation_out=need_propagation_out,
    )

    if scalp > 0:
        sam3_features, sam3_pos = sam3_features[:-scalp], sam3_pos[:-scalp]
        interactive_features, interactive_pos = (
            interactive_features[:-scalp],
            interactive_pos[:-scalp],
        )
        propagation_features, propagation_pos = (
            propagation_features[:-scalp],
            propagation_pos[:-scalp],
        )

    output: dict[str, object] = {}
    if need_sam3_out:
        sam3_last = sam3_features[-1]
        output.update(
            {
                "vision_features": _feature_tensor(sam3_last),
                "vision_mask": _feature_mask(sam3_last),
                "vision_pos_enc": sam3_pos,
                "backbone_fpn": sam3_features,
            }
        )
    if need_interactive_out:
        interactive_last = interactive_features[-1]
        output["interactive"] = {
            "vision_features": _feature_tensor(interactive_last),
            "vision_mask": _feature_mask(interactive_last),
            "vision_pos_enc": interactive_pos,
            "backbone_fpn": interactive_features,
        }
    if need_propagation_out:
        propagation_last = propagation_features[-1]
        output["sam2_backbone_out"] = {
            "vision_features": _feature_tensor(propagation_last),
            "vision_mask": _feature_mask(propagation_last),
            "vision_pos_enc": propagation_pos,
            "backbone_fpn": propagation_features,
        }
    return output


class SAM3VLBackbone(nn.Module):
    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text: _LanguageBackbone | None,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp: int = 0,
    ) -> None:
        super().__init__()
        if compile_visual:
            _raise_vl_unsupported(
                "sam3_mlx.model.vl_combiner.SAM3VLBackbone(compile_visual=True)",
                detail="compile_visual is not implemented in MLX.",
                alternative="compile_visual=False",
            )

        self.vision_backbone: Sam3DualViTDetNeck = visual
        self.language_backbone = text
        self.scalp = scalp
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone

    def __call__(
        self,
        samples: mx.array,
        captions: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
        additional_text: Sequence[str] | None = None,
    ) -> dict[str, object]:
        return self.forward(samples, captions, input_boxes, additional_text)

    def forward(
        self,
        samples: mx.array,
        captions: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
        additional_text: Sequence[str] | None = None,
    ) -> dict[str, object]:
        output = self.forward_image(samples)
        output.update(self.forward_text(captions, input_boxes, additional_text))
        return output

    def forward_image(self, samples: mx.array) -> dict[str, object]:
        return activation_ckpt_wrapper(self._forward_image_no_act_ckpt)(
            samples=samples,
            act_ckpt_enable=(
                self.act_ckpt_whole_vision_backbone and getattr(self, "training", False)
            ),
        )

    def _forward_image_no_act_ckpt(self, samples: mx.array) -> dict[str, object]:
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(
            samples
        )

        if self.scalp > 0:
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        return {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

    def forward_text(
        self,
        captions: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
        additional_text: Sequence[str] | None = None,
        device: str | None = None,
    ) -> dict[str, mx.array]:
        return activation_ckpt_wrapper(self._forward_text_no_ack_ckpt)(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text,
            device=device,
            act_ckpt_enable=(
                self.act_ckpt_whole_language_backbone
                and getattr(self, "training", False)
            ),
        )

    def _forward_text_no_ack_ckpt(
        self,
        captions: list[str] | PreencodedText,
        input_boxes: list[object] | None = None,
        additional_text: Sequence[str] | None = None,
        device: str | None = None,
    ) -> dict[str, mx.array]:
        del device
        output: dict[str, mx.array] = {}

        text_to_encode = copy(captions)
        if additional_text is not None:
            typed_text = cast(list[str], text_to_encode)
            typed_text.extend(additional_text)
            text_to_encode = typed_text

        language_backbone = cast(_LanguageBackbone, self.language_backbone)
        text_attention_mask, text_memory, text_embeds = language_backbone(
            text_to_encode, input_boxes
        )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[
                -len(additional_text) :
            ]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds
        return output


class SAM3VLBackboneTri(SAM3VLBackbone):
    def __init__(
        self,
        visual: Sam3TriViTDetNeck,
        text: _LanguageBackbone | None,
        compile_visual: bool = False,
        scalp: int = 0,
    ) -> None:
        super().__init__(
            visual=cast(Sam3DualViTDetNeck, visual),
            text=text,
            compile_visual=compile_visual,
            scalp=scalp,
        )
        if not isinstance(self.vision_backbone, Sam3TriViTDetNeck):
            raise TypeError(
                "SAM3VLBackboneTri requires Sam3TriViTDetNeck, got "
                f"{type(self.vision_backbone)!r}."
            )

    def forward_image(
        self,
        samples: mx.array,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ) -> dict[str, object]:
        return activation_ckpt_wrapper(self._forward_image_tri_no_act_ckpt)(
            samples=samples,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
            act_ckpt_enable=(
                self.act_ckpt_whole_vision_backbone and getattr(self, "training", False)
            ),
        )

    def _forward_image_tri_no_act_ckpt(
        self,
        samples: mx.array,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ) -> dict[str, object]:
        return _forward_tri_vision(
            cast(_TriVisionBackbone, self.vision_backbone),
            samples,
            scalp=self.scalp,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )


class VisionOnly(nn.Module):
    def __init__(
        self,
        visual: _VisionBackbone,
        n_features: int,
        forward_in_chunk_for_eval: bool = False,
        eval_chunk_size: int = 4,
        eval_cast_to_cpu: bool = False,
        scalp: int = 0,
        compile_mode: str | None = None,
        compile_extra_args: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        if compile_mode is not None or compile_extra_args is not None:
            _raise_vl_unsupported(
                "sam3_mlx.model.vl_combiner.VisionOnly(compile_mode)",
                detail="VisionOnly compile mode is not ported to MLX.",
                alternative="compile_mode=None",
            )
        self.vision_backbone = visual
        self.n_features = n_features
        self.forward_in_chunk_for_eval = forward_in_chunk_for_eval
        self.eval_chunk_size = eval_chunk_size
        self.eval_cast_to_cpu = eval_cast_to_cpu
        self.scalp = scalp
        self.should_compile = False
        self.compiled = False

    def _compile(self) -> None:
        pass

    def forward_image(self, samples: mx.array) -> dict[str, object]:
        self._compile()
        features, pos = self.vision_backbone.forward(samples)
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        elif self.scalp < 0:
            features.pop(self.scalp)
            pos.pop(self.scalp)

        last = features[-1]
        return {
            "vision_features": _feature_tensor(last),
            "vision_mask": _feature_mask(last),
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }

    def forward_text(
        self,
        captions: Sequence[str],
        input_boxes: object | None = None,
        additional_text: Sequence[str] | None = None,
        device: str | None = None,
    ) -> dict[str, mx.array]:
        del input_boxes, additional_text, device
        bs = len(captions)
        return {
            "language_features": mx.zeros((0, bs, self.n_features)),
            "language_mask": mx.zeros((bs, 0), dtype=mx.bool_),
        }


class TriHeadVisionOnly(VisionOnly):
    def __init__(
        self,
        visual: Sam3TriViTDetNeck,
        n_features: int,
        forward_in_chunk_for_eval: bool = False,
        eval_chunk_size: int = 4,
        eval_cast_to_cpu: bool = False,
        scalp: int = 0,
        compile_mode: str | None = None,
        compile_extra_args: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            visual=cast(_VisionBackbone, visual),
            n_features=n_features,
            forward_in_chunk_for_eval=forward_in_chunk_for_eval,
            eval_chunk_size=eval_chunk_size,
            eval_cast_to_cpu=eval_cast_to_cpu,
            scalp=scalp,
            compile_mode=compile_mode,
            compile_extra_args=compile_extra_args,
        )
        if not isinstance(self.vision_backbone, Sam3TriViTDetNeck):
            raise TypeError(
                "TriHeadVisionOnly requires Sam3TriViTDetNeck, got "
                f"{type(self.vision_backbone)!r}."
            )

    def forward_image(
        self,
        samples: mx.array,
        *,
        need_sam3_out: bool = True,
        need_interactive_out: bool = True,
        need_propagation_out: bool = True,
    ) -> dict[str, object]:
        return _forward_tri_vision(
            cast(_TriVisionBackbone, self.vision_backbone),
            samples,
            scalp=self.scalp,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )
