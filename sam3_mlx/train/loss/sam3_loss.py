# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

"""MLX loss-wrapper orchestration ported from official SAM3 training code.

The official wrapper is a Torch ``nn.Module`` and normalizes losses with
``torch.distributed`` when requested. The MLX fork keeps the image-safe loss
composition behavior and treats ``global``/``local`` normalization as the same
single-process boundary. Distributed training remains outside this port.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from sam3_mlx.model.model_misc import SAM3Output
from sam3_mlx.train.loss.loss_fns import CORE_LOSS_KEY, Det2TrkAssoc, Masks


MLX_SAM3_LOSS_BASE_COMMIT = "4794409a19afd9e3faeac66a2f1c4373ddf10f5b"


def _as_float_array(value) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(mx.float32)
    return mx.array(value, dtype=mx.float32)


def _zero(dtype=mx.float32) -> mx.array:
    return mx.array(0.0, dtype=dtype)


class DummyLoss(nn.Module):
    """Eval placeholder loss matching the official always-zero contract."""

    def __init__(
        self,
        core_loss_key: str = CORE_LOSS_KEY,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__()
        del device, kwargs
        self.core_loss_key = core_loss_key

    def __call__(self, *args, **kwargs):
        del args, kwargs
        return {self.core_loss_key: _zero()}

    def accumulate(self, out_dict):
        """Called by iterative loss code to ensure a core loss is present."""

        if self.core_loss_key not in out_dict:
            out_dict[self.core_loss_key] = _zero()
        return out_dict


class Sam3LossWrapper(nn.Module):
    def __init__(
        self,
        loss_fns_find,
        normalization="global",
        matcher=None,
        o2m_matcher=None,
        o2m_weight=1.0,
        use_o2m_matcher_on_o2m_aux=True,
        loss_fn_semantic_seg=None,
        normalize_by_valid_object_num=False,
        normalize_by_stage_num=False,
        scale_by_find_batch_size=False,
    ):
        super().__init__()
        if normalization not in ["global", "local", "none"]:
            raise AssertionError("normalization must be one of global, local, none.")
        self.loss_fns_find = loss_fns_find
        self.normalization = normalization
        self.normalize_by_valid_object_num = normalize_by_valid_object_num
        self.normalize_by_stage_num = normalize_by_stage_num
        self.matcher = matcher
        self.o2m_matcher = o2m_matcher
        self.o2m_weight = o2m_weight
        self.use_o2m_matcher_on_o2m_aux = use_o2m_matcher_on_o2m_aux
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.scale_by_find_batch_size = scale_by_find_batch_size

    def _get_num_boxes(self, targets):
        if self.normalize_by_valid_object_num:
            boxes_hw = _as_float_array(targets["boxes"]).reshape(-1, 4)
            num_boxes = mx.sum(mx.all(boxes_hw[:, 2:] > 0, axis=-1).astype(mx.float32))
        else:
            num_boxes = mx.sum(_as_float_array(targets["num_boxes"]))

        if self.normalization in ["global", "local"]:
            return mx.maximum(num_boxes, mx.array(1.0, dtype=mx.float32))
        return mx.array(1.0, dtype=mx.float32)

    def _get_o2m_indices(
        self,
        is_aux,
        o2m_out,
        o2m_targets,
        o2m_out_is_valid,
        o2m_target_is_valid_padded,
    ):
        use_o2m_matcher = self.use_o2m_matcher_on_o2m_aux or not is_aux
        matcher = self.o2m_matcher if use_o2m_matcher else self.matcher
        if matcher is None:
            raise ValueError("o2m outputs require a matcher in Sam3LossWrapper.")
        return matcher(
            o2m_out,
            o2m_targets,
            out_is_valid=o2m_out_is_valid,
            target_is_valid_padded=o2m_target_is_valid_padded,
        )

    def compute_loss(self, nested_out, targets):
        num_boxes = self._get_num_boxes(targets)
        o2m_out_is_valid = nested_out.get("o2m_out_is_valid", None)
        o2m_target_is_valid_padded = nested_out.get("o2m_target_is_valid_padded", None)

        output_list = [(nested_out, "", False)]
        if "aux_outputs" in nested_out:
            output_list.extend(
                (aux_out, f"_aux_{i}", True)
                for i, aux_out in enumerate(nested_out["aux_outputs"])
            )
        if "first_stage" in nested_out:
            output_list.append((nested_out["first_stage"], "_fs", True))

        losses = {}
        total_core_loss = _zero()
        for out, suffix, is_aux in output_list:
            if "indices" not in out:
                raise KeyError("Sam3LossWrapper expects outputs to contain indices.")
            indices = out["indices"]
            has_o2m_out = "pred_logits_o2m" in out
            if has_o2m_out:
                o2m_out = {
                    k[: -len("_o2m")]: v for k, v in out.items() if k.endswith("_o2m")
                }
                o2m_targets = targets
                o2m_indices = self._get_o2m_indices(
                    is_aux,
                    o2m_out,
                    o2m_targets,
                    o2m_out_is_valid,
                    o2m_target_is_valid_padded,
                )

            for loss_fn in self.loss_fns_find:
                loss_dict = loss_fn(
                    outputs=out,
                    targets=targets,
                    indices=indices,
                    num_boxes=num_boxes,
                    is_aux=is_aux,
                )
                total_core_loss = total_core_loss + loss_dict.pop(CORE_LOSS_KEY)
                losses.update(
                    {f"{key}{suffix}": value for key, value in loss_dict.items()}
                )

                compute_o2m_loss = has_o2m_out
                if isinstance(loss_fn, Masks):
                    compute_o2m_loss = compute_o2m_loss and "pred_masks" in o2m_out
                if isinstance(loss_fn, Det2TrkAssoc):
                    compute_o2m_loss = False
                if compute_o2m_loss:
                    loss_dict = loss_fn(
                        outputs=o2m_out,
                        targets=o2m_targets,
                        indices=o2m_indices,
                        num_boxes=num_boxes,
                        is_aux=is_aux,
                    )
                    for key in list(loss_dict):
                        loss_dict[key] = loss_dict[key] * self.o2m_weight
                    total_core_loss = total_core_loss + loss_dict.pop(CORE_LOSS_KEY)
                    losses.update(
                        {
                            f"{key}{suffix}_o2m": value
                            for key, value in loss_dict.items()
                        }
                    )

        losses[CORE_LOSS_KEY] = total_core_loss
        return losses

    def __call__(self, find_stages: SAM3Output, find_targets):
        if find_stages.loss_stages is not None:
            find_targets = [find_targets[i] for i in find_stages.loss_stages]
        with SAM3Output.iteration_mode(
            find_stages,
            iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
        ) as find_stages:
            if len(find_stages) != len(find_targets):
                raise AssertionError("find_stages and find_targets length mismatch.")
            total_losses = {}
            for stage_outputs, stage_targets in zip(find_stages, find_targets):
                stage_targets = [stage_targets] * len(stage_outputs)
                for outputs, targets in zip(stage_outputs, stage_targets):
                    cur_losses = self.compute_loss(outputs, targets)

                    if self.loss_fn_semantic_seg is not None:
                        cur_semantic_losses = self.loss_fn_semantic_seg(
                            outputs,
                            targets,
                        )
                        cur_losses[CORE_LOSS_KEY] = cur_losses[
                            CORE_LOSS_KEY
                        ] + cur_semantic_losses.pop(CORE_LOSS_KEY)
                        overlap = set(cur_losses).intersection(set(cur_semantic_losses))
                        if overlap:
                            raise AssertionError(
                                f"semantic and find losses overlap: {sorted(overlap)}"
                            )
                        cur_losses.update(cur_semantic_losses)

                    if self.normalize_by_stage_num:
                        cur_losses[CORE_LOSS_KEY] = cur_losses[CORE_LOSS_KEY] / len(
                            find_stages
                        )

                    if self.scale_by_find_batch_size:
                        batch_size = targets["num_boxes"].shape[0]
                        cur_losses[CORE_LOSS_KEY] = cur_losses[CORE_LOSS_KEY] * (
                            batch_size**0.5
                        )

                    for key, value in cur_losses.items():
                        if key not in total_losses:
                            total_losses[key] = value
                        else:
                            total_losses[key] = total_losses[key] + value

        return total_losses


__all__ = [
    "DummyLoss",
    "MLX_SAM3_LOSS_BASE_COMMIT",
    "Sam3LossWrapper",
]
