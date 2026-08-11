from pathlib import Path

import numpy as np
import pytest
import mlx.core as mx
from mlx import nn

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model_builder import (
    Sam3CheckpointLoadReport,
    build_tracker,
    _audit_sam3_image_checkpoint_load,  # pyright: ignore[reportPrivateUsage]
    _load_checkpoint,  # pyright: ignore[reportPrivateUsage]
    _load_multiplex_checkpoint,  # pyright: ignore[reportPrivateUsage]
    _load_multiplex_tracker_checkpoint,  # pyright: ignore[reportPrivateUsage]
    _load_tracker_checkpoint,  # pyright: ignore[reportPrivateUsage]
    _is_generated_checkpoint_key,  # pyright: ignore[reportPrivateUsage]
)
from sam3_mlx.model.data_misc import reshape_array
from tests._mlx_runtime import flat_parameters, save_safetensors


class _TinyCheckpointModel(nn.Module):
    checkpoint_load_report: Sam3CheckpointLoadReport | None

    def __init__(self) -> None:
        super().__init__()
        self.inst_interactive_predictor = None
        self.head = nn.Linear(3, 2)
        self.scale = mx.ones((1,))
        self.checkpoint_load_report = None


class _WeightLeaf(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = mx.zeros(shape)


class _Sam2ConvStage(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.dconv_2x2 = _WeightLeaf(shape)


class _TinyConvLayoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inst_interactive_predictor = None
        self.backbone = {
            "vision_backbone": {
                "sam2_convs": [
                    _Sam2ConvStage((1, 1, 1, 1)),
                    _Sam2ConvStage((3, 2, 2, 4)),
                ]
            }
        }


class _TinyInteractivePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = {"no_mem_embed": mx.zeros((1, 1, 2))}


class _TinyInteractiveCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = _WeightLeaf((1,))
        self.inst_interactive_predictor = _TinyInteractivePredictor()


class _TinyTrackerCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.no_mem_embed = mx.zeros((1, 1, 2))


class _TinyMultiplexCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = {
            "backbone": {
                "language_backbone": {
                    "encoder": {
                        "text_projection": mx.zeros((2, 1)),
                    },
                    "resizer": {
                        "weight": mx.zeros((2, 1)),
                        "bias": mx.zeros((2,)),
                    },
                }
            }
        }
        self.tracker = {
            "model": {
                "output_valid_embed": mx.zeros((2, 2)),
            }
        }


class _TinyMultiplexTrackerCheckpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_valid_embed = mx.zeros((2, 2))


def test_checkpoint_audit_reports_loaded_missing_extra_and_shape_mismatch():
    model = _TinyCheckpointModel()
    weights = {
        "head.weight": mx.ones((2, 3)),
        "head.bias": mx.ones((3,)),
        "extra.weight": mx.ones((1,)),
    }

    report = _audit_sam3_image_checkpoint_load(model, weights)

    assert report.loaded == ("head.weight",)
    assert report.missing == ("scale",)
    assert report.extra == ("extra.weight",)
    assert len(report.shape_mismatched) == 1
    mismatch = report.shape_mismatched[0]
    assert mismatch.key == "head.bias"
    assert mismatch.model_shape == (2,)
    assert mismatch.checkpoint_shape == (3,)


def test_generated_checkpoint_allowlist_is_exact():
    assert _is_generated_checkpoint_key("backbone.language_backbone.encoder.attn_mask")
    assert not _is_generated_checkpoint_key("detector.attn_mask")


def test_load_checkpoint_rejects_partial_checkpoint_without_mutating_model(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "partial.safetensors"
    original = np.asarray(flat_parameters(model)["head.weight"]).copy()
    replacement = reshape_array(mx.arange(6), 2, 3).astype(mx.float32)
    save_safetensors(
        checkpoint_path,
        {"head.weight": replacement},
    )

    with pytest.raises(ValueError, match="missing_required=2"):
        _load_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        np.asarray(flat_parameters(model)["head.weight"]),
        original,
    )


def test_load_checkpoint_allows_explicit_development_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "partial-development.safetensors"
    replacement = reshape_array(mx.arange(6), 2, 3).astype(mx.float32)
    save_safetensors(checkpoint_path, {"head.weight": replacement})

    report = _load_checkpoint(model, checkpoint_path, strict=False)

    assert report.loaded == ("head.weight",)
    assert report.missing == ("head.bias", "scale")
    assert model.checkpoint_load_report is report
    np.testing.assert_array_equal(
        np.asarray(flat_parameters(model)["head.weight"]),
        np.asarray(replacement),
    )


def test_load_checkpoint_rejects_unreviewed_source_weight(tmp_path: Path) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "unexpected.safetensors"
    save_safetensors(
        checkpoint_path,
        {
            "head.weight": mx.ones((2, 3)),
            "head.bias": mx.ones((2,)),
            "scale": mx.ones((1,)),
            "unexpected.weight": mx.ones((1,)),
        },
    )

    with pytest.raises(ValueError, match="without a reviewed model mapping"):
        _load_checkpoint(model, checkpoint_path)


def test_load_checkpoint_rejects_shape_mismatch_without_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "bad-shape.safetensors"
    original_weight = np.asarray(flat_parameters(model)["head.weight"]).copy()
    replacement = reshape_array(mx.arange(6), 2, 3).astype(mx.float32)
    save_safetensors(
        checkpoint_path,
        {
            "head.weight": replacement,
            "head.bias": mx.ones((3,)),
        },
    )

    with pytest.raises(ValueError, match="shape-mismatched weights"):
        _load_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        np.asarray(flat_parameters(model)["head.weight"]),
        original_weight,
    )


def test_load_checkpoint_normalizes_known_conv_layout_before_audit(
    tmp_path: Path,
) -> None:
    model = _TinyConvLayoutModel()
    checkpoint_path = tmp_path / "conv-layout.safetensors"
    key = "backbone.vision_backbone.sam2_convs.1.dconv_2x2.weight"
    torch_layout = reshape_array(mx.arange(4 * 3 * 2 * 2), 4, 3, 2, 2).astype(
        mx.float32
    )
    save_safetensors(checkpoint_path, {key: torch_layout})

    report = _load_checkpoint(model, checkpoint_path, strict=False)

    assert report.loaded == (key,)
    assert report.shape_mismatched == ()
    np.testing.assert_array_equal(
        np.asarray(flat_parameters(model)[key]),
        np.asarray(torch_layout).transpose(1, 2, 3, 0),
    )


def test_load_checkpoint_merges_explicit_interactive_checkpoint(tmp_path: Path) -> None:
    model = _TinyInteractiveCheckpointModel()
    checkpoint_path = tmp_path / "base.safetensors"
    interactive_checkpoint_path = tmp_path / "interactive.safetensors"
    save_safetensors(
        checkpoint_path,
        {"base.weight": mx.array([5.0], dtype=mx.float32)},
    )
    save_safetensors(
        interactive_checkpoint_path,
        {
            "tracker_model.no_memory_embedding": mx.array(
                [[[7.0, 8.0]]],
                dtype=mx.float32,
            )
        },
    )

    report = _load_checkpoint(
        model,
        checkpoint_path,
        interactive_checkpoint_path=interactive_checkpoint_path,
    )

    assert report.loaded == (
        "base.weight",
        "inst_interactive_predictor.model.no_mem_embed",
    )
    params = flat_parameters(model)
    np.testing.assert_array_equal(
        np.asarray(params["base.weight"]),
        np.array([5.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(params["inst_interactive_predictor.model.no_mem_embed"]),
        np.array([[[7.0, 8.0]]], dtype=np.float32),
    )


def test_load_checkpoint_rejects_interactive_checkpoint_without_interactivity(
    tmp_path: Path,
) -> None:
    model = _TinyCheckpointModel()
    checkpoint_path = tmp_path / "base.safetensors"
    interactive_checkpoint_path = tmp_path / "interactive.safetensors"
    save_safetensors(
        checkpoint_path,
        {"head.weight": mx.ones((2, 3))},
    )
    save_safetensors(
        interactive_checkpoint_path,
        {"tracker_model.no_memory_embedding": mx.ones((1, 1, 2))},
    )

    with pytest.raises(
        ValueError,
        match="interactive_checkpoint_path requires",
    ):
        _load_checkpoint(
            model,
            checkpoint_path,
            interactive_checkpoint_path=interactive_checkpoint_path,
        )


def test_load_checkpoint_rejects_interactive_shape_mismatch_without_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyInteractiveCheckpointModel()
    checkpoint_path = tmp_path / "base.safetensors"
    interactive_checkpoint_path = tmp_path / "interactive-bad-shape.safetensors"
    original_base = np.asarray(flat_parameters(model)["base.weight"]).copy()
    original_interactive = np.asarray(
        flat_parameters(model)["inst_interactive_predictor.model.no_mem_embed"]
    ).copy()
    save_safetensors(
        checkpoint_path,
        {"base.weight": mx.array([5.0], dtype=mx.float32)},
    )
    save_safetensors(
        interactive_checkpoint_path,
        {
            "tracker_model.no_memory_embedding": mx.array(
                [[[7.0, 8.0, 9.0]]],
                dtype=mx.float32,
            )
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "inst_interactive_predictor.model.no_mem_embed: "
            r"model \(1, 1, 2\), checkpoint \(1, 1, 3\)"
        ),
    ):
        _load_checkpoint(
            model,
            checkpoint_path,
            interactive_checkpoint_path=interactive_checkpoint_path,
        )

    params = flat_parameters(model)
    np.testing.assert_array_equal(
        np.asarray(params["base.weight"]),
        original_base,
    )
    np.testing.assert_array_equal(
        np.asarray(params["inst_interactive_predictor.model.no_mem_embed"]),
        original_interactive,
    )


def test_load_tracker_checkpoint_maps_official_alias_and_loads(tmp_path: Path) -> None:
    model = _TinyTrackerCheckpointModel()
    checkpoint_path = tmp_path / "tracker.safetensors"
    save_safetensors(
        checkpoint_path,
        {
            "tracker_model.no_memory_embedding": mx.array(
                [[[7.0, 8.0]]],
                dtype=mx.float32,
            )
        },
    )

    report = _load_tracker_checkpoint(model, checkpoint_path)

    assert report.loaded == ("no_mem_embed",)
    np.testing.assert_array_equal(
        np.asarray(flat_parameters(model)["no_mem_embed"]),
        np.array([[[7.0, 8.0]]], dtype=np.float32),
    )


def test_load_tracker_checkpoint_rejects_shape_mismatch_without_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyTrackerCheckpointModel()
    checkpoint_path = tmp_path / "tracker-bad.safetensors"
    original = np.asarray(flat_parameters(model)["no_mem_embed"]).copy()
    save_safetensors(
        checkpoint_path,
        {
            "tracker_model.no_memory_embedding": mx.array(
                [[[7.0, 8.0, 9.0]]],
                dtype=mx.float32,
            )
        },
    )

    with pytest.raises(ValueError, match="shape-mismatched weights"):
        _load_tracker_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        np.asarray(flat_parameters(model)["no_mem_embed"]),
        original,
    )


def test_load_multiplex_checkpoint_loads_detector_and_tracker_weights(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexCheckpointModel()
    checkpoint_path = tmp_path / "multiplex.safetensors"
    text_projection = mx.array([[5.0, 6.0]], dtype=mx.float32)
    text_resizer = mx.array([[7.0], [8.0]], dtype=mx.float32)
    text_resizer_bias = mx.array([9.0, 10.0], dtype=mx.float32)
    output_valid = mx.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=mx.float32,
    )
    save_safetensors(
        checkpoint_path,
        {
            "detector_model.text_encoder.text_projection.weight": text_projection,
            "detector_model.text_projection.weight": text_resizer,
            "detector_model.text_projection.bias": text_resizer_bias,
            "tracker_model.output_valid_embed": output_valid,
        },
    )

    report = _load_multiplex_checkpoint(model, checkpoint_path)

    assert report.loaded == (
        "detector.backbone.language_backbone.encoder.text_projection",
        "detector.backbone.language_backbone.resizer.bias",
        "detector.backbone.language_backbone.resizer.weight",
        "tracker.model.output_valid_embed",
    )
    params = flat_parameters(model)
    np.testing.assert_array_equal(
        np.asarray(
            params["detector.backbone.language_backbone.encoder.text_projection"]
        ),
        np.array([[5.0], [6.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(params["detector.backbone.language_backbone.resizer.weight"]),
        np.asarray(text_resizer),
    )
    np.testing.assert_array_equal(
        np.asarray(params["detector.backbone.language_backbone.resizer.bias"]),
        np.asarray(text_resizer_bias),
    )
    np.testing.assert_array_equal(
        np.asarray(params["tracker.model.output_valid_embed"]),
        np.asarray(output_valid),
    )


def test_load_multiplex_checkpoint_requires_text_resizer_weights(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexCheckpointModel()
    checkpoint_path = tmp_path / "multiplex-missing-resizer.safetensors"
    save_safetensors(
        checkpoint_path,
        {
            "detector_model.text_encoder.text_projection.weight": mx.array(
                [[5.0, 6.0]],
                dtype=mx.float32,
            ),
            "tracker_model.output_valid_embed": mx.ones((2, 2), dtype=mx.float32),
        },
    )

    with pytest.raises(ValueError, match="required VE text resizer weights"):
        _load_multiplex_checkpoint(model, checkpoint_path)


def test_load_multiplex_checkpoint_rejects_shape_mismatch_without_partial_load(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexCheckpointModel()
    checkpoint_path = tmp_path / "multiplex-bad.safetensors"
    original_text = np.asarray(
        flat_parameters(model)[
            "detector.backbone.language_backbone.encoder.text_projection"
        ]
    ).copy()
    save_safetensors(
        checkpoint_path,
        {
            "detector_model.text_encoder.text_projection.weight": mx.array(
                [[5.0, 6.0]],
                dtype=mx.float32,
            ),
            "tracker_model.output_valid_embed": mx.ones((3, 2), dtype=mx.float32),
        },
    )

    with pytest.raises(ValueError, match="shape-mismatched weights"):
        _load_multiplex_checkpoint(model, checkpoint_path)

    np.testing.assert_array_equal(
        np.asarray(
            flat_parameters(model)[
                "detector.backbone.language_backbone.encoder.text_projection"
            ]
        ),
        original_text,
    )


def test_load_multiplex_checkpoint_rejects_pytorch_checkpoint_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="PyTorch SAM 3.1 multiplex"):
        _load_multiplex_checkpoint(
            _TinyMultiplexCheckpointModel(),
            tmp_path / "sam3.1_multiplex.pt",
        )


def test_load_multiplex_tracker_checkpoint_loads_direct_tracker_model(
    tmp_path: Path,
) -> None:
    model = _TinyMultiplexTrackerCheckpointModel()
    checkpoint_path = tmp_path / "multiplex-tracker.safetensors"
    output_valid = mx.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=mx.float32,
    )
    save_safetensors(
        checkpoint_path,
        {"tracker_model.output_valid_embed": output_valid},
    )

    report = _load_multiplex_tracker_checkpoint(model, checkpoint_path)

    assert report.loaded == ("output_valid_embed",)
    np.testing.assert_array_equal(
        np.asarray(flat_parameters(model)["output_valid_embed"]),
        np.asarray(output_valid),
    )


def test_build_tracker_rejects_checkpoint_with_unmapped_backbone(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "tracker.safetensors"
    save_safetensors(
        checkpoint_path,
        {"tracker_model.no_memory_embedding": mx.zeros((1, 1, 256))},
    )

    with pytest.raises(Sam3MlxUnsupportedError, match="detector/tracker-neck"):
        build_tracker(
            apply_temporal_disambiguation=False,
            with_backbone=True,
            checkpoint_path=checkpoint_path,
        )
