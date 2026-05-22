import importlib
import sys
import types

import numpy as np
import pytest
from PIL import Image


_PROCESSOR_MODULES = (
    "mlx_sam3p1.model.sam3_image_processor",
    "mlx_sam3p1.model.box_ops",
    "mlx_sam3p1.model.data_misc",
)
_MISSING = object()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _install_fake_mlx(monkeypatch):
    class FakeArray(np.ndarray):
        def __new__(cls, value, dtype=None):
            return np.asarray(value, dtype=dtype).view(cls)

    def wrap(value, dtype=None):
        return np.asarray(value, dtype=dtype).view(FakeArray)

    core = types.ModuleType("mlx.core")
    core.array = FakeArray
    core.bool_ = np.bool_
    core.float32 = np.float32
    core.int64 = np.int64
    core.sigmoid = lambda x: wrap(_sigmoid(x))
    core.stack = lambda arrays, axis=0: wrap(np.stack(arrays, axis=axis))
    core.zeros = lambda shape, dtype=None: wrap(np.zeros(shape, dtype=dtype))
    core.eval = lambda *args, **kwargs: None

    class Upsample:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __call__(self, x):
            return x

    nn = types.ModuleType("mlx.nn")
    nn.Upsample = Upsample

    mlx = types.ModuleType("mlx")
    mlx.core = core
    mlx.nn = nn

    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)


@pytest.fixture()
def processor_module(monkeypatch):
    originals = {name: sys.modules.get(name, _MISSING) for name in _PROCESSOR_MODULES}
    for name in _PROCESSOR_MODULES:
        sys.modules.pop(name, None)

    _install_fake_mlx(monkeypatch)
    module = importlib.import_module("mlx_sam3p1.model.sam3_image_processor")
    yield module

    for name, original in originals.items():
        if original is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class _GroundingModel:
    inst_interactive_predictor = None

    def __init__(self, outputs=None):
        self.outputs = outputs or {}

    def call_grounding(self, **kwargs):
        self.call_kwargs = kwargs
        return self.outputs


def _identity_interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    return input


def test_call_grounding_uses_plural_mask_logits_and_full_boolean_filter(
    processor_module, monkeypatch
):
    outputs = {
        "pred_boxes": processor_module.mx.array(
            [
                [
                    [0.5, 0.5, 0.2, 0.4],
                    [0.1, 0.1, 0.2, 0.2],
                    [0.2, 0.3, 0.4, 0.2],
                ]
            ],
            dtype=processor_module.mx.float32,
        ),
        "pred_logits": processor_module.mx.array([[[2.0], [-2.0], [1.0]]]),
        "pred_masks": processor_module.mx.array(
            [
                [
                    [[1.0, -1.0], [0.0, 2.0]],
                    [[-3.0, -3.0], [-3.0, -3.0]],
                    [[0.25, 0.5], [-0.5, 1.0]],
                ]
            ]
        ),
        "presence_logit_dec": processor_module.mx.array([10.0]),
        "semantic_seg": processor_module.mx.array([[[[0.0, 1.0], [2.0, 3.0]]]]),
    }
    processor = processor_module.Sam3Processor(
        _GroundingModel(outputs), confidence_threshold=0.5
    )
    state = {
        "backbone_out": {"image": object()},
        "geometric_prompt": object(),
        "original_height": 20,
        "original_width": 10,
    }

    def fail_numpy_array(*args, **kwargs):
        raise AssertionError("_call_grounding must not filter via host NumPy")

    monkeypatch.setattr(processor_module, "interpolate", _identity_interpolate)
    monkeypatch.setattr(
        processor_module, "np", types.SimpleNamespace(array=fail_numpy_array)
    )

    result = processor._call_grounding(state)

    expected_scores = (
        _sigmoid(np.array([2.0, -2.0, 1.0])) * _sigmoid(np.array([10.0]))[0]
    )[[0, 2]]
    expected_boxes = np.array(
        [
            [4.0, 6.0, 6.0, 14.0],
            [0.0, 4.0, 4.0, 8.0],
        ]
    )
    expected_masks_logits = _sigmoid(
        np.asarray(outputs["pred_masks"])[0, [0, 2]][:, None]
    )

    assert "masks_logits" in result
    assert "mask_logits" not in result
    np.testing.assert_allclose(np.asarray(result["scores"]), expected_scores, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(result["boxes"]), expected_boxes, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(result["masks_logits"]), expected_masks_logits, rtol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(result["masks"]), expected_masks_logits > 0.5
    )


def test_call_grounding_rejects_batch_outputs(processor_module):
    outputs = {
        "pred_boxes": processor_module.mx.array(
            [
                [[0.5, 0.5, 0.2, 0.2]],
                [[0.5, 0.5, 0.2, 0.2]],
            ]
        ),
        "pred_logits": processor_module.mx.array([[[2.0]], [[2.0]]]),
        "pred_masks": processor_module.mx.array(
            [
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 1.0], [1.0, 1.0]]],
            ]
        ),
        "presence_logit_dec": processor_module.mx.array([10.0, 10.0]),
        "semantic_seg": processor_module.mx.array(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        ),
    }
    processor = processor_module.Sam3Processor(
        _GroundingModel(outputs), confidence_threshold=0.5
    )
    state = {
        "backbone_out": {"image": object()},
        "geometric_prompt": object(),
        "original_height": 20,
        "original_width": 10,
    }

    with pytest.raises(
        NotImplementedError,
        match="Batch grounding is not supported.*batch size 2",
    ):
        processor._call_grounding(state)


def test_set_image_batch_fails_with_clear_single_image_contract(processor_module):
    processor = processor_module.Sam3Processor(_GroundingModel())

    with pytest.raises(
        NotImplementedError,
        match="set_image_batch is not supported.*use set_image",
    ):
        processor.set_image_batch([Image.new("RGB", (2, 2))])


def test_reset_all_prompts_removes_plural_and_legacy_mask_logits(processor_module):
    processor = processor_module.Sam3Processor(_GroundingModel())
    image_features = object()
    state = {
        "backbone_out": {
            "language_features": object(),
            "language_mask": object(),
            "language_embeds": object(),
            "image_features": image_features,
        },
        "geometric_prompt": object(),
        "boxes": object(),
        "masks": object(),
        "masks_logits": object(),
        "mask_logits": object(),
        "scores": object(),
    }

    processor.reset_all_prompts(state)

    assert state == {"backbone_out": {"image_features": image_features}}
