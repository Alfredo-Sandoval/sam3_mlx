import pytest
from importlib import import_module

import mlx.core as mx

import sam3_mlx
from sam3_mlx._unsupported import (
    REASONS,
    UNSUPPORTED_METADATA_ATTR,
    Sam3MlxUnsupportedError,
    raise_unsupported,
    unsupported_function,
)
from sam3_mlx.agent import _unsupported as agent_unsupported
from sam3_mlx.eval import _unsupported as eval_unsupported
from sam3_mlx.perflib import fa3
from sam3_mlx.perflib.triton.connected_components import connected_components_triton
from sam3_mlx.perflib.triton import connected_components as triton_cc
from sam3_mlx.perflib.triton import nms as triton_nms
from sam3_mlx.perflib.triton.nms import nms_triton
from sam3_mlx.train import _unsupported as train_unsupported
from sam3_mlx.train import train, trainer
from sam3_mlx.train.optim import optimizer
from sam3_mlx.train.utils import checkpoint_utils, distributed, logger, train_utils
from tests._paths import REPO_ROOT

sigmoid_focal_loss_module = import_module("sam3_mlx.train.loss.sigmoid_focal_loss")
BLOCKED_ACCELERATOR = "cu" + "da"


def _assert_unsupported(call, *, reason: str, feature_fragment: str, message: str):
    with pytest.raises(Sam3MlxUnsupportedError, match=message) as exc_info:
        call()

    assert exc_info.value.reason == reason
    assert feature_fragment in exc_info.value.feature
    return exc_info.value


def test_canonical_unsupported_error_preserves_notimplemented_contract():
    assert sam3_mlx.Sam3MlxUnsupportedError is Sam3MlxUnsupportedError

    with pytest.raises(
        Sam3MlxUnsupportedError,
        match=(
            r"sam3_mlx\.example is not supported in sam3_mlx "
            r"\(triton-kernel\).*Use sam3_mlx\.native instead"
        ),
    ) as exc_info:
        raise_unsupported(
            "sam3_mlx.example",
            reason="triton-kernel",
            alternative="sam3_mlx.native",
            detail="Triton kernels are unavailable in MLX.",
            upstream_commit="2814fa6",
        )

    error = exc_info.value
    assert isinstance(error, NotImplementedError)
    assert error.feature == "sam3_mlx.example"
    assert error.reason == "triton-kernel"
    assert error.alternative == "sam3_mlx.native"
    assert error.detail == "Triton kernels are unavailable in MLX."
    assert error.upstream_commit == "2814fa6"


def test_unknown_reason_is_rejected_at_construction():
    assert "triton-kernel" in REASONS

    with pytest.raises(ValueError, match="Unknown unsupported reason"):
        raise_unsupported("sam3_mlx.bad", reason="triton-ish")


@pytest.mark.parametrize(
    ("helper", "reason", "message"),
    [
        (agent_unsupported, "agent-llm", "external LLM services"),
        (eval_unsupported, "eval-stack", "evaluation surface"),
        (train_unsupported, "training-loop", "Full training datasets"),
    ],
)
def test_domain_helpers_raise_canonical_error(helper, reason, message):
    with pytest.raises(Sam3MlxUnsupportedError, match=message) as exc_info:
        helper.raise_unsupported("domain.feature")

    assert exc_info.value.feature == "domain.feature"
    assert exc_info.value.reason == reason
    assert exc_info.value.upstream_commit == "2814fa619404a722d03e9a012e083e4f293a4e53"


def test_unsupported_function_metadata_and_raise_contract():
    @unsupported_function(
        "sam3_mlx.decorated",
        reason="torch-compile",
        alternative="compile=False",
        detail="torch.compile is unavailable in MLX.",
    )
    def decorated(value):
        return value

    info = getattr(decorated, UNSUPPORTED_METADATA_ATTR)
    assert info.feature == "sam3_mlx.decorated"
    assert info.reason == "torch-compile"
    assert info.alternative == "compile=False"

    with pytest.raises(Sam3MlxUnsupportedError, match="torch.compile is unavailable"):
        decorated(object())


@pytest.mark.parametrize(
    ("fn", "feature", "alternative"),
    [
        (
            nms_triton,
            "sam3_mlx.perflib.triton.nms.nms_triton",
            "sam3_mlx.perflib.nms",
        ),
        (
            connected_components_triton,
            "sam3_mlx.perflib.triton.connected_components.connected_components_triton",
            "sam3_mlx.perflib.connected_components",
        ),
    ],
)
def test_triton_public_functions_fail_with_canonical_triton_kernel_error(
    fn, feature, alternative
):
    info = getattr(fn, UNSUPPORTED_METADATA_ATTR)
    assert info.feature == feature
    assert info.reason == "triton-kernel"
    assert info.alternative == alternative

    with pytest.raises(Sam3MlxUnsupportedError, match="Triton kernels") as exc_info:
        fn(None, None, None)

    assert exc_info.value.feature == feature
    assert exc_info.value.reason == "triton-kernel"
    assert exc_info.value.alternative == alternative


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (triton_nms._nms_suppression_kernel, (None, None, 0, 0, 0)),
        (triton_cc._any_combine, (None, None)),
        (triton_cc.tl_any, (None,)),
        (triton_cc._init_labels_kernel, (None, None, 0, 0)),
        (triton_cc.find, (None, None, None)),
        (triton_cc.union, (None, None, None, None)),
        (
            triton_cc._merge_helper,
            (
                None,
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0,
                0,
                0,
            ),
        ),
        (triton_cc._local_prop_kernel, (None, None, 0, 0, 0, 0)),
        (triton_cc._pointer_jump_kernel, (None, None, 0, 0)),
        (triton_cc._count_labels_kernel, (None, None, 0, 0)),
        (triton_cc._broadcast_sizes_kernel, (None, None, None, 0, 0)),
    ],
)
def test_private_triton_kernel_shims_preserve_fail_fast_metadata(fn, args):
    info = getattr(fn, UNSUPPORTED_METADATA_ATTR)
    assert info.reason == "triton-kernel"
    assert info.upstream_commit == "2814fa619404a722d03e9a012e083e4f293a4e53"

    with pytest.raises(Sam3MlxUnsupportedError, match="Triton kernels"):
        fn(*args)


@pytest.mark.parametrize(
    ("call", "args", "feature", "reason", "message"),
    [
        (
            fa3.flash_attn_func_op,
            (None, None, None),
            "flash_attn_func_op",
            "flash-attn-3",
            "FlashAttention 3",
        ),
        (
            distributed.convert_to_distributed_tensor,
            (None,),
            "convert_to_distributed_tensor",
            "torch-distributed",
            "torch.distributed",
        ),
        (
            train.single_proc_run,
            (0, 0, None, 1),
            "single_proc_run",
            "training-loop",
            "Hydra",
        ),
        (
            trainer.print_model_summary,
            (None,),
            "print_model_summary",
            "training-loop",
            "Torch trainer",
        ),
        (
            checkpoint_utils.load_checkpoint,
            ([],),
            "load_checkpoint",
            "training-loop",
            "torch.load",
        ),
        (
            sigmoid_focal_loss_module._inner_focal_loss_fwd,
            (),
            "_inner_focal_loss_fwd",
            "triton-kernel",
            "Triton kernels",
        ),
        (
            optimizer.construct_optimizer,
            (None, None),
            "construct_optimizer",
            "training-loop",
            "torch.optim",
        ),
        (
            train_utils.setup_distributed_backend,
            (None, None),
            "setup_distributed_backend",
            "training-loop",
            "distributed training",
        ),
        (
            logger.make_tensorboard_logger,
            ("logs",),
            "make_tensorboard_logger",
            "training-loop",
            "tensorboard",
        ),
    ],
)
def test_migrated_helper_islands_raise_canonical_error(
    call, args, feature, reason, message
):
    with pytest.raises(Sam3MlxUnsupportedError, match=message) as exc_info:
        call(*args)

    assert exc_info.value.feature == feature
    assert exc_info.value.reason == reason


CANONICAL_UPSTREAM_COMMIT = "2814fa619404a722d03e9a012e083e4f293a4e53"

# Load-bearing public guards that *must* stay registered. If one of these is
# accidentally removed (e.g. someone deletes the @unsupported_function decorator
# while porting), this test fires. Add to the set when promoting a new public
# guard; do not remove entries lightly.
REQUIRED_REGISTRY_FEATURES = frozenset(
    {
        "sam3_mlx.perflib.triton.nms.nms_triton",
        "sam3_mlx.perflib.triton.connected_components.connected_components_triton",
    }
)


def test_registry_walker_returns_bounded_well_formed_entries():
    from sam3_mlx._unsupported import unsupported_features

    import_errors: list[tuple[str, BaseException]] = []
    features = unsupported_features(
        on_import_error=lambda n, e: import_errors.append((n, e))
    )

    assert not import_errors, f"Unexpected import errors: {import_errors!r}"

    feature_names = {info.feature for info in features}
    missing = REQUIRED_REGISTRY_FEATURES - feature_names
    assert not missing, (
        f"Required @unsupported_function entries disappeared from the registry: {sorted(missing)}"
    )
    # Floor catches "decorator removed from a whole module" regressions without
    # breaking on intentional one-off cleanups. Update the floor if a deliberate
    # cleanup legitimately lowers the count.
    assert len(features) >= len(REQUIRED_REGISTRY_FEATURES) + 10, (
        f"Registry has only {len(features)} entries; expected at least "
        f"{len(REQUIRED_REGISTRY_FEATURES) + 10}. Did a module's decorators get stripped?"
    )

    for info in features:
        assert info.reason in REASONS, (
            f"{info.feature} has unknown reason {info.reason!r}"
        )
        assert info.feature.startswith("sam3_mlx."), info.feature
        assert info.upstream_commit == CANONICAL_UPSTREAM_COMMIT, (
            f"{info.feature} pinned to {info.upstream_commit!r}, "
            f"expected canonical {CANONICAL_UPSTREAM_COMMIT!r}"
        )
        if info.alternative is not None:
            assert info.alternative.startswith("sam3_mlx."), (
                f"{info.feature} has alternative {info.alternative!r}; "
                "alternatives must point at an in-package replacement."
            )

    features_again = unsupported_features()
    assert [f.feature for f in features_again] == [f.feature for f in features], (
        "Walker output should be deterministic across calls."
    )


def test_registry_entries_raise_when_invoked():
    import importlib

    from sam3_mlx._unsupported import unsupported_features

    for info in unsupported_features():
        module_name, _, attr = info.feature.rpartition(".")
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
        with pytest.raises(Sam3MlxUnsupportedError) as exc_info:
            fn()
        # The whole point of the registry is "what the metadata advertises is what
        # the raise delivers." Pin every contract field, not just feature/reason.
        assert exc_info.value.feature == info.feature
        assert exc_info.value.reason == info.reason
        assert exc_info.value.alternative == info.alternative, (
            f"{info.feature}: alternative drift "
            f"(advertised={info.alternative!r}, raised={exc_info.value.alternative!r})"
        )
        assert exc_info.value.upstream_commit == info.upstream_commit, (
            f"{info.feature}: upstream_commit drift "
            f"(advertised={info.upstream_commit!r}, raised={exc_info.value.upstream_commit!r})"
        )


def test_torch_compile_guard_raises_canonical_error():
    from sam3_mlx.model.sam3_video_inference import Sam3VideoInference

    with pytest.raises(Sam3MlxUnsupportedError, match="torch.compile") as exc_info:
        Sam3VideoInference(image_model=None, compile_model=True)

    assert exc_info.value.feature == (
        "sam3_mlx.model.sam3_video_inference.Sam3VideoInference(compile_model=True)"
    )
    assert exc_info.value.reason == "torch-compile"


@pytest.mark.parametrize(
    ("call", "reason", "feature_fragment", "message"),
    [
        (
            lambda: import_module("sam3_mlx.model_builder").build_sam3_image_model(
                device="tpu",
                load_from_HF=False,
            ),
            "unsupported-device",
            "device='tpu'",
            "explicit MLX runtime",
        ),
        (
            lambda: import_module("sam3_mlx.model_builder").build_sam3_image_model(
                compile=True,
                load_from_HF=False,
            ),
            "torch-compile",
            "build_sam3_image_model",
            "torch.compile",
        ),
        (
            lambda: import_module("sam3_mlx.model_builder").build_sam3_video_predictor(
                gpus_to_use=[0],
                load_from_HF=False,
            ),
            "video-multi-gpu",
            "gpus_to_use",
            "not supported",
        ),
        (
            lambda: import_module("sam3_mlx.model_builder").build_sam3_video_model(
                has_presence_token=False,
                load_from_HF=False,
            ),
            "video-multiplex",
            "has_presence_token=False",
            "presence-token",
        ),
        # build_sam3_multiplex_video_predictor(load_from_HF=False, use_fa3=False)
        # now supports checkpoint-free and local converted-checkpoint construction.
        # The default still fails fast because automatic HF download/conversion is
        # not wired into the MLX runtime.
        (
            lambda: import_module("sam3_mlx.model_builder").build_sam3_predictor(),
            "video-multiplex",
            "build_sam3_multiplex_video_predictor",
            "download/conversion",
        ),
        # Sam3TrackerPredictor.__init__ is now a real MLX construction layer
        # for cached-feature single-object tracking. Deferred predictor paths
        # are covered by tests/port/tracker/test_tracker_predictor.py.
    ],
)
def test_model_builder_runtime_guards_raise_canonical_error(
    call, reason, feature_fragment, message
):
    error = _assert_unsupported(
        call,
        reason=reason,
        feature_fragment=feature_fragment,
        message=message,
    )
    assert error.upstream_commit == "2814fa619404a722d03e9a012e083e4f293a4e53"


@pytest.mark.parametrize(
    ("call", "reason", "feature_fragment", "message"),
    [
        (
            lambda: import_module(
                "sam3_mlx.model.model_misc"
            ).MultiheadAttentionWrapper(
                dims=4,
                num_heads=2,
            )(
                mx.zeros((1, 2, 4)),
                mx.zeros((1, 2, 4)),
                mx.zeros((1, 2, 4)),
                use_fa3=True,
            ),
            "flash-attn-3",
            "MultiheadAttentionWrapper(use_fa3=True)",
            "FlashAttention 3",
        ),
        (
            lambda: import_module(
                "sam3_mlx.model.model_misc"
            ).MultiheadAttentionWrapper(
                dims=4,
                num_heads=2,
            )(
                mx.zeros((1, 2, 4)),
                mx.zeros((1, 2, 4)),
                mx.zeros((1, 2, 4)),
                attn_type=import_module(
                    "sam3_mlx.model.model_misc"
                ).AttentionType.Xformer,
            ),
            "xformers",
            "MultiheadAttentionWrapper(attn_type='Xformer')",
            "xformers",
        ),
        (
            lambda: import_module("sam3_mlx.sam.transformer").Attention(
                embedding_dim=4,
                num_heads=2,
                use_fa3=True,
            ),
            "flash-attn-3",
            "Attention(use_fa3=True)",
            "FlashAttention 3",
        ),
        (
            lambda: import_module("sam3_mlx.model.decoder").functional_attention(
                mx.zeros((1, 1, 4)),
                mx.zeros((1, 1, 4)),
                mx.zeros((1, 1, 4)),
                dropout=0.0,
                num_heads=2,
                use_fa3=True,
                rope_k_repeat=False,
            ),
            "flash-attn-3",
            "functional_attention(use_fa3=True)",
            "FlashAttention 3",
        ),
        (
            lambda: import_module("sam3_mlx.model.decoder").SimpleRoPEAttention(
                d_model=4,
                num_heads=2,
                dropout_p=0.0,
                use_fa3=True,
            ),
            "flash-attn-3",
            "SimpleRoPEAttention(use_fa3=True)",
            "FlashAttention 3",
        ),
    ],
)
def test_attention_runtime_guards_raise_canonical_error(
    call, reason, feature_fragment, message
):
    error = _assert_unsupported(
        call,
        reason=reason,
        feature_fragment=feature_fragment,
        message=message,
    )
    assert error.upstream_commit == "2814fa619404a722d03e9a012e083e4f293a4e53"


def test_real_rope_all_keys_excluded_guard_raises_canonical_error():
    from sam3_mlx.model.decoder import functional_attention

    error = _assert_unsupported(
        lambda: functional_attention(
            mx.zeros((1, 1, 4)),
            mx.zeros((1, 1, 4)),
            mx.zeros((1, 1, 4)),
            dropout=0.0,
            num_heads=2,
            num_k_exclude_rope=1,
            freqs_cis=mx.zeros((1, 2)),
            use_rope_real=True,
            rope_k_repeat=False,
        ),
        reason="video-multiplex",
        feature_fragment="use_rope_real=True",
        message="Real RoPE",
    )
    assert error.upstream_commit == "2814fa619404a722d03e9a012e083e4f293a4e53"


def test_video_inference_device_guard_raises_canonical_error():
    from sam3_mlx.model.sam3_video_inference import Sam3VideoInference

    error = _assert_unsupported(
        lambda: Sam3VideoInference(image_model=None).to(BLOCKED_ACCELERATOR),
        reason="unsupported-device",
        feature_fragment=f"device='{BLOCKED_ACCELERATOR}'",
        message="explicit MLX runtime",
    )
    assert error.upstream_commit == "2814fa619404a722d03e9a012e083e4f293a4e53"


@pytest.mark.parametrize(
    ("call", "reason", "feature_fragment", "message"),
    [
        (
            lambda: import_module("sam3_mlx.sam.rope").init_t_xy(
                1,
                1,
                device=BLOCKED_ACCELERATOR,
            ),
            "unsupported-device",
            f"device='{BLOCKED_ACCELERATOR}'",
            "RoPE helpers",
        ),
        (
            lambda: import_module("sam3_mlx.model.sam3_image_processor").Sam3Processor(
                model=None,
                device="tpu",
            ),
            "unsupported-device",
            "device='tpu'",
            "explicit MLX runtime",
        ),
        (
            lambda: import_module("sam3_mlx.model.vl_combiner").SAM3VLBackbone(
                visual=None,
                text=None,
                compile_visual=True,
            ),
            "torch-compile",
            "compile_visual=True",
            "compile_visual",
        ),
        (
            lambda: import_module(
                "sam3_mlx.model.sam3_video_predictor"
            ).Sam3VideoPredictorMultiGPU(),
            "video-multi-gpu",
            "Sam3VideoPredictorMultiGPU",
            "multi-GPU",
        ),
        (
            lambda: import_module("sam3_mlx.model.io_utils").TorchCodecDecoder(
                source="video.mp4"
            ),
            "torchcodec",
            "TorchCodecDecoder",
            "TorchCodec",
        ),
        (
            lambda: import_module("sam3_mlx.model.sam1_task_predictor")
            .SAM3InteractiveImageModel()
            .forward_image(mx.zeros((1, 3, 16, 16))),
            "image-interactivity",
            "forward_image",
            "attached",
        ),
        (
            lambda: import_module(
                "sam3_mlx.model.sam3_image"
            ).Sam3ImageOnVideoMultiGPU(),
            "video-multi-gpu",
            "Sam3ImageOnVideoMultiGPU",
            "all-gather",
        ),
    ],
)
def test_runtime_surface_guards_raise_canonical_error(
    call, reason, feature_fragment, message
):
    error = _assert_unsupported(
        call,
        reason=reason,
        feature_fragment=feature_fragment,
        message=message,
    )
    assert error.upstream_commit == "2814fa619404a722d03e9a012e083e4f293a4e53"


@pytest.mark.parametrize(
    ("call", "reason", "feature_fragment", "message"),
    [
        (
            lambda: import_module("sam3_mlx.model.data_misc")
            .NestedTensor(
                mx.zeros((1, 1)),
                None,
            )
            .to("cpu"),
            "unsupported-device",
            "NestedTensor.to(device='cpu')",
            "explicit MLX device",
        ),
        (
            lambda: import_module("sam3_mlx.model.data_misc")
            .NestedTensor(
                mx.zeros((1, 1)),
                None,
            )
            .pin_memory("cpu"),
            "training-loop",
            "NestedTensor.pin_memory(device='cpu')",
            "PyTorch CPU-pinning",
        ),
        (
            lambda: import_module("sam3_mlx.agent.helpers.boxes").BoxMode.convert(
                [0.0, 0.0, 10.0, 20.0],
                import_module("sam3_mlx.agent.helpers.boxes").BoxMode.XYXY_ABS,
                import_module("sam3_mlx.agent.helpers.boxes").BoxMode.XYWHA_ABS,
            ),
            "port-gap",
            "BoxMode.convert",
            "not supported yet",
        ),
        (
            lambda: import_module("sam3_mlx.model.edt").edt_kernel(),
            "triton-kernel",
            "edt_kernel",
            "Triton kernel",
        ),
        (
            lambda: import_module("sam3_mlx.model.necks").Sam3DualViTDetNeck(
                trunk=type("Trunk", (), {"channel_list": [8]})(),
                position_encoding=None,
                d_model=4,
                scale_factors=(3.0,),
            ),
            "port-gap",
            "scale_factor=3.0",
            "Scale factor 3.0",
        ),
        (
            lambda: import_module(
                "sam3_mlx.perflib.associate_det_trk"
            ).associate_det_trk(
                mx.zeros((1, 4, 4)),
                mx.zeros((1, 5, 4)),
            ),
            "port-gap",
            "mask_resize=True",
            "mask resizing",
        ),
        (
            lambda: import_module("sam3_mlx.train.data.torch_dataset").TorchDataset(
                dataset=[0, 1],
                batch_size=1,
                num_workers=1,
                shuffle=False,
                pin_memory=False,
                drop_last=False,
            ),
            "training-loop",
            "TorchDataset(num_workers)",
            "worker processes",
        ),
        (
            lambda: import_module("sam3_mlx.train.loss.loss_fns").IABCEMdetr(
                pos_weight=1.0,
                use_separate_loss_for_det_and_trk=True,
            ),
            "training-loop",
            "use_separate_loss_for_det_and_trk=True",
            "video/tracking",
        ),
        (
            lambda: import_module("sam3_mlx.train.loss.loss_fns").Det2TrkAssoc(),
            "training-loop",
            "Det2TrkAssoc",
            "video/tracking association loss",
        ),
        (
            lambda: import_module("sam3_mlx.visualization_utils").save_masklet_video(
                video_frames=None,
                outputs=None,
                out_path="out.mp4",
            ),
            "port-gap",
            "save_masklet_video",
            "video encoding",
        ),
    ],
)
def test_misc_port_boundary_guards_raise_canonical_error(
    call, reason, feature_fragment, message
):
    error = _assert_unsupported(
        call,
        reason=reason,
        feature_fragment=feature_fragment,
        message=message,
    )
    assert error.upstream_commit == "2814fa619404a722d03e9a012e083e4f293a4e53"


def test_no_bare_notimplementederror_in_runtime_surface_guards():
    runtime_files = [
        "sam3_mlx/model/io_utils.py",
        "sam3_mlx/model/sam1_task_predictor.py",
        "sam3_mlx/model/sam3_image.py",
        "sam3_mlx/model/sam3_image_processor.py",
        "sam3_mlx/model/sam3_multiplex_video_predictor.py",
        "sam3_mlx/model/sam3_tracker_base.py",
        "sam3_mlx/model/sam3_tracking_predictor.py",
        "sam3_mlx/model/sam3_video_predictor.py",
        "sam3_mlx/model/utils/misc.py",
        "sam3_mlx/model/utils/sam2_utils.py",
        "sam3_mlx/model/vl_combiner.py",
        "sam3_mlx/sam/rope.py",
    ]

    offenders = [
        path
        for path in runtime_files
        if "NotImplementedError" in (REPO_ROOT / path).read_text()
    ]
    assert offenders == [], (
        "Runtime-facing unsupported guards must use Sam3MlxUnsupportedError. "
        f"Stale NotImplementedError references found in: {offenders}"
    )


def test_no_residual_compile_mode_notimplementederror_in_package():
    """Regression guard: every compile_mode/compile_model guard must route through raise_unsupported."""
    import re

    pkg = REPO_ROOT / "sam3_mlx"
    pattern = re.compile(
        r"raise\s+NotImplementedError\([^)]*compile_(?:mode|model)", re.DOTALL
    )
    offenders = []
    for path in pkg.rglob("*.py"):
        text = path.read_text()
        if pattern.search(text):
            offenders.append(str(path.relative_to(pkg.parent)))
    assert offenders == [], (
        "torch.compile guards must use raise_unsupported(..., reason='torch-compile'). "
        f"Stale inline raises found in: {offenders}"
    )


def test_no_bare_notimplementederror_in_trackeval_shims():
    """Regression guard: trackeval shims must route through raise_unsupported / @unsupported_function."""
    trackeval = REPO_ROOT / "sam3_mlx" / "eval" / "hota_eval_toolkit" / "trackeval"
    offenders = []
    for path in trackeval.rglob("*.py"):
        if "raise NotImplementedError" in path.read_text():
            offenders.append(str(path.relative_to(trackeval.parent.parent.parent)))
    assert offenders == [], (
        "trackeval shims must use raise_unsupported(..., reason='eval-stack') or "
        "@unsupported_function. Stale inline raises found in: "
        f"{offenders}"
    )


def test_trackeval_base_metric_method_raises_canonical_error():
    from sam3_mlx.eval.hota_eval_toolkit.trackeval.metrics._base_metric import (
        _BaseMetric,
    )

    instance = _BaseMetric.__new__(_BaseMetric)
    with pytest.raises(Sam3MlxUnsupportedError, match="metric computation") as exc_info:
        instance.eval_sequence(data=None)
    assert exc_info.value.reason == "eval-stack"
    assert exc_info.value.feature.endswith("._BaseMetric.eval_sequence")


def test_upstream_md_unsupported_section_is_in_sync_with_registry():
    """The autogenerated table in UPSTREAM.md must match the live registry."""
    from sam3_mlx._unsupported import (
        _UNSUPPORTED_DOCS_MARKER_BEGIN,
        _UNSUPPORTED_DOCS_MARKER_END,
        render_unsupported_markdown,
    )

    upstream_md = REPO_ROOT / "UPSTREAM.md"
    text = upstream_md.read_text()
    assert _UNSUPPORTED_DOCS_MARKER_BEGIN in text
    assert _UNSUPPORTED_DOCS_MARKER_END in text

    _, _, rest = text.partition(_UNSUPPORTED_DOCS_MARKER_BEGIN)
    inside, _, _ = rest.partition(_UNSUPPORTED_DOCS_MARKER_END)
    actual = inside.lstrip("\n")
    expected = render_unsupported_markdown()
    assert actual == expected, (
        "UPSTREAM.md autogenerated section is stale. "
        "Run `python -m sam3_mlx._unsupported --write UPSTREAM.md`."
    )
