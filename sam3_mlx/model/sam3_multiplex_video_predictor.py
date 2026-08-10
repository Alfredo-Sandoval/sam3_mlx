from __future__ import annotations

from typing import Protocol, cast

from sam3_mlx.model.lifecycle_predictor import LifecycleSafeSam3BasePredictor


class _WarmUpCompilationHook(Protocol):
    def __call__(self) -> object: ...


class Sam3MultiplexVideoPredictor(LifecycleSafeSam3BasePredictor):
    """Torch-free SAM 3.1 multiplex predictor wrapper.

    The official class adds Torch autocast and optional warm-up compilation around
    the shared request/session API. The MLX port keeps the same constructor
    surface and session behavior without entering a Torch-only autocast context.
    """

    def __init__(
        self,
        model: object | None,
        session_expiration_sec: int = 1200,
        default_output_prob_thresh: float = 0.5,
        async_loading_frames: bool = True,
        warm_up: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.session_expiration_sec = session_expiration_sec
        self.default_output_prob_thresh = default_output_prob_thresh
        self.async_loading_frames = async_loading_frames
        self.warm_up = bool(warm_up)
        if self.warm_up:
            self._run_mlx_warm_up()

    def _run_mlx_warm_up(self) -> None:
        if self.model is None:
            raise ValueError("warm_up=True requires a model instance.")
        model = self.model
        setattr(model, "_warm_up_complete", False)
        warm_up_compilation = cast(
            _WarmUpCompilationHook | None,
            getattr(model, "warm_up_compilation", None),
        )
        if warm_up_compilation is not None:
            warm_up_compilation()
        setattr(model, "_warm_up_complete", True)
