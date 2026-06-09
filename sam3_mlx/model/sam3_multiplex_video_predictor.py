from __future__ import annotations

from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor


class Sam3MultiplexVideoPredictor(Sam3BasePredictor):
    """Torch-free SAM 3.1 multiplex predictor wrapper.

    The official class adds Torch autocast and optional warm-up compilation around
    the shared request/session API. The MLX port keeps the same constructor
    surface and session behavior without entering a Torch-only autocast context.
    """

    def __init__(
        self,
        model,
        session_expiration_sec=1200,
        default_output_prob_thresh=0.5,
        async_loading_frames=True,
        warm_up=False,
    ):
        super().__init__()
        self.model = model
        self.session_expiration_sec = session_expiration_sec
        self.default_output_prob_thresh = default_output_prob_thresh
        self.async_loading_frames = async_loading_frames
        self.warm_up = bool(warm_up)
        if self.warm_up:
            self._run_mlx_warm_up()

    def _run_mlx_warm_up(self):
        if self.model is None:
            raise ValueError("warm_up=True requires a model instance.")
        self.model._warm_up_complete = False
        warm_up_compilation = getattr(self.model, "warm_up_compilation", None)
        if warm_up_compilation is not None:
            warm_up_compilation()
        self.model._warm_up_complete = True

    def _extend_expiration_time(self, session):
        super()._extend_expiration_time(session)
        if self.session_expiration_sec:
            session["expiration_sec"] = self.session_expiration_sec
