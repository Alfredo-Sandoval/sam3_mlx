from threading import Event, Thread

import pytest

from sam3_mlx._unsupported import Sam3MlxUnsupportedError
from sam3_mlx.model.lifecycle_predictor import LifecycleSafeSam3BasePredictor
from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor
from sam3_mlx.model.sam3_video_predictor import Sam3VideoPredictor


class _ClosableFrames:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FailingFrames(_ClosableFrames):
    def close(self) -> None:
        self.closed = True
        raise OSError("decoder close failed")


class _BlockingInitModel:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.frames = _ClosableFrames()
        self.init_calls = 0

    def init_state(self, **_kwargs):
        self.init_calls += 1
        self.started.set()
        assert self.release.wait(timeout=2)
        return {
            "frames": self.frames,
            "orig_height": 1,
            "orig_width": 1,
            "num_frames": 1,
        }


class _ImmediateModel:
    def __init__(self, *, frames=None) -> None:
        self.frames = frames or _ClosableFrames()
        self.init_calls = 0

    def init_state(self, **_kwargs):
        self.init_calls += 1
        return {
            "frames": self.frames,
            "orig_height": 1,
            "orig_width": 1,
            "num_frames": 1,
        }


def _predictor(model):
    predictor = Sam3BasePredictor()
    predictor.model = model
    return predictor


def test_lifecycle_compatibility_class_adds_no_divergent_overrides():
    assert LifecycleSafeSam3BasePredictor.__bases__ == (Sam3BasePredictor,)
    divergent_members = {
        name
        for name, value in LifecycleSafeSam3BasePredictor.__dict__.items()
        if callable(value) or isinstance(value, (classmethod, property, staticmethod))
    }
    assert divergent_members == set()


def test_shutdown_prevents_inflight_session_publication_and_disposes_state():
    model = _BlockingInitModel()
    predictor = _predictor(model)
    errors: list[str] = []

    def start() -> None:
        try:
            predictor.start_session("resource", session_id="late")
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = Thread(target=start)
    worker.start()
    assert model.started.wait(timeout=2)

    predictor.shutdown()
    model.release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == [
        "Predictor was shut down while the session was loading; "
        "the loaded state was disposed."
    ]
    assert model.frames.closed is True
    assert predictor.is_shutdown is True
    assert predictor._all_inference_states == {}
    assert predictor._reserved_session_ids == set()
    assert predictor._cancelled_session_ids == set()


def test_close_session_prevents_inflight_session_publication():
    model = _BlockingInitModel()
    predictor = _predictor(model)
    errors: list[str] = []

    def start() -> None:
        try:
            predictor.start_session("resource", session_id="closing")
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = Thread(target=start)
    worker.start()
    assert model.started.wait(timeout=2)

    assert predictor.close_session("closing") == {"is_success": True}
    model.release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert errors == [
        "Session closing was closed while it was loading; "
        "the loaded state was disposed."
    ]
    assert model.frames.closed is True
    assert predictor._all_inference_states == {}
    assert predictor._reserved_session_ids == set()
    assert predictor._cancelled_session_ids == set()


def test_shutdown_is_terminal_and_idempotent():
    model = _ImmediateModel()
    predictor = _predictor(model)
    predictor.start_session("resource", session_id="live")

    predictor.shutdown()
    predictor.shutdown()

    assert model.frames.closed is True
    assert predictor._all_inference_states == {}
    with pytest.raises(RuntimeError, match="cannot start new sessions"):
        predictor.start_session("resource", session_id="after-shutdown")
    assert model.init_calls == 1


def test_close_provider_failure_still_clears_and_closes_session():
    model = _ImmediateModel(frames=_FailingFrames())
    predictor = _predictor(model)
    predictor.start_session("resource", session_id="bad-close")
    session = predictor._all_inference_states["bad-close"]

    with pytest.raises(RuntimeError, match="Failed to close frame provider"):
        predictor.close_session("bad-close", run_gc_collect=False)

    assert predictor._all_inference_states == {}
    assert model.frames.closed is True
    assert session["closed"] is True
    assert session["state"] == {}


def test_selected_frame_async_folder_loading_fails_before_unbounded_preload(tmp_path):
    model = _ImmediateModel()
    predictor = Sam3VideoPredictor(
        video_model=model,
        async_loading_frames=True,
    )

    with pytest.raises(Sam3MlxUnsupportedError, match="bounded-memory contract"):
        predictor.start_session(tmp_path)

    assert model.init_calls == 0
