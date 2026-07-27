"""Lifecycle-safe predictor base for shared MLX inference services."""

from __future__ import annotations

from threading import Event, RLock
import time
import uuid
from typing import Any

from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor


class LifecycleSafeSam3BasePredictor(Sam3BasePredictor):
    """Close the start-session/shutdown race in the upstream-compatible base.

    ``Sam3BasePredictor.start_session`` intentionally performs potentially slow
    frame loading outside the global session lock. Without a terminal predictor
    state, ``shutdown()`` can clear the reservation table and return while an
    in-flight loader later publishes a new live session. A second caller can
    also reuse that cleared ID, causing one state to overwrite another.

    This subclass makes shutdown terminal, rechecks the terminal state before
    publishing a loaded session, and disposes any state that completed loading
    after shutdown began. Public predictor classes inherit this hardened base;
    the compatibility implementation remains available for internal fixtures.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predictor_shutdown = False

    @property
    def is_shutdown(self) -> bool:
        with self._sessions_lock:
            return self._predictor_shutdown

    def start_session(
        self,
        resource_path,
        session_id: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> dict[str, str]:
        if self.model is None:
            raise RuntimeError("Sam3BasePredictor.model must be initialized.")
        if session_id is None:
            session_id = str(uuid.uuid4())

        with self._sessions_lock:
            if self._predictor_shutdown:
                raise RuntimeError(
                    "Predictor has been shut down and cannot start new sessions."
                )
            if (
                session_id in self._all_inference_states
                or session_id in self._reserved_session_ids
            ):
                raise ValueError(f"Session ID already exists: {session_id}")
            self._reserved_session_ids.add(session_id)

        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type

        try:
            inference_state = self.model.init_state(**init_kwargs)
        except BaseException:
            with self._sessions_lock:
                self._reserved_session_ids.discard(session_id)
            raise

        now = time.monotonic()
        cancel_event = Event()
        if isinstance(inference_state, dict):
            inference_state["_cancel_event"] = cancel_event
        session: dict[str, Any] = {
            "state": inference_state,
            "session_id": session_id,
            "created_monotonic": now,
            "last_used_monotonic": now,
            "lock": RLock(),
            "cancel_event": cancel_event,
            "state_version": 0,
            "propagation_active": False,
            "closing": False,
            "closed": False,
        }

        publish_error: RuntimeError | None = None
        with self._sessions_lock:
            self._reserved_session_ids.discard(session_id)
            if self._predictor_shutdown:
                publish_error = RuntimeError(
                    "Predictor was shut down while the session was loading; "
                    "the loaded state was disposed."
                )
            elif session_id in self._all_inference_states:
                # Defensive guard: a reservation should make this unreachable,
                # but never overwrite a live state if an implementation changes.
                publish_error = RuntimeError(
                    f"Session ID became occupied while loading: {session_id}"
                )
            else:
                self._all_inference_states[session_id] = session

        if publish_error is not None:
            self._dispose_session(session)
            raise publish_error
        return {"session_id": session_id}

    def shutdown(self) -> None:
        """Atomically stop publication and dispose all currently live sessions."""

        with self._sessions_lock:
            if self._predictor_shutdown:
                return
            self._predictor_shutdown = True
            sessions = list(self._all_inference_states.values())
            for session in sessions:
                with session["lock"]:
                    session["closing"] = True
            self._all_inference_states.clear()
            # In-flight starters retain their local reservation identity and will
            # observe _predictor_shutdown before publication. Clearing this set
            # here prevents stale bookkeeping from surviving shutdown.
            self._reserved_session_ids.clear()

        for session in sessions:
            self._dispose_session(session)
