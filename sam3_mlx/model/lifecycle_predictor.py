"""Lifecycle-safe predictor base for shared MLX inference services."""

from __future__ import annotations

import gc
from threading import Event, RLock
import time
import uuid
from typing import Any

from sam3_mlx.model.sam3_base_predictor import Sam3BasePredictor


class LifecycleSafeSam3BasePredictor(Sam3BasePredictor):
    """Close publication races around loading, closing, and shutdown.

    Session initialization is intentionally performed outside the global registry
    lock. This subclass keeps the expensive load concurrent while ensuring that a
    close or shutdown request observed during the load prevents publication.
    """

    def __init__(self) -> None:
        super().__init__()
        self._predictor_shutdown = False
        self._cancelled_session_ids: set[str] = set()

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
            # Defensive cleanup for an ID whose prior loader failed after close.
            self._cancelled_session_ids.discard(session_id)
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
                self._cancelled_session_ids.discard(session_id)
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
            cancelled = session_id in self._cancelled_session_ids
            self._cancelled_session_ids.discard(session_id)
            if self._predictor_shutdown:
                publish_error = RuntimeError(
                    "Predictor was shut down while the session was loading; "
                    "the loaded state was disposed."
                )
            elif cancelled:
                publish_error = RuntimeError(
                    f"Session {session_id} was closed while it was loading; "
                    "the loaded state was disposed."
                )
            elif session_id in self._all_inference_states:
                publish_error = RuntimeError(
                    f"Session ID became occupied while loading: {session_id}"
                )
            else:
                self._all_inference_states[session_id] = session

        if publish_error is not None:
            try:
                self._dispose_session(session)
            except BaseException as exc:
                raise RuntimeError(
                    f"Failed to dispose unpublished session {session_id}."
                ) from exc
            raise publish_error
        return {"session_id": session_id}

    def close_session(
        self,
        session_id: str,
        run_gc_collect: bool = True,
    ) -> dict[str, bool]:
        session = None
        with self._sessions_lock:
            if session_id in self._reserved_session_ids:
                # Keep the reservation until its loader returns, but make
                # publication impossible.
                self._cancelled_session_ids.add(session_id)
                return {"is_success": True}
            session = self._all_inference_states.get(session_id)
            if session is not None:
                with session["lock"]:
                    session["closing"] = True
                    self._all_inference_states.pop(session_id, None)
        if session is not None:
            self._dispose_session(session)
            if run_gc_collect:
                gc.collect()
        return {"is_success": True}

    @staticmethod
    def _dispose_session(session: dict[str, Any]) -> None:
        """Dispose deterministically and mark closed even if a provider fails."""

        close_error: BaseException | None = None
        with session["lock"]:
            session["closing"] = True
            session["cancel_event"].set()
            session["propagation_active"] = False
            state = session.get("state")
            if isinstance(state, dict):
                frames = state.get("frames")
                close = getattr(frames, "close", None)
                if callable(close):
                    try:
                        close()
                    except BaseException as exc:
                        close_error = exc
                state.clear()
            session["closed"] = True
        if close_error is not None:
            raise RuntimeError(
                f"Failed to close frame provider for session {session['session_id']}."
            ) from close_error

    def _live_session_frame_counts(self) -> list[tuple[str, int]]:
        """Return a race-safe snapshot for diagnostics."""

        with self._sessions_lock:
            sessions = list(self._all_inference_states.items())
        snapshot: list[tuple[str, int]] = []
        for session_id, session in sessions:
            with session["lock"]:
                if session["closing"] or session["closed"]:
                    continue
                state = session.get("state")
                if isinstance(state, dict) and "num_frames" in state:
                    snapshot.append((session_id, int(state["num_frames"])))
        return snapshot

    def shutdown(self) -> None:
        """Atomically stop publication and attempt disposal of every live session."""

        with self._sessions_lock:
            if self._predictor_shutdown:
                return
            self._predictor_shutdown = True
            sessions = list(self._all_inference_states.values())
            for session in sessions:
                with session["lock"]:
                    session["closing"] = True
            self._all_inference_states.clear()
            self._reserved_session_ids.clear()
            self._cancelled_session_ids.clear()

        errors: list[BaseException] = []
        for session in sessions:
            try:
                self._dispose_session(session)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"Predictor shutdown encountered {len(errors)} disposal error(s)."
            ) from errors[0]
