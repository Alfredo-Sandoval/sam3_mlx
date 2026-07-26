from __future__ import annotations

import gc
import inspect
from threading import Event, RLock
import time
import uuid
from typing import Any

from sam3_mlx._unsupported import raise_unsupported


class Sam3BasePredictor:
    """Torch-free request dispatcher matching the official SAM3 video API."""

    def __init__(self) -> None:
        self.model = None
        self._all_inference_states: dict[str, dict[str, Any]] = {}
        self._sessions_lock = RLock()
        self._reserved_session_ids: set[str] = set()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = request["type"]
        if request_type == "start_session":
            return self.start_session(
                resource_path=request["resource_path"],
                session_id=request.get("session_id"),
                offload_video_to_cpu=request.get("offload_video_to_cpu", False),
                offload_state_to_cpu=request.get("offload_state_to_cpu", False),
            )
        if request_type == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_index"],
                text=request.get("text"),
                points=request.get("points"),
                point_labels=request.get("point_labels"),
                clear_old_points=request.get("clear_old_points", True),
                bounding_boxes=request.get("bounding_boxes"),
                bounding_box_labels=request.get("bounding_box_labels"),
                clear_old_boxes=request.get("clear_old_boxes", True),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    getattr(self, "default_output_prob_thresh", 0.5),
                ),
                obj_id=request.get("obj_id"),
                rel_coordinates=request.get("rel_coordinates", True),
            )
        if request_type == "remove_object":
            return self.remove_object(
                session_id=request["session_id"],
                frame_idx=request.get("frame_index", 0),
                obj_id=request["obj_id"],
            )
        if request_type == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        if request_type == "cancel_propagation":
            return self.cancel_propagation(session_id=request["session_id"])
        if request_type == "close_session":
            if "clear_cache_threshold" in request:
                raise_unsupported(
                    "Sam3BasePredictor.close_session(clear_cache_threshold)",
                    reason="port-gap",
                    detail=(
                        "MLX does not expose the upstream CUDA cache-percentage "
                        "contract, so this control cannot be honored."
                    ),
                    alternative="Omit clear_cache_threshold.",
                )
            return self.close_session(
                session_id=request["session_id"],
                run_gc_collect=request.get("run_gc_collect", True),
            )
        raise RuntimeError(f"invalid request type: {request_type}")

    def handle_stream_request(self, request: dict[str, Any]):
        request_type = request["type"]
        if request_type == "propagate_in_video":
            yield from self.propagate_in_video(
                session_id=request["session_id"],
                propagation_direction=request.get("propagation_direction", "both"),
                start_frame_idx=request.get("start_frame_index"),
                max_frame_num_to_track=request.get("max_frame_num_to_track"),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    getattr(self, "default_output_prob_thresh", 0.5),
                ),
            )
            return
        raise RuntimeError(f"invalid request type: {request_type}")

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
        session = {
            "state": inference_state,
            "session_id": session_id,
            "created_monotonic": now,
            "last_used_monotonic": now,
            "lock": RLock(),
            "cancel_event": cancel_event,
        }
        with self._sessions_lock:
            self._reserved_session_ids.discard(session_id)
            self._all_inference_states[session_id] = session
        return {"session_id": session_id}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: str | None = None,
        points=None,
        point_labels=None,
        clear_old_points: bool = True,
        bounding_boxes=None,
        bounding_box_labels=None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: int | None = None,
        rel_coordinates: bool = True,
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        with session["lock"]:
            self._extend_expiration_time(session)
            prompt_kwargs = dict(
                inference_state=session["state"],
                frame_idx=frame_idx,
                text_str=text,
                points=points,
                point_labels=point_labels,
                clear_old_points=clear_old_points,
                boxes_xywh=bounding_boxes,
                box_labels=bounding_box_labels,
                clear_old_boxes=clear_old_boxes,
                output_prob_thresh=output_prob_thresh,
                rel_coordinates=rel_coordinates,
            )
            if obj_id is not None:
                prompt_kwargs["obj_id"] = obj_id

            signature = inspect.signature(self.model.add_prompt)
            valid_params = set(signature.parameters)
            if not clear_old_points and "clear_old_points" not in valid_params:
                raise_unsupported(
                    "Sam3BasePredictor.add_prompt(clear_old_points=False)",
                    reason="video-tracker",
                    detail="The selected-frame MLX path resets point prompts per request.",
                    alternative="clear_old_points=True",
                )
            if not clear_old_boxes and "clear_old_boxes" not in valid_params:
                raise_unsupported(
                    "Sam3BasePredictor.add_prompt(clear_old_boxes=False)",
                    reason="video-tracker",
                    detail="The selected-frame MLX path resets box prompts per request.",
                    alternative="clear_old_boxes=True",
                )
            if obj_id is not None and "obj_id" not in valid_params:
                raise_unsupported(
                    "Sam3BasePredictor.add_prompt(obj_id)",
                    reason="video-tracker",
                    detail="The selected model cannot honor caller-supplied object IDs.",
                    alternative="Omit obj_id.",
                )
            filtered_kwargs = {
                key: value for key, value in prompt_kwargs.items() if key in valid_params
            }

            frame_idx, outputs = self.model.add_prompt(**filtered_kwargs)
        return {"frame_index": frame_idx, "outputs": outputs}

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        with session["lock"]:
            self._extend_expiration_time(session)
            result = self.model.remove_object(
                session["state"],
                obj_id=obj_id,
                frame_idx=frame_idx,
                is_user_action=is_user_action,
            )
        if result is None or (isinstance(result, tuple) and result[1] is None):
            import numpy as np

            state = session["state"]
            outputs = {
                "out_obj_ids": np.zeros(0, dtype=np.int64),
                "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
                "out_binary_masks": np.zeros(
                    (
                        0,
                        int(state["orig_height"]),
                        int(state["orig_width"]),
                    ),
                    dtype=bool,
                ),
            }
        elif isinstance(result, tuple):
            _, outputs = result
        else:
            outputs = result
        return {"frame_index": frame_idx, "outputs": outputs}

    def cancel_propagation(self, session_id: str) -> dict[str, bool]:
        session = self._get_session(session_id)
        with session["lock"]:
            self._extend_expiration_time(session)
            session["cancel_event"].set()
            if hasattr(self.model, "cancel_propagation"):
                self.model.cancel_propagation(session["state"])
        return {"is_success": True}

    def propagate_in_video(
        self,
        session_id: str,
        propagation_direction: str = "both",
        start_frame_idx: int | None = None,
        max_frame_num_to_track: int | None = None,
        output_prob_thresh: float = 0.5,
        **kwargs,
    ):
        session = self._get_session(session_id)
        with session["lock"]:
            self._extend_expiration_time(session)
            session["cancel_event"].clear()
            if propagation_direction not in {"both", "forward", "backward"}:
                raise ValueError(
                    f"invalid propagation direction: {propagation_direction}"
                )
            signature = inspect.signature(self.model.propagate_in_video)
            propagate_kwargs = {
                "inference_state": session["state"],
                "start_frame_idx": start_frame_idx,
                "max_frame_num_to_track": max_frame_num_to_track,
            }
            if "output_prob_thresh" in signature.parameters:
                propagate_kwargs["output_prob_thresh"] = output_prob_thresh
            for key, value in kwargs.items():
                if key in signature.parameters:
                    propagate_kwargs[key] = value
        directions = []
        if propagation_direction in {"both", "forward"}:
            directions.append(False)
        if propagation_direction in {"both", "backward"}:
            directions.append(True)
        for reverse in directions:
            generator = self.model.propagate_in_video(
                **propagate_kwargs,
                reverse=reverse,
            )
            while not session["cancel_event"].is_set():
                try:
                    with session["lock"]:
                        frame_idx, outputs = next(generator)
                        self._extend_expiration_time(session)
                except StopIteration:
                    break
                yield {"frame_index": frame_idx, "outputs": outputs}
            if session["cancel_event"].is_set():
                return

    def reset_session(self, session_id: str) -> dict[str, bool]:
        session = self._get_session(session_id)
        with session["lock"]:
            self._extend_expiration_time(session)
            self.model.reset_state(session["state"])
        return {"is_success": True}

    def close_session(
        self,
        session_id: str,
        run_gc_collect: bool = True,
    ) -> dict[str, bool]:
        with self._sessions_lock:
            session = self._all_inference_states.pop(session_id, None)
        if session is not None:
            self._dispose_session(session)
            if run_gc_collect:
                gc.collect()
        return {"is_success": True}

    def _get_session(self, session_id: str) -> dict[str, Any]:
        expired_session = None
        with self._sessions_lock:
            session = self._all_inference_states.get(session_id)
            if session is not None and self._session_is_expired(session):
                expired_session = self._all_inference_states.pop(session_id, None)
                session = None
        if expired_session is not None:
            self._dispose_session(expired_session)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def _extend_expiration_time(self, session: dict[str, Any]) -> None:
        session["last_used_monotonic"] = time.monotonic()

    def _session_is_expired(self, session: dict[str, Any]) -> bool:
        expiration_sec = getattr(self, "session_expiration_sec", None)
        if expiration_sec is None or expiration_sec <= 0:
            return False
        idle_seconds = time.monotonic() - session["last_used_monotonic"]
        return idle_seconds >= expiration_sec

    @staticmethod
    def _dispose_session(session: dict[str, Any]) -> None:
        with session["lock"]:
            session["cancel_event"].set()
            state = session.get("state")
            if isinstance(state, dict):
                state.clear()

    def shutdown(self) -> None:
        with self._sessions_lock:
            sessions = list(self._all_inference_states.values())
            self._all_inference_states.clear()
            self._reserved_session_ids.clear()
        for session in sessions:
            self._dispose_session(session)
