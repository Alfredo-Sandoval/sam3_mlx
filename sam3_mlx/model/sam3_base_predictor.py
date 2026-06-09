from __future__ import annotations

import gc
import inspect
import time
import uuid
from typing import Any

_CLEAR_CACHE_THRESHOLD = 80


class Sam3BasePredictor:
    """Torch-free request dispatcher matching the official SAM3 video API."""

    def __init__(self) -> None:
        self.model = None
        self._all_inference_states: dict[str, dict[str, Any]] = {}

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
            return self.close_session(
                session_id=request["session_id"],
                run_gc_collect=request.get("run_gc_collect", True),
                clear_cache_threshold=int(
                    request.get("clear_cache_threshold", _CLEAR_CACHE_THRESHOLD)
                ),
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
        init_kwargs = {
            "resource_path": resource_path,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
        }
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        inference_state = self.model.init_state(**init_kwargs)

        if session_id is None:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
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
        self._extend_expiration_time(session)
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
        self._extend_expiration_time(session)
        if propagation_direction not in {"both", "forward", "backward"}:
            raise ValueError(f"invalid propagation direction: {propagation_direction}")
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
        if propagation_direction in {"both", "forward"}:
            for frame_idx, outputs in self.model.propagate_in_video(
                **propagate_kwargs,
                reverse=False,
            ):
                yield {"frame_index": frame_idx, "outputs": outputs}
        if propagation_direction in {"both", "backward"}:
            for frame_idx, outputs in self.model.propagate_in_video(
                **propagate_kwargs,
                reverse=True,
            ):
                yield {"frame_index": frame_idx, "outputs": outputs}

    def reset_session(self, session_id: str) -> dict[str, bool]:
        session = self._get_session(session_id)
        self._extend_expiration_time(session)
        self.model.reset_state(session["state"])
        return {"is_success": True}

    def close_session(
        self,
        session_id: str,
        run_gc_collect: bool = True,
        clear_cache_threshold: int = _CLEAR_CACHE_THRESHOLD,
    ) -> dict[str, bool]:
        del clear_cache_threshold
        session = self._all_inference_states.pop(session_id, None)
        if session is not None:
            state = session.get("state")
            if isinstance(state, dict):
                state.clear()
            session.clear()
            if run_gc_collect:
                gc.collect()
        return {"is_success": True}

    def _get_session(self, session_id: str) -> dict[str, Any]:
        session = self._all_inference_states.get(session_id)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def _extend_expiration_time(self, session: dict[str, Any]) -> None:
        session["last_use_time"] = time.time()

    def shutdown(self) -> None:
        self._all_inference_states.clear()
