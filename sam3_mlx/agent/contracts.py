"""Shared typed contracts for the optional agent orchestration surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NotRequired, Protocol, TypedDict

from PIL import Image


type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)


class ContentPart(TypedDict):
    type: str
    text: NotRequired[str]
    image: NotRequired[str]


class Message(TypedDict):
    role: str
    content: str | list[ContentPart]


class AgentOutput(TypedDict):
    orig_img_h: int
    orig_img_w: int
    pred_boxes: list[list[float]]
    pred_masks: list[object]
    pred_scores: list[float]
    original_image_path: NotRequired[str]
    output_image_path: NotRequired[str]
    text_prompt: NotRequired[str]
    image_path: NotRequired[str]


type AgentInferenceResult = tuple[list[Message], AgentOutput, Image.Image]


class GenerateRequest(Protocol):
    def __call__(self, messages: Sequence[Message]) -> str | None: ...


class SegmentService(Protocol):
    def __call__(
        self,
        *,
        image_path: str,
        text_prompt: str,
        output_folder_path: str,
    ) -> str: ...


class ImageProcessor(Protocol):
    def set_image(self, image: Image.Image) -> Mapping[str, object]: ...

    def set_text_prompt(
        self, *, state: Mapping[str, object], prompt: str
    ) -> Mapping[str, object]: ...


class Visualize(Protocol):
    def __call__(
        self, input_json: Mapping[str, object], zoom_in_index: int | None = None
    ) -> Image.Image | tuple[Image.Image, Image.Image]: ...
