"""LLM client compatibility surface."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from sam3_mlx.agent.contracts import ContentPart, JsonValue, Message


class _ChatClient(Protocol):
    def chat(self, *, messages: object, sampling_params: object) -> object: ...


class _GeneratedText(Protocol):
    text: str


class _RequestOutput(Protocol):
    outputs: Sequence[_GeneratedText]


def _content_part_json(part: ContentPart) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {"type": part["type"]}
    if "text" in part:
        result["text"] = part["text"]
    if "image" in part:
        result["image"] = part["image"]
    return result


def _message_content_json(content: str | list[ContentPart]) -> JsonValue:
    if isinstance(content, str):
        return content
    return [_content_part_json(part) for part in content]


def get_image_base64_and_mime(image_path: str | Path) -> tuple[str, str]:
    """Convert an image file to a base64 string and MIME type."""
    path = Path(image_path)
    ext = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime_type = mime_types.get(ext, "image/jpeg")
    return base64.b64encode(path.read_bytes()).decode("utf-8"), mime_type


def _process_messages_for_openai(
    messages: Sequence[Message],
) -> list[dict[str, JsonValue]]:
    """Translate official SAM3 image message parts into OpenAI image_url parts."""
    processed_messages: list[dict[str, JsonValue]] = []
    for message in messages:
        processed_message: dict[str, JsonValue] = {"role": message["role"]}
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            processed_message["content"] = _message_content_json(content)
            processed_messages.append(processed_message)
            continue

        processed_content: list[JsonValue] = []
        for part in content:
            if part.get("type") == "image":
                image_path = part.get("image")
                if not isinstance(image_path, str):
                    raise TypeError("Image message parts must contain a string path.")
                base64_image, mime_type = get_image_base64_and_mime(image_path)
                processed_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    }
                )
            else:
                processed_content.append(_content_part_json(part))
        processed_message["content"] = processed_content
        processed_messages.append(processed_message)
    return processed_messages


def _chat_completions_url(server_url: str | None) -> str:
    base_url = (
        server_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    )
    return f"{base_url.rstrip('/')}/chat/completions"


def _extract_response_text(response_payload: object) -> str | None:
    if not isinstance(response_payload, dict):
        raise TypeError("LLM response payload must be a JSON object.")
    payload = cast(dict[str, object], response_payload)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = cast(list[object], choices)[0]
    if not isinstance(first, dict):
        raise TypeError("LLM response choices must contain JSON objects.")
    message = cast(dict[str, object], first).get("message")
    if not isinstance(message, dict):
        return None
    content = cast(dict[str, object], message).get("content")
    if content is not None and not isinstance(content, str):
        raise TypeError("LLM response message content must be a string or null.")
    return content


def send_generate_request(
    messages: Sequence[Message],
    server_url: str | None = None,
    model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key: str | None = None,
    max_tokens: int = 4096,
) -> str | None:
    """
    Send an OpenAI-compatible chat completion request.

    This mirrors the official SAM3 agent client shape from
    ``third_party/facebook-sam3`` while avoiding a hard dependency on the
    ``openai`` package in the MLX runtime. ``server_url`` should point at an
    OpenAI-compatible server root such as ``http://127.0.0.1:8000/v1``. If it is
    omitted, ``OPENAI_BASE_URL`` is used and then the public OpenAI API root.
    """
    if isinstance(max_tokens, bool):
        raise TypeError("max_tokens must be an integer, not bool.")
    processed_messages = _process_messages_for_openai(list(messages))
    payload = {
        "model": model,
        "messages": processed_messages,
        "max_completion_tokens": max_tokens,
        "n": 1,
    }
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    request = urllib.request.Request(
        _chat_completions_url(server_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM request failed with HTTP {exc.code} from {request.full_url}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed for {request.full_url}: {exc}") from exc

    return _extract_response_text(response_payload)


def send_direct_request(
    llm: _ChatClient,
    messages: Sequence[Message],
    sampling_params: object,
) -> str | None:
    """Run the official-shaped direct vLLM chat path with processed image parts."""
    processed_messages = _process_messages_for_openai(messages)
    outputs_value = llm.chat(
        messages=processed_messages,
        sampling_params=sampling_params,
    )
    if not isinstance(outputs_value, Sequence) or not outputs_value:
        return None
    first = cast(Sequence[object], outputs_value)[0]
    if not hasattr(first, "outputs"):
        raise RuntimeError(f"Unexpected direct LLM output format: {outputs_value!r}")
    outputs = cast(_RequestOutput, first).outputs
    if not outputs:
        raise RuntimeError(f"Unexpected direct LLM output format: {outputs_value!r}")
    return outputs[0].text
