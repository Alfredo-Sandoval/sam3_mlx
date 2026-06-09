"""LLM client compatibility surface."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


def get_image_base64_and_mime(image_path):
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
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate official SAM3 image message parts into OpenAI image_url parts."""
    processed_messages: list[dict[str, Any]] = []
    for message in messages:
        processed_message = dict(message)
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            processed_messages.append(processed_message)
            continue

        processed_content: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                image_path = part["image"]
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
                processed_content.append(part)
        processed_message["content"] = processed_content
        processed_messages.append(processed_message)
    return processed_messages


def _chat_completions_url(server_url: str | None) -> str:
    base_url = (
        server_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    )
    return f"{base_url.rstrip('/')}/chat/completions"


def _extract_response_text(response_payload: dict[str, Any]) -> Optional[str]:
    choices = response_payload.get("choices")
    if not choices:
        return None
    message = choices[0].get("message", {})
    return message.get("content")


def send_generate_request(
    messages,
    server_url=None,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key=None,
    max_tokens=4096,
):
    """
    Send an OpenAI-compatible chat completion request.

    This mirrors the official SAM3 agent client shape from
    ``third_party/facebook-sam3`` while avoiding a hard dependency on the
    ``openai`` package in the MLX runtime. ``server_url`` should point at an
    OpenAI-compatible server root such as ``http://127.0.0.1:8000/v1``. If it is
    omitted, ``OPENAI_BASE_URL`` is used and then the public OpenAI API root.
    """
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
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any,
) -> Optional[str]:
    """Run the official-shaped direct vLLM chat path with processed image parts."""
    processed_messages = _process_messages_for_openai(messages)
    outputs = llm.chat(
        messages=processed_messages,
        sampling_params=sampling_params,
    )
    if not outputs:
        return None
    try:
        return outputs[0].outputs[0].text
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(f"Unexpected direct LLM output format: {outputs!r}") from exc
