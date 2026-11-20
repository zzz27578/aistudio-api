"""OpenAI JSON and SSE serialization over the shared generation result."""

from __future__ import annotations

import base64
import json
import time
import uuid

from aistudio_api.api.responses import _coerce_usage_int
from aistudio_api.domain.models import GeneratedImage, ModelOutput


def response_identity() -> tuple[str, int]:
    return f"chatcmpl-{uuid.uuid4().hex}", int(time.time())


def to_openai_usage(usage: dict[str, object] | None) -> dict[str, object]:
    raw = usage or {}
    completion_details = raw.get("completion_tokens_details")
    details = completion_details if isinstance(completion_details, dict) else {}
    prompt = _coerce_usage_int(raw.get("prompt_tokens"))
    completion = _coerce_usage_int(raw.get("completion_tokens"))
    if not completion:
        completion = _coerce_usage_int(
            details.get("visible_tokens")
        ) + _coerce_usage_int(details.get("reasoning_tokens"))
    result: dict[str, object] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": _coerce_usage_int(raw.get("total_tokens"))
        or prompt + completion,
    }
    if "reasoning_tokens" in details:
        result["completion_tokens_details"] = {
            "reasoning_tokens": _coerce_usage_int(details.get("reasoning_tokens"))
        }
    if raw.get("cached_tokens") is not None:
        result["prompt_tokens_details"] = {
            "cached_tokens": _coerce_usage_int(raw.get("cached_tokens"))
        }
    return result


def to_openai_finish_reason(reason: object, *, has_tools: bool = False) -> str:
    normalized = str(reason).upper()
    if normalized in ("2", "MAX_TOKENS"):
        return "length"
    if normalized in {
        "3",
        "4",
        "7",
        "8",
        "9",
        "11",
        "14",
        "17",
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }:
        return "content_filter"
    if normalized not in ("1", "22", "STOP"):
        raise ValueError(
            f"Upstream generation did not complete successfully ({reason})."
        )
    return "tool_calls" if has_tools else "stop"


def google_signature(signature: str) -> dict[str, object]:
    return {"google": {"thought_signature": signature}}


def to_openai_tools(calls: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for call in calls:
        arguments = call.get("args", call.get("arguments", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        tool: dict[str, object] = {
            "id": str(
                call.get("call_id") or call.get("id") or f"call_{uuid.uuid4().hex}"
            ),
            "type": "function",
            "function": {"name": str(call.get("name") or ""), "arguments": arguments},
        }
        signature = str(call.get("thought_signature") or "")
        if signature:
            tool["extra_content"] = google_signature(signature)
        result.append(tool)
    return result


def to_openai_images(images: list[GeneratedImage]) -> list[dict[str, object]]:
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image.mime};base64,{base64.b64encode(image.data).decode('ascii')}"
            },
            **(
                {"extra_content": google_signature(image.thought_signature)}
                if image.thought_signature
                else {}
            ),
        }
        for image in images
    ]


def to_openai_completion(
    output: ModelOutput, requested_model: str
) -> dict[str, object]:
    response_id, created = response_identity()
    candidate = output.candidates[0] if output.candidates else None
    tools = to_openai_tools(output.function_calls)
    message: dict[str, object] = {"role": "assistant", "content": output.text or None}
    if output.thinking:
        message["reasoning_content"] = output.thinking
    if tools:
        message["tool_calls"] = tools
    if output.images:
        # Explicit extension; this does not implement the Images API.
        message["images"] = to_openai_images(output.images)
    if candidate:
        signatures = {
            key: value
            for key, value in {
                "thought_signature": candidate.text_signature,
                "thinking_signature": candidate.thinking_signature,
                "standalone_signature": candidate.standalone_signature,
            }.items()
            if value
        }
        if signatures:
            message["extra_content"] = {"google": signatures}
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": requested_model.removeprefix("models/"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": to_openai_finish_reason(
                    candidate.finish_reason if candidate else None,
                    has_tools=bool(tools),
                ),
            }
        ],
        "usage": to_openai_usage(output.usage),
    }


def openai_sse_chunk(
    response_id: str,
    created: int,
    model: str,
    delta: dict[str, object] | None = None,
    *,
    finish_reason: str | None = None,
    usage: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model.removeprefix("models/"),
        "choices": []
        if usage is not None
        else [
            {
                "index": 0,
                "delta": delta or {},
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = to_openai_usage(usage)
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
