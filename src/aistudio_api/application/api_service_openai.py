"""OpenAI protocol adapter, sharing Gemini's request and account engine."""

from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import json
import socket
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
from urllib.parse import urljoin

import httpx
from fastapi import HTTPException

from aistudio_api.api.responses_openai import (
    openai_sse_chunk,
    response_identity,
    to_openai_completion,
    to_openai_finish_reason,
    to_openai_images,
    to_openai_tools,
)
from aistudio_api.api.schemas import (
    GeminiContent,
    GeminiFunctionCall,
    GeminiFunctionResponse,
    GeminiGenerateContentRequest,
    GeminiGenerationConfig,
    GeminiInlineData,
    GeminiPart,
    GeminiTool,
)
from aistudio_api.api.schemas_openai import OpenAIChatRequest, OpenAIMessage
from aistudio_api.application.api_service_gemini import (
    _RequestStreamingResponse,
    classify_gemini_error_payload,
    run_generation,
    stream_generation,
)
from aistudio_api.domain.errors import TransportError
from aistudio_api.domain.models import GeneratedImage
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.gateway.model_catalog import model_catalog

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES_PER_REQUEST = 8
IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def openai_error_payload(exc: Exception) -> tuple[int, dict[str, object]]:
    status, message, _ = classify_gemini_error_payload(exc)
    error_type = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "invalid_request_error",
        429: "rate_limit_error",
    }.get(status, "server_error")
    detail = exc.detail if isinstance(exc, HTTPException) else None
    detail_map = detail if isinstance(detail, dict) else {}
    return status, {
        "error": {
            "message": message,
            "type": error_type,
            "param": detail_map.get("param"),
            "code": detail_map.get("code"),
        }
    }


def _bad_request(message: str, param: str | None = None) -> HTTPException:
    return HTTPException(400, detail={"message": message, "param": param})


def _google_metadata(extra: object) -> dict[str, object]:
    if not isinstance(extra, dict):
        return {}
    google = extra.get("google")
    return google if isinstance(google, dict) else {}


async def _download_image(url: str) -> tuple[str, bytes]:
    """Fetch bounded public images, pinning each validated DNS result to the request."""
    async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
        for _ in range(4):
            try:
                target = httpx.URL(url)
            except httpx.InvalidURL as exc:
                raise _bad_request("Invalid image URL", "messages") from exc
            if (
                target.scheme not in ("http", "https")
                or not target.host
                or target.userinfo
            ):
                raise _bad_request(
                    "Images require a public HTTP(S) or base64 data URL", "messages"
                )
            port = target.port or (443 if target.scheme == "https" else 80)
            try:
                addresses = await asyncio.get_running_loop().getaddrinfo(
                    target.host,
                    port,
                    type=socket.SOCK_STREAM,
                )
            except OSError as exc:
                raise _bad_request("Could not resolve image host", "messages") from exc
            ips = [str(item[4][0]) for item in addresses]
            if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
                raise _bad_request(
                    "Image URL must resolve to public addresses", "messages"
                )
            # Re-resolving the hostname inside the HTTP client would permit DNS
            # rebinding. Connect to the checked address while retaining TLS SNI.
            pinned = target.copy_with(host=ips[0])
            authority = target.netloc.decode("ascii")
            try:
                async with client.stream(
                    "GET",
                    pinned,
                    headers={"Host": authority},
                    extensions={"sni_hostname": target.host},
                ) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("location")
                        if not location:
                            raise _bad_request(
                                "Image redirect has no location", "messages"
                            )
                        url = urljoin(str(target), location)
                        continue
                    if response.status_code != 200:
                        raise _bad_request(
                            f"Image download returned HTTP {response.status_code}",
                            "messages",
                        )
                    mime = (
                        response.headers.get("content-type", "")
                        .split(";", 1)[0]
                        .strip()
                        .lower()
                    )
                    if mime not in IMAGE_MIME_TYPES:
                        raise _bad_request("Unsupported image content type", "messages")
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_IMAGE_BYTES:
                            raise _bad_request("Image exceeds 10 MiB", "messages")
                    if not data:
                        raise _bad_request("Image is empty", "messages")
                    return mime, bytes(data)
            except httpx.HTTPError as exc:
                raise _bad_request("Image download failed", "messages") from exc
    raise _bad_request("Too many image redirects", "messages")


async def _image_part(value: object) -> GeminiPart:
    image = value if isinstance(value, dict) else {}
    url = image.get("url")
    if not isinstance(url, str) or not url:
        raise _bad_request("image_url.url is required", "messages")
    if image.get("detail", "auto") not in ("auto", "low", "high"):
        raise _bad_request("Invalid image_url.detail", "messages")
    if url.startswith("data:"):
        header, separator, encoded = url.partition(",")
        mime = header.removeprefix("data:").removesuffix(";base64").lower()
        if (
            not separator
            or not header.endswith(";base64")
            or mime not in IMAGE_MIME_TYPES
        ):
            raise _bad_request("Expected a base64 image data URL", "messages")
        if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise _bad_request("Image exceeds 10 MiB", "messages")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise _bad_request("Invalid image base64", "messages") from exc
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise _bad_request("Image is empty or exceeds 10 MiB", "messages")
    else:
        mime, data = await _download_image(url)
    return GeminiPart(
        inlineData=GeminiInlineData(
            mimeType=mime, data=base64.b64encode(data).decode("ascii")
        )
    )


async def _content_parts(message: OpenAIMessage) -> list[GeminiPart]:
    if isinstance(message.content, str):
        return [GeminiPart(text=message.content)] if message.content else []
    parts: list[GeminiPart] = []
    for part in message.content or []:
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(GeminiPart(text=str(part["text"])))
        elif part.get("type") == "image_url" and message.role in ("user", "tool"):
            converted = await _image_part(part.get("image_url"))
            signature = _google_metadata(part.get("extra_content")).get(
                "thought_signature"
            )
            if isinstance(signature, str):
                converted.thoughtSignature = signature
            parts.append(converted)
        else:
            raise _bad_request("Unsupported content part for this role", "messages")
    return parts


def _function_declarations(req: OpenAIChatRequest) -> list[GeminiTool] | None:
    declarations: list[dict[str, object]] = []
    for tool in req.tools or []:
        function = tool.get("function")
        if (
            tool.get("type") != "function"
            or not isinstance(function, dict)
            or not function.get("name")
        ):
            raise _bad_request("Only named function tools are supported", "tools")
        if function.get("strict") is True:
            raise _bad_request("Strict tool validation is not supported", "tools")
        declarations.append(
            {key: value for key, value in function.items() if key != "strict"}
        )
    return [GeminiTool(functionDeclarations=declarations)] if declarations else None


def _tool_config(req: OpenAIChatRequest) -> dict[str, object] | None:
    if req.parallel_tool_calls is False:
        raise _bad_request(
            "parallel_tool_calls=false is not supported by this backend",
            "parallel_tool_calls",
        )
    choice = req.tool_choice
    if choice is None:
        return None
    if isinstance(choice, str):
        modes = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
        if choice not in modes:
            raise _bad_request("Unsupported tool_choice", "tool_choice")
        return {"functionCallingConfig": {"mode": modes[choice]}}
    function = choice.get("function")
    if (
        choice.get("type") != "function"
        or not isinstance(function, dict)
        or not function.get("name")
    ):
        raise _bad_request("Invalid tool_choice", "tool_choice")
    name = function["name"]
    names = set()
    for tool in req.tools or []:
        declaration = tool.get("function")
        if isinstance(declaration, dict):
            names.add(str(declaration.get("name")))
    if name not in names:
        raise _bad_request("tool_choice must name a declared function", "tool_choice")
    return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}


async def to_gemini_request(req: OpenAIChatRequest) -> GeminiGenerateContentRequest:
    if req.n != 1:
        raise _bad_request("Only n=1 is supported", "n")
    if (
        req.max_tokens is not None
        and req.max_completion_tokens is not None
        and req.max_tokens != req.max_completion_tokens
    ):
        raise _bad_request(
            "Conflicting max_tokens and max_completion_tokens", "max_completion_tokens"
        )
    if req.logprobs or req.top_logprobs is not None:
        raise _bad_request("OpenAI logprobs are not supported", "logprobs")
    options = req.stream_options or {}
    if set(options) - {"include_usage"} or (
        "include_usage" in options and not isinstance(options["include_usage"], bool)
    ):
        raise _bad_request("Unsupported stream_options", "stream_options")
    total_images = sum(
        1
        for msg in req.messages
        if isinstance(msg.content, list)
        for part in msg.content
        if part.get("type") == "image_url"
    )
    if total_images > MAX_IMAGES_PER_REQUEST:
        raise _bad_request("At most 8 images are supported per request", "messages")
    contents: list[GeminiContent] = []
    system: list[GeminiPart] = []
    call_names: dict[str, str] = {}
    for message in req.messages:
        parts = await _content_parts(message)
        if message.role in ("system", "developer"):
            if message.tool_calls or message.reasoning_content:
                raise _bad_request(
                    "System messages cannot contain tool calls or reasoning", "messages"
                )
            system.extend(parts)
            continue
        if message.role == "assistant":
            metadata = _google_metadata(message.extra_content)
            if message.reasoning_content:
                parts.insert(
                    0,
                    GeminiPart(
                        text=message.reasoning_content,
                        thought=True,
                        thoughtSignature=str(metadata.get("thinking_signature") or "")
                        or None,
                    ),
                )
            text_signature = metadata.get("thought_signature")
            if isinstance(text_signature, str):
                text_parts = [
                    part for part in parts if part.text is not None and not part.thought
                ]
                if text_parts:
                    text_parts[0].thoughtSignature = text_signature
            for call in message.tool_calls or []:
                function = call.get("function")
                call_id = call.get("id")
                if (
                    call.get("type", "function") != "function"
                    or not isinstance(function, dict)
                    or not function.get("name")
                    or not isinstance(call_id, str)
                    or not call_id
                ):
                    raise _bad_request("Invalid assistant tool_call", "messages")
                name = str(function["name"])
                if call_id in call_names:
                    raise _bad_request("Duplicate tool_call id in history", "messages")
                call_names[call_id] = name
                arguments = function.get("arguments", "{}")
                try:
                    args = (
                        json.loads(arguments)
                        if isinstance(arguments, str)
                        else arguments
                    )
                except ValueError as exc:
                    raise _bad_request(
                        "Tool arguments must be valid JSON", "messages"
                    ) from exc
                if not isinstance(args, dict):
                    raise _bad_request(
                        "Tool arguments must be a JSON object", "messages"
                    )
                signature = _google_metadata(call.get("extra_content")).get(
                    "thought_signature"
                )
                parts.append(
                    GeminiPart(
                        functionCall=GeminiFunctionCall(
                            name=name, args=args, id=call_id
                        ),
                        thoughtSignature=signature
                        if isinstance(signature, str)
                        else None,
                    )
                )
            signatures = [metadata.get("standalone_signature")]
            extra_signatures = metadata.get("thought_signatures")
            if isinstance(extra_signatures, list):
                signatures.extend(extra_signatures)
            parts.extend(
                GeminiPart(thoughtSignature=sig)
                for sig in signatures
                if isinstance(sig, str) and sig
            )
        elif message.role == "tool":
            call_id = message.tool_call_id
            if not call_id or call_id not in call_names:
                raise _bad_request(
                    "tool_call_id must match an earlier assistant tool call", "messages"
                )
            response_value: object = message.content
            if isinstance(response_value, str):
                with suppress(ValueError):
                    response_value = json.loads(response_value)
            elif isinstance(response_value, list):
                response_value = {
                    "result": "\n".join(
                        part.text for part in parts if part.text is not None
                    )
                }
            if not isinstance(response_value, dict):
                response_value = {"result": response_value}
            media = [part for part in parts if part.inlineData is not None]
            parts = [
                GeminiPart(
                    functionResponse=GeminiFunctionResponse(
                        name=call_names[call_id],
                        id=call_id,
                        response=response_value,
                        parts=media or None,
                    )
                )
            ]
        elif message.tool_calls or message.reasoning_content:
            raise _bad_request(
                "Only assistant messages may contain tool calls or reasoning",
                "messages",
            )
        if not parts:
            raise _bad_request("Message must contain content or tool calls", "messages")
        role = "model" if message.role == "assistant" else "user"
        if contents and contents[-1].role == role:
            contents[-1].parts.extend(parts)
        else:
            contents.append(GeminiContent(role=role, parts=parts))
    if not contents:
        raise _bad_request("At least one non-system message is required", "messages")
    config = GeminiGenerationConfig(
        temperature=req.temperature,
        topP=req.top_p,
        topK=req.top_k,
        maxOutputTokens=req.max_completion_tokens or req.max_tokens,
        stopSequences=[req.stop] if isinstance(req.stop, str) else req.stop,
        seed=req.seed,
        presencePenalty=req.presence_penalty,
        frequencyPenalty=req.frequency_penalty,
    )
    response_format = req.response_format or {}
    kind = response_format.get("type", "text")
    if kind in ("json_object", "json_schema"):
        config.responseMimeType = "application/json"
        if kind == "json_schema":
            definition = response_format.get("json_schema")
            if not isinstance(definition, dict) or not isinstance(
                definition.get("schema"), dict
            ):
                raise _bad_request("json_schema.schema is required", "response_format")
            if definition.get("strict") is True:
                raise _bad_request(
                    "Strict JSON Schema validation is not supported", "response_format"
                )
            config.responseSchema = definition["schema"]
    elif kind != "text":
        raise _bad_request("Unsupported response_format", "response_format")
    return GeminiGenerateContentRequest(
        contents=contents,
        systemInstruction=GeminiContent(parts=system) if system else None,
        tools=_function_declarations(req),
        toolConfig=_tool_config(req),
        generationConfig=config,
        reasoningEffort=req.reasoning_effort,
    )


def _safe_finish(reason: object, *, has_tools: bool) -> str:
    try:
        return to_openai_finish_reason(reason, has_tools=has_tools)
    except ValueError as exc:
        raise HTTPException(
            502, detail={"message": str(exc), "code": "upstream_incomplete"}
        ) from exc


async def _openai_stream(
    req: OpenAIChatRequest,
    converted: GeminiGenerateContentRequest,
    client: AIStudioClient,
) -> AsyncGenerator[str | bytes, None]:
    response_id, created = response_identity()
    final_reason: object = None
    usage: dict[str, object] = {}
    tools_seen: dict[str, tuple[int, str, str]] = {}
    generated_tool_ids: dict[tuple[str, int], str] = {}
    pending_signature: str | None = None
    try:
        yield openai_sse_chunk(
            response_id, created, req.model, {"role": "assistant", "content": ""}
        )
        async with aclosing(stream_generation(req.model, converted, client)) as events:
            async for kind, payload in events:
                if kind == "usage":
                    usage = payload if isinstance(payload, dict) else {}
                    continue
                if kind == "finish_reason":
                    final_reason = payload
                    continue
                if kind in ("done", "response_id"):
                    continue
                if kind == "thought_signature":
                    if pending_signature:
                        yield openai_sse_chunk(
                            response_id,
                            created,
                            req.model,
                            {
                                "extra_content": {
                                    "google": {
                                        "thought_signatures": [pending_signature]
                                    }
                                }
                            },
                        )
                    pending_signature = str(payload)
                    continue
                delta: dict[str, object] = {}
                if kind in ("body", "thinking"):
                    delta["content" if kind == "body" else "reasoning_content"] = str(
                        payload
                    )
                    if pending_signature:
                        key = (
                            "thought_signature"
                            if kind == "body"
                            else "thinking_signature"
                        )
                        delta["extra_content"] = {"google": {key: pending_signature}}
                        pending_signature = None
                elif kind == "tool_calls" and isinstance(payload, list):
                    raw_calls = [
                        dict(item) for item in payload if isinstance(item, dict)
                    ]
                    occurrences: dict[str, int] = {}
                    for raw_call in raw_calls:
                        name = str(raw_call.get("name") or "")
                        occurrence = occurrences.get(name, 0)
                        occurrences[name] = occurrence + 1
                        if not (raw_call.get("call_id") or raw_call.get("id")):
                            raw_call["call_id"] = generated_tool_ids.setdefault(
                                (name, occurrence),
                                f"call_{response_id}_{len(generated_tool_ids)}",
                            )
                    calls = to_openai_tools(raw_calls)
                    outgoing: list[dict[str, object]] = []
                    for call in calls:
                        function = call["function"]
                        if not isinstance(function, dict):
                            continue
                        call_id, arguments = str(call["id"]), str(function["arguments"])
                        if call_id in tools_seen:
                            index, previous, name = tools_seen[call_id]
                            if name != function["name"] or not arguments.startswith(
                                previous
                            ):
                                raise TransportError(
                                    "Upstream changed an already emitted tool call."
                                )
                            suffix = arguments[len(previous) :]
                            if not suffix and not call.get("extra_content"):
                                continue
                            item: dict[str, object] = {
                                "index": index,
                                "function": {"arguments": suffix},
                            }
                            if call.get("extra_content"):
                                item["extra_content"] = call["extra_content"]
                        else:
                            index = len(tools_seen)
                            item = {"index": index, **call}
                        tools_seen[call_id] = (index, arguments, str(function["name"]))
                        outgoing.append(item)
                    if outgoing:
                        delta["tool_calls"] = outgoing
                        if pending_signature:
                            delta["extra_content"] = {
                                "google": {"thought_signatures": [pending_signature]}
                            }
                            pending_signature = None
                elif kind == "images" and isinstance(payload, list):
                    delta["images"] = to_openai_images(
                        [
                            image
                            for image in payload
                            if isinstance(image, GeneratedImage)
                        ]
                    )
                if not delta:
                    continue
                yield openai_sse_chunk(response_id, created, req.model, delta)
        finish = _safe_finish(final_reason, has_tools=bool(tools_seen))
        if pending_signature:
            yield openai_sse_chunk(
                response_id,
                created,
                req.model,
                {
                    "extra_content": {
                        "google": {"thought_signatures": [pending_signature]}
                    }
                },
            )
        yield openai_sse_chunk(response_id, created, req.model, finish_reason=finish)
        if (req.stream_options or {}).get("include_usage"):
            yield openai_sse_chunk(response_id, created, req.model, usage=usage)
        yield "data: [DONE]\n\n"
    except Exception as exc:
        _, error = openai_error_payload(exc)
        yield "data: " + json.dumps(error, ensure_ascii=False) + "\n\n"


async def handle_openai_chat(req: OpenAIChatRequest, client: AIStudioClient):
    model_catalog.resolve_variant(req.model)
    converted = await to_gemini_request(req)
    if req.stream:
        return _RequestStreamingResponse(
            _openai_stream(req, converted, client), media_type="text/event-stream"
        )
    output = await run_generation(req.model, converted, client)
    _safe_finish(
        output.candidates[0].finish_reason if output.candidates else None,
        has_tools=bool(output.function_calls),
    )
    return to_openai_completion(output, req.model)
