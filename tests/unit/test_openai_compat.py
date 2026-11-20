"""Offline protocol contracts; no Google requests, signer or credentials."""

from __future__ import annotations

import base64
import json
import socket
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from aistudio_api.api.app import app
from aistudio_api.api.dependencies import get_client
from aistudio_api.api.schemas_openai import OpenAIChatRequest
from aistudio_api.application import api_service_gemini as engine
from aistudio_api.application.api_service_openai import (
    _download_image,
    to_gemini_request,
)
from aistudio_api.application.chat_service import normalize_gemini_request
from aistudio_api.config import settings
from aistudio_api.domain.models import Candidate, ModelOutput
from aistudio_api.infrastructure.gateway.client import AIStudioClient, PreparedRequest
from aistudio_api.infrastructure.gateway.model_catalog import (
    AliasPolicy,
    ModelPolicy,
    ThinkingLevel,
    model_catalog,
    wrapper_alias_id,
)
from aistudio_api.infrastructure.gateway.model_discovery import model_discovery

SIGNATURE = base64.b64encode(b"opaque-tool-signature").decode("ascii")


@pytest.fixture
def offline_openai(monkeypatch, tmp_path):
    monkeypatch.setenv("AISTUDIO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "api_keys", frozenset())
    monkeypatch.setattr(engine, "ensure_active_account", AsyncMock())
    monkeypatch.setattr(engine, "_record_request_rotator_event", MagicMock())
    monkeypatch.setattr(engine.runtime_state, "record", MagicMock())
    discovered = [{"id": "gemini-test", "thinkingLevels": [1, 3]}]
    monkeypatch.setattr(
        model_discovery, "get_models", AsyncMock(return_value=discovered)
    )
    model_catalog.sync(discovered)
    client = MagicMock(spec=AIStudioClient)
    client.session = None
    client.prepare_request = AsyncMock(
        return_value=PreparedRequest(url="https://offline.test", headers={}, body="[]")
    )
    client.generate_content = AsyncMock(
        return_value=ModelOutput(
            candidates=[
                Candidate(text="answer", thinking="reasoning", finish_reason=1)
            ],
            usage={
                "prompt_tokens": 8,
                "completion_tokens": 5,
                "total_tokens": 13,
                "cached_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        )
    )
    monkeypatch.setitem(app.dependency_overrides, get_client, lambda: client)
    return client


def _http_client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://offline.test"
    )


def _chunks(text: str):
    return [
        json.loads(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.asyncio
async def test_nonstream_chat_shares_engine_and_usage(offline_openai):
    async with _http_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-test",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    result = response.json()
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "reasoning",
    }
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens_details"] == {"reasoning_tokens": 2}
    assert result["usage"]["prompt_tokens_details"] == {"cached_tokens": 3}
    assert (
        offline_openai.prepare_request.call_args.kwargs["model"] == "models/gemini-test"
    )
    ensure_account = engine.ensure_active_account
    assert isinstance(ensure_account, AsyncMock)
    ensure_account.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_and_openai_routes_coexist(offline_openai):
    async with _http_client() as client:
        response = await client.post(
            "/v1beta/models/gemini-test:generateContent",
            json={"contents": [{"parts": [{"text": "hi"}]}]},
        )
    assert response.status_code == 200
    assert response.json()["candidates"][0]["content"]["parts"][-1]["text"] == "answer"
    assert "choices" not in response.json()


@pytest.mark.asyncio
async def test_alias_priority_and_model_visibility_shared(offline_openai):
    model_catalog.update(
        "gemini-test",
        ModelPolicy(
            visible=False,
            thinking_aliases={ThinkingLevel.HIGH: AliasPolicy(enabled=True)},
        ),
    )
    async with _http_client() as client:
        listed = (await client.get("/v1/models")).json()["data"]
        assert [item["id"] for item in listed] == ["gemini-test-thinkhigh"]
        assert (await client.get("/v1/models/gemini-test")).status_code == 404
        assert (await client.get("/v1/models/gemini-test-thinkhigh")).status_code == 200
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-test-thinkhigh",
                "reasoning_effort": "low",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["model"] == "gemini-test-thinkhigh"
    config = offline_openai.prepare_request.call_args.kwargs[
        "generation_config_overrides"
    ]
    assert config["thinking_config"] == [1, None, None, 3]
    ensure_account = engine.ensure_active_account
    assert isinstance(ensure_account, AsyncMock)
    assert ensure_account.call_args.kwargs["model"] == "models/gemini-test"


@pytest.mark.asyncio
async def test_disabled_model_rejected_before_account_selection(offline_openai):
    model_catalog.update("gemini-test", ModelPolicy(enabled=False))
    async with _http_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-test",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 403
    assert response.json()["error"]["type"] == "permission_error"
    ensure_account = engine.ensure_active_account
    assert isinstance(ensure_account, AsyncMock)
    ensure_account.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_history_uses_real_calls_and_results():
    request = OpenAIChatRequest.model_validate(
        {
            "model": "gemini-test",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "developer", "content": "developer"},
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"nested":{"cities":["Paris"]}}',
                            },
                            "extra_content": {
                                "google": {"thought_signature": SIGNATURE}
                            },
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "clock", "arguments": "{}"},
                        },
                    ],
                    "extra_content": {"google": {"standalone_signature": SIGNATURE}},
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "content": '{"temperature":20}',
                },
                {"role": "tool", "tool_call_id": "call_b", "content": "noon"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        }
    )
    converted = await to_gemini_request(request)
    assert converted.systemInstruction is not None
    assert [part.text for part in converted.systemInstruction.parts] == [
        "system",
        "developer",
    ]
    assistant = converted.contents[1]
    assert assistant.role == "model"
    assert assistant.parts[0].functionCall is not None
    assert assistant.parts[0].functionCall.id == "call_a"
    assert assistant.parts[0].functionCall.args == {"nested": {"cities": ["Paris"]}}
    assert assistant.parts[0].thoughtSignature == SIGNATURE
    assert assistant.parts[-1].thoughtSignature == SIGNATURE
    responses = converted.contents[2].parts
    assert [
        part.functionResponse.id for part in responses if part.functionResponse
    ] == ["call_a", "call_b"]
    assert responses[-1].functionResponse is not None
    assert responses[-1].functionResponse.response == {"result": "noon"}
    normalized = normalize_gemini_request(converted, "gemini-test")
    assert normalized.tools is not None
    assert "city" in str(normalized.tools)


@pytest.mark.asyncio
async def test_data_image_input_maps_to_inline_data():
    encoded = base64.b64encode(b"image-fixture").decode("ascii")
    converted = await to_gemini_request(
        OpenAIChatRequest.model_validate(
            {
                "model": "gemini-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )
    image = converted.contents[0].parts[1].inlineData
    assert image is not None and image.mimeType == "image/png" and image.data == encoded


@pytest.mark.asyncio
async def test_remote_image_blocks_private_addresses_without_fetch(monkeypatch):
    loop = __import__("asyncio").get_running_loop()
    monkeypatch.setattr(
        loop,
        "getaddrinfo",
        AsyncMock(
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
            ]
        ),
    )
    with pytest.raises(HTTPException, match="400"):
        await _download_image("http://example.test/image.png")


@pytest.mark.asyncio
async def test_remote_image_pins_public_address_and_sni(monkeypatch):
    loop = __import__("asyncio").get_running_loop()
    monkeypatch.setattr(
        loop,
        "getaddrinfo",
        AsyncMock(
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ]
        ),
    )
    received = []

    def serve(request):
        received.append(request)
        return httpx.Response(
            200, headers={"content-type": "image/png"}, content=b"image-fixture"
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(serve), **kwargs),
    )
    assert await _download_image("https://example.test/image.png") == (
        "image/png",
        b"image-fixture",
    )
    assert received[0].url.host == "93.184.216.34"
    assert received[0].headers["host"] == "example.test"
    assert received[0].extensions["sni_hostname"] == "example.test"


@pytest.mark.asyncio
async def test_stream_has_stable_identity_tools_usage_and_signature(offline_openai):
    async def generate(**_kwargs):
        yield "thinking", "reason"
        yield (
            "tool_calls",
            [
                {
                    "name": "weather",
                    "args": {"city": "Paris"},
                    "call_id": "call_a",
                    "thought_signature": SIGNATURE,
                },
                {"name": "clock", "args": {}, "call_id": "call_b"},
            ],
        )
        yield "usage", {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        yield "finish_reason", 1
        yield "done", None

    offline_openai.stream_generate_content = generate
    async with _http_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-test",
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    chunks = _chunks(response.text)
    assert len({chunk["id"] for chunk in chunks}) == 1
    assert len({chunk["created"] for chunk in chunks}) == 1
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    tool_delta = next(
        chunk["choices"][0]["delta"]["tool_calls"]
        for chunk in chunks
        if chunk["choices"] and "tool_calls" in chunk["choices"][0]["delta"]
    )
    assert [call["index"] for call in tool_delta] == [0, 1]
    assert [call["id"] for call in tool_delta] == ["call_a", "call_b"]
    assert tool_delta[0]["extra_content"]["google"]["thought_signature"] == SIGNATURE
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["total_tokens"] == 7
    assert response.text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(("reason", "expected"), [(2, "length"), (3, "content_filter")])
async def test_finish_reasons_are_not_all_stop(offline_openai, reason, expected):
    offline_openai.generate_content.return_value = ModelOutput(
        candidates=[Candidate(text="partial", finish_reason=reason)]
    )
    async with _http_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-test",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == expected


@pytest.mark.asyncio
async def test_stream_failure_after_content_has_error_and_no_success_done(
    offline_openai,
):
    async def generate(**_kwargs):
        yield "body", "partial"
        raise HTTPException(502, detail="upstream disconnected")

    offline_openai.stream_generate_content = generate
    async with _http_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-test",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    chunks = _chunks(response.text)
    assert chunks[-1]["error"]["type"] == "server_error"
    assert "[DONE]" not in response.text
    assert not any(
        chunk.get("choices", [{}])[0].get("finish_reason")
        for chunk in chunks
        if chunk.get("choices")
    )


@pytest.mark.asyncio
async def test_repeated_tool_snapshot_without_upstream_id_is_not_duplicated(
    offline_openai,
):
    async def generate(**_kwargs):
        yield "tool_calls", [{"name": "weather", "args": {"city": "Paris"}}]
        yield "tool_calls", [{"name": "weather", "args": {"city": "Paris"}}]
        yield "finish_reason", 1
        yield "done", None

    offline_openai.stream_generate_content = generate
    async with _http_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-test",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    tools = [
        call
        for chunk in _chunks(response.text)
        if chunk.get("choices")
        for call in chunk["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert len(tools) == 1
    assert tools[0]["index"] == 0 and tools[0]["id"].startswith("call_")


@pytest.mark.asyncio
async def test_auth_and_validation_use_openai_error_shape(offline_openai, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", frozenset({"offline-secret"}))
    async with _http_client() as client:
        unauth = await client.get("/v1/models")
        invalid = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer offline-secret"},
            json={"model": "gemini-test", "messages": []},
        )
    assert unauth.status_code == 401 and unauth.headers["www-authenticate"] == "Bearer"
    assert unauth.json()["error"]["type"] == "authentication_error"
    assert (
        invalid.status_code == 400
        and invalid.json()["error"]["type"] == "invalid_request_error"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"n": 2},
        {"parallel_tool_calls": False},
        {"max_tokens": 10, "max_completion_tokens": 20},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {}, "strict": True},
            }
        },
    ],
)
async def test_unsupported_semantics_are_explicit(offline_openai, extra):
    async with _http_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-test",
                "messages": [{"role": "user", "content": "hi"}],
                **extra,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    offline_openai.generate_content.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_wrapper_alias_uses_shared_engine_and_preserves_usage(
    offline_openai, stream
):
    model_catalog.update(
        "gemini-test",
        ModelPolicy(text_wrapper=AliasPolicy(enabled=True)),
    )
    wrapped_call = {
        "name": "emit_complete_response",
        "args": {"display_stream": "wrapped answer"},
    }
    offline_openai.generate_content.return_value = ModelOutput(
        candidates=[Candidate(function_calls=[wrapped_call], finish_reason=1)],
        usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    )

    async def generate(**_kwargs):
        yield "tool_calls", [wrapped_call]
        yield "finish_reason", 1
        yield "usage", {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
        yield "done", None

    offline_openai.stream_generate_content = generate
    async with _http_client() as client:
        listed = (await client.get("/v1/models")).json()["data"]
        name = wrapper_alias_id("gemini-test")
        assert name in [item["id"] for item in listed]
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": name,
                "stream": stream,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    if stream:
        chunks = _chunks(response.text)
        assert any(
            chunk.get("choices")
            and chunk["choices"][0]["delta"].get("content") == "wrapped answer"
            for chunk in chunks
        )
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["usage"]["total_tokens"] == 7
    else:
        assert response.json()["choices"][0]["message"]["content"] == "wrapped answer"
        assert response.json()["usage"]["total_tokens"] == 7
    contents = offline_openai.prepare_request.call_args.kwargs["contents"]
    assert contents[1].parts[0].function_call[0] == "emit_complete_response"


@pytest.mark.asyncio
async def test_wrapper_rejects_reserved_tool_name_with_openai_error(offline_openai):
    model_catalog.update(
        "gemini-test", ModelPolicy(text_wrapper=AliasPolicy(enabled=True))
    )
    async with _http_client() as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": wrapper_alias_id("gemini-test"),
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"type": "function", "function": {"name": "emit_complete_response"}}
                ],
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    offline_openai.prepare_request.assert_not_awaited()
