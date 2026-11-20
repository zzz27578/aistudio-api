"""AstrBot-style tool loops through wrapper aliases, using an offline gateway."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from aistudio_api.api.app import app
from aistudio_api.api.dependencies import get_client
from aistudio_api.application import api_service_gemini as engine
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

BASE = "gemini-test"
MODEL = wrapper_alias_id(BASE, ThinkingLevel.HIGH)
SIGNATURE = base64.b64encode(b"real-external-tool-signature").decode("ascii")
IMAGE_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aH0cAAAAASUVORK5CYII="
)
IMAGE_DATA = base64.b64encode(IMAGE_BYTES).decode("ascii")
USAGE = {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}
EXTERNAL_CALL = {
    "name": "lookup",
    "args": {"query": "hello"},
    "call_id": "call_lookup",
    "thought_signature": SIGNATURE,
}
FINAL_CALL = {
    "name": "emit_complete_response",
    "args": {"display_stream": "completed answer"},
}
DECLARATION = {
    "name": "lookup",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


@pytest.fixture
def gateway(monkeypatch, tmp_path):
    monkeypatch.setenv("AISTUDIO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "api_keys", frozenset())
    monkeypatch.setattr(settings, "web_password", None)
    monkeypatch.setattr(engine, "ensure_active_account", AsyncMock())
    monkeypatch.setattr(engine, "_record_request_rotator_event", MagicMock())
    monkeypatch.setattr(engine.runtime_state, "record", MagicMock())
    model_catalog.sync([{"id": BASE, "thinkingLevels": [3]}])
    model_catalog.update(
        BASE,
        ModelPolicy(
            wrapper_thinking_aliases={ThinkingLevel.HIGH: AliasPolicy(enabled=True)}
        ),
    )
    client = MagicMock(spec=AIStudioClient)
    client.session = None
    client.prepare_request = AsyncMock(
        return_value=PreparedRequest(url="https://offline.test", headers={}, body="[]")
    )
    monkeypatch.setitem(app.dependency_overrides, get_client, lambda: client)
    return client


def upstream_output(client, call):
    client.generate_content = AsyncMock(
        return_value=ModelOutput(
            candidates=[Candidate(function_calls=[dict(call)], finish_reason=1)],
            usage=dict(USAGE),
        )
    )

    async def events(**_kwargs):
        yield "tool_calls", [dict(call)]
        yield "finish_reason", 1
        yield "usage", dict(USAGE)
        yield "done", None

    client.stream_generate_content = events


def payload(protocol, stream, *, after_tool=False, mode=None):
    if protocol == "openai":
        messages: list[dict[str, object]] = [{"role": "user", "content": "hello"}]
        if after_tool:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_lookup",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"query":"hello"}',
                                },
                                "extra_content": {
                                    "google": {"thought_signature": SIGNATURE}
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_lookup",
                        "content": '{"found":"result"}',
                    },
                ]
            )
        result: dict[str, object] = {
            "model": MODEL,
            "messages": messages,
            "tools": [{"type": "function", "function": DECLARATION}],
            "stream": stream,
            "stream_options": {"include_usage": True},
        }
        if mode is not None:
            result["tool_choice"] = (
                {"type": "function", "function": {"name": "lookup"}}
                if mode == "specific"
                else mode
            )
        return "/v1/chat/completions", result
    contents: list[dict[str, object]] = [{"role": "user", "parts": [{"text": "hello"}]}]
    if after_tool:
        contents.extend(
            [
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "lookup",
                                "args": {"query": "hello"},
                                "id": "call_lookup",
                            },
                            "thoughtSignature": SIGNATURE,
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "lookup",
                                "id": "call_lookup",
                                "response": {"found": "result"},
                            }
                        }
                    ],
                },
            ]
        )
    result = {
        "contents": contents,
        "tools": [{"functionDeclarations": [DECLARATION]}],
    }
    if mode is not None:
        config: dict[str, object] = {
            "mode": {"auto": "AUTO", "none": "NONE"}.get(mode, "ANY")
        }
        if mode == "specific":
            config["allowedFunctionNames"] = ["lookup"]
        result["toolConfig"] = {"functionCallingConfig": config}
    method = "streamGenerateContent" if stream else "generateContent"
    return f"/v1beta/models/{MODEL}:{method}", result


async def invoke(protocol, stream, *, after_tool=False, mode=None):
    path, body = payload(protocol, stream, after_tool=after_tool, mode=mode)
    return await invoke_body(path, body, stream)


async def invoke_body(path, body, stream):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://offline.test"
    ) as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    if not stream:
        return [response.json()]
    values = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert values and not any("error" in value for value in values)
    return values


def image_payload(protocol, stream, scenario, with_tools):
    path, body = payload(protocol, stream, after_tool=scenario == "tool_result")
    if not with_tools:
        body.pop("tools")
    if protocol == "openai":
        image = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{IMAGE_DATA}"},
        }
        if scenario == "tool_result":
            messages = body["messages"]
            assert isinstance(messages, list)
            last_message = messages[-1]
            assert isinstance(last_message, dict)
            last_message["content"] = [
                {"type": "text", "text": "tool image"},
                image,
            ]
        else:
            image_turn = {
                "role": "user",
                "content": [{"type": "text", "text": "image question"}, image],
            }
            plain_turn = {"role": "user", "content": "plain question"}
            signed_answer = {
                "role": "assistant",
                "content": "prior answer",
                "reasoning_content": "prior thought",
                "extra_content": {
                    "google": {
                        "thought_signature": SIGNATURE,
                        "thinking_signature": SIGNATURE,
                    }
                },
            }
            body["messages"] = [
                image_turn if scenario == "historical" else plain_turn,
                signed_answer,
                plain_turn if scenario == "historical" else image_turn,
            ]
    else:
        image = {"inlineData": {"mimeType": "image/png", "data": IMAGE_DATA}}
        if scenario == "tool_result":
            contents = body["contents"]
            assert isinstance(contents, list)
            last_content = contents[-1]
            assert isinstance(last_content, dict)
            parts = last_content["parts"]
            assert isinstance(parts, list)
            response_part = parts[0]
            assert isinstance(response_part, dict)
            response = response_part["functionResponse"]
            assert isinstance(response, dict)
            response["parts"] = [image]
        else:
            image_turn = {
                "role": "user",
                "parts": [{"text": "image question"}, image],
            }
            plain_turn = {"role": "user", "parts": [{"text": "plain question"}]}
            signed_answer = {
                "role": "model",
                "parts": [
                    {
                        "text": "prior thought",
                        "thought": True,
                        "thoughtSignature": SIGNATURE,
                    },
                    {"text": "prior answer", "thoughtSignature": SIGNATURE},
                ],
            }
            body["contents"] = [
                image_turn if scenario == "historical" else plain_turn,
                signed_answer,
                plain_turn if scenario == "historical" else image_turn,
            ]
    return path, body


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
async def test_wrapper_external_tool_roundtrip_and_final_answer(
    gateway, protocol, stream
):
    upstream_output(gateway, EXTERNAL_CALL)
    values = await invoke(protocol, stream)
    if protocol == "openai":
        choices = [choice for value in values for choice in value.get("choices", [])]
        field = "delta" if stream else "message"
        calls = [
            call for choice in choices for call in choice[field].get("tool_calls", [])
        ]
        assert calls[0]["id"] == "call_lookup"
        assert calls[0]["function"]["name"] == "lookup"
        assert calls[0]["extra_content"]["google"]["thought_signature"] == SIGNATURE
        assert choices[-1]["finish_reason"] == "tool_calls"
    else:
        parts = [
            part
            for value in values
            for candidate in value.get("candidates", [])
            for part in candidate["content"]["parts"]
        ]
        tool = next(part for part in parts if "functionCall" in part)
        assert tool["functionCall"] == {
            "name": "lookup",
            "args": {"query": "hello"},
            "id": "call_lookup",
        }
        assert tool["thoughtSignature"] == SIGNATURE

    upstream_output(gateway, FINAL_CALL)
    values = await invoke(protocol, stream, after_tool=True)
    contents = gateway.prepare_request.call_args.kwargs["contents"]
    calls = [part for turn in contents for part in turn.parts if part.function_call]
    responses = [
        part for turn in contents for part in turn.parts if part.function_response
    ]
    assert calls[0].function_call == ("lookup", {"query": "hello"}, "call_lookup")
    assert calls[0].thought_signature == SIGNATURE
    assert responses[0].function_response == (
        "lookup",
        {"found": "result"},
        "call_lookup",
    )
    if protocol == "openai":
        choices = [choice for value in values for choice in value.get("choices", [])]
        field = "delta" if stream else "message"
        assert (
            "".join(choice[field].get("content") or "" for choice in choices)
            == "completed answer"
        )
        assert choices[-1]["finish_reason"] == "stop"
        assert values[-1]["usage"]["total_tokens"] == 12
        assert not any(choice[field].get("tool_calls") for choice in choices)
    else:
        parts = [
            part
            for value in values
            for candidate in value.get("candidates", [])
            for part in candidate["content"]["parts"]
        ]
        assert "".join(part.get("text", "") for part in parts) == "completed answer"
        assert not any(part.get("functionCall") for part in parts)
        assert values[-1]["usageMetadata"]["totalTokenCount"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai", "gemini"])
@pytest.mark.parametrize(
    ("mode", "allowed"),
    [
        ("auto", {"lookup", "emit_complete_response"}),
        ("none", {"emit_complete_response"}),
        ("required", {"lookup"}),
        ("specific", {"lookup"}),
    ],
)
async def test_wrapper_preserves_client_tool_selection(
    gateway, protocol, mode, allowed
):
    upstream_output(gateway, FINAL_CALL if mode == "none" else EXTERNAL_CALL)
    await invoke(protocol, False, mode=mode)
    prepared = gateway.prepare_request.call_args.kwargs
    config = prepared["tool_config"]
    assert config[1][0] == 2
    assert set(config[1][1]) == allowed
    assert prepared["generation_config_overrides"]["thinking_config"] == [
        1,
        None,
        None,
        3,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai", "gemini"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scenario", ["current", "historical", "tool_result"])
@pytest.mark.parametrize("with_tools", [False, True])
async def test_wrapper_media_input_preserves_history_and_unwraps_final_text(
    gateway, protocol, stream, scenario, with_tools
):
    upstream_output(gateway, FINAL_CALL)
    path, body = image_payload(protocol, stream, scenario, with_tools)
    values = await invoke_body(path, body, stream)
    contents = gateway.prepare_request.call_args.kwargs["contents"]
    assert [turn.role for turn in contents] == ["user", "model", "user"]
    images = [part for turn in contents for part in turn.parts if part.inline_data]
    assert len(images) == 1
    assert images[0].inline_data == ("image/png", IMAGE_DATA)
    assert base64.b64decode(images[0].inline_data[1]) == IMAGE_BYTES
    calls = [part for turn in contents for part in turn.parts if part.function_call]
    responses = [
        part for turn in contents for part in turn.parts if part.function_response
    ]
    if scenario == "tool_result":
        assert calls[0].function_call == ("lookup", {"query": "hello"}, "call_lookup")
        assert calls[0].thought_signature == SIGNATURE
        assert responses[0].function_response == (
            "lookup",
            {"result": "tool image"} if protocol == "openai" else {"found": "result"},
            "call_lookup",
        )
        assert contents[-1].parts[-1].inline_data == ("image/png", IMAGE_DATA)
    else:
        assert not calls and not responses
        prior = contents[1].parts
        assert prior[0].text == "prior thought" and prior[0].thought is True
        assert prior[0].thought_signature == SIGNATURE
        assert prior[1].text == "prior answer"
        assert prior[1].thought_signature == SIGNATURE
        image_index = 0 if scenario == "historical" else 2
        assert contents[image_index].parts[-1].inline_data == ("image/png", IMAGE_DATA)
        assert contents[2 - image_index].parts[0].text == "plain question"
    if protocol == "openai":
        choices = [choice for value in values for choice in value.get("choices", [])]
        field = "delta" if stream else "message"
        assert (
            "".join(choice[field].get("content") or "" for choice in choices)
            == "completed answer"
        )
        assert choices[-1]["finish_reason"] == "stop"
        assert not any(choice[field].get("tool_calls") for choice in choices)
        assert values[-1]["usage"]["total_tokens"] == 12
    else:
        parts = [
            part
            for value in values
            for candidate in value.get("candidates", [])
            for part in candidate["content"]["parts"]
        ]
        assert "".join(part.get("text", "") for part in parts) == "completed answer"
        assert not any(part.get("functionCall") for part in parts)
        assert values[-1]["usageMetadata"]["totalTokenCount"] == 12
