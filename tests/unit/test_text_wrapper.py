"""Text wrapper contracts using local values only; no signer or model calls."""

import base64

import pytest

from aistudio_api.api.schemas import GeminiGenerateContentRequest
from aistudio_api.application.chat_service import normalize_gemini_request
from aistudio_api.application.text_wrapper import (
    WRAPPER_TOOL_NAME,
    TextWrapperStream,
    unwrap_text_output,
    wrap_text_request,
)
from aistudio_api.domain.errors import TransportError
from aistudio_api.domain.models import Candidate, ModelOutput


def request(**updates):
    payload = {
        "systemInstruction": {"parts": [{"text": "Keep the system instruction."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "question one"}]},
            {
                "role": "model",
                "parts": [
                    {
                        "text": "prior thought",
                        "thought": True,
                        "thoughtSignature": "thought-sig",
                    },
                    {"text": "prior answer", "thoughtSignature": "answer-sig"},
                ],
            },
            {"role": "user", "parts": [{"text": "question two"}]},
        ],
        "generationConfig": {
            "temperature": 0.25,
            "thinkingConfig": {"thinkingLevel": "HIGH"},
        },
    }
    payload.update(updates)
    return GeminiGenerateContentRequest.model_validate(payload)


def call(text, **updates):
    return {"name": WRAPPER_TOOL_NAME, "args": {"display_stream": text}, **updates}


def test_request_conversion_preserves_system_config_signatures_and_original():
    original = request()
    before = original.model_dump()
    transformed = wrap_text_request(original)
    assert original.model_dump() == before
    assert transformed.systemInstruction is not None
    assert transformed.generationConfig is not None
    assert transformed.systemInstruction.parts[0].text == "Keep the system instruction."
    assert len(transformed.systemInstruction.parts) == 2
    assert transformed.generationConfig.temperature == 0.25
    assert transformed.contents[0].parts[0].text == "Init"
    function_response = transformed.contents[2].parts[0].functionResponse
    assert function_response is not None
    assert function_response.response == {"user_input": "question one"}
    model = transformed.contents[3]
    assert model.parts[0].thoughtSignature == "thought-sig"
    assert model.parts[1].functionCall is not None
    assert model.parts[1].functionCall.args == {"display_stream": "prior answer"}
    assert model.parts[1].thoughtSignature == "answer-sig"
    placeholder = transformed.contents[1].parts[0].thoughtSignature
    assert placeholder is not None
    assert base64.b64decode(placeholder) == b"skip_thought_signature_validator"
    normalized = normalize_gemini_request(transformed, "models/gemini-4-hydrogen")
    assert normalized.tools
    assert normalized.tool_config
    assert normalized.contents[3].parts[1].function_call is not None
    assert normalized.contents[3].parts[1].function_call[0] == WRAPPER_TOOL_NAME
    assert normalized.generation_config_overrides is not None
    assert normalized.generation_config_overrides["thinking_config"] == [
        1,
        None,
        None,
        3,
    ]


@pytest.mark.parametrize(
    "updates",
    [
        {"tools": [{"googleSearch": {}}]},
        {"tools": [{"functionDeclarations": [{"name": "real_tool"}]}]},
        {"toolConfig": {"functionCallingConfig": {"mode": "NONE"}}},
        {"generationConfig": {"responseSchema": {}}},
        {"generationConfig": {"responseMimeType": "application/json"}},
        {"generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}},
        {"generationConfig": {"speechConfig": {}}},
        {
            "contents": [
                {"parts": [{"inlineData": {"mimeType": "image/png", "data": "AA=="}}]}
            ]
        },
        {
            "contents": [
                {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "real", "args": {}}}],
                }
            ]
        },
        {"contents": [{"role": "tool", "parts": [{"text": "result"}]}]},
    ],
)
def test_unsupported_combinations_fail_explicitly(updates):
    with pytest.raises(ValueError):
        wrap_text_request(request(**updates))


def test_nonstream_restores_text_without_forging_terminal_or_usage():
    original = ModelOutput(
        model="models/gemini-4-hydrogen",
        response_id="r-1",
        usage={"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
        candidates=[
            Candidate(
                text="stray outside text",
                thinking="reasoning",
                thinking_signature="thinking-sig",
                function_calls=[call("answer", thought_signature="tool-sig")],
                finish_reason=2,
                finish_message="token limit",
            )
        ],
    )
    restored = unwrap_text_output(original)
    assert restored.text == "answer"
    assert restored.function_calls == []
    assert restored.thinking == "reasoning"
    assert restored.candidates[0].thinking_signature == "thinking-sig"
    assert restored.candidates[0].standalone_signature == "tool-sig"
    assert restored.candidates[0].finish_reason == 2
    assert restored.candidates[0].finish_message == "token limit"
    assert restored.response_id == "r-1"
    assert restored.usage == original.usage
    assert original.function_calls


def test_nonstream_plain_fallback_retains_safety_and_unknown_completion():
    for reason in (None, 3, 99):
        original = ModelOutput(
            candidates=[Candidate(text="plain fallback", finish_reason=reason)]
        )
        restored = unwrap_text_output(original)
        assert restored.text == "plain fallback"
        assert restored.candidates[0].finish_reason == reason


def test_stream_buffers_content_but_preserves_terminal_order_and_signatures():
    stream = TextWrapperStream()
    usage = {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5}
    assert stream.feed("response_id", "r-1") == [("response_id", "r-1")]
    assert stream.feed("thinking", "thought") == [("thinking", "thought")]
    assert stream.feed("body", "stray") == []
    assert stream.feed("tool_calls", [call("hello ", thought_signature="sig-1")]) == []
    assert stream.feed("tool_calls", [call("world")]) == []
    assert stream.feed("finish_reason", 2) == []
    assert stream.feed("usage", usage) == []
    assert stream.feed("done", None) == []
    assert stream.finish() == [
        ("body", "hello world"),
        ("thought_signature", "sig-1"),
        ("finish_reason", 2),
        ("usage", usage),
        ("done", None),
    ]
    assert stream.finish() == []


def test_stream_does_not_invent_success_or_deduplicate_repeated_delta_text():
    stream = TextWrapperStream()
    stream.feed("tool_calls", [call("repeat")])
    stream.feed("tool_calls", [call("repeat")])
    assert stream.finish() == [("body", "repeatrepeat")]
    fallback = TextWrapperStream()
    fallback.feed("body", "plain")
    fallback.feed("finish_reason", 3)
    assert fallback.finish() == [("body", "plain"), ("finish_reason", 3)]


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "other", "args": {}},
        {"name": WRAPPER_TOOL_NAME, "arguments": '{"display_stream":"half'},
    ],
)
def test_malformed_or_unexpected_tools_raise_instead_of_returning_success(bad):
    with pytest.raises(TransportError):
        TextWrapperStream().feed("tool_calls", [bad])
    with pytest.raises(TransportError):
        unwrap_text_output(ModelOutput(candidates=[Candidate(function_calls=[bad])]))
