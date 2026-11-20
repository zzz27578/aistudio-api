"""Offline contracts for media inputs carried by experimental wrapper aliases."""

from __future__ import annotations

import base64

import pytest

from aistudio_api.api.schemas import GeminiGenerateContentRequest
from aistudio_api.application.chat_service import normalize_gemini_request
from aistudio_api.application.text_wrapper import WRAPPER_TOOL_NAME, wrap_text_request

IMAGE = base64.b64encode(b"image-input").decode("ascii")
AUDIO = base64.b64encode(b"audio-input").decode("ascii")
VIDEO = base64.b64encode(b"video-input").decode("ascii")
SIGNATURE = base64.b64encode(b"original-model-signature").decode("ascii")
DECLARATION = {
    "name": "lookup",
    "parameters": {
        "type": "OBJECT",
        "properties": {"query": {"type": "STRING"}},
    },
}


def request(contents, **updates):
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": "Keep this system instruction."}]},
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024,
            "stopSequences": ["end"],
            "thinkingConfig": {"thinkingLevel": "HIGH", "includeThoughts": True},
            "responseModalities": ["TEXT"],
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
        ],
    }
    payload.update(updates)
    return GeminiGenerateContentRequest.model_validate(payload)


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
def test_wrapper_image_uses_native_history_without_losing_order_or_config(
    historical, with_tools
):
    image_turn = {
        "role": "user",
        "parts": [
            {"text": "before image"},
            {"inlineData": {"mimeType": "image/png", "data": IMAGE}},
            {"text": "after image"},
        ],
    }
    plain_turn = {"role": "user", "parts": [{"text": "plain question"}]}
    signed_answer = {
        "role": "model",
        "parts": [
            {"text": "prior thought", "thought": True, "thoughtSignature": SIGNATURE},
            {"text": "prior answer", "thoughtSignature": SIGNATURE},
        ],
    }
    req = request(
        [
            image_turn if historical else plain_turn,
            signed_answer,
            plain_turn if historical else image_turn,
        ],
        tools=[{"functionDeclarations": [DECLARATION]}] if with_tools else None,
    )
    original = req.model_dump()
    wrapped = wrap_text_request(req)
    normalized = normalize_gemini_request(wrapped, "gemini-test")

    assert req.model_dump() == original
    assert wrapped.contents == req.contents
    assert [turn.role for turn in normalized.contents] == ["user", "model", "user"]
    image_index = 0 if historical else 2
    parts = normalized.contents[image_index].parts
    assert [part.text for part in parts] == ["before image", None, "after image"]
    assert parts[1].inline_data == ("image/png", IMAGE)
    assert normalized.contents[2 - image_index].parts[0].text == "plain question"
    prior = normalized.contents[1].parts
    assert prior[0].thought is True and prior[0].text == "prior thought"
    assert prior[0].thought_signature == SIGNATURE
    assert prior[1].text == "prior answer"
    assert prior[1].thought_signature == SIGNATURE
    assert wrapped.generationConfig == req.generationConfig
    assert wrapped.safetySettings == req.safetySettings
    assert normalized.generation_config_overrides is not None
    assert normalized.generation_config_overrides["thinking_config"] == [
        1,
        None,
        None,
        3,
    ]
    assert normalized.generation_config_overrides["max_tokens"] == 1024
    assert normalized.generation_config_overrides["stop_sequences"] == ["end"]
    assert normalized.temperature == 0.3
    assert wrapped.tools is not None
    names = [
        declaration["name"]
        for tool in wrapped.tools
        for declaration in tool.functionDeclarations or []
    ]
    assert names == (["lookup"] if with_tools else []) + [WRAPPER_TOOL_NAME]
    assert wrapped.toolConfig == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": names}
    }


def test_wrapper_audio_video_and_metadata_survive_normalization():
    media_parts = [
        {"text": "narration", "speechMetadata": {"speaker": "Alice", "style": "calm"}},
        {"inlineData": {"mimeType": "audio/wav", "data": AUDIO}},
        {
            "inlineData": {"mimeType": "video/mp4", "data": VIDEO},
            "videoMetadata": {"startOffset": "1s", "endOffset": "3s", "fps": 2.0},
        },
        {"videoMetadata": {"startOffset": "4s", "endOffset": "5s", "fps": 1.0}},
    ]
    req = request([{"role": "user", "parts": media_parts}])
    original = req.model_dump()
    wrapped = wrap_text_request(req)
    normalized = normalize_gemini_request(wrapped, "gemini-test")
    assert req.model_dump() == original
    assert wrapped.contents == req.contents
    normalized_parts = normalized.contents[0].parts
    assert len(normalized_parts) == 4
    assert normalized_parts[0].text == "narration"
    assert normalized_parts[0].speech_metadata == {"speaker": "Alice", "style": "calm"}
    assert normalized_parts[1].inline_data == ("audio/wav", AUDIO)
    assert normalized_parts[2].inline_data == ("video/mp4", VIDEO)
    assert normalized_parts[2].video_metadata == {
        "start_offset": "1s",
        "end_offset": "3s",
        "fps": 2.0,
    }
    assert normalized_parts[3].video_metadata == {
        "start_offset": "4s",
        "end_offset": "5s",
        "fps": 1.0,
    }


@pytest.mark.parametrize(
    ("mime_type", "data"),
    [("image/png", IMAGE), ("audio/wav", AUDIO), ("video/mp4", VIDEO)],
)
def test_wrapper_tool_result_preserves_inline_media_and_response_id(mime_type, data):
    req = request(
        [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "lookup",
                            "id": "call_lookup",
                            "response": {"found": True},
                            "parts": [
                                {"text": "before media"},
                                {"inlineData": {"mimeType": mime_type, "data": data}},
                                {"text": "after media"},
                            ],
                        }
                    }
                ],
            }
        ]
    )
    original = req.model_dump()
    wrapped = wrap_text_request(req)
    normalized = normalize_gemini_request(wrapped, "gemini-test")
    assert req.model_dump() == original
    assert wrapped.contents == req.contents
    parts = normalized.contents[0].parts
    assert parts[0].function_response == ("lookup", {"found": True}, "call_lookup")
    assert [part.text for part in parts[1:]] == ["before media", None, "after media"]
    assert parts[2].inline_data == (mime_type, data)


@pytest.mark.parametrize(
    "part",
    [
        {"text": "narration", "speechMetadata": {"speaker": "Alice"}},
        {
            "inlineData": {"mimeType": "video/mp4", "data": VIDEO},
            "videoMetadata": {"fps": 2.0},
        },
        {"videoMetadata": {"fps": 2.0}},
    ],
)
def test_wrapper_rejects_tool_result_metadata_that_normalization_would_drop(part):
    req = request(
        [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "lookup",
                            "response": {},
                            "parts": [part],
                        }
                    }
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="Text wrapper does not support"):
        wrap_text_request(req)


@pytest.mark.parametrize(
    ("role", "part", "message"),
    [
        (
            "user",
            {"text": "caption", "inlineData": {"mimeType": "image/png", "data": IMAGE}},
            "Media data and text must be separate parts",
        ),
        (
            "user",
            {"speechMetadata": {"speaker": "Alice"}},
            "speechMetadata requires a text part",
        ),
        (
            "user",
            {"text": "caption", "videoMetadata": {"fps": 2.0}},
            "videoMetadata requires media or a standalone metadata part",
        ),
        (
            "model",
            {"functionCall": {"name": "lookup"}, "videoMetadata": {"fps": 2.0}},
            "videoMetadata requires media or a standalone metadata part",
        ),
        (
            "user",
            {
                "functionResponse": {"name": "lookup", "response": {}},
                "videoMetadata": {"fps": 2.0},
            },
            "videoMetadata requires media or a standalone metadata part",
        ),
    ],
)
def test_wrapper_rejects_part_combinations_that_normalization_would_drop(
    role, part, message
):
    req = request([{"role": role, "parts": [part]}])
    original = req.model_dump()
    with pytest.raises(ValueError, match=message):
        wrap_text_request(req)
    assert req.model_dump() == original


def test_wrapper_native_history_is_deep_copied():
    req = request(
        [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "lookup",
                            "id": "call_lookup",
                            "args": {"query": ["original"]},
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
                            "response": {"found": ["original"]},
                            "parts": [
                                {"inlineData": {"mimeType": "image/png", "data": IMAGE}}
                            ],
                        }
                    }
                ],
            },
        ],
        tools=[{"functionDeclarations": [DECLARATION]}],
        toolConfig={"functionCallingConfig": {"mode": "AUTO"}},
    )
    original = req.model_dump()
    wrapped = wrap_text_request(req)
    assert wrapped.contents == req.contents
    assert wrapped.contents is not req.contents
    call = wrapped.contents[0].parts[0].functionCall
    assert call is not None and isinstance(call.args, dict)
    call.args["query"] = ["changed"]
    wrapped.contents[0].parts[0].thoughtSignature = "changed"
    response = wrapped.contents[1].parts[0].functionResponse
    assert response is not None and isinstance(response.response, dict)
    response.response["found"] = ["changed"]
    assert response.parts is not None
    media = response.parts[0].inlineData
    assert media is not None
    media.data = "changed"
    assert wrapped.systemInstruction is not None
    wrapped.systemInstruction.parts[0].text = "changed"
    assert wrapped.tools is not None
    declarations = wrapped.tools[0].functionDeclarations
    assert declarations is not None
    declarations[0]["parameters"] = {"type": "STRING"}
    assert wrapped.toolConfig is not None
    wrapped.toolConfig["functionCallingConfig"] = {"mode": "NONE"}
    assert req.model_dump() == original


@pytest.mark.parametrize("nested", [False, True])
def test_wrapper_file_uri_remains_explicitly_unsupported(nested):
    file_part = {
        "fileData": {"mimeType": "image/png", "fileUri": "gs://offline/image.png"}
    }
    part = (
        {"functionResponse": {"name": "lookup", "response": {}, "parts": [file_part]}}
        if nested
        else file_part
    )
    req = request([{"role": "user", "parts": [part]}])
    with pytest.raises(ValueError, match="fileData is not supported yet"):
        wrap_text_request(req)


@pytest.mark.parametrize(
    "part",
    [
        {"inlineData": {"mimeType": "image/png", "data": IMAGE}},
        {"videoMetadata": {"fps": 1.0}},
        {"text": "narration", "speechMetadata": {"speaker": "Alice"}},
    ],
)
def test_wrapper_does_not_expand_media_support_to_system_instruction(part):
    req = request(
        [{"role": "user", "parts": [{"text": "question"}]}],
        systemInstruction={"parts": [part]},
    )
    with pytest.raises(ValueError, match="Text wrapper does not support"):
        wrap_text_request(req)


@pytest.mark.parametrize(
    "part",
    [
        {"executableCode": {"language": "PYTHON", "code": "print(1)"}},
        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "1"}},
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_wrapper_does_not_expand_media_support_to_code_history(part, nested):
    contents = (
        [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "lookup",
                            "response": {},
                            "parts": [part],
                        }
                    }
                ],
            }
        ]
        if nested
        else [{"role": "model", "parts": [part]}]
    )
    req = request(contents)
    with pytest.raises(ValueError, match="Text wrapper does not support"):
        wrap_text_request(req)


def test_wrapper_media_does_not_enable_builtin_tools():
    req = request(
        [
            {
                "role": "user",
                "parts": [{"inlineData": {"mimeType": "image/png", "data": IMAGE}}],
            }
        ],
        tools=[{"codeExecution": {}}],
    )
    with pytest.raises(ValueError, match="application function tools only"):
        wrap_text_request(req)
