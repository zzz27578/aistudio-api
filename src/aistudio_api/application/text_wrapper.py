"""Experimental text/tool wrapping shared by Gemini and OpenAI adapters.

This changes the output carrier; it does not resume interrupted generations.
Request transformation happens before normal normalization and signing.
"""

from __future__ import annotations

import base64
import copy
import json
from dataclasses import dataclass, field, replace

from aistudio_api.api.schemas import (
    GeminiContent,
    GeminiFunctionCall,
    GeminiFunctionResponse,
    GeminiGenerateContentRequest,
    GeminiPart,
    GeminiTool,
)
from aistudio_api.domain.errors import TransportError
from aistudio_api.domain.models import Candidate, ModelOutput

WRAPPER_TOOL_NAME = "emit_complete_response"
_SKIP_SIGNATURE = base64.b64encode(b"skip_thought_signature_validator").decode("ascii")
_WRAPPER_INSTRUCTION = (
    "Return the complete user-visible answer by calling emit_complete_response once. "
    "Put the answer in its display_stream string. Do not emit the answer outside "
    "this function call. The conversation is represented by this function and "
    "its user_input responses. Follow all system instructions and safety requirements."
)
_TOOL_WRAPPER_INSTRUCTION = (
    "Use the application's function tools when needed. To give the final answer "
    "to the user, call emit_complete_response once and put the complete answer "
    "in display_stream. That function is an internal output carrier, not an "
    "application action. Do not call it in the same turn as an application tool. "
    "Preserve the existing conversation and follow all system instructions and "
    "safety requirements."
)
_MAX_BUFFERED_CHARACTERS = 8 * 1024 * 1024
StreamEvent = tuple[str, object | None]


def _require_text_parts(
    content: GeminiContent, *, allow_functions: bool = False
) -> None:
    for part in content.parts:
        values = part.model_dump(exclude_none=True)
        allowed = {"text", "thought", "thoughtSignature"}
        if allow_functions:
            allowed.update(("functionCall", "functionResponse"))
        if set(values) - allowed:
            raise ValueError("Text wrapper models only accept text conversation parts")
        if part.functionCall is not None and content.role not in ("model", "assistant"):
            raise ValueError("Function calls must be in model history")
        if part.functionResponse is not None:
            if (content.role or "user") != "user":
                raise ValueError("Function responses must be in user history")
            if part.functionResponse.parts:
                _require_text_parts(GeminiContent(parts=part.functionResponse.parts))


def _validate_request(req: GeminiGenerateContentRequest) -> list[str]:
    if not req.contents:
        raise ValueError("contents is required")
    names: list[str] = []
    for tool in req.tools or []:
        if set(tool.model_dump(exclude_none=True)) - {"functionDeclarations"}:
            raise ValueError(
                "Text wrapper models support application function tools only"
            )
        for declaration in tool.functionDeclarations or []:
            name = declaration.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("Application function tools require a name")
            if name == WRAPPER_TOOL_NAME:
                raise ValueError("emit_complete_response is reserved for text wrapping")
            if name in names:
                raise ValueError("Duplicate application function name")
            names.append(name)
    config = req.generationConfig
    if config is not None:
        if config.responseSchema is not None or (
            config.responseMimeType is not None
            and config.responseMimeType.lower() != "text/plain"
        ):
            raise ValueError("Text wrapper models do not support structured output")
        if config.responseModalities is not None and any(
            str(value).upper() != "TEXT" for value in config.responseModalities
        ):
            raise ValueError("Text wrapper models only support TEXT output")
        if any(
            value is not None
            for value in (config.imageConfig, config.speechConfig, config.videoConfig)
        ):
            raise ValueError("Text wrapper models do not support media generation")
    if req.systemInstruction is not None:
        _require_text_parts(req.systemInstruction)
    for content in req.contents:
        if (content.role or "user").lower() not in ("user", "model", "assistant"):
            raise ValueError("Text wrapper history only accepts user and model roles")
        _require_text_parts(content, allow_functions=True)
        if (content.role or "user").lower() == "user" and any(
            part.thought or part.thoughtSignature for part in content.parts
        ):
            raise ValueError("User text cannot carry model thoughts or signatures")
    return names


def _wrapper_tool() -> GeminiTool:
    return GeminiTool(
        functionDeclarations=[
            {
                "name": WRAPPER_TOOL_NAME,
                "description": "Return the complete user-visible answer.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"display_stream": {"type": "STRING"}},
                    "required": ["display_stream"],
                },
            }
        ]
    )


def _wrapper_tool_config(
    req: GeminiGenerateContentRequest, names: list[str]
) -> dict[str, object]:
    config = copy.deepcopy(req.toolConfig or {})
    raw = config.pop("functionCallingConfig", None)
    snake_raw = config.pop("function_calling_config", None)
    if raw is None:
        raw = snake_raw
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("functionCallingConfig must be an object")
    if set(raw) - {"mode", "allowedFunctionNames", "allowed_function_names"}:
        raise ValueError("Unsupported function calling configuration for text wrapping")
    modes = {0: "AUTO", 1: "AUTO", 2: "ANY", 3: "NONE", 4: "VALIDATED"}
    raw_mode = raw.get("mode", "AUTO")
    mode = modes.get(raw_mode) if isinstance(raw_mode, int) else str(raw_mode).upper()
    if mode == "MODE_UNSPECIFIED":
        mode = "AUTO"
    if mode not in ("AUTO", "ANY", "NONE", "VALIDATED"):
        raise ValueError("Unsupported function calling mode for text wrapping")
    allowed = raw.get("allowedFunctionNames", raw.get("allowed_function_names"))
    if allowed is not None and (
        not isinstance(allowed, list)
        or any(not isinstance(name, str) or name not in names for name in allowed)
    ):
        raise ValueError(
            "allowedFunctionNames must name declared application functions"
        )
    selected = list(allowed) if isinstance(allowed, list) and allowed else list(names)
    if mode == "NONE":
        # Disable application actions while retaining our private output carrier.
        selected = [WRAPPER_TOOL_NAME]
    elif mode == "ANY":
        # A required/specific application tool must not be satisfied by final text.
        if not selected:
            raise ValueError("Required application tool is not declared")
    else:
        selected.append(WRAPPER_TOOL_NAME)
    config["functionCallingConfig"] = {"mode": "ANY", "allowedFunctionNames": selected}
    return config


def _wrap_with_application_tools(
    req: GeminiGenerateContentRequest, names: list[str]
) -> GeminiGenerateContentRequest:
    system = (
        req.systemInstruction.model_copy(deep=True)
        if req.systemInstruction
        else GeminiContent(parts=[])
    )
    system.parts.append(GeminiPart(text=_TOOL_WRAPPER_INSTRUCTION))
    return req.model_copy(
        update={
            "contents": [content.model_copy(deep=True) for content in req.contents],
            "systemInstruction": system,
            "tools": [
                *(tool.model_copy(deep=True) for tool in req.tools or []),
                _wrapper_tool(),
            ],
            "toolConfig": _wrapper_tool_config(req, names),
        }
    )


@dataclass
class _TextTurn:
    role: str
    text: str
    thoughts: list[GeminiPart] = field(default_factory=list)
    signature: str | None = None


def wrap_text_request(
    req: GeminiGenerateContentRequest,
) -> GeminiGenerateContentRequest:
    """Build a private tool chain without mutating or discarding client fields."""
    names = _validate_request(req)
    if names or any(
        part.functionCall is not None or part.functionResponse is not None
        for content in req.contents
        for part in content.parts
    ):
        # Real tool transcripts must keep their original parts, IDs and signatures.
        return _wrap_with_application_tools(req, names)
    turns: list[_TextTurn] = []
    for content in req.contents:
        role = "user" if (content.role or "user").lower() == "user" else "model"
        turn = _TextTurn(
            role=role,
            text="".join(part.text or "" for part in content.parts if not part.thought),
            thoughts=[
                part.model_copy(deep=True) for part in content.parts if part.thought
            ],
            signature=next(
                (
                    part.thoughtSignature
                    for part in reversed(content.parts)
                    if not part.thought and part.thoughtSignature
                ),
                None,
            ),
        )
        if turns and turns[-1].role == role:
            previous = turns[-1]
            if turn.text:
                previous.text = (
                    f"{previous.text}\n\n{turn.text}" if previous.text else turn.text
                )
            previous.thoughts.extend(turn.thoughts)
            previous.signature = turn.signature or previous.signature
        else:
            turns.append(turn)

    contents = [
        GeminiContent(role="user", parts=[GeminiPart(text="Init")]),
        GeminiContent(
            role="model",
            parts=[
                GeminiPart(
                    functionCall=GeminiFunctionCall(
                        name=WRAPPER_TOOL_NAME, args={"display_stream": ""}
                    ),
                    thoughtSignature=_SKIP_SIGNATURE,
                )
            ],
        ),
    ]
    if turns[0].role == "model":
        contents.append(_user_turn("(start)"))
    for turn in turns:
        if turn.role == "user":
            contents.append(_user_turn(turn.text))
        else:
            contents.append(
                GeminiContent(
                    role="model",
                    parts=[
                        *turn.thoughts,
                        GeminiPart(
                            functionCall=GeminiFunctionCall(
                                name=WRAPPER_TOOL_NAME,
                                args={"display_stream": turn.text},
                            ),
                            thoughtSignature=turn.signature or _SKIP_SIGNATURE,
                        ),
                    ],
                )
            )
    if contents[-1].role == "model":
        contents.append(_user_turn("continue"))

    system = (
        req.systemInstruction.model_copy(deep=True)
        if req.systemInstruction is not None
        else GeminiContent(parts=[])
    )
    system.parts.append(GeminiPart(text=_WRAPPER_INSTRUCTION))
    return req.model_copy(
        update={
            "contents": contents,
            "systemInstruction": system,
            "tools": [_wrapper_tool()],
            "toolConfig": _wrapper_tool_config(req, names),
        }
    )


def _user_turn(text: str) -> GeminiContent:
    return GeminiContent(
        role="user",
        parts=[
            GeminiPart(
                functionResponse=GeminiFunctionResponse(
                    name=WRAPPER_TOOL_NAME, response={"user_input": text}
                )
            )
        ],
    )


def _wrapper_call_text(call: dict[str, object]) -> str:
    if call.get("name") != WRAPPER_TOOL_NAME:
        raise TransportError("Text wrapper returned an unexpected tool call")
    args = call.get("args", call.get("arguments"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError as exc:
            raise TransportError(
                "Text wrapper returned incomplete tool arguments"
            ) from exc
    if not isinstance(args, dict) or not isinstance(args.get("display_stream"), str):
        raise TransportError("Text wrapper did not return a display_stream string")
    return args["display_stream"]


def _unwrap_candidate(candidate: Candidate) -> Candidate:
    if (
        candidate.images
        or candidate.reasoning_images
        or candidate.function_responses
        or candidate.code_output
    ):
        raise TransportError("Text wrapper returned an unsupported non-text response")
    private = [
        call
        for call in candidate.function_calls
        if call.get("name") == WRAPPER_TOOL_NAME
    ]
    external = [
        call
        for call in candidate.function_calls
        if call.get("name") != WRAPPER_TOOL_NAME
    ]
    wrapped = "".join(_wrapper_call_text(call) for call in private)
    return replace(
        candidate,
        text=wrapped if private else candidate.text,
        function_calls=external,
    )


def unwrap_text_output(output: ModelOutput) -> ModelOutput:
    """Restore text and retain genuine terminal, usage and signature metadata."""
    return replace(
        output,
        candidates=[_unwrap_candidate(candidate) for candidate in output.candidates],
    )


class TextWrapperStream:
    """Buffer only text content, emitting the original terminal events at flush.

    Decoded upstream tool argument strings are deltas. They are concatenated as
    received; substring-based deduplication would corrupt valid repeated text.
    A failed generation must not call finish(), or a partial answer could appear
    complete. The caller owns cancellation, retries and SSE keepalives.
    """

    def __init__(self) -> None:
        self._wrapped: list[str] = []
        self._fallback: list[str] = []
        self._terminal: dict[str, object | None] = {}
        self._size = 0
        self._saw_wrapper = False
        self._finished = False

    def _append(self, target: list[str], text: str) -> None:
        self._size += len(text)
        if self._size > _MAX_BUFFERED_CHARACTERS:
            raise TransportError("Text wrapper response exceeded its buffer limit")
        target.append(text)

    def feed(self, event_type: str, payload: object | None) -> list[StreamEvent]:
        if self._finished:
            raise RuntimeError("Text wrapper stream is already finished")
        if event_type == "body":
            if isinstance(payload, str):
                self._append(self._fallback, payload)
            return []
        if event_type == "tool_calls":
            if not isinstance(payload, list) or any(
                not isinstance(call, dict) for call in payload
            ):
                raise TransportError("Text wrapper returned invalid tool calls")
            external: list[dict[str, object]] = []
            for call in payload:
                if call.get("name") != WRAPPER_TOOL_NAME:
                    external.append(call)
                    continue
                self._saw_wrapper = True
                self._append(self._wrapped, _wrapper_call_text(call))
            return [("tool_calls", external)] if external else []
        if event_type in (
            "images",
            "reasoning_images",
            "function_responses",
            "code_output",
        ):
            raise TransportError(
                "Text wrapper returned an unsupported non-text response"
            )
        if event_type in ("finish_reason", "usage", "done"):
            self._terminal[event_type] = payload
            return []
        return [(event_type, payload)]

    def finish(self) -> list[StreamEvent]:
        if self._finished:
            return []
        self._finished = True
        text = "".join(self._wrapped if self._saw_wrapper else self._fallback)
        events: list[StreamEvent] = []
        if text:
            events.append(("body", text))
        events.extend(
            (kind, self._terminal[kind])
            for kind in ("finish_reason", "usage", "done")
            if kind in self._terminal
        )
        self._wrapped.clear()
        self._fallback.clear()
        self._terminal.clear()
        return events
