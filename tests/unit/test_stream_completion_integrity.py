"""Offline stream contracts for complete envelopes and prematurely closed bodies."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest

from aistudio_api.domain.errors import TransportError
from aistudio_api.infrastructure.gateway.client import AIStudioClient, PreparedRequest
from aistudio_api.infrastructure.gateway.streaming import StreamingGateway


def response_wire(*, finish: int | None = None) -> str:
    candidate: list[object] = [[[[None, "received text"]], "model"], finish]
    return ")]}'\n" + json.dumps([[[[candidate], None, [3, 2, 5]]]])


async def collect_events(raw: str) -> list[tuple[str, object | None]]:
    class MockSession:
        async def send_streaming_request(
            self, **_kwargs: object
        ) -> AsyncGenerator[tuple[str, object], None]:
            yield "status", 200
            yield "chunk", raw.encode("utf-8")

    gateway = StreamingGateway(session=MockSession())  # type: ignore[arg-type]
    request = PreparedRequest(url="https://example.test", headers={}, body="{}")
    return [event async for event in gateway.stream_chat(request=request, model="test")]


@pytest.mark.asyncio
async def test_complete_envelope_without_finish_remains_valid():
    events = await collect_events(response_wire())
    assert ("body", "received text") in events
    assert ("finish_reason", 1) in events
    assert events[-1] == ("done", None)


@pytest.mark.asyncio
async def test_partial_envelope_is_not_reported_as_stop():
    with pytest.raises(TransportError, match="complete envelope"):
        await collect_events(response_wire()[:-2])


@pytest.mark.asyncio
async def test_explicit_terminal_survives_missing_outer_closers():
    events = await collect_events(response_wire(finish=2)[:-2])
    assert ("finish_reason", 2) in events


@pytest.mark.asyncio
async def test_unparseable_success_body_is_an_error():
    with pytest.raises(TransportError, match="complete envelope"):
        await collect_events("unexpected body")


@pytest.mark.asyncio
async def test_repaired_finish_number_cannot_become_a_false_stop():
    raw = response_wire(finish=12).split(", 12")[0] + ", 1"
    with pytest.raises(TransportError, match="complete envelope"):
        await collect_events(raw)


@pytest.mark.asyncio
async def test_complete_tool_does_not_mask_a_later_truncated_tool():
    first = [[[[None, None, None, ["lookup", {"query": "ok"}, "call1"]]], "model"]]
    second = [
        [[[None, None, None, ["lookup", {"query": "unfinished"}, "call2"]]], "model"]
    ]
    raw = ")]}'\n" + json.dumps([[[[first], None], [[second], None]]])
    raw = raw.split("unfinished")[0] + "unfin"
    with pytest.raises(TransportError, match="complete envelope"):
        await collect_events(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("truncated", [False, True])
async def test_nonstream_requires_a_complete_generation_envelope(truncated: bool):
    raw = response_wire()
    if truncated:
        raw = raw[:-2]

    class MockSession:
        async def send_request(self, **_kwargs: object) -> tuple[int, bytes]:
            return 200, raw.encode("utf-8")

    client = AIStudioClient()
    client._session = MockSession()  # type: ignore[assignment]
    request = PreparedRequest(url="https://example.test", headers={}, body="{}")
    if truncated:
        with pytest.raises(TransportError, match="incomplete generation JSON"):
            await client.generate_content(model="test", prepared=request)
    else:
        output = await client.generate_content(model="test", prepared=request)
        assert output.text == "received text"
        assert output.candidates[0].finish_reason == 1
