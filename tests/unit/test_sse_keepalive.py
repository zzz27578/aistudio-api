"""Keepalive timeout does not cancel a generation, but closing the client does."""

import asyncio

import pytest

from aistudio_api.application.stream_lifecycle import with_sse_keepalive


@pytest.mark.asyncio
async def test_keepalive_keeps_read_pending_and_closes_it_on_disconnect():
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        started.set()
        try:
            await release.wait()
            yield "data: answer\n\n"
            await release.wait()
        finally:
            closed.set()

    iterator = with_sse_keepalive(source(), interval=0.01)
    assert await anext(iterator) == ": keepalive\n\n"
    assert started.is_set() and not closed.is_set()
    release.set()
    assert await anext(iterator) == "data: answer\n\n"
    await iterator.aclose()
    assert closed.is_set()


@pytest.mark.asyncio
async def test_closing_while_pending_cancels_source():
    pending = asyncio.Event()
    closed = asyncio.Event()

    async def source():
        try:
            await pending.wait()
            yield "never"
        finally:
            closed.set()

    iterator = with_sse_keepalive(source(), interval=0.01)
    assert await anext(iterator) == ": keepalive\n\n"
    await iterator.aclose()
    assert closed.is_set()
