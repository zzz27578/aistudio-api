"""SSE keepalives that preserve pending reads and cancel them on disconnect."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from anyio import CancelScope


async def with_sse_keepalive(
    content: AsyncGenerator[str | bytes, None], *, interval: float = 10.0
) -> AsyncGenerator[str | bytes, None]:
    pending: asyncio.Task[str | bytes] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(content))
            try:
                chunk = await asyncio.wait_for(
                    asyncio.shield(pending), timeout=interval
                )
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            except StopAsyncIteration:
                return
            pending = None
            yield chunk
    finally:
        with CancelScope(shield=True):
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await content.aclose()
