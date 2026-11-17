"""GitHub comparison and failure feedback using a hermetic transport."""

import httpx
import pytest

from aistudio_api.api.app import app
from aistudio_api.config import settings
from aistudio_api.infrastructure.gateway.update_checker import UpdateChecker

CURRENT = "a" * 40
LATEST = "b" * 40


@pytest.mark.parametrize(
    ("latest", "compare", "expected"),
    [
        (CURRENT, "identical", "up_to_date"),
        (LATEST, "ahead", "update_available"),
        (LATEST, "diverged", "diverged"),
        (LATEST, "behind", "local_ahead"),
    ],
)
async def test_comparison_and_cache(monkeypatch, latest, compare, expected):
    calls = []

    def respond(req):
        calls.append(req.url.path)
        if req.url.path.endswith("/aistudio-api"):
            data = {"default_branch": "main"}
        elif "/compare/" in req.url.path:
            data = {"status": compare, "ahead_by": 2}
        else:
            data = {
                "sha": latest,
                "commit": {
                    "message": "new commit",
                    "committer": {"date": "2026-11-16T00:00:00Z"},
                },
            }
        return httpx.Response(200, json=data)

    real_client = httpx.AsyncClient

    def factory(**kwargs):
        return real_client(transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    monkeypatch.setenv("AISTUDIO_UPSTREAM_REVISION", CURRENT)
    checker = UpdateChecker()
    result = await checker.check()
    assert result.status == expected
    assert result.current_revision == CURRENT
    assert result.latest_revision == latest
    count = len(calls)
    assert await checker.check() == result
    assert len(calls) == count


async def test_network_failure_has_actionable_response(monkeypatch):
    from unittest.mock import AsyncMock

    from aistudio_api.api.routes_updates import update_checker

    monkeypatch.setattr(settings, "web_password", None)
    monkeypatch.setattr(
        update_checker, "check", AsyncMock(side_effect=httpx.ConnectError("offline"))
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/system/check-update")
        assert response.status_code == 502
        assert "稍后重试" in response.json()["error"]["message"]
