"""Bounded, cached comparison against the fixed upstream GitHub repository."""

from __future__ import annotations

import asyncio
import os
import time

import httpx
from pydantic import BaseModel

REPOSITORY = "DeconstructedCube/aistudio-api"
REPOSITORY_URL = f"https://github.com/{REPOSITORY}"
API_URL = f"https://api.github.com/repos/{REPOSITORY}"


class UpdateStatus(BaseModel):
    current_revision: str
    custom_build: str
    latest_revision: str
    branch: str
    status: str
    checked_at: float
    commit_message: str
    committed_at: str
    commit_url: str
    compare_url: str | None = None
    ahead_by: int | None = None


class UpdateChecker:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._cached: UpdateStatus | None = None

    async def check(self) -> UpdateStatus:
        async with self._lock:
            current = os.getenv("AISTUDIO_UPSTREAM_REVISION", "unknown")
            build = os.getenv("AISTUDIO_CUSTOM_BUILD", "")
            if (
                self._cached
                and self._cached.current_revision == current
                and time.time() - self._cached.checked_at < 60
            ):
                return self._cached
            async with httpx.AsyncClient(
                timeout=10,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "aistudio-api-update-check",
                },
            ) as client:
                repo = await client.get(API_URL)
                repo.raise_for_status()
                branch = str(repo.json()["default_branch"])
                response = await client.get(f"{API_URL}/commits/{branch}")
                response.raise_for_status()
                data = response.json()
                latest = str(data["sha"])
                status = "unknown"
                ahead: int | None = None
                compare_url: str | None = None
                if current == latest:
                    status = "up_to_date"
                    ahead = 0
                elif len(current) == 40 and all(
                    c in "0123456789abcdef" for c in current
                ):
                    comparison = await client.get(
                        f"{API_URL}/compare/{current}...{latest}"
                    )
                    comparison.raise_for_status()
                    info = comparison.json()
                    comparison_status = str(info.get("status", "unknown"))
                    status = {
                        "ahead": "update_available",
                        "identical": "up_to_date",
                        "behind": "local_ahead",
                        "diverged": "diverged",
                    }.get(comparison_status, "unknown")
                    ahead = int(info.get("ahead_by", 0))
                    compare_url = f"{REPOSITORY_URL}/compare/{current}...{latest}"
                self._cached = UpdateStatus(
                    current_revision=current,
                    custom_build=build,
                    latest_revision=latest,
                    branch=branch,
                    status=status,
                    checked_at=time.time(),
                    commit_message=str(data["commit"]["message"]),
                    committed_at=str(data["commit"]["committer"]["date"]),
                    commit_url=f"{REPOSITORY_URL}/commit/{latest}",
                    compare_url=compare_url,
                    ahead_by=ahead,
                )
                return self._cached


update_checker = UpdateChecker()
