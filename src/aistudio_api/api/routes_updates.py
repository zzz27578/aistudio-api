"""Console update checks (never mutates the deployment)."""

import httpx
from fastapi import APIRouter, HTTPException

from aistudio_api.infrastructure.gateway.update_checker import (
    UpdateStatus,
    update_checker,
)

router = APIRouter()


@router.get("/api/system/check-update", response_model=UpdateStatus)
async def check_update() -> UpdateStatus:
    try:
        return await update_checker.check()
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(
            502, "无法检查 GitHub 更新，请稍后重试（网络异常或请求限额）"
        ) from exc
