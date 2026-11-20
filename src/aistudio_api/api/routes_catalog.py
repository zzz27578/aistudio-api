"""Authenticated model administration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from aistudio_api.api.dependencies import get_runtime_state
from aistudio_api.infrastructure.gateway.model_catalog import (
    ModelPolicy,
    ThinkingLevel,
    alias_id,
    model_catalog,
    model_id,
    supported_levels,
    supports_text_wrapper,
    wrapper_alias_id,
)
from aistudio_api.infrastructure.gateway.model_discovery import model_discovery

if TYPE_CHECKING:
    from aistudio_api.api.state import RuntimeState

router = APIRouter(prefix="/api/model-catalog")


@router.get("")
async def list_catalog(
    refresh: bool = False,
    runtime_state: RuntimeState = Depends(get_runtime_state),
) -> dict[str, object]:
    session = runtime_state.client.session if runtime_state.client else None
    discovered = await model_discovery.get_models(
        session=session, force_refresh=refresh
    )
    state = model_catalog.sync(discovered, refreshed=refresh)
    entries: list[dict[str, object]] = []
    for info in discovered:
        base = model_id(info)
        policy = state.models.get(base, ModelPolicy())
        levels = supported_levels(info)
        entries.append(
            {
                "id": base,
                "metadata": info,
                "policy": policy.model_dump(mode="json"),
                "supported_levels": levels,
                "aliases": {level: alias_id(base, level) for level in ThinkingLevel},
                "supports_text_wrapper": supports_text_wrapper(info),
                "wrapper_alias": wrapper_alias_id(base),
                "wrapper_thinking_aliases": {
                    level: wrapper_alias_id(base, level) for level in ThinkingLevel
                },
            }
        )
    return {
        "models": entries,
        "refreshed_at": state.refreshed_at,
        "source": model_discovery.source,
        "refresh_error": model_discovery.refresh_error,
    }


@router.put("/{base}")
async def update_model(base: str, policy: ModelPolicy) -> dict[str, bool]:
    model_catalog.update(base, policy)
    return {"ok": True}
