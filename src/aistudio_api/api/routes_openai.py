"""OpenAI routes coexist with Gemini on the same application and port."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from aistudio_api.api.dependencies import get_client, get_runtime_state
from aistudio_api.api.schemas_openai import OpenAIChatRequest
from aistudio_api.api.state import RuntimeState
from aistudio_api.application.api_service_openai import handle_openai_chat
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.gateway.model_catalog import model_catalog
from aistudio_api.infrastructure.gateway.model_discovery import model_discovery

router = APIRouter()


def _model_object(item: dict[str, object]) -> dict[str, object]:
    return {
        "id": str(item.get("id") or item.get("name") or "").removeprefix("models/"),
        "object": "model",
        "created": 0,
        "owned_by": "google",
    }


async def _public_models(state: RuntimeState) -> list[dict[str, object]]:
    session = state.client.session if state.client else None
    discovered = await model_discovery.get_models(session=session)
    return model_catalog.public_models(discovered)


@router.get("/v1/models")
async def list_openai_models(state: RuntimeState = Depends(get_runtime_state)):
    return {
        "object": "list",
        "data": [_model_object(item) for item in await _public_models(state)],
    }


@router.get("/v1/models/{model_id:path}")
async def get_openai_model(
    model_id: str, state: RuntimeState = Depends(get_runtime_state)
):
    clean = model_id.removeprefix("models/")
    for item in await _public_models(state):
        model = _model_object(item)
        if model["id"] == clean:
            return model
    raise HTTPException(404, detail="Model not found")


@router.post("/v1/chat/completions")
async def create_chat_completion(
    req: OpenAIChatRequest, client: AIStudioClient = Depends(get_client)
):
    return await handle_openai_chat(req, client)
