"""Model policy contracts, thinking wire values and routing (no upstream calls)."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

from aistudio_api.api.app import app
from aistudio_api.api.schemas import (
    GeminiCountTokensRequest,
    GeminiGenerateContentRequest,
)
from aistudio_api.application import api_service_gemini as service
from aistudio_api.application.chat_service import normalize_gemini_request
from aistudio_api.config import settings
from aistudio_api.domain.models import Candidate, ModelOutput
from aistudio_api.infrastructure.gateway.model_catalog import (
    AliasPolicy,
    ModelCatalog,
    ModelPolicy,
    ThinkingLevel,
    alias_id,
    model_catalog,
    wrapper_alias_id,
)
from aistudio_api.infrastructure.gateway.model_discovery import model_discovery

BASE = "gemini-4-hydrogen"
MODELS: list[dict[str, object]] = [
    {
        "id": BASE,
        "displayName": "Hydrogen",
        "thinkingLevels": [1, 2, 3, 4],
        "aliases": ["hydrogen-latest"],
        "supportedGenerationMethods": ["generateContent"],
    }
]


@pytest.fixture(autouse=True)
def catalog_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AISTUDIO_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(model_discovery, "get_models", AsyncMock(return_value=MODELS))
    monkeypatch.setattr(settings, "web_password", None)
    monkeypatch.setattr(settings, "api_keys", frozenset())
    model_catalog.sync(MODELS)


def enabled_aliases(*, visible=True):
    return ModelPolicy(
        visible=visible,
        thinking_aliases={level: AliasPolicy(enabled=True) for level in ThinkingLevel},
    )


def request():
    return GeminiGenerateContentRequest.model_validate(
        {
            "contents": [{"parts": [{"text": "hello"}]}],
            "generationConfig": {
                "temperature": 0.4,
                "thinkingConfig": {
                    "thinkingLevel": "HIGH",
                    "thinkingBudget": 100,
                    "includeThoughts": True,
                },
            },
            "reasoningEffort": "HIGH",
        }
    )


def test_visibility_is_independent_and_settings_survive_refresh():
    model_catalog.update(BASE, enabled_aliases(visible=False))
    model_catalog.sync([{**MODELS[0], "description": "new metadata"}])
    restarted = ModelCatalog()
    assert not restarted.load().models[BASE].visible
    assert restarted.resolve(BASE) == (f"models/{BASE}", None)
    public = restarted.public_models(MODELS)
    assert [m["id"] for m in public] == [
        alias_id(BASE, level) for level in ThinkingLevel
    ]
    model_catalog.update(BASE, ModelPolicy(enabled=False))
    assert model_catalog.public_models(MODELS) == []
    for name in (BASE, "hydrogen-latest", alias_id(BASE, ThinkingLevel.HIGH)):
        with pytest.raises(HTTPException) as error:
            restarted.resolve(name)
        assert error.value.status_code == 403


@pytest.mark.parametrize(
    ("level", "code"), [("LOW", 1), ("MEDIUM", 2), ("HIGH", 3), ("MINIMAL", 4)]
)
def test_alias_overrides_client_budget_and_effort_without_mutating_request(level, code):
    model_catalog.update(BASE, enabled_aliases())
    original = request()
    path, resolved = service.resolve_catalog_request(
        alias_id(BASE, ThinkingLevel(level)), original
    )
    normalized = normalize_gemini_request(resolved, path)
    assert normalized.generation_config_overrides is not None
    assert resolved.generationConfig is not None
    assert original.generationConfig is not None
    assert isinstance(original.generationConfig.thinkingConfig, dict)
    assert normalized.model == f"models/{BASE}"
    assert normalized.generation_config_overrides["thinking_config"] == [
        1,
        None,
        None,
        code,
    ]
    assert resolved.generationConfig.temperature == 0.4
    assert original.generationConfig.thinkingConfig["thinkingBudget"] == 100
    assert resolved.reasoningEffort is None


def test_unknown_or_unsupported_aliases_rejected():
    with pytest.raises(HTTPException) as error:
        model_catalog.resolve(alias_id(BASE, ThinkingLevel.LOW))
    assert error.value.status_code == 404
    model_catalog.sync([{**MODELS[0], "thinkingLevels": [3]}])
    with pytest.raises(HTTPException) as error:
        model_catalog.update(BASE, enabled_aliases())
    assert error.value.status_code == 400


async def test_public_list_detail_and_admin_auth(monkeypatch):
    model_catalog.update(BASE, enabled_aliases(visible=False))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/v1beta/models")
        assert response.status_code == 200
        assert len(response.json()["models"]) == 4
        response = await client.get(
            f"/v1beta/models/{alias_id(BASE, ThinkingLevel.LOW)}"
        )
        assert response.status_code == 200
        assert response.json()["name"].endswith("-thinklow")
        assert (await client.get(f"/v1beta/models/{BASE}")).status_code == 404
        monkeypatch.setattr(settings, "web_password", "admin-test")
        monkeypatch.setattr(settings, "api_keys", frozenset({"api-test"}))
        for path in ("/api/model-catalog", "/api/system/check-update"):
            assert (await client.get(path)).status_code == 401
            assert (
                await client.get(path, headers={"Authorization": "Bearer api-test"})
            ).status_code == 401
        assert (
            await client.get(
                "/api/model-catalog", headers={"Authorization": "Bearer admin-test"}
            )
        ).status_code == 200
        assert (
            await client.put(f"/api/model-catalog/{BASE}", json={"enabled": False})
        ).status_code == 401


@pytest.mark.parametrize("stream", [False, True])
async def test_alias_resolved_before_account_selection_and_signing(monkeypatch, stream):
    model_catalog.update(BASE, enabled_aliases())
    ensure = AsyncMock(return_value=None)
    monkeypatch.setattr(service, "ensure_active_account", ensure)
    client = MagicMock()
    client.prepare_request = AsyncMock(return_value=MagicMock(ctx=None))
    client.generate_content = AsyncMock(
        return_value=ModelOutput(candidates=[Candidate(text="answer", finish_reason=1)])
    )

    async def events(**kwargs):
        yield "body", "answer"
        yield "finish_reason", 1
        yield "done", None

    client.stream_generate_content = events
    name = alias_id(BASE, ThinkingLevel.LOW)
    if stream:
        assert [
            item async for item in service.stream_generation(name, request(), client)
        ]
    else:
        await service.run_generation(name, request(), client)
    ensure.assert_awaited_once_with(0, model=f"models/{BASE}")
    prepared = client.prepare_request.call_args.kwargs
    assert prepared["model"] == f"models/{BASE}"
    assert prepared["generation_config_overrides"]["thinking_config"] == [
        1,
        None,
        None,
        1,
    ]


async def test_count_tokens_routes_to_base_and_disabled_models_never_select_accounts(
    monkeypatch,
):
    model_catalog.update(BASE, enabled_aliases())
    ensure = AsyncMock()
    monkeypatch.setattr(service, "ensure_active_account", ensure)
    client = MagicMock()
    client.count_tokens = AsyncMock(return_value={"totalTokens": 1})
    req = GeminiCountTokensRequest(generateContentRequest=request())
    await service.handle_gemini_count_tokens(
        alias_id(BASE, ThinkingLevel.MEDIUM), req, client
    )
    ensure.assert_awaited_once_with(0, model=f"models/{BASE}")
    assert client.count_tokens.call_args.kwargs["model"] == f"models/{BASE}"
    ensure.reset_mock()
    model_catalog.update(BASE, ModelPolicy(enabled=False))
    for stream in (True, False):
        with pytest.raises(HTTPException):
            await service.handle_gemini_generate_content(
                BASE, request(), client, stream=stream
            )
    with pytest.raises(HTTPException):
        await service.handle_gemini_count_tokens(BASE, req, client)
    ensure.assert_not_called()


def test_corrupt_settings_fail_closed(tmp_path):
    path = tmp_path / "model_catalog.json"
    path.write_text("broken", encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        model_catalog.resolve(BASE)
    assert error.value.status_code == 503
    assert path.read_text(encoding="utf-8") == "broken"


def test_wrapper_aliases_are_independent_persistent_and_policy_gated():
    model_catalog.update(
        BASE,
        ModelPolicy(
            visible=False,
            text_wrapper=AliasPolicy(enabled=True),
            wrapper_thinking_aliases={
                ThinkingLevel.HIGH: AliasPolicy(enabled=True, visible=False),
                ThinkingLevel.LOW: AliasPolicy(enabled=True),
            },
        ),
    )
    restarted = ModelCatalog()
    basic = restarted.resolve_variant(wrapper_alias_id(BASE))
    assert basic.model == f"models/{BASE}"
    assert basic.thinking_level is None
    assert basic.text_wrapper
    combined = restarted.resolve_variant(wrapper_alias_id(BASE, ThinkingLevel.HIGH))
    assert combined.thinking_level == ThinkingLevel.HIGH
    assert combined.text_wrapper
    assert [item["id"] for item in restarted.public_models(MODELS)] == [
        wrapper_alias_id(BASE),
        wrapper_alias_id(BASE, ThinkingLevel.LOW),
    ]
    with pytest.raises(HTTPException) as error:
        restarted.resolve(wrapper_alias_id(BASE))
    assert error.value.status_code == 400
    with pytest.raises(HTTPException) as error:
        restarted.resolve(alias_id(BASE, ThinkingLevel.HIGH))
    assert error.value.status_code == 404
    model_catalog.update(BASE, ModelPolicy(enabled=False))
    with pytest.raises(HTTPException) as error:
        restarted.resolve_variant(wrapper_alias_id(BASE, ThinkingLevel.HIGH))
    assert error.value.status_code == 403


def test_wrapper_unsupported_models_and_disappearing_levels_fail_closed():
    model_catalog.update(
        BASE,
        ModelPolicy(
            wrapper_thinking_aliases={ThinkingLevel.HIGH: AliasPolicy(enabled=True)}
        ),
    )
    model_catalog.sync([{**MODELS[0], "thinkingLevels": [1]}])
    with pytest.raises(HTTPException) as error:
        model_catalog.resolve_variant(wrapper_alias_id(BASE, ThinkingLevel.HIGH))
    assert error.value.status_code == 400
    model_catalog.sync([{**MODELS[0], "is_image_model": True}])
    with pytest.raises(HTTPException) as error:
        model_catalog.update(BASE, ModelPolicy(text_wrapper=AliasPolicy(enabled=True)))
    assert error.value.status_code == 400
    assert all(
        not item.get("textWrapper")
        for item in model_catalog.public_models([{**MODELS[0], "is_image_model": True}])
    )


def test_wrapper_alias_conflicts_do_not_shadow_real_upstream_names():
    metadata = [{**MODELS[0], "aliases": [wrapper_alias_id(BASE)]}]
    model_catalog.sync(metadata)
    with pytest.raises(HTTPException) as error:
        model_catalog.update(BASE, ModelPolicy(text_wrapper=AliasPolicy(enabled=True)))
    assert error.value.status_code == 400
    variant = model_catalog.resolve_variant(wrapper_alias_id(BASE))
    assert variant.model == f"models/{BASE}"
    assert not variant.text_wrapper


async def test_wrapper_catalog_contract_and_hidden_metadata(monkeypatch):
    model_catalog.update(
        BASE, ModelPolicy(text_wrapper=AliasPolicy(enabled=True, visible=False))
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/model-catalog")
        assert response.status_code == 200
        entry = response.json()["models"][0]
        assert entry["supports_text_wrapper"] is True
        assert entry["wrapper_alias"] == wrapper_alias_id(BASE)
        assert entry["wrapper_thinking_aliases"]["HIGH"] == wrapper_alias_id(
            BASE, ThinkingLevel.HIGH
        )
        response = await client.get(f"/v1beta/models/{wrapper_alias_id(BASE)}")
        assert response.status_code == 404
