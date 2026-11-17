"""Persistent model availability, discovery visibility and thinking aliases."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from enum import StrEnum

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from aistudio_api.config import resolve_data_dir
from aistudio_api.infrastructure.utils.common import atomic_write_json


class ThinkingLevel(StrEnum):
    MINIMAL = "MINIMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


WIRE_LEVELS = {
    1: ThinkingLevel.LOW,
    2: ThinkingLevel.MEDIUM,
    3: ThinkingLevel.HIGH,
    4: ThinkingLevel.MINIMAL,
}


class AliasPolicy(BaseModel):
    enabled: bool = False
    visible: bool = True


class ModelPolicy(BaseModel):
    enabled: bool = True
    visible: bool = True
    thinking_aliases: dict[ThinkingLevel, AliasPolicy] = Field(default_factory=dict)


class CatalogState(BaseModel):
    models: dict[str, ModelPolicy] = Field(default_factory=dict)
    metadata: dict[str, dict[str, object]] = Field(default_factory=dict)
    refreshed_at: float | None = None


def model_id(info: Mapping[str, object]) -> str:
    return str(info.get("id") or info.get("name") or "").removeprefix("models/")


def supported_levels(info: Mapping[str, object]) -> list[ThinkingLevel]:
    raw = info.get("thinkingLevels")
    values = raw if isinstance(raw, list) else []
    advertised = {
        WIRE_LEVELS[v]
        for v in values
        if isinstance(v, int) and not isinstance(v, bool) and v in WIRE_LEVELS
    }
    return [level for level in ThinkingLevel if level in advertised]


def alias_id(base: str, level: ThinkingLevel) -> str:
    return f"{base}-think{level.value.lower()}"


class ModelCatalog:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    def load(self) -> CatalogState:
        path = resolve_data_dir() / "model_catalog.json"
        if not path.exists():
            return CatalogState()
        try:
            return CatalogState.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise HTTPException(503, "Model settings could not be read") from exc

    def _save(self, state: CatalogState) -> None:
        atomic_write_json(
            resolve_data_dir() / "model_catalog.json", state.model_dump(mode="json")
        )

    def sync(
        self, discovered: list[dict[str, object]], *, refreshed: bool = False
    ) -> CatalogState:
        with self._lock:
            state = self.load()
            metadata = {
                model_id(info): dict(info) for info in discovered if model_id(info)
            }
            if metadata != state.metadata or refreshed:
                state.metadata = metadata
                state.refreshed_at = time.time()
                self._save(state)
            return state

    def update(self, base: str, policy: ModelPolicy) -> None:
        with self._lock:
            state = self.load()
            info = state.metadata.get(base)
            if info is None:
                raise HTTPException(
                    404, "Model not found; refresh the model list first"
                )
            for level, alias in policy.thinking_aliases.items():
                if alias.enabled and level not in supported_levels(info):
                    raise HTTPException(
                        400, f"Upstream does not advertise {level} for this model"
                    )
                if alias.enabled and alias_id(base, level) in state.metadata:
                    raise HTTPException(
                        400, "Thinking alias conflicts with an upstream model"
                    )
            state.models[base] = policy
            self._save(state)

    def resolve(self, requested: str) -> tuple[str, ThinkingLevel | None]:
        """Resolve and gate before account selection, including upstream aliases."""
        clean = requested.removeprefix("models/")
        state = self.load()
        base = clean
        level: ThinkingLevel | None = None
        if clean not in state.metadata:
            for candidate in ThinkingLevel:
                suffix = f"-think{candidate.value.lower()}"
                if clean.endswith(suffix):
                    base = clean.removesuffix(suffix)
                    level = candidate
                    break
            if level is None:
                for name, info in state.metadata.items():
                    aliases = info.get("aliases")
                    if isinstance(aliases, list) and clean in aliases:
                        base = name
                        break
        policy = state.models.get(base, ModelPolicy())
        if not policy.enabled:
            raise HTTPException(403, "Model is disabled")
        if level is not None:
            alias = policy.thinking_aliases.get(level, AliasPolicy())
            if not alias.enabled or base not in state.metadata:
                raise HTTPException(404, "Thinking alias is not enabled")
            if level not in supported_levels(state.metadata[base]):
                raise HTTPException(
                    400, "Upstream no longer advertises this thinking level"
                )
        return f"models/{base}", level

    def public_models(
        self, discovered: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        state = self.sync(discovered)
        result: list[dict[str, object]] = []
        for info in discovered:
            base = model_id(info)
            policy = state.models.get(base, ModelPolicy())
            if not policy.enabled:
                continue
            if policy.visible:
                result.append(dict(info))
            for level in supported_levels(info):
                alias = policy.thinking_aliases.get(level, AliasPolicy())
                if alias.enabled and alias.visible:
                    name = alias_id(base, level)
                    if name in state.metadata:
                        continue
                    result.append(
                        {
                            **info,
                            "id": name,
                            "name": f"models/{name}",
                            "displayName": f"{info.get('displayName', base)} · {level}",
                            "description": f"{base}; fixed thinkingLevel={level}",
                        }
                    )
        return result


model_catalog = ModelCatalog()
