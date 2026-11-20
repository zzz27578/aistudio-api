"""Persistent model availability, discovery visibility and thinking aliases."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
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
    text_wrapper: AliasPolicy = Field(default_factory=AliasPolicy)
    wrapper_thinking_aliases: dict[ThinkingLevel, AliasPolicy] = Field(
        default_factory=dict
    )


@dataclass(frozen=True)
class ResolvedModel:
    model: str
    thinking_level: ThinkingLevel | None = None
    text_wrapper: bool = False


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


TEXT_WRAPPER_SUFFIX = "-抗截断"


def wrapper_alias_id(base: str, level: ThinkingLevel | None = None) -> str:
    return (
        f"{alias_id(base, level) if level is not None else base}{TEXT_WRAPPER_SUFFIX}"
    )


def supports_text_wrapper(info: Mapping[str, object]) -> bool:
    """Only generation models with text output qualify for this text strategy."""
    if any(
        info.get(flag) is True
        for flag in ("is_image_model", "is_audio_model", "is_video_model")
    ) or info.get("category") in ("image", "audio", "video", "decision"):
        return False
    name = model_id(info).lower()
    if any(
        part in name for part in ("image", "imagen", "tts", "veo", "lyria", "decision")
    ):
        return False
    methods = info.get("supportedGenerationMethods")
    return not isinstance(methods, list) or "generateContent" in methods


def _upstream_names(metadata: Mapping[str, dict[str, object]]) -> dict[str, str]:
    names = {name: name for name in metadata}
    for name, info in metadata.items():
        aliases = info.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                if isinstance(alias, str):
                    names.setdefault(alias.removeprefix("models/"), name)
    return names


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
            upstream_names = _upstream_names(state.metadata)
            for level, alias in policy.thinking_aliases.items():
                if alias.enabled and level not in supported_levels(info):
                    raise HTTPException(
                        400, f"Upstream does not advertise {level} for this model"
                    )
                if alias.enabled and alias_id(base, level) in upstream_names:
                    raise HTTPException(
                        400, "Thinking alias conflicts with an upstream model"
                    )
            wrapper_enabled = policy.text_wrapper.enabled or any(
                alias.enabled for alias in policy.wrapper_thinking_aliases.values()
            )
            if wrapper_enabled and not supports_text_wrapper(info):
                raise HTTPException(400, "Text wrapping is unavailable for this model")
            wrappers = [
                (None, policy.text_wrapper),
                *policy.wrapper_thinking_aliases.items(),
            ]
            for level, alias in wrappers:
                if not alias.enabled:
                    continue
                if level is not None and level not in supported_levels(info):
                    raise HTTPException(
                        400, f"Upstream does not advertise {level} for this model"
                    )
                if wrapper_alias_id(base, level) in upstream_names:
                    raise HTTPException(
                        400, "Text wrapper alias conflicts with an upstream model"
                    )
            state.models[base] = policy
            self._save(state)

    def resolve(self, requested: str) -> tuple[str, ThinkingLevel | None]:
        """Resolve non-wrapped operations without silently discarding a strategy."""
        variant = self.resolve_variant(requested)
        if variant.text_wrapper:
            raise HTTPException(
                400, "Text wrapper aliases require a generation request"
            )
        return variant.model, variant.thinking_level

    def resolve_variant(self, requested: str) -> ResolvedModel:
        """Resolve strategy and gate before account selection, including aliases."""
        clean = requested.removeprefix("models/")
        state = self.load()
        upstream_names = _upstream_names(state.metadata)
        base = upstream_names.get(clean, clean)
        level: ThinkingLevel | None = None
        text_wrapper = False
        if clean not in upstream_names:
            if base.endswith(TEXT_WRAPPER_SUFFIX):
                base = base.removesuffix(TEXT_WRAPPER_SUFFIX)
                text_wrapper = True
            for candidate in ThinkingLevel:
                suffix = f"-think{candidate.value.lower()}"
                if base.endswith(suffix):
                    base = base.removesuffix(suffix)
                    level = candidate
                    break
            base = upstream_names.get(base, base)
        policy = state.models.get(base, ModelPolicy())
        if not policy.enabled:
            raise HTTPException(403, "Model is disabled")
        if text_wrapper:
            wrapper_policy = (
                policy.wrapper_thinking_aliases.get(level, AliasPolicy())
                if level is not None
                else policy.text_wrapper
            )
            if not wrapper_policy.enabled or base not in state.metadata:
                raise HTTPException(404, "Text wrapper alias is not enabled")
            if not supports_text_wrapper(state.metadata[base]):
                raise HTTPException(400, "Text wrapping is unavailable for this model")
        if level is not None:
            if not text_wrapper:
                alias = policy.thinking_aliases.get(level, AliasPolicy())
                if not alias.enabled or base not in state.metadata:
                    raise HTTPException(404, "Thinking alias is not enabled")
            if level not in supported_levels(state.metadata[base]):
                raise HTTPException(
                    400, "Upstream no longer advertises this thinking level"
                )
        return ResolvedModel(f"models/{base}", level, text_wrapper)

    def public_models(
        self, discovered: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        state = self.sync(discovered)
        upstream_names = _upstream_names(state.metadata)
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
                    if name in upstream_names:
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
            if supports_text_wrapper(info):
                wrappers = [
                    (None, policy.text_wrapper),
                    *(
                        (
                            level,
                            policy.wrapper_thinking_aliases.get(level, AliasPolicy()),
                        )
                        for level in supported_levels(info)
                    ),
                ]
                for level, alias in wrappers:
                    name = wrapper_alias_id(base, level)
                    if not alias.enabled or not alias.visible or name in upstream_names:
                        continue
                    result.append(
                        {
                            **info,
                            "id": name,
                            "name": f"models/{name}",
                            "aliases": [],
                            "displayName": f"{info.get('displayName', base)} · {level + ' · ' if level else ''}抗截断（实验文本包装）",
                            "description": (
                                f"{base}; experimental text tool wrapping; buffered response; "
                                "no automatic continuation; text only; no external tools or schema"
                                + (f"; fixed thinkingLevel={level}" if level else "")
                            ),
                            "supportedGenerationMethods": [
                                "generateContent",
                            ],
                            "textWrapper": True,
                            "baseModel": base,
                            "streamingMode": "buffered",
                        }
                    )
        return result


model_catalog = ModelCatalog()
