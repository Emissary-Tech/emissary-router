from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


ENV_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
UNRESOLVED_ENV_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")
ProviderType = Literal["anthropic", "openrouter", "google"]
PROVIDER_TYPES_BY_NAME: dict[str, ProviderType] = {
    "anthropic": "anthropic",
    "openrouter": "openrouter",
    "google": "google",
}


def infer_provider_type(name: str, config: "ProviderConfig") -> ProviderType:
    if config.type:
        return config.type
    if name in PROVIDER_TYPES_BY_NAME:
        return PROVIDER_TYPES_BY_NAME[name]
    raise ValueError(
        f"provider '{name}' needs an explicit type; expected one of "
        f"{sorted(set(PROVIDER_TYPES_BY_NAME.values()))}"
    )


def user_config_path() -> Path:
    return Path(os.environ.get("ROUTER_CONFIG", "~/.config/router/config.yaml")).expanduser()


def user_pricing_path() -> Path:
    return Path(os.environ.get("ROUTER_PRICING", "~/.config/router/pricing.yaml")).expanduser()


def _interpolate_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(0))

        return ENV_RE.sub(repl, value)
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    return value


def _find_unresolved_env(value: Any, path: str = "") -> list[str]:
    if isinstance(value, str):
        if UNRESOLVED_ENV_RE.search(value):
            return [path or "<root>"]
        return []
    if isinstance(value, list):
        unresolved: list[str] = []
        for index, item in enumerate(value):
            unresolved.extend(_find_unresolved_env(item, f"{path}[{index}]"))
        return unresolved
    if isinstance(value, dict):
        unresolved = []
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            unresolved.extend(_find_unresolved_env(item, child))
        return unresolved
    return []


def unresolved_env_paths(path: Path) -> list[str]:
    return _find_unresolved_env(_interpolate_env(_read_raw_mapping(path)))


def _read_raw_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    return raw


def _read_mapping(path: Path, strict_env: bool = True) -> dict[str, Any]:
    raw = _read_raw_mapping(path)
    interpolated = _interpolate_env(raw)
    unresolved = _find_unresolved_env(interpolated)
    if unresolved and strict_env:
        raise ValueError(
            f"{path} contains unresolved environment variables at: {', '.join(unresolved)}"
        )
    return interpolated


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8788
    auth_key: str | None = None

    @model_validator(mode="after")
    def enforce_loopback_without_key(self) -> "ServerConfig":
        if not self.auth_key and self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("server.auth_key is required when binding outside loopback")
        return self


class CacheConfig(BaseModel):
    strip_dynamic_attribution: bool = True


class ProviderConfig(BaseModel):
    type: ProviderType | None = None
    api_key: str | None = None
    base_url: str | None = None
    cache: CacheConfig = Field(default_factory=CacheConfig)


class ModelProviderTargetConfig(BaseModel):
    model_id: str


class ModelConfig(BaseModel):
    provider: str
    model_id: str | None = None
    providers: dict[str, ModelProviderTargetConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selected_provider_target(self) -> "ModelConfig":
        if self.model_id is None and self.provider not in self.providers:
            raise ValueError(
                "model_id is required unless models.<name>.providers contains the selected provider"
            )
        return self


class ResolvedModel(BaseModel):
    name: str
    provider: str
    model_id: str


class PolicyConfig(BaseModel):
    name: Literal["cheap_first", "argmax"] = "cheap_first"
    tau: float = 0.85
    candidates: list[str] = Field(default_factory=list)


class RouterConfig(BaseModel):
    default: str
    enabled: list[str]
    policy: PolicyConfig = Field(default_factory=PolicyConfig)


class ClassifierConfig(BaseModel):
    url: str
    api_key: str | None = None
    model: str
    timeout_seconds: float = 30


class TelemetryConfig(BaseModel):
    enabled: bool = True
    log_path: str = "~/.local/state/router/events.jsonl"
    include_classifier_input: bool = False


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    router: RouterConfig
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelConfig]
    classifier: ClassifierConfig
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="after")
    def validate_references(self) -> "AppConfig":
        for provider_name, provider in self.providers.items():
            infer_provider_type(provider_name, provider)

        missing_providers = {
            model.provider
            for model in self.models.values()
            if model.provider not in self.providers
        }
        if missing_providers:
            raise ValueError(f"unknown providers referenced: {sorted(missing_providers)}")

        all_models = set(self.models)
        referenced = set(self.router.enabled)
        referenced.add(self.router.default)
        referenced.update(self.router.policy.candidates)
        missing_models = sorted(referenced - all_models)
        if missing_models:
            raise ValueError(f"unknown models referenced: {missing_models}")
        return self

    def resolve_model(self, name: str) -> ResolvedModel:
        model = self.models[name]
        target = model.providers.get(model.provider)
        model_id = target.model_id if target else model.model_id
        if model_id is None:
            raise ValueError(f"model {name} has no model_id for provider {model.provider}")
        return ResolvedModel(
            name=name,
            provider=model.provider,
            model_id=model_id,
        )


class TokenPricing(BaseModel):
    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float | None = None


class PricingConfig(BaseModel):
    pricing: dict[str, TokenPricing]


def load_config(path: Path | None = None, strict_env: bool = True) -> AppConfig:
    resolved = path or user_config_path()
    return AppConfig.model_validate(_read_mapping(resolved, strict_env=strict_env))


def load_pricing(path: Path | None = None, strict_env: bool = True) -> PricingConfig:
    resolved = path or user_pricing_path()
    return PricingConfig.model_validate(_read_mapping(resolved, strict_env=strict_env))
