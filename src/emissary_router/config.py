from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from emissary_router.catalog import CATALOG, PROVIDER_ENV, ROUTER_API_KEY_ENV, ProviderName

DEFAULT_CLASSIFICATION_URL = "https://api.withemissary.com/v1/classification"


def emissary_router_home() -> Path:
    return Path(os.environ.get("EMISSARY_ROUTER_HOME", "~/.emissary-router")).expanduser()


def user_config_path() -> Path:
    configured = os.environ.get("EMISSARY_ROUTER_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return emissary_router_home() / "config.yaml"


def load_env_files(paths: list[Path] | None = None) -> None:
    """Load local env files without overriding already-exported variables."""
    env_paths = paths or [emissary_router_home() / ".env", Path.cwd() / ".env"]
    for path in env_paths:
        _load_env_file(path.expanduser())


def _load_env_file(path: Path) -> None:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _strip_env_quotes(value.strip())


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_raw_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    return raw


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8788
    auth_key: str | None = None

    @model_validator(mode="after")
    def enforce_loopback_without_key(self) -> "ServerConfig":
        if not self.auth_key and self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("server.auth_key is required when binding outside loopback")
        return self


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strip_dynamic_attribution: bool = True


class ProviderConfig(BaseModel):
    """Internal provider settings built from the catalog and environment."""
    model_config = ConfigDict(extra="forbid")

    type: ProviderName | None = None
    api_key: str | None = None
    base_url: str | None = None
    cache: CacheConfig = Field(default_factory=CacheConfig)


class ResolvedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: ProviderName
    model_id: str


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = DEFAULT_CLASSIFICATION_URL
    router_model: str = "emissary-model-router-shared"
    timeout_seconds: float = 30


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    log_path: str = "~/.emissary-router/events.jsonl"
    retention_days: int | None = Field(default=30, ge=0)
    max_events: int | None = Field(default=50000, ge=0)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Each value is: false (off), true (on, recommended provider), or a provider
    # name (on, that provider). Provider must be one the model supports.
    models: dict[str, bool | ProviderName]
    default: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    router: RouterConfig = Field(default_factory=RouterConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="after")
    def validate_catalog_references(self) -> "AppConfig":
        unknown_models = sorted(set(self.models) - set(CATALOG))
        if unknown_models:
            raise ValueError(f"unknown models: {unknown_models}")
        for name, value in self.models.items():
            if isinstance(value, str):
                supported = CATALOG[name].providers
                if value not in supported:
                    raise ValueError(
                        f"provider '{value}' is not available for {name}; "
                        f"supported: {sorted(supported)}"
                    )
        if self.default not in CATALOG:
            raise ValueError(f"default model is not in the built-in catalog: {self.default}")
        if self.default not in self.enabled_models():
            raise ValueError("default model must be enabled")
        return self

    def enabled_models(self) -> list[str]:
        # Truthy values (true or a provider name) are enabled; false is off.
        return [name for name in CATALOG if self.models.get(name, False)]

    def resolve_model(self, name: str) -> ResolvedModel:
        spec = CATALOG[name]
        value = self.models.get(name)
        provider = value if isinstance(value, str) else spec.default_provider
        return ResolvedModel(name=spec.name, provider=provider, model_id=spec.providers[provider])

    def required_provider_env(self) -> dict[ProviderName, str]:
        providers = {self.resolve_model(name).provider for name in self.enabled_models()}
        return {provider: PROVIDER_ENV[provider] for provider in sorted(providers)}


def missing_runtime_env(config: AppConfig) -> list[str]:
    required = [ROUTER_API_KEY_ENV, *config.required_provider_env().values()]
    return [name for name in required if not os.environ.get(name)]


def load_config(path: Path | None = None, strict_env: bool = True) -> AppConfig:
    del strict_env  # Kept for older call sites; config no longer interpolates secrets.
    load_env_files()
    resolved = path or user_config_path()
    return AppConfig.model_validate(_read_raw_mapping(resolved))
