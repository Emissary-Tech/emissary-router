from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from emissary_router.catalog import (
    CATALOG,
    PROVIDER_ENV,
    ROUTER_API_KEY_ENV,
    ProviderName,
    cost_score,
)

DEFAULT_CLASSIFICATION_URL = "https://api.withemissary.com/v1/classification"


def emissary_router_home() -> Path:
    return Path(os.environ.get("EMISSARY_ROUTER_HOME", "~/.emissary-router")).expanduser()


def user_config_path() -> Path:
    configured = os.environ.get("EMISSARY_ROUTER_CONFIG")
    if configured:
        return Path(configured).expanduser()
    return emissary_router_home() / "config.json"


def user_env_path() -> Path:
    return emissary_router_home() / ".env"


def ensure_user_config() -> bool:
    """Write the default config to the user's home if it does not exist yet.

    Returns True if a new file was created.
    """
    from emissary_router.defaults import CONFIG_TEMPLATE

    path = user_config_path()
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE)
    return True


def _parse_env_lines(text: str) -> "list[tuple[str, str]]":
    """Parse `.env` text into (key, value) pairs. Shared by the reader and loader."""
    pairs: list[tuple[str, str]] = []
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
        if key:
            pairs.append((key, _strip_env_quotes(value.strip())))
    return pairs


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict without mutating os.environ."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    return dict(_parse_env_lines(text))


def _format_env_value(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("environment value must not contain a newline")
    if value == value.strip() and not any(ch in value for ch in " \t\"'"):
        return value
    if '"' in value:
        raise ValueError("environment value must not contain a double-quote")
    return f'"{value}"'


def write_env_file(path: Path, values: dict[str, str]) -> None:
    """Write key=value lines and restrict permissions (the file holds secrets)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{key}={_format_env_value(value)}\n" for key, value in values.items())
    path.write_text(body)
    try:
        path.chmod(0o600)
    except OSError:
        pass


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
    for key, value in _parse_env_lines(text):
        if key not in os.environ:
            os.environ[key] = value


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


class ModelEntry(BaseModel):
    """Per-model config. `model_id` is owned by the catalog and never set here."""
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    provider: ProviderName | None = None


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # `url` is an internal endpoint: it has a default and can be overridden, but is
    # intentionally not shown in the shipped config. `router_model` is user-facing.
    url: str = DEFAULT_CLASSIFICATION_URL
    router_model: str = "emissary-model-router-shared"
    timeout_seconds: float = 30
    max_retries: int = Field(default=5, ge=0)
    retry_backoff_seconds: float = Field(default=0.5, ge=0.0)


class TelemetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    db_path: str = "~/.emissary-router/events.sqlite3"
    retention_days: int | None = Field(default=30, ge=0)
    max_events: int | None = Field(default=50000, ge=0)


class DemoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Conference split-screen demo (default Sonnet vs routed). Makes real model calls,
    # so it sits behind the same auth as the dashboard; mounted only when enabled.
    enabled: bool = True
    streaming: bool = False  # default UI mode; the page can still toggle it live


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Per model: an object {enabled, provider}. Shorthands are accepted and coerced:
    # true/false -> {enabled: ...}; a provider name -> {provider: ...}.
    models: dict[str, ModelEntry]
    default: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    policy: Literal["deviate_if_confident", "cache_aware"] = "deviate_if_confident"
    router: RouterConfig = Field(default_factory=RouterConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_model_entries(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        coerced: dict[str, Any] = {}
        for name, entry in value.items():
            if isinstance(entry, bool):
                coerced[name] = {"enabled": entry}
            elif isinstance(entry, str):
                coerced[name] = {"provider": entry}
            else:
                coerced[name] = entry
        return coerced

    @model_validator(mode="after")
    def validate_catalog_references(self) -> "AppConfig":
        unknown_models = sorted(set(self.models) - set(CATALOG))
        if unknown_models:
            raise ValueError(f"unknown models: {unknown_models}")
        for name, entry in self.models.items():
            if entry.provider is not None:
                supported = CATALOG[name].providers
                if entry.provider not in supported:
                    raise ValueError(
                        f"provider '{entry.provider}' is not available for {name}; "
                        f"supported: {sorted(supported)}"
                    )
        if self.default not in CATALOG:
            raise ValueError(f"default model is not in the built-in catalog: {self.default}")
        if self.default not in self.enabled_models():
            raise ValueError("default model must be enabled")
        return self

    def enabled_models(self) -> list[str]:
        # Ordered cheap -> expensive by price, derived from the catalog (not dict order),
        # which is the order the routing policy scans.
        names = [name for name in CATALOG if name in self.models and self.models[name].enabled]
        return sorted(names, key=lambda name: cost_score(CATALOG[name]))

    def resolve_model(self, name: str) -> ResolvedModel:
        spec = CATALOG[name]
        entry = self.models.get(name)
        provider = entry.provider if entry and entry.provider else spec.default_provider
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
