from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from emissary_router.catalog import CATALOG
from emissary_router.config import (
    AppConfig,
    DEFAULT_CLASSIFICATION_URL,
    load_env_files,
    missing_runtime_env,
    user_config_path,
)


def _config(**overrides) -> AppConfig:
    raw = {
        "models": {
            "claude-sonnet-4.6": True,
            "claude-haiku-4.5": True,
            "gemini-3.1-flash-lite": True,
        },
        "default": "claude-sonnet-4.6",
        "confidence": 0.8,
    }
    raw.update(overrides)
    return AppConfig.model_validate(raw)


def test_catalog_order_is_cheap_to_expensive() -> None:
    assert list(CATALOG) == [
        "gemini-3.1-flash-lite",
        "claude-haiku-4.5",
        "claude-sonnet-4.6",
    ]


def test_enabled_models_follow_catalog_order() -> None:
    config = _config(
        models={
            "claude-sonnet-4.6": True,
            "claude-haiku-4.5": False,
            "gemini-3.1-flash-lite": True,
        }
    )

    assert config.enabled_models() == ["gemini-3.1-flash-lite", "claude-sonnet-4.6"]
    assert config.resolve_model("gemini-3.1-flash-lite").provider == "openrouter"


def test_model_value_true_uses_default_provider() -> None:
    config = _config()
    sonnet = config.resolve_model("claude-sonnet-4.6")
    assert sonnet.provider == "anthropic"
    assert sonnet.model_id == "claude-sonnet-4-6"


def test_model_value_provider_overrides_transport() -> None:
    config = _config(
        models={
            "claude-sonnet-4.6": True,
            "claude-haiku-4.5": "openrouter",
            "gemini-3.1-flash-lite": True,
        }
    )
    haiku = config.resolve_model("claude-haiku-4.5")
    assert haiku.provider == "openrouter"
    assert haiku.model_id == "anthropic/claude-haiku-4.5"


def test_provider_override_changes_required_env() -> None:
    # All Claude routed through OpenRouter -> only the OpenRouter key is required.
    config = _config(
        models={
            "claude-sonnet-4.6": "openrouter",
            "claude-haiku-4.5": "openrouter",
            "gemini-3.1-flash-lite": True,
        }
    )
    assert config.required_provider_env() == {"openrouter": "OPENROUTER_API_KEY"}


def test_unsupported_provider_for_model_fails_validation() -> None:
    # Gemini is OpenRouter-only in V1.
    with pytest.raises(ValidationError):
        _config(
            models={"gemini-3.1-flash-lite": "anthropic", "claude-sonnet-4.6": True}
        )


def test_router_config_defaults_classification_endpoint() -> None:
    config = _config()

    assert config.router.url == DEFAULT_CLASSIFICATION_URL
    assert config.router.router_model == "emissary-model-router-shared"


def test_default_user_path_uses_emissary_router_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", "/tmp/emissary-router-test")
    monkeypatch.delenv("EMISSARY_ROUTER_CONFIG", raising=False)

    assert user_config_path().as_posix() == "/tmp/emissary-router-test/config.yaml"


def test_server_auth_key_allows_non_loopback_binding() -> None:
    config = _config(server={"host": "0.0.0.0", "auth_key": "router-key"})

    assert config.server.auth_key == "router-key"


def test_unknown_model_fails_validation() -> None:
    with pytest.raises(ValidationError):
        _config(models={"claude-sonnet-4.6": True, "not-a-model": True})


def test_default_must_be_enabled() -> None:
    with pytest.raises(ValidationError):
        _config(models={"claude-sonnet-4.6": False}, default="claude-sonnet-4.6")


def test_confidence_must_be_probability() -> None:
    with pytest.raises(ValidationError):
        _config(confidence=1.5)


def test_env_loader_does_not_override_existing_values(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=from-file\nB='quoted value'\n")
    monkeypatch.setenv("A", "already-set")
    monkeypatch.delenv("B", raising=False)

    load_env_files([env_file])

    assert os.environ["A"] == "already-set"
    assert os.environ["B"] == "quoted value"


def test_missing_runtime_env_reports_router_and_enabled_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ["EMISSARY_ROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]:
        monkeypatch.delenv(key, raising=False)

    config = _config()

    assert missing_runtime_env(config) == [
        "EMISSARY_ROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ]
