from __future__ import annotations

import pytest
from pydantic import ValidationError

from emissary_router.config import (
    AppConfig,
    DEFAULT_CLASSIFICATION_URL,
    ProviderConfig,
    infer_provider_type,
    user_config_path,
    user_pricing_path,
)


def test_provider_type_is_inferred_from_builtin_name() -> None:
    assert infer_provider_type("anthropic", ProviderConfig(api_key="k")) == "anthropic"
    assert infer_provider_type("google", ProviderConfig(api_key="k")) == "google"
    assert infer_provider_type("openrouter", ProviderConfig(api_key="k")) == "openrouter"


def test_custom_provider_name_requires_explicit_type() -> None:
    with pytest.raises(ValueError):
        infer_provider_type("company_proxy", ProviderConfig(api_key="k"))


def test_model_first_config_accepts_api_key_only_provider() -> None:
    config = AppConfig.model_validate(
        {
            "router": {
                "url": "http://127.0.0.1:9999",
                "router_model": "router",
                "default": "claude-sonnet-4.6",
                "enabled": ["claude-sonnet-4.6"],
                "policy": {"name": "argmax"},
            },
            "providers": {"anthropic": {"api_key": "k"}},
            "models": {
                "claude-sonnet-4.6": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4-6",
                }
            },
        }
    )

    assert config.providers["anthropic"].api_key == "k"


def test_router_config_defaults_classification_endpoint() -> None:
    config = AppConfig.model_validate(
        {
            "router": {
                "api_key": "router-key",
                "default": "claude-sonnet-4.6",
                "enabled": ["claude-sonnet-4.6"],
                "policy": {"name": "argmax"},
            },
            "providers": {"anthropic": {"api_key": "k"}},
            "models": {
                "claude-sonnet-4.6": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4-6",
                }
            },
        }
    )

    assert config.router.url == DEFAULT_CLASSIFICATION_URL
    assert config.router.router_model == "emissary-model-router-shared"


def test_default_user_paths_use_emissary_router_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", "/tmp/emissary-router-test")
    monkeypatch.delenv("EMISSARY_ROUTER_CONFIG", raising=False)
    monkeypatch.delenv("EMISSARY_ROUTER_PRICING", raising=False)

    assert user_config_path().as_posix() == "/tmp/emissary-router-test/config.yaml"
    assert user_pricing_path().as_posix() == "/tmp/emissary-router-test/pricing.yaml"


def test_server_auth_key_allows_non_loopback_binding() -> None:
    config = AppConfig.model_validate(
        {
            "server": {"host": "0.0.0.0", "auth_key": "router-key"},
            "router": {
                "url": "http://127.0.0.1:9999",
                "router_model": "router",
                "default": "claude-sonnet-4.6",
                "enabled": ["claude-sonnet-4.6"],
                "policy": {"name": "argmax"},
            },
            "providers": {"anthropic": {"api_key": "k"}},
            "models": {
                "claude-sonnet-4.6": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4-6",
                }
            },
        }
    )

    assert config.server.auth_key == "router-key"


def test_unknown_provider_without_type_fails_config_validation() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "router": {
                    "url": "http://127.0.0.1:9999",
                    "router_model": "router",
                    "default": "model",
                    "enabled": ["model"],
                    "policy": {"name": "argmax"},
                },
                "providers": {"company_proxy": {"api_key": "k"}},
                "models": {
                    "model": {
                        "provider": "company_proxy",
                        "model_id": "upstream-model",
                    }
                },
            }
        )
