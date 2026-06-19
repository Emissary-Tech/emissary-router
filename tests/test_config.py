from __future__ import annotations

import pytest
from pydantic import ValidationError

from router.config import AppConfig, ProviderConfig, infer_provider_type


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
            "classifier": {"url": "http://127.0.0.1:9999", "model": "router"},
        }
    )

    assert config.providers["anthropic"].api_key == "k"


def test_server_auth_key_allows_non_loopback_binding() -> None:
    config = AppConfig.model_validate(
        {
            "server": {"host": "0.0.0.0", "auth_key": "router-key"},
            "router": {
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
            "classifier": {"url": "http://127.0.0.1:9999", "model": "router"},
        }
    )

    assert config.server.auth_key == "router-key"


def test_unknown_provider_without_type_fails_config_validation() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "router": {
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
                "classifier": {"url": "http://127.0.0.1:9999", "model": "router"},
            }
        )
