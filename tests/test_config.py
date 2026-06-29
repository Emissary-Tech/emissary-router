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


def test_glm_and_kimi_are_openrouter_only_with_expected_pricing() -> None:
    glm = CATALOG["glm-5.2"]
    assert glm.providers == {"openrouter": "z-ai/glm-5.2"}
    assert glm.default_provider == "openrouter"
    assert (glm.pricing.input, glm.pricing.output, glm.pricing.cache_read) == (0.94, 3.00, 0.18)
    # OpenRouter implicit caching: no write premium, so cache_write == input price.
    assert glm.pricing.cache_write_5m == glm.pricing.input

    kimi = CATALOG["kimi-k2.7-code"]
    assert kimi.providers == {"openrouter": "moonshotai/kimi-k2.7-code"}
    assert kimi.default_provider == "openrouter"
    assert (kimi.pricing.input, kimi.pricing.output, kimi.pricing.cache_read) == (0.74, 3.50, 0.15)
    assert kimi.pricing.cache_write_5m == kimi.pricing.input


def test_glm_and_kimi_are_selectable_in_config() -> None:
    config = _config(
        models={
            "claude-sonnet-4.6": True,
            "glm-5.2": {"enabled": True, "provider": "openrouter"},
            "kimi-k2.7-code": True,
        },
    )
    assert "glm-5.2" in config.enabled_models()
    assert config.resolve_model("kimi-k2.7-code").model_id == "moonshotai/kimi-k2.7-code"


def test_catalog_contains_supported_models() -> None:
    # Catalog insertion order is cosmetic: routing derives order from cost_score
    # (see test_routing_order_is_by_price_not_dict_order), so assert membership, not order.
    assert set(CATALOG) == {
        "gemini-3.1-flash-lite",
        "glm-5.2",
        "kimi-k2.7-code",
        "claude-haiku-4.5",
        "claude-sonnet-4.6",
    }


def test_cost_score_orders_catalog_cheap_to_expensive() -> None:
    from emissary_router.catalog import cost_score

    assert (
        cost_score(CATALOG["gemini-3.1-flash-lite"])
        < cost_score(CATALOG["glm-5.2"])
        < cost_score(CATALOG["kimi-k2.7-code"])
        < cost_score(CATALOG["claude-haiku-4.5"])
        < cost_score(CATALOG["claude-sonnet-4.6"])
    )


def test_routing_order_is_by_price_not_dict_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # Reverse the catalog's insertion order; routing must still go cheapest-first.
    from emissary_router import config as config_module

    reordered = dict(reversed(list(CATALOG.items())))
    monkeypatch.setattr(config_module, "CATALOG", reordered)

    config = _config(models={name: True for name in reordered})
    assert config.enabled_models() == [
        "gemini-3.1-flash-lite",
        "glm-5.2",
        "kimi-k2.7-code",
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


def test_default_user_path_is_always_config_json(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(home))
    monkeypatch.delenv("EMISSARY_ROUTER_CONFIG", raising=False)

    assert user_config_path() == home / "config.json"

    # A legacy config.yaml is ignored; config.json is still the default.
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}")
    assert user_config_path() == home / "config.json"


def test_nested_model_entry_and_shorthands() -> None:
    config = _config(
        models={
            "claude-sonnet-4.6": {"enabled": True, "provider": "openrouter"},  # nested
            "claude-haiku-4.5": True,  # bool shorthand
            "gemini-3.1-flash-lite": "openrouter",  # provider-string shorthand
        }
    )
    assert config.resolve_model("claude-sonnet-4.6").provider == "openrouter"
    assert config.resolve_model("claude-haiku-4.5").provider == "anthropic"  # default
    assert config.enabled_models() == [
        "gemini-3.1-flash-lite",
        "claude-haiku-4.5",
        "claude-sonnet-4.6",
    ]


def test_nested_entry_disabled() -> None:
    config = _config(
        models={
            "claude-sonnet-4.6": {"enabled": True},
            "claude-haiku-4.5": {"enabled": False},
            "gemini-3.1-flash-lite": True,
        }
    )
    assert "claude-haiku-4.5" not in config.enabled_models()


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


def test_policy_defaults_and_rejects_unknown() -> None:
    assert _config().policy == "deviate_if_confident"
    with pytest.raises(ValidationError):
        _config(policy="argmax")


def test_env_loader_does_not_override_existing_values(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=from-file\nB='quoted value'\n")
    monkeypatch.setenv("A", "already-set")
    monkeypatch.delenv("B", raising=False)

    load_env_files([env_file])

    assert os.environ["A"] == "already-set"
    assert os.environ["B"] == "quoted value"


def test_write_env_file_round_trips_and_rejects_newline(tmp_path) -> None:
    from emissary_router.config import read_env_file, write_env_file

    path = tmp_path / ".env"
    write_env_file(path, {"A": "plain-key", "B": "has space", "C": ""})
    assert read_env_file(path) == {"A": "plain-key", "B": "has space", "C": ""}
    assert oct(path.stat().st_mode)[-3:] == "600"

    with pytest.raises(ValueError):
        write_env_file(path, {"X": "line1\nline2"})


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
