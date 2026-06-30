from __future__ import annotations

from emissary_router.config import AppConfig
from emissary_router.providers.thinking import always_on_reasoning_models
from emissary_router.routing.policy import choose_model


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {
                "claude-sonnet-4.6": True,
                "claude-haiku-4.5": True,
                "gemini-3.1-flash-lite": True,
            },
            "default": "claude-sonnet-4.6",
            "confidence": 0.8,
        }
    )


def test_deviates_to_cheapest_confident_model() -> None:
    decision = choose_model(
        _config(),
        {
            "gemini-3.1-flash-lite": 0.81,
            "claude-haiku-4.5": 0.9,
            "claude-sonnet-4.6": 0.2,
        },
    )

    assert decision.model_name == "gemini-3.1-flash-lite"
    assert decision.reason == "deviate_if_confident:p>=0.8"


def test_falls_back_to_default_when_no_model_is_confident() -> None:
    decision = choose_model(
        _config(),
        {
            "gemini-3.1-flash-lite": 0.2,
            "claude-haiku-4.5": 0.3,
            "claude-sonnet-4.6": 0.4,
        },
    )

    assert decision.model_name == "claude-sonnet-4.6"
    assert decision.reason == "default"


def _config_with_kimi() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {
                "gemini-3.1-flash-lite": True,
                "kimi-k2.7-code": True,
                "claude-haiku-4.5": True,
                "claude-sonnet-4.6": True,
            },
            "default": "claude-sonnet-4.6",
            "confidence": 0.8,
        }
    )


def test_background_skips_always_on_reasoning_models() -> None:
    config = _config_with_kimi()
    probs = {
        "gemini-3.1-flash-lite": 0.2,
        "kimi-k2.7-code": 0.95,  # confident, cheapest confident -> would win normally
        "claude-haiku-4.5": 0.1,
        "claude-sonnet-4.6": 0.1,
    }

    # main call: kimi wins
    assert choose_model(config, probs).model_name == "kimi-k2.7-code"

    # background call: kimi (always-on reasoning) is skipped -> nobody else confident -> default
    decision = choose_model(config, probs, skip_models=always_on_reasoning_models())
    assert decision.model_name == "claude-sonnet-4.6"
    assert decision.reason == "default"
    assert "kimi-k2.7-code" in always_on_reasoning_models()
