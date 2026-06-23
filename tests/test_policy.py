from __future__ import annotations

from emissary_router.config import AppConfig
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
