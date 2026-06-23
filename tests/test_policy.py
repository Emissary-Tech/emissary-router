from __future__ import annotations

from emissary_router.caching.ledger import CacheLedger
from emissary_router.caching.usage import Usage
from emissary_router.config import AppConfig
from emissary_router.routing.cache_cost import RequestCostFeatures
from emissary_router.routing.policy import choose_model


def _config(policy: str = "deviate_if_confident") -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {
                "claude-sonnet-4.6": True,
                "claude-haiku-4.5": True,
                "gemini-3.1-flash-lite": True,
            },
            "default": "claude-sonnet-4.6",
            "confidence": 0.8,
            "policy": policy,
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


def test_cache_aware_keeps_default_when_candidate_below_confidence() -> None:
    config = _config(policy="cache_aware")
    ledger = CacheLedger()
    features = _features()

    decision = choose_model(
        config,
        {
            "gemini-3.1-flash-lite": 0.2,
            "claude-haiku-4.5": 0.79,
            "claude-sonnet-4.6": 0.2,
        },
        cost_features=features,
        cache_ledger=ledger,
    )

    assert decision.model_name == "claude-sonnet-4.6"
    assert decision.reason == "cache_aware:default_after_cache"


def test_cache_aware_keeps_warm_default_when_candidate_cold_costs_more() -> None:
    config = _config(policy="cache_aware")
    ledger = CacheLedger()
    features = _features(prefix_tokens=30000, input_tokens=31000)
    ledger.observe(
        config.resolve_model("claude-sonnet-4.6"),
        features,
        Usage(cache_creation_input_tokens=30000),
    )

    decision = choose_model(
        config,
        {
            "gemini-3.1-flash-lite": 0.2,
            "claude-haiku-4.5": 0.95,
            "claude-sonnet-4.6": 0.1,
        },
        cost_features=features,
        cache_ledger=ledger,
    )

    assert decision.model_name == "claude-sonnet-4.6"
    assert decision.cache_prediction["warm"] is True


def test_cache_aware_deviates_when_confident_candidate_is_warm_and_cheaper() -> None:
    config = _config(policy="cache_aware")
    ledger = CacheLedger()
    features = _features(prefix_tokens=30000, input_tokens=31000)
    ledger.observe(
        config.resolve_model("claude-haiku-4.5"),
        features,
        Usage(cache_creation_input_tokens=30000),
    )

    decision = choose_model(
        config,
        {
            "gemini-3.1-flash-lite": 0.2,
            "claude-haiku-4.5": 0.95,
            "claude-sonnet-4.6": 0.1,
        },
        cost_features=features,
        cache_ledger=ledger,
    )

    assert decision.model_name == "claude-haiku-4.5"
    assert decision.reason == "cache_aware:p>=0.8:cheaper_after_cache"
    assert decision.cache_prediction["warm"] is True


def _features(prefix_tokens: int = 10000, input_tokens: int = 11000) -> RequestCostFeatures:
    return RequestCostFeatures(
        session_id="session-1",
        prefix_hash="prefix",
        estimated_input_tokens=input_tokens,
        estimated_cacheable_prefix_tokens=prefix_tokens,
        estimated_fresh_input_tokens=max(input_tokens - prefix_tokens, 0),
        expected_output_tokens=1024,
    )
