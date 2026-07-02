from __future__ import annotations

from emissary_router.caching.ledger import CacheLedger
from emissary_router.caching.usage import Usage
from emissary_router.config import AppConfig
from emissary_router.providers.thinking import always_on_reasoning_models
from emissary_router.routing.cache_cost import (
    CachePrediction,
    EstimatedCost,
    RequestCostFeatures,
    is_cheaper,
)
from emissary_router.routing.policy import choose_model


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


def _features(prefix_tokens: int = 10000, input_tokens: int = 11000) -> RequestCostFeatures:
    return RequestCostFeatures(
        session_id="session-1",
        prefix_hash="prefix",
        estimated_input_tokens=input_tokens,
        estimated_cacheable_prefix_tokens=prefix_tokens,
        estimated_fresh_input_tokens=max(input_tokens - prefix_tokens, 0),
        expected_output_tokens=1024,
    )


def _estimated(model_name: str, total_usd: float) -> EstimatedCost:
    return EstimatedCost(
        model_name=model_name,
        total_usd=total_usd,
        input_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        expected_output_tokens=0,
        cache_prediction=CachePrediction(warm=False, cached_tokens=0, confidence=None, reason="test"),
    )


# --- price-ordered fallback path (no cost features / ledger available) ---


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


# --- background guard (skip_models) ---


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


def test_background_with_always_on_default_falls_back_to_cheapest_usable() -> None:
    config = AppConfig.model_validate(
        {
            "models": {
                "gemini-3.1-flash-lite": True,
                "kimi-k2.7-code": True,
                "claude-sonnet-4.6": True,
            },
            "default": "kimi-k2.7-code",
            "confidence": 0.8,
        }
    )
    probs = {m: 0.1 for m in config.enabled_models()}  # nobody confident

    decision = choose_model(config, probs, skip_models=always_on_reasoning_models())
    assert decision.model_name == "gemini-3.1-flash-lite"  # cheapest usable
    assert decision.reason == "default_unsuitable:cheapest_alternative"

    # main call (no skip): the configured default serves as usual
    assert choose_model(config, probs).model_name == "kimi-k2.7-code"


def test_background_serves_default_when_everything_is_skipped() -> None:
    config = AppConfig.model_validate(
        {
            "models": {"kimi-k2.7-code": True},
            "default": "kimi-k2.7-code",
            "confidence": 0.8,
        }
    )
    decision = choose_model(
        config, {"kimi-k2.7-code": 0.1}, skip_models=always_on_reasoning_models()
    )
    assert decision.model_name == "kimi-k2.7-code"  # nothing better exists
    assert decision.reason == "default"


def test_background_guard_applies_on_the_cache_aware_path_too() -> None:
    config = _config_with_kimi()
    probs = {
        "gemini-3.1-flash-lite": 0.2,
        "kimi-k2.7-code": 0.95,
        "claude-haiku-4.5": 0.1,
        "claude-sonnet-4.6": 0.1,
    }
    decision = choose_model(
        config,
        probs,
        skip_models=always_on_reasoning_models(),
        cost_features=_features(),
        cache_ledger=CacheLedger(),
    )
    assert decision.model_name == "claude-sonnet-4.6"
    assert decision.reason == "cache_aware:no_confident_candidate"


# --- cache-aware default path (cost features + ledger present, as in the gateway) ---


def test_cache_aware_keeps_default_when_candidate_below_confidence() -> None:
    decision = choose_model(
        _config(),
        {
            "gemini-3.1-flash-lite": 0.2,
            "claude-haiku-4.5": 0.79,
            "claude-sonnet-4.6": 0.2,
        },
        cost_features=_features(),
        cache_ledger=CacheLedger(),
    )

    assert decision.model_name == "claude-sonnet-4.6"
    assert decision.reason == "cache_aware:no_confident_candidate"


def test_cache_aware_keeps_warm_default_when_candidate_cold_costs_more() -> None:
    config = _config()
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
    assert decision.reason == "cache_aware:warm_default_cheaper"
    assert decision.cache_prediction["warm"] is True


def test_cache_aware_deviates_when_confident_candidate_is_warm_and_cheaper() -> None:
    config = _config()
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
    assert decision.reason == "cache_aware:candidate_cheaper"
    assert decision.cache_prediction["warm"] is True


def test_cache_aware_keeps_warm_default_when_history_dominates_prefix() -> None:
    """Regression: the warm cache covers the whole conversation history, not just the
    small system+tools prefix. A cheaper-when-cold candidate must NOT win and bust that
    cache. Before the fix, the warm credit was clamped to system+tools and the default
    was overpriced ~5x, so the policy wrongly deviated."""
    config = _config()
    ledger = CacheLedger()
    # system+tools prefix is small (8k); the conversation history dominates input (68k).
    features = _features(prefix_tokens=8000, input_tokens=68000)
    # Last turn the provider reported a large warm read on the default.
    ledger.observe(
        config.resolve_model("claude-sonnet-4.6"),
        features,
        Usage(input_tokens=1000, cache_read_input_tokens=67000),
    )

    decision = choose_model(
        config,
        {
            "gemini-3.1-flash-lite": 0.2,
            "claude-haiku-4.5": 0.95,  # confident and cheaper *when cold*
            "claude-sonnet-4.6": 0.1,
        },
        cost_features=features,
        cache_ledger=ledger,
    )

    assert decision.model_name == "claude-sonnet-4.6"
    assert decision.reason == "cache_aware:warm_default_cheaper"
    assert decision.cache_prediction["warm"] is True
    # The warm credit must reflect the observed cache (67k), not the system+tools prefix.
    assert decision.estimated_costs["claude-sonnet-4.6"]["cache_read_tokens"] == 67000


def test_cache_aware_uses_any_positive_savings_without_margin() -> None:
    baseline = _estimated("claude-sonnet-4.6", total_usd=1.0)
    candidate = _estimated("claude-haiku-4.5", total_usd=0.999999)

    assert is_cheaper(candidate, baseline) is True
