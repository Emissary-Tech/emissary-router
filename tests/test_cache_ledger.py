from __future__ import annotations

from emissary_router.caching.ledger import CacheLedger
from emissary_router.caching.usage import Usage
from emissary_router.config import AppConfig
from emissary_router.routing.cache_cost import RequestCostFeatures


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {
                "claude-sonnet-4.6": True,
                "claude-haiku-4.5": True,
                "gemini-3.1-flash-lite": True,
            },
            "default": "claude-sonnet-4.6",
        }
    )


def _features(session_id: str | None = "session-1") -> RequestCostFeatures:
    return RequestCostFeatures(
        session_id=session_id,
        prefix_hash="prefix",
        estimated_input_tokens=12000,
        estimated_cacheable_prefix_tokens=10000,
        estimated_fresh_input_tokens=2000,
        expected_output_tokens=1024,
    )


def test_predicts_anthropic_creation_as_warm() -> None:
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(model, features, Usage(cache_creation_input_tokens=10000))

    prediction = ledger.predict(model, features)
    assert prediction.warm is True
    assert prediction.cached_tokens == 10000
    assert prediction.confidence == "predictable"


def test_best_effort_creation_without_read_is_not_predicted_warm() -> None:
    config = _config()
    model = config.resolve_model("gemini-3.1-flash-lite")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(model, features, Usage(cache_creation_input_tokens=10000))

    prediction = ledger.predict(model, features)
    assert prediction.warm is False
    assert prediction.reason == "best_effort_unconfirmed"


def test_no_session_does_not_cross_pollinate_cache_state() -> None:
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features(session_id=None)

    ledger.observe(model, features, Usage(cache_creation_input_tokens=10000))

    prediction = ledger.predict(model, features)
    assert prediction.warm is False
    assert prediction.reason == "no_session"


def test_zero_cache_observation_clears_previous_prediction() -> None:
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(model, features, Usage(cache_creation_input_tokens=10000))
    assert ledger.predict(model, features).warm is True

    ledger.observe(model, features, Usage())

    prediction = ledger.predict(model, features)
    assert prediction.warm is False
    assert prediction.reason == "cold"
