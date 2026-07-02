from __future__ import annotations

from emissary_router.caching.ledger import CacheLedger
from emissary_router.caching.usage import Usage
from emissary_router.config import AppConfig
from emissary_router.routing.cache_cost import (
    RequestCostFeatures,
    extract_request_cost_features,
)


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


def test_predict_credits_full_observed_cache_not_just_prefix() -> None:
    """Regression: a warm read covers tools+system+most of the history, not only the
    system+tools prefix. predict() must surface the observed cache size (bounded by the
    request's total input), not clamp it back down to estimated_cacheable_prefix_tokens."""
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = RequestCostFeatures(
        session_id="session-1",
        prefix_hash="prefix",
        estimated_input_tokens=68000,
        estimated_cacheable_prefix_tokens=8000,  # system+tools only
        estimated_fresh_input_tokens=60000,
        expected_output_tokens=1024,
    )

    ledger.observe(model, features, Usage(input_tokens=1000, cache_read_input_tokens=67000))

    prediction = ledger.predict(model, features)
    assert prediction.warm is True
    assert prediction.cached_tokens == 67000  # not clamped to 8000


def test_best_effort_creation_without_read_is_not_predicted_warm() -> None:
    config = _config()
    model = config.resolve_model("gemini-3.1-flash-lite")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(model, features, Usage(cache_creation_input_tokens=10000))

    prediction = ledger.predict(model, features)
    assert prediction.warm is False
    assert prediction.reason == "best_effort_unconfirmed"


def test_expected_output_tracks_observed_outputs() -> None:
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features()

    assert ledger.expected_output_tokens() == 1024  # seed default

    for _ in range(30):
        ledger.observe(model, features, Usage(output_tokens=200, cache_creation_input_tokens=10000))

    # The rolling estimate converges toward observed 200, well under the 1024 seed.
    assert 150 < ledger.expected_output_tokens() < 350


def test_background_calls_do_not_move_output_estimate() -> None:
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features()

    seed = ledger.expected_output_tokens()
    for _ in range(30):
        ledger.observe(
            model,
            features,
            Usage(output_tokens=50, cache_creation_input_tokens=10000),
            is_main=False,
        )

    # Background outputs (title/summary) must not drag the main-call estimate down.
    assert ledger.expected_output_tokens() == seed
    # ...but the cache entry is still recorded for every call.
    assert ledger.predict(model, features).warm is True


def test_expected_output_bounded_by_max_tokens() -> None:
    body = {"max_tokens": 50, "system": "s", "messages": [{"role": "user", "content": "hi"}]}
    features = extract_request_cost_features(body, {}, expected_output_tokens=5000)
    assert features.expected_output_tokens == 50  # output can't exceed max_tokens


def test_expected_output_uses_rolling_estimate_under_ceiling() -> None:
    body = {"max_tokens": 32000, "system": "s", "messages": [{"role": "user", "content": "hi"}]}
    features = extract_request_cost_features(body, {}, expected_output_tokens=300)
    assert features.expected_output_tokens == 300  # rolling estimate, below the ceiling


def test_expected_output_defaults_without_rolling_data() -> None:
    body = {"max_tokens": 32000, "system": "s", "messages": []}
    features = extract_request_cost_features(body, {})
    assert features.expected_output_tokens == 1024  # seed default, under the ceiling


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
    # REAL evidence of a gone cache: a successful response (input/output tokens
    # present) that reports zero cache tokens.
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(model, features, Usage(cache_creation_input_tokens=10000))
    assert ledger.predict(model, features).warm is True

    ledger.observe(model, features, Usage(input_tokens=12000, output_tokens=50))

    prediction = ledger.predict(model, features)
    assert prediction.warm is False
    assert prediction.reason == "cold"


def test_error_shaped_all_zero_usage_does_not_wipe_warm_entry() -> None:
    # Provider error paths (429/500, severed streams) report an all-zero Usage().
    # That is NO evidence about the provider-side cache — a transient error must not
    # wipe warm state and trigger a pointless model switch on the next turn.
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(model, features, Usage(cache_creation_input_tokens=10000))
    assert ledger.predict(model, features).warm is True

    ledger.observe(model, features, Usage())  # all-zero: error shape

    assert ledger.predict(model, features).warm is True  # still warm
