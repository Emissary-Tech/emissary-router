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


def test_base64_image_counts_flat_not_as_text() -> None:
    # A 1MB screenshot is ~1.5k provider tokens, not 250k. Counting the data URL as
    # text made a single image look more expensive than the warm-cache credit and
    # busted the cache with a pointless deviation.
    body = {
        "system": "x" * 32000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this screenshot?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "A" * 1_000_000,
                        },
                    },
                ],
            }
        ],
    }
    features = extract_request_cost_features(body, {"x-claude-code-session-id": "s1"})
    assert features.estimated_input_tokens < 20000


def test_leading_inline_system_counts_toward_cacheable_prefix() -> None:
    # Providers hoist LEADING role:"system" messages into the top-level system prompt,
    # so they are part of the cacheable prefix — the estimate must mirror that.
    headers = {"x-claude-code-session-id": "s1"}
    inline = extract_request_cost_features(
        {"messages": [{"role": "system", "content": "S" * 32000}, {"role": "user", "content": "hi"}]},
        headers,
    )
    top_level = extract_request_cost_features(
        {"system": "S" * 32000, "messages": [{"role": "user", "content": "hi"}]},
        headers,
    )
    assert inline.estimated_cacheable_prefix_tokens == top_level.estimated_cacheable_prefix_tokens
    # ...and it is part of the prefix identity, not the conversation
    assert inline.prefix_hash != extract_request_cost_features(
        {"messages": [{"role": "user", "content": "hi"}]}, headers
    ).prefix_hash


def test_prefix_hash_ignores_conversation_content() -> None:
    headers = {"x-claude-code-session-id": "s1"}
    first = extract_request_cost_features(
        {"system": "S", "tools": [{"name": "t"}], "messages": [{"role": "user", "content": "hi"}]},
        headers,
    )
    second = extract_request_cost_features(
        {"system": "S", "tools": [{"name": "t"}], "messages": [{"role": "user", "content": "a completely different, much longer turn"}]},
        headers,
    )
    assert first.prefix_hash == second.prefix_hash


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


def test_compact_shrink_replaces_ratcheted_cache_size() -> None:
    # After /compact the provider reports a small read/write for the shrunken prompt.
    # The entry must follow the LATEST observation — keeping the old 150k figure would
    # credit a cache that no longer matches this prefix.
    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(model, features, Usage(input_tokens=1000, cache_read_input_tokens=150000))
    ledger.observe(model, features, Usage(input_tokens=1000, cache_read_input_tokens=8000))

    prediction = ledger.predict(model, features)
    # bounded by estimated_cacheable_prefix_tokens (10000) — not the stale 150000
    assert prediction.cached_tokens == 10000


def test_best_effort_read_confirmation_is_sticky() -> None:
    # Once a real cache read confirmed same-host routing holds for this
    # (session, provider, model, prefix), a later creation-only turn (the provider
    # re-built part of the cache) must not flip the entry back to unconfirmed.
    config = _config()
    model = config.resolve_model("gemini-3.1-flash-lite")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(model, features, Usage(input_tokens=100, cache_read_input_tokens=8000))
    assert ledger.predict(model, features).warm is True

    ledger.observe(model, features, Usage(input_tokens=100, cache_creation_input_tokens=9000))
    assert ledger.predict(model, features).warm is True


def test_observed_at_anchors_ttl_at_request_start() -> None:
    # The provider refreshed its cache TTL when it READ the cache (request start),
    # not when the stream finished minutes later. An observation anchored at a
    # request start older than the TTL must already be expired.
    import time

    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    features = _features()

    ledger.observe(
        model,
        features,
        Usage(input_tokens=100, cache_creation_input_tokens=10000),
        observed_at=time.time() - 301,
    )

    prediction = ledger.predict(model, features)
    assert prediction.warm is False
    assert prediction.reason == "expired"


def test_expired_entries_are_swept_on_observe() -> None:
    # Rotated sessions never call predict() on their old keys, so lazy pops alone
    # leak entries forever; past the threshold observe() sweeps everything expired.
    import time

    config = _config()
    model = config.resolve_model("claude-sonnet-4.6")
    ledger = CacheLedger()
    stale = time.time() - 3600
    for index in range(300):
        features = RequestCostFeatures(
            session_id=f"session-{index}",
            prefix_hash="prefix",
            estimated_input_tokens=12000,
            estimated_cacheable_prefix_tokens=10000,
            estimated_fresh_input_tokens=2000,
            expected_output_tokens=1024,
        )
        ledger.observe(
            model, features, Usage(cache_creation_input_tokens=10000), observed_at=stale
        )

    live = _features()
    ledger.observe(model, live, Usage(cache_creation_input_tokens=10000))

    assert len(ledger._entries) < 300
    assert ledger.predict(model, live).warm is True


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
