from __future__ import annotations

import asyncio

from starlette.responses import JSONResponse

from emissary_router.caching.ledger import CacheLedger
from emissary_router.caching.usage import Usage
from emissary_router.config import AppConfig
from emissary_router.pipeline import RouterPipeline
from emissary_router.routing.cache_cost import extract_request_cost_features


class _ConfidentKimiClassifier:
    async def predict(self, _input):
        return {
            "gemini-3.1-flash-lite": 0.1,
            "glm-5.2": 0.1,
            "kimi-k2.7-code": 0.95,
            "claude-haiku-4.5": 0.1,
            "claude-sonnet-4.6": 0.1,
        }


class _FakeProvider:
    def __init__(self, name: str):
        self.name = name
        self.calls = []

    async def messages(self, request, model, context, on_complete):
        self.calls.append(model.name)
        on_complete(Usage(input_tokens=10, output_tokens=2), {"http_status": 200})
        return JSONResponse({"ok": True})


def _config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {
                "gemini-3.1-flash-lite": True,
                "glm-5.2": True,
                "kimi-k2.7-code": True,
                "claude-haiku-4.5": True,
                "claude-sonnet-4.6": True,
            },
            "default": "claude-sonnet-4.6",
            "confidence": 0.8,
        }
    )


def _pipeline() -> tuple[RouterPipeline, dict[str, _FakeProvider]]:
    pipe = RouterPipeline(_config())
    pipe._classifier = _ConfidentKimiClassifier()
    providers = {name: _FakeProvider(name) for name in ("anthropic", "openrouter")}
    pipe._providers = providers
    return pipe, providers


def _served(providers: dict[str, _FakeProvider]) -> list[str]:
    return [name for p in providers.values() for name in p.calls]


# ~290k estimated tokens: over every window except the 1M ones.
_BIG = "x" * 1_160_000


def test_oversized_request_routes_to_cheapest_model_that_fits() -> None:
    # kimi is confident but its 262k window can't hold the request; sonnet (default,
    # 200k without the 1M beta) can't either. Deterministic outcome: the cheapest
    # 1M-window model serves — no luck, no provider 400.
    pipe, providers = _pipeline()
    body = {
        "system": _BIG,
        "max_tokens": 32000,
        "tools": [{"name": "t", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "hi"}],
    }

    resp = asyncio.run(pipe.handle_messages(body, {"x-claude-code-session-id": "s1"}))

    assert resp.status_code == 200
    assert _served(providers) == ["gemini-3.1-flash-lite"]


def test_context_1m_beta_lets_native_sonnet_hold_the_request() -> None:
    # Same oversized request, but Claude Code is in 1M mode (anthropic-beta header):
    # the native sonnet default fits and serves as usual.
    pipe, providers = _pipeline()
    body = {
        "system": _BIG,
        "max_tokens": 32000,
        "tools": [{"name": "t", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = {
        "x-claude-code-session-id": "s1",
        "anthropic-beta": "context-1m-2025-08-07,interleaved-thinking-2025-05-14",
    }

    resp = asyncio.run(pipe.handle_messages(body, headers))

    assert resp.status_code == 200
    assert _served(providers) == ["claude-sonnet-4.6"]


def test_ledger_observation_overrides_token_underestimate() -> None:
    # chars/4 undercounts CJK-heavy history; the provider-reported cache size from
    # last turn is the real number and must drive the guard.
    pipe, _ = _pipeline()
    config = pipe._config
    body = {
        "system": "s" * 40_000,  # estimate ~10k tokens
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = {"x-claude-code-session-id": "s1"}
    features = extract_request_cost_features(body, headers)
    # Last turn sonnet actually read a 280k-token cache for this same prefix.
    pipe.cache_ledger.observe(
        config.resolve_model("claude-sonnet-4.6"),
        features,
        Usage(input_tokens=1000, cache_read_input_tokens=280_000),
    )

    oversized = pipe._oversized_models(body, headers, features)

    assert "kimi-k2.7-code" in oversized
    assert "claude-haiku-4.5" in oversized
    assert "claude-sonnet-4.6" in oversized  # no 1M beta on this request
    assert "gemini-3.1-flash-lite" not in oversized
    assert "glm-5.2" not in oversized


def test_openrouter_reserves_requested_output_anthropic_does_not() -> None:
    # Measured behavior: OpenRouter validates input + max_tokens against the window;
    # native Anthropic validates input only.
    pipe, _ = _pipeline()
    body = {
        "system": "x" * 960_000,  # ~240k tokens
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = {"x-claude-code-session-id": "s1"}
    features = extract_request_cost_features(body, headers)

    oversized = pipe._oversized_models(body, headers, features)
    assert "kimi-k2.7-code" in oversized  # 240k + 32k reserve > 262,144

    body_small_output = {**body, "max_tokens": 1000}
    features2 = extract_request_cost_features(body_small_output, headers)
    oversized2 = pipe._oversized_models(body_small_output, headers, features2)
    assert "kimi-k2.7-code" not in oversized2  # 240k + 1k fits


def test_observed_context_is_scoped_to_session_and_freshness() -> None:
    config = _config()
    ledger = CacheLedger()
    body = {"system": "s" * 40_000, "messages": [{"role": "user", "content": "hi"}]}
    features = extract_request_cost_features(body, {"x-claude-code-session-id": "s1"})
    other_session = extract_request_cost_features(body, {"x-claude-code-session-id": "s2"})

    ledger.observe(
        config.resolve_model("claude-sonnet-4.6"),
        features,
        Usage(input_tokens=1000, cache_read_input_tokens=280_000),
    )

    assert ledger.observed_context_tokens(features) == 280_000
    assert ledger.observed_context_tokens(other_session) == 0

    import time

    ledger.observe(
        config.resolve_model("claude-sonnet-4.6"),
        features,
        Usage(input_tokens=1000, cache_read_input_tokens=280_000),
        observed_at=time.time() - 3600,  # expired
    )
    assert ledger.observed_context_tokens(features) == 0
