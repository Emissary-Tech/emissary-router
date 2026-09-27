from __future__ import annotations

import asyncio
import dataclasses
import json
import sqlite3
import time

from starlette.responses import JSONResponse

from emissary_router import catalog
from emissary_router.caching.ledger import CacheLedger
from emissary_router.caching.usage import Usage
from emissary_router.config import AppConfig
from emissary_router.pipeline import RouterPipeline
from emissary_router.routing.cache_cost import RequestCostFeatures
from emissary_router.routing.context_guard import fits_context, oversized_models
from emissary_router.telemetry import SqliteStore


def _config(models: dict, default: str) -> AppConfig:
    return AppConfig.model_validate({"models": models, "default": default, "confidence": 0.8})


def _features(input_tokens: int, expected_output: int = 1024) -> RequestCostFeatures:
    return RequestCostFeatures(
        session_id="s1",
        prefix_hash="prefix",
        estimated_input_tokens=input_tokens,
        estimated_cacheable_prefix_tokens=min(input_tokens, 10_000),
        estimated_fresh_input_tokens=max(input_tokens - 10_000, 0),
        expected_output_tokens=expected_output,
    )


SONNET_HAIKU = {"claude-sonnet-5": True, "claude-haiku-4.5": True}


# --- the fit check itself ---------------------------------------------------------


def test_fit_keeps_a_five_percent_margin() -> None:
    # 200K window -> 190K usable after the margin
    assert fits_context(200_000, 189_000, 1_000)
    assert not fits_context(200_000, 189_001, 1_000)


def test_unknown_window_always_fits() -> None:
    assert fits_context(None, 10**9, 10**6)


# --- which candidates get excluded -------------------------------------------------


def test_small_window_candidate_is_excluded() -> None:
    config = _config(SONNET_HAIKU, "claude-sonnet-5")
    assert oversized_models(config, {"max_tokens": 32000}, _features(195_000)) == {
        "claude-haiku-4.5"
    }


def test_candidate_that_fits_is_kept() -> None:
    config = _config(SONNET_HAIKU, "claude-sonnet-5")
    assert oversized_models(config, {"max_tokens": 32000}, _features(180_000)) == frozenset()


def test_default_is_never_excluded() -> None:
    # haiku is the client's chosen default: an overflow there stays the client's to
    # handle (normalized 400 -> compaction), the guard only filters deviations.
    config = _config(SONNET_HAIKU, "claude-haiku-4.5")
    assert oversized_models(config, {"max_tokens": 32000}, _features(500_000)) == frozenset()


def test_unknown_catalog_window_never_excludes(monkeypatch) -> None:
    haiku = catalog.CATALOG["claude-haiku-4.5"]
    monkeypatch.setitem(
        catalog.CATALOG, "claude-haiku-4.5", dataclasses.replace(haiku, context_window=None)
    )
    config = _config(SONNET_HAIKU, "claude-sonnet-5")
    assert oversized_models(config, {"max_tokens": 32000}, _features(500_000)) == frozenset()


def test_openrouter_reserves_the_requested_max_tokens() -> None:
    # OpenRouter validates input + max_tokens against the window: 170K + 32K > 190K.
    via_openrouter = _config(
        {"claude-sonnet-5": True, "claude-haiku-4.5": {"enabled": True, "provider": "openrouter"}},
        "claude-sonnet-5",
    )
    assert oversized_models(via_openrouter, {"max_tokens": 32000}, _features(170_000)) == {
        "claude-haiku-4.5"
    }
    # Native Anthropic reserves only the expected output, so the same request fits.
    native = _config(SONNET_HAIKU, "claude-sonnet-5")
    assert oversized_models(native, {"max_tokens": 32000}, _features(170_000)) == frozenset()


def test_kimi_window_on_openrouter() -> None:
    config = _config({"claude-sonnet-5": True, "kimi-k2.7-code": True}, "claude-sonnet-5")
    # 262,144 * 0.95 = 249,037 usable
    assert oversized_models(config, {"max_tokens": 32000}, _features(220_000)) == {"kimi-k2.7-code"}
    assert oversized_models(config, {"max_tokens": 16000}, _features(220_000)) == frozenset()


# --- the provider-reported size floor ----------------------------------------------


def test_observed_cache_size_floors_the_estimate() -> None:
    config = _config(SONNET_HAIKU, "claude-sonnet-5")
    ledger = CacheLedger()
    features = _features(10_000)
    ledger.observe(
        config.resolve_model("claude-sonnet-5"),
        features,
        Usage(input_tokens=100, cache_read_input_tokens=195_000),
    )
    # our chars/4 estimate says 10K, the provider said the conversation is 195K
    assert oversized_models(config, {"max_tokens": 32000}, features, ledger) == {
        "claude-haiku-4.5"
    }


def test_freshest_observation_wins_after_compaction() -> None:
    config = _config(SONNET_HAIKU, "claude-sonnet-5")
    ledger = CacheLedger()
    features = _features(10_000)
    now = time.time()
    # an older, large observation on haiku (before /compact) ...
    ledger.observe(
        config.resolve_model("claude-haiku-4.5"),
        features,
        Usage(input_tokens=100, cache_read_input_tokens=195_000),
        observed_at=now - 60,
    )
    # ... and the newest, small one on sonnet (after /compact)
    ledger.observe(
        config.resolve_model("claude-sonnet-5"),
        features,
        Usage(input_tokens=100, cache_read_input_tokens=20_000),
        observed_at=now,
    )
    assert ledger.observed_context_tokens(features) == 20_000
    assert oversized_models(config, {"max_tokens": 32000}, features, ledger) == frozenset()


# --- end to end through the pipeline -------------------------------------------------


class _ConfidentHaikuClassifier:
    async def predict(self, _input):
        return {"claude-sonnet-5": 0.1, "claude-haiku-4.5": 0.95}


class _FakeProvider:
    name = "anthropic"

    def __init__(self):
        self.calls = []

    async def messages(self, request, model, context, on_complete):
        self.calls.append(model)
        on_complete(Usage(input_tokens=10, output_tokens=2), {"http_status": 200})
        return JSONResponse({"ok": True})


def test_pipeline_keeps_an_oversized_request_on_the_default(tmp_path) -> None:
    store = SqliteStore(tmp_path / "e.sqlite3")
    pipe = RouterPipeline(_config(SONNET_HAIKU, "claude-sonnet-5"), store=store)
    pipe._classifier = _ConfidentHaikuClassifier()
    fake = _FakeProvider()
    pipe._providers = {"anthropic": fake}
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "system": "x" * 800_000,  # ~200K tokens: more than haiku can hold
        "max_tokens": 32000,
    }

    resp = asyncio.run(pipe.handle_messages(body, {"x-claude-code-session-id": "s1"}))

    assert resp.status_code == 200
    # haiku was confident and cheaper, but can't hold the request
    assert [m.name for m in fake.calls] == ["claude-sonnet-5"]
    with sqlite3.connect(tmp_path / "e.sqlite3") as conn:
        served, reason, raw = conn.execute(
            "SELECT served_model, route_reason, raw_event FROM events"
        ).fetchone()
    assert served == "claude-sonnet-5"
    assert reason == "cache_aware:no_confident_candidate"
    assert json.loads(raw) == {"context_excluded": ["claude-haiku-4.5"]}


def test_pipeline_deviates_normally_when_the_request_fits(tmp_path) -> None:
    store = SqliteStore(tmp_path / "e.sqlite3")
    pipe = RouterPipeline(_config(SONNET_HAIKU, "claude-sonnet-5"), store=store)
    pipe._classifier = _ConfidentHaikuClassifier()
    fake = _FakeProvider()
    pipe._providers = {"anthropic": fake}
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 32000}

    resp = asyncio.run(pipe.handle_messages(body, {"x-claude-code-session-id": "s1"}))

    assert resp.status_code == 200
    assert [m.name for m in fake.calls] == ["claude-haiku-4.5"]
    with sqlite3.connect(tmp_path / "e.sqlite3") as conn:
        (raw,) = conn.execute("SELECT raw_event FROM events").fetchone()
    assert raw is None
