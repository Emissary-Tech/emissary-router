from __future__ import annotations

import asyncio

import httpx
from starlette.responses import JSONResponse

from emissary_router.caching.usage import Usage
from emissary_router.cli import _warn_missing_env
from emissary_router.config import AppConfig
from emissary_router.pipeline import RouterPipeline
from emissary_router.routing.cache_cost import extract_request_cost_features
from emissary_router.telemetry import SqliteStore


class _FailingClassifier:
    """Simulates the classifier being unreachable (transport error after retries)."""

    async def predict(self, _input):
        raise httpx.ConnectError("classifier unreachable")


class _PartialClassifier:
    """Returns probabilities missing a label required by config."""

    async def predict(self, _input):
        return {"claude-sonnet-4.6": 0.9}  # missing gemini-3.1-flash-lite


class _ConfidentHaikuClassifier:
    async def predict(self, _input):
        return {
            "claude-sonnet-4.6": 0.1,
            "claude-haiku-4.5": 0.95,
            "gemini-3.1-flash-lite": 0.1,
        }


class _FakeProvider:
    name = "anthropic"

    def __init__(self):
        self.calls = []

    async def messages(self, request, model, context, on_complete):
        self.calls.append(model)
        on_complete(Usage(input_tokens=10, output_tokens=2), {"http_status": 200})
        return JSONResponse({"ok": True})


def _config():
    return AppConfig.model_validate(
        {
            "models": {"claude-sonnet-4.6": True, "gemini-3.1-flash-lite": True},
            "default": "claude-sonnet-4.6",
        }
    )


def _config_with_haiku():
    return AppConfig.model_validate(
        {
            "models": {
                "claude-sonnet-4.6": True,
                "claude-haiku-4.5": True,
                "gemini-3.1-flash-lite": False,
            },
            "default": "claude-sonnet-4.6",
            "policy": "cache_aware",
        }
    )


def test_classifier_failure_falls_back_to_default(tmp_path):
    # When the classifier is unreachable, the request is served by the default
    # model (fail-open) rather than returning 502.
    store = SqliteStore(tmp_path / "e.sqlite3")
    pipe = RouterPipeline(_config(), store=store)
    pipe._classifier = _FailingClassifier()
    fake = _FakeProvider()
    pipe._providers = {"anthropic": fake}

    resp = asyncio.run(
        pipe.handle_messages(
            {"messages": [{"role": "user", "content": "hi"}]},
            {"x-claude-code-session-id": "s1"},
        )
    )
    assert resp.status_code == 200
    assert [m.name for m in fake.calls] == ["claude-sonnet-4.6"]
    rows = store.list_events()
    assert len(rows) == 1
    assert rows[0]["served_model"] == "claude-sonnet-4.6"
    assert rows[0]["route_reason"] == "fallback: router_issue"
    assert rows[0]["session_id"] == "s1"


def test_missing_labels_still_502_and_recorded(tmp_path):
    # A missing-labels classifier response is a real config/classifier mismatch:
    # fail loud with 502, and still record a telemetry row for it.
    store = SqliteStore(tmp_path / "e.sqlite3")
    pipe = RouterPipeline(_config(), store=store)
    pipe._classifier = _PartialClassifier()

    resp = asyncio.run(
        pipe.handle_messages(
            {"messages": [{"role": "user", "content": "hi"}]},
            {"x-claude-code-session-id": "s1"},
        )
    )
    assert resp.status_code == 502
    rows = store.list_events()
    assert len(rows) == 1
    assert rows[0]["served_model"] == "(routing error)"
    assert rows[0]["http_status"] == 502


def test_pipeline_passes_cache_state_to_cache_aware_policy(tmp_path):
    store = SqliteStore(tmp_path / "e.sqlite3")
    config = _config_with_haiku()
    pipe = RouterPipeline(config, store=store)
    pipe._classifier = _ConfidentHaikuClassifier()
    fake = _FakeProvider()
    pipe._providers = {"anthropic": fake}
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "system": "x" * 120000,
        "max_tokens": 1,
    }
    headers = {"x-claude-code-session-id": "s1"}
    features = extract_request_cost_features(body, headers)
    pipe._cache_ledger.observe(
        config.resolve_model("claude-sonnet-4.6"),
        features,
        Usage(cache_creation_input_tokens=30000),
    )

    resp = asyncio.run(pipe.handle_messages(body, headers))

    assert resp.status_code == 200
    assert [m.name for m in fake.calls] == ["claude-sonnet-4.6"]
    rows = store.list_events()
    # Haiku was confident, but the warm sonnet cache made the default cheaper after cache.
    assert rows[0]["route_reason"] == "cache_aware:warm_default_cheaper"


def test_warn_missing_env_is_provider_aware(capsys, monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "EMISSARY_ROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # OpenRouter-only setup: must not demand an Anthropic key
    cfg = AppConfig.model_validate(
        {
            "models": {
                "claude-haiku-4.5": {"enabled": True, "provider": "openrouter"},
                "gemini-3.1-flash-lite": True,
            },
            "default": "claude-haiku-4.5",
        }
    )
    _warn_missing_env(cfg)
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in err
    assert "EMISSARY_ROUTER_API_KEY" in err
    assert "ANTHROPIC_API_KEY" not in err
