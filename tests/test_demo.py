from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from emissary_router.caching.usage import Usage
from emissary_router.config import AppConfig
from emissary_router.dashboard import build_demo_router
from emissary_router.pipeline import RouterPipeline


class _ConfidentHaiku:
    async def predict(self, _input):
        return {"gemini-3.1-flash-lite": 0.2, "claude-haiku-4.5": 0.95, "claude-sonnet-4.6": 0.3}


class _Unconfident:
    async def predict(self, _input):
        return {"gemini-3.1-flash-lite": 0.2, "claude-haiku-4.5": 0.3, "claude-sonnet-4.6": 0.4}


class _FakeProvider:
    name = "anthropic"

    def __init__(self):
        self.bodies = []

    async def messages(self, request, model, context, on_complete):
        self.bodies.append(request.body)
        on_complete(Usage(input_tokens=1000, output_tokens=200), {"http_status": 200})
        return JSONResponse({"content": [{"type": "text", "text": f"answer:{model.name}"}]})


def _pipeline(classifier):
    config = AppConfig.model_validate(
        {
            "models": {
                "claude-sonnet-4.6": True,
                "claude-haiku-4.5": True,
                "gemini-3.1-flash-lite": True,
            },
            "default": "claude-sonnet-4.6",
        }
    )
    pipe = RouterPipeline(config)
    pipe._classifier = classifier
    pipe._providers = {"anthropic": _FakeProvider()}
    return pipe


def test_compare_routes_to_cheaper_model_and_reports_savings():
    pipe = _pipeline(_ConfidentHaiku())
    result = asyncio.run(pipe.compare("what is a hash map?"))

    assert result["baseline_model"] == "claude-sonnet-4.6"
    assert result["baseline"]["model"] == "claude-sonnet-4.6"
    assert result["baseline"]["answer"] == "answer:claude-sonnet-4.6"
    assert result["routed"]["model"] == "claude-haiku-4.5"
    assert result["routed"]["answer"] == "answer:claude-haiku-4.5"
    assert result["routed"]["route_reason"].startswith("deviate_if_confident")
    # same tokens, cheaper model -> positive savings
    assert result["baseline"]["cost_usd"] > result["routed"]["cost_usd"] > 0
    assert result["savings_pct"] > 0


def test_compare_escalates_to_sonnet_when_not_confident():
    pipe = _pipeline(_Unconfident())
    result = asyncio.run(pipe.compare("prove sqrt(2) is irrational"))

    assert result["routed"]["model"] == "claude-sonnet-4.6"  # same as baseline
    assert result["savings_pct"] == 0


def test_compare_applies_same_settings_to_both_sides():
    pipe = _pipeline(_ConfidentHaiku())
    asyncio.run(pipe.compare("hi", max_tokens=64000, effort="high"))

    bodies = pipe._providers["anthropic"].bodies
    assert len(bodies) == 2  # baseline + routed, identical settings
    for body in bodies:
        assert body["max_tokens"] == 64000
        assert body["output_config"] == {"effort": "high"}


class _RecordingPipeline:
    def __init__(self):
        self.kwargs = None

    async def compare(self, query, max_tokens=32000, effort=None):
        self.kwargs = {"query": query, "max_tokens": max_tokens, "effort": effort}
        return {
            "query": query, "baseline_model": "claude-sonnet-4.6",
            "baseline": {"model": "claude-sonnet-4.6", "answer": "a", "cost_usd": 0.002, "latency_ms": 10},
            "routed": {"model": "claude-haiku-4.5", "answer": "b", "cost_usd": 0.0007, "latency_ms": 8,
                       "route_reason": "deviate_if_confident:p>=0.8"},
            "savings_pct": 65,
        }


def test_compare_endpoint_passes_effort_and_clamps_max_tokens():
    app = FastAPI()
    app.include_router(build_demo_router(auth_key=None))
    rec = _RecordingPipeline()
    app.state.pipeline = rec
    client = TestClient(app)

    client.post("/api/demo/compare", json={"query": "hi", "effort": "medium", "max_tokens": 64000})
    assert rec.kwargs["effort"] == "medium"
    assert rec.kwargs["max_tokens"] == 64000

    # invalid effort dropped, oversized max_tokens clamped to the ceiling
    client.post("/api/demo/compare", json={"query": "hi", "effort": "bogus", "max_tokens": 999999})
    assert rec.kwargs["effort"] is None
    assert rec.kwargs["max_tokens"] == 64000


class _StubPipeline:
    async def compare(self, query, max_tokens=32000, effort=None):
        return {
            "query": query,
            "baseline_model": "claude-sonnet-4.6",
            "baseline": {"model": "claude-sonnet-4.6", "answer": "a", "cost_usd": 0.002, "latency_ms": 10},
            "routed": {"model": "claude-haiku-4.5", "answer": "b", "cost_usd": 0.0007, "latency_ms": 8,
                       "route_reason": "deviate_if_confident:p>=0.8"},
            "savings_pct": 65,
        }


def _client():
    app = FastAPI()
    app.include_router(build_demo_router(auth_key=None, streaming_default=False))
    app.state.pipeline = _StubPipeline()
    return TestClient(app)


def test_demo_page_served():
    body = _client().get("/demo").text
    assert "Emissary routed" in body
    assert "/api/demo/compare" in body


def test_compare_endpoint_returns_result():
    resp = _client().post("/api/demo/compare", json={"query": "hi"})
    assert resp.status_code == 200
    assert resp.json()["savings_pct"] == 65


def test_compare_endpoint_rejects_empty_query():
    resp = _client().post("/api/demo/compare", json={"query": "   "})
    assert resp.status_code == 400
