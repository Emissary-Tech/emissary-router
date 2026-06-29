from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, StreamingResponse

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
        on_complete(
            Usage(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=500),
            {"http_status": 200},
        )
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


def _turn(text):
    return [{"role": "user", "content": text}]


def test_chat_routes_to_cheaper_model_and_reports_savings():
    pipe = _pipeline(_ConfidentHaiku())
    result = asyncio.run(pipe.chat(_turn("what is a hash map?"), _turn("what is a hash map?")))

    assert result["baseline_model"] == "claude-sonnet-4.6"
    assert result["baseline"]["model"] == "claude-sonnet-4.6"
    assert result["baseline"]["answer"] == "answer:claude-sonnet-4.6"
    assert result["routed"]["model"] == "claude-haiku-4.5"
    assert result["routed"]["answer"] == "answer:claude-haiku-4.5"
    assert result["routed"]["route_reason"].startswith("deviate_if_confident")
    assert result["baseline"]["cost_usd"] > result["routed"]["cost_usd"] > 0
    assert result["savings_pct"] > 0


def test_chat_reports_router_and_model_latency_split():
    pipe = _pipeline(_ConfidentHaiku())
    result = asyncio.run(pipe.chat(_turn("hi"), _turn("hi")))

    routed = result["routed"]
    assert routed["router_ms"] >= 0
    assert routed["total_ms"] >= routed["model_ms"]
    # baseline has no router step
    assert result["baseline"]["router_ms"] == 0.0
    assert result["baseline"]["total_ms"] == result["baseline"]["model_ms"]


def test_chat_escalates_to_sonnet_when_not_confident():
    pipe = _pipeline(_Unconfident())
    result = asyncio.run(pipe.chat(_turn("prove sqrt(2) is irrational"), _turn("prove sqrt(2) is irrational")))

    assert result["routed"]["model"] == "claude-sonnet-4.6"  # same as baseline
    assert result["savings_pct"] == 0


def test_chat_side_runs_only_one_side():
    pipe = _pipeline(_ConfidentHaiku())
    base = asyncio.run(pipe.chat_side("baseline", _turn("hi"), _turn("hi")))
    routed = asyncio.run(pipe.chat_side("routed", _turn("hi"), _turn("hi"), session_id="s1", policy="cache_aware"))
    assert base["model"] == "claude-sonnet-4.6"
    assert base["router_ms"] == 0.0
    assert routed["model"] == "claude-haiku-4.5"
    assert routed["route_reason"]  # routed side carries the routing reason


def test_demo_system_prompt_has_date_and_conditional_search():
    from emissary_router.pipeline import _demo_system
    base = _demo_system(with_search=False)
    assert "Current date:" in base
    assert "fabricate" in base
    assert "web_search" not in base  # no tools block without search
    searched = _demo_system(with_search=True)
    assert "web_search" in searched
    assert "trust the results" in searched


def test_chat_includes_dated_system_prompt():
    pipe = _pipeline(_ConfidentHaiku())
    asyncio.run(pipe.chat(_turn("hi"), _turn("hi")))
    for body in pipe._providers["anthropic"].bodies:
        assert "Current date:" in body["system"]
        assert "web_search" not in body["system"]  # chat path has no search block


def test_chat_resolves_reasoning_per_model():
    # routed side is haiku (no effort param) -> thinking budget; baseline sonnet -> effort
    pipe = _pipeline(_ConfidentHaiku())
    asyncio.run(pipe.chat(_turn("hi"), _turn("hi"), max_tokens=64000, effort="high"))

    bodies = pipe._providers["anthropic"].bodies
    assert len(bodies) == 2
    assert all(b["max_tokens"] == 64000 for b in bodies)  # same token budget on both
    assert {"effort": "high"} in [b.get("output_config") for b in bodies]  # sonnet keeps effort
    # haiku gets a thinking budget of half max_tokens instead of effort
    assert any((b.get("thinking") or {}).get("budget_tokens") == 32000 for b in bodies)


def test_assistant_turn_content_preserves_thinking_for_tool_continuation():
    from emissary_router.pipeline import _assistant_turn_content
    tool_uses = [{"type": "tool_use", "id": "t1", "name": "web_search", "input": {"query": "x"}}]
    content = _assistant_turn_content(
        "Searching.", {"thinking": "let me think", "signature": "sig123"}, tool_uses
    )
    assert content[0] == {"type": "thinking", "thinking": "let me think", "signature": "sig123"}
    assert content[1] == {"type": "text", "text": "Searching."}
    assert content[2] == tool_uses[0]


def test_assistant_turn_content_drops_unsigned_thinking_and_empty_text():
    from emissary_router.pipeline import _assistant_turn_content
    # No signature (e.g. OpenRouter) -> no thinking block replayed; empty text -> skipped.
    tool_uses = [{"type": "tool_use", "id": "t1", "name": "web_search", "input": {}}]
    content = _assistant_turn_content("", {"thinking": "x", "signature": ""}, tool_uses)
    assert content == tool_uses


def test_demo_reasoning_off_adds_no_reasoning():
    from emissary_router.pipeline import _demo_reasoning_for_model
    body = {"max_tokens": 32000}
    _demo_reasoning_for_model(body, "claude-haiku-4.5")
    assert "thinking" not in body and "output_config" not in body


def test_chat_sends_history_with_cache_breakpoint():
    pipe = _pipeline(_ConfidentHaiku())
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and now?"},
    ]
    asyncio.run(pipe.chat(history, history))
    for body in pipe._providers["anthropic"].bodies:
        msgs = body["messages"]
        assert len(msgs) == 3
        assert msgs[0] == {"role": "user", "content": "hi"}  # earlier turns untouched
        # the last message carries an ephemeral cache breakpoint
        assert msgs[-1]["content"][0]["text"] == "and now?"
        assert msgs[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_chat_observes_served_model_into_ledger():
    pipe = _pipeline(_ConfidentHaiku())
    result = asyncio.run(pipe.chat(_turn("hi"), _turn("hi"), session_id="s1", policy="cache_aware"))
    assert result["routed"]["model"] == "claude-haiku-4.5"
    # the served model's cache was recorded for this session, so next turn can route warm
    assert len(pipe._cache_ledger._entries) == 1


class _StreamProvider:
    name = "anthropic"

    async def messages(self, request, model, context, on_complete):
        async def gen():
            yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n\n'
            yield 'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n\n'
            on_complete(Usage(input_tokens=100, output_tokens=2), {"http_status": 200})

        return StreamingResponse(gen(), media_type="text/event-stream")


def test_stream_chat_yields_deltas_and_done():
    pipe = _pipeline(_ConfidentHaiku())
    pipe._providers = {"anthropic": _StreamProvider()}

    async def collect():
        return [ev async for ev in pipe.stream_chat(_turn("hi"), _turn("hi"), session_id="s")]

    evs = asyncio.run(collect())
    base = "".join(e["text"] for e in evs if e.get("side") == "baseline" and e["type"] == "delta")
    routed = "".join(e["text"] for e in evs if e.get("side") == "routed" and e["type"] == "delta")
    assert base == "Hello"
    assert routed == "Hello"
    dones = [e for e in evs if e["type"] == "done"]
    assert {e["side"] for e in dones} == {"baseline", "routed"}
    assert any(e["type"] == "meta" and e["side"] == "routed" for e in evs)
    routed_done = next(e for e in dones if e["side"] == "routed")
    assert routed_done["total_ms"] >= routed_done["model_ms"]  # router time included


def _sse(ev):
    return "data: " + __import__("json").dumps(ev) + "\n\n"


class _ToolStreamProvider:
    """Round 1: the model calls web_search; round 2 (once a tool_result is in the
    conversation): it answers. Deterministic per side via conversation state."""
    name = "anthropic"

    async def messages(self, request, model, context, on_complete):
        msgs = request.body.get("messages") or []
        answered = any(
            isinstance(m.get("content"), list)
            and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"])
            for m in msgs
        )

        async def gen():
            if not answered:
                yield _sse({"type": "content_block_start", "index": 0,
                            "content_block": {"type": "tool_use", "id": "t1", "name": "web_search"}})
                yield _sse({"type": "content_block_delta", "index": 0,
                            "delta": {"type": "input_json_delta", "partial_json": '{"query": "weather"}'}})
                yield _sse({"type": "message_delta", "delta": {"stop_reason": "tool_use"}})
                on_complete(Usage(input_tokens=100, output_tokens=10), {"http_status": 200})
            else:
                yield _sse({"type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": "It is sunny"}})
                yield _sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"}})
                on_complete(Usage(input_tokens=50, output_tokens=5), {"http_status": 200})

        return StreamingResponse(gen(), media_type="text/event-stream")


def test_stream_chat_runs_web_search_tool_loop(monkeypatch):
    import emissary_router.demo_search as ds
    calls = []

    async def fake_search(query):
        calls.append(query)
        return "RESULTS for " + query

    monkeypatch.setattr(ds, "web_search", fake_search)

    pipe = _pipeline(_ConfidentHaiku())
    pipe._providers = {"anthropic": _ToolStreamProvider()}

    async def collect():
        return [ev async for ev in pipe.stream_chat(_turn("weather?"), _turn("weather?"), search=True)]

    evs = asyncio.run(collect())
    tool_evs = [e for e in evs if e["type"] == "tool"]
    assert any(e["query"] == "weather" for e in tool_evs)  # tool event surfaced
    assert "weather" in calls  # the search actually ran
    routed_text = "".join(e["text"] for e in evs if e.get("side") == "routed" and e["type"] == "delta")
    assert "sunny" in routed_text  # answer streamed after the tool round
    routed_done = next(e for e in evs if e["type"] == "done" and e["side"] == "routed")
    assert routed_done["searches"] >= 1


def test_stream_endpoint_emits_sse():
    resp = _client().post(
        "/api/demo/stream",
        json={"baseline": [{"role": "user", "content": "hi"}], "routed": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.text
    assert "data:" in body
    assert "claude-haiku-4.5" in body


def _fake_openrouter_httpx(monkeypatch, lines, status=200):
    """Patch demo_stream's httpx so the OpenRouter passthrough reads a canned OpenAI SSE
    stream (line-by-line, like a real upstream) instead of hitting the network."""
    import emissary_router.demo_stream as dstream

    class _Resp:
        status_code = status

        async def aiter_lines(self):
            for line in lines:
                yield line

        async def aread(self):
            return b"error body"

    class _Stream:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, *a, **k):
            return _Stream()

    monkeypatch.setattr(dstream.httpx, "AsyncClient", _Client)


def _openrouter_model():
    from emissary_router.config import AppConfig

    config = AppConfig.model_validate(
        {"models": {"gemini-3.1-flash-lite": True, "claude-sonnet-4.6": True}, "default": "claude-sonnet-4.6"}
    )
    return config.resolve_model("gemini-3.1-flash-lite")  # OpenRouter-only in the catalog


def _or_provider():
    from emissary_router.config import ProviderConfig
    from emissary_router.providers.openrouter import OpenRouterProvider

    return OpenRouterProvider(ProviderConfig(type="openrouter", api_key="k"))


def _drain(provider, model, body):
    import emissary_router.demo_stream as dstream

    async def collect():
        return [ev async for ev in dstream.stream_model(provider, model, body, {})]

    return asyncio.run(collect())


def test_openrouter_passthrough_streams_token_by_token(monkeypatch):
    # Two separate content deltas must arrive as two text events (real streaming), not one
    # buffered blob — this is the whole point of the passthrough.
    _fake_openrouter_httpx(monkeypatch, [
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}',
        "data: [DONE]",
    ])
    evs = _drain(_or_provider(), _openrouter_model(),
                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64})

    text_evs = [e for e in evs if e["type"] == "text"]
    assert [e["text"] for e in text_evs] == ["Hel", "lo"]  # streamed in pieces
    done = evs[-1]
    assert done["type"] == "done"
    assert done["stop_reason"] == "end_turn"
    assert done["usage"].output_tokens == 2
    assert done["status"] == 200


def test_openrouter_passthrough_accumulates_streamed_tool_call(monkeypatch):
    # OpenAI streams tool-call arguments as partial-JSON deltas; the passthrough must
    # reassemble them into one completed tool_call event.
    _fake_openrouter_httpx(monkeypatch, [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"web_search","arguments":"{\\"que"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ry\\":\\"x\\"}"}}]}}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ])
    evs = _drain(_or_provider(), _openrouter_model(),
                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64})

    tool = next(e for e in evs if e["type"] == "tool_call")
    assert tool["name"] == "web_search"
    assert tool["input"] == {"query": "x"}
    assert evs[-1]["stop_reason"] == "tool_use"  # finish_reason tool_calls -> tool_use


def test_openrouter_passthrough_surfaces_upstream_error(monkeypatch):
    _fake_openrouter_httpx(monkeypatch, [], status=400)
    evs = _drain(_or_provider(), _openrouter_model(),
                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64})
    assert len(evs) == 1
    assert evs[0]["type"] == "done" and evs[0]["status"] == 400


class _RecordingPipeline:
    def __init__(self):
        self.kwargs = None

    async def stream_chat(self, baseline_messages, routed_messages, session_id=None, max_tokens=32000, effort=None, policy=None, search=False):
        yield {"side": "routed", "type": "meta", "model": "claude-haiku-4.5", "router_ms": 2}
        yield {"side": "baseline", "type": "delta", "text": "hi"}
        yield {"side": "baseline", "type": "done", "model": "claude-sonnet-4.6", "cost_usd": 0.002,
               "model_ms": 10, "router_ms": 0.0, "total_ms": 10}
        yield {"side": "routed", "type": "done", "model": "claude-haiku-4.5", "cost_usd": 0.0007,
               "model_ms": 8, "router_ms": 2, "total_ms": 10}

    async def chat(self, baseline_messages, routed_messages, session_id=None, max_tokens=32000, effort=None, policy=None):
        self.kwargs = {"baseline": baseline_messages, "routed": routed_messages, "session_id": session_id,
                       "max_tokens": max_tokens, "effort": effort, "policy": policy}
        return {
            "baseline_model": "claude-sonnet-4.6",
            "baseline": {"model": "claude-sonnet-4.6", "answer": "a", "cost_usd": 0.002,
                         "router_ms": 0.0, "model_ms": 10, "total_ms": 10},
            "routed": {"model": "claude-haiku-4.5", "answer": "b", "cost_usd": 0.0007,
                       "router_ms": 2, "model_ms": 8, "total_ms": 10,
                       "route_reason": "deviate_if_confident:p>=0.8"},
            "savings_pct": 65,
        }

    async def chat_side(self, side, baseline_messages, routed_messages, session_id=None, max_tokens=32000, effort=None, policy=None):
        self.kwargs = {"side": side, "baseline": baseline_messages, "routed": routed_messages,
                       "session_id": session_id, "max_tokens": max_tokens, "effort": effort, "policy": policy}
        model = "claude-sonnet-4.6" if side == "baseline" else "claude-haiku-4.5"
        return {"model": model, "answer": side, "cost_usd": 0.001, "router_ms": 0, "model_ms": 5, "total_ms": 5}


def _client(pipeline=None):
    app = FastAPI()
    app.include_router(build_demo_router(auth_key=None))
    app.state.pipeline = pipeline or _RecordingPipeline()
    return TestClient(app)


def test_demo_page_served():
    body = _client().get("/demo").text
    assert "Emissary routed" in body
    assert "/api/demo/chat" in body


def test_chat_endpoint_returns_result():
    resp = _client().post("/api/demo/chat", json={"baseline": [{"role": "user", "content": "hi"}],
                                                  "routed": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    assert resp.json()["savings_pct"] == 65


def test_chat_endpoint_side_runs_one_side():
    rec = _RecordingPipeline()
    client = _client(rec)
    msg = [{"role": "user", "content": "hi"}]
    resp = client.post("/api/demo/chat", json={"baseline": msg, "routed": msg, "side": "baseline"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "claude-sonnet-4.6"  # one side only, no baseline/routed wrapper
    assert rec.kwargs["side"] == "baseline"


def test_set_search_key_writes_env_and_masks(tmp_path, monkeypatch):
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    client = _client()

    r = client.put("/api/demo/search-key", json={"key": "tvly-abcd1234"})
    assert r.status_code == 200
    assert r.json()["set"] is True
    assert r.json()["hint"].endswith("1234")
    assert "tvly-abcd1234" not in r.json()["hint"]  # never echo the full key

    assert os.environ["TAVILY_API_KEY"] == "tvly-abcd1234"  # applied live
    env_text = (tmp_path / ".env").read_text()
    assert "TAVILY_API_KEY=tvly-abcd1234" in env_text  # persisted to .env, not config

    got = client.get("/api/demo/search-key").json()
    assert got["set"] is True
    assert "tvly-abcd1234" not in got["hint"]


def test_set_search_key_rejects_empty_and_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    client = _client()
    assert client.put("/api/demo/search-key", json={"key": ""}).status_code == 400
    assert client.put("/api/demo/search-key", json={"key": "bad\nkey"}).status_code == 400


def test_chat_endpoint_rejects_missing_messages():
    assert _client().post("/api/demo/chat", json={"baseline": [], "routed": []}).status_code == 400
    assert _client().post("/api/demo/chat", json={"routed": [{"role": "user", "content": "hi"}]}).status_code == 400


def test_chat_endpoint_passes_options():
    rec = _RecordingPipeline()
    client = _client(rec)
    msg = [{"role": "user", "content": "hi"}]

    client.post("/api/demo/chat", json={"baseline": msg, "routed": msg, "effort": "medium",
                                        "max_tokens": 64000, "policy": "cache_aware", "session_id": "s1"})
    assert rec.kwargs["effort"] == "medium"
    assert rec.kwargs["max_tokens"] == 64000
    assert rec.kwargs["policy"] == "cache_aware"
    assert rec.kwargs["session_id"] == "s1"

    # invalid effort/policy dropped, oversized max_tokens clamped to the ceiling
    client.post("/api/demo/chat", json={"baseline": msg, "routed": msg, "effort": "bogus",
                                        "max_tokens": 999999, "policy": "bogus"})
    assert rec.kwargs["effort"] is None
    assert rec.kwargs["policy"] is None
    assert rec.kwargs["max_tokens"] == 64000
