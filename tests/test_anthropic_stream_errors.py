"""Streaming requests: an upstream error must come back under its REAL status, never as a
200 whose SSE body is a bare error JSON (Claude Code reads that as an empty/malformed
response and only retries)."""
from __future__ import annotations

import asyncio
import json

from starlette.responses import StreamingResponse

from emissary_router.config import ProviderConfig, ResolvedModel
from emissary_router.providers.anthropic import AnthropicProvider
from emissary_router.schemas import AnthropicRequest, RequestContext

SSE = (
    b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant",'
    b'"model":"claude-haiku-4-5","content":[],"usage":{"input_tokens":12,"cache_read_input_tokens":30,'
    b'"cache_creation_input_tokens":5,"output_tokens":1}}}\n\n'
    b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}\n\n'
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


class FakeResponse:
    def __init__(self, status, body, content_type):
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._body = body
        self.closed = False

    async def aread(self):
        return self._body

    async def aclose(self):
        self.closed = True

    async def aiter_raw(self):
        # two arbitrary chunk boundaries, to exercise the relay's buffering
        yield self._body[:40]
        yield self._body[40:]


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, timeout=None, response=None):
        self.closed = False
        FakeClient.instances.append(self)

    def build_request(self, method, url, headers=None, json=None):
        return {"method": method, "url": url, "headers": headers, "json": json}

    async def send(self, request, stream=False):
        return FakeClient.next_response

    async def aclose(self):
        self.closed = True


def _run(monkeypatch, response):
    FakeClient.next_response = response
    FakeClient.instances = []
    monkeypatch.setattr("emissary_router.providers.anthropic.httpx.AsyncClient", FakeClient)
    completions = []

    async def go():
        provider = AnthropicProvider(ProviderConfig(api_key="k"))
        request = AnthropicRequest(body={"max_tokens": 64000, "stream": True,
                                         "messages": [{"role": "user", "content": "hi"}]}, headers={})
        model = ResolvedModel(name="claude-haiku-4.5", provider="anthropic", model_id="claude-haiku-4-5")
        context = RequestContext(request_id="req", conversation_id=None, classifier_input="", requested_model=None)
        resp = await provider.messages(request, model, context,
                                       on_complete=lambda usage, meta: completions.append((usage, meta)))
        body = b""
        if isinstance(resp, StreamingResponse):
            async for chunk in resp.body_iterator:
                body += chunk
        return resp, body

    resp, body = asyncio.run(go())
    return resp, body, completions


def test_upstream_400_on_a_streaming_request_keeps_its_status_and_body(monkeypatch):
    err = json.dumps({"type": "error", "error": {"type": "invalid_request_error",
                                                 "message": "thinking.enabled.budget_tokens: Input should be >= 1024"}}).encode()
    resp, body, completions = _run(monkeypatch, FakeResponse(400, err, "application/json"))
    assert not isinstance(resp, StreamingResponse)
    assert resp.status_code == 400
    assert resp.media_type == "application/json"
    assert json.loads(resp.body)["error"]["type"] == "invalid_request_error"
    # telemetry: the failed call is recorded with its status, zero usage, the error text
    (usage, meta), = completions
    assert meta["http_status"] == 400 and meta["stream"] is True and "budget_tokens" in meta["error"]
    assert usage.input_tokens == 0 and usage.output_tokens == 0
    # nothing left open
    assert FakeClient.next_response.closed and all(c.closed for c in FakeClient.instances)


def test_healthy_stream_is_relayed_and_usage_recorded(monkeypatch):
    resp, body, completions = _run(monkeypatch, FakeResponse(200, SSE, "text/event-stream"))
    assert isinstance(resp, StreamingResponse)
    assert body == SSE
    (usage, meta), = completions
    assert meta["http_status"] == 200 and meta["error"] is None
    assert (usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens, usage.output_tokens) == (12, 30, 5, 7)
    assert FakeClient.next_response.closed and all(c.closed for c in FakeClient.instances)
