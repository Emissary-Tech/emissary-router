from __future__ import annotations

import asyncio

from emissary_router.config import ProviderConfig, ResolvedModel
from emissary_router.providers.anthropic import AnthropicProvider
from emissary_router.providers.thinking import (
    max_effort_for_model,
    normalize_anthropic_thinking_for_model,
)
from emissary_router.schemas import AnthropicRequest, RequestContext


def test_anthropic_sse_usage_parses_cache_tokens() -> None:
    sse = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"usage":{'
        '"input_tokens":10,'
        '"cache_read_input_tokens":9000,'
        '"cache_creation_input_tokens":500,'
        '"output_tokens":0}}}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","usage":{"output_tokens":7}}\n\n'
    )

    usage = AnthropicProvider._usage_from_sse(sse)

    assert usage.input_tokens == 10
    assert usage.cache_read_input_tokens == 9000
    assert usage.cache_creation_input_tokens == 500
    assert usage.output_tokens == 7


def test_strip_cch_attribution_from_system_text() -> None:
    body = {
        "system": "stable prefix\nClaude Code attribution cch=abc123\nmore stable prefix",
    }

    AnthropicProvider._strip_cch_attribution(body)

    assert "cch=" not in body["system"]
    assert "stable prefix" in body["system"]
    assert "more stable prefix" in body["system"]


def test_model_thinking_capabilities_expose_max_effort() -> None:
    assert max_effort_for_model("claude-sonnet-4.6") == "max"
    assert max_effort_for_model("gemini-3.1-flash-lite") == "high"
    assert max_effort_for_model("unknown-model") == "high"


def test_haiku_adaptive_thinking_maps_to_enabled_budget() -> None:
    body = {
        "max_tokens": 4096,
        "thinking": {"type": "adaptive", "effort": "max"},
        "effort": "max",
    }

    changes = normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5")

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4095}
    assert "effort" not in body
    assert changes == [
        "thinking=adaptive->enabled(budget=4095)",
        "effort=max dropped",
    ]


def test_sonnet_adaptive_thinking_is_preserved_and_effort_is_clamped() -> None:
    body = {
        "max_tokens": 4096,
        "thinking": {"type": "adaptive", "effort": "xhigh"},
        "reasoning_effort": "xhigh",
    }

    changes = normalize_anthropic_thinking_for_model(body, "claude-sonnet-4.6")

    assert body["thinking"] == {"type": "adaptive", "effort": "max"}
    assert body["reasoning_effort"] == "max"
    assert changes == [
        "thinking.effort=xhigh->max",
        "reasoning_effort=xhigh->max",
    ]


def test_anthropic_provider_sanitizes_haiku_adaptive_before_send(monkeypatch) -> None:
    captured: dict = {}

    class FakeResponse:
        status_code = 200
        headers: dict = {}
        content = b"{}"
        text = "{}"

        def json(self) -> dict:
            return {"usage": {}}

    class FakeClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, headers, json):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("emissary_router.providers.anthropic.httpx.AsyncClient", FakeClient)

    async def run() -> None:
        provider = AnthropicProvider(ProviderConfig(api_key="k"))
        request = AnthropicRequest(
            body={
                "max_tokens": 4096,
                "messages": [],
                "thinking": {"type": "adaptive", "effort": "max"},
            },
            headers={},
        )
        model = ResolvedModel(
            name="claude-haiku-4.5",
            provider="anthropic",
            model_id="claude-haiku-4-5",
        )
        context = RequestContext(
            request_id="req",
            conversation_id=None,
            classifier_input="",
            requested_model=None,
        )

        await provider.messages(request, model, context)

    asyncio.run(run())

    assert captured["json"]["model"] == "claude-haiku-4-5"
    assert captured["json"]["thinking"] == {"type": "enabled", "budget_tokens": 4095}
