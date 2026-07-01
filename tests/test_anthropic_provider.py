from __future__ import annotations

import asyncio

from emissary_router.config import ProviderConfig, ResolvedModel
from emissary_router.providers.anthropic import AnthropicProvider
from emissary_router.providers.thinking import (
    SYNTHETIC_THINKING_SIGNATURE,
    max_effort_for_model,
    normalize_anthropic_thinking_for_model,
)
from emissary_router.schemas import AnthropicRequest, RequestContext


def test_clamp_max_tokens_to_served_model_ceiling() -> None:
    # A fable-configured Claude Code sends max_tokens=128000; haiku-4.5's ceiling is
    # 64000 (live-verified 400 above it). Clamp must also shrink an explicit thinking
    # budget so budget < max_tokens still holds.
    from emissary_router.providers.thinking import clamp_max_tokens_for_model

    body = {"max_tokens": 128000, "thinking": {"type": "enabled", "budget_tokens": 127999}}
    clamp_max_tokens_for_model(body, "claude-haiku-4.5")
    assert body["max_tokens"] == 64000
    assert body["thinking"]["budget_tokens"] == 63999

    # Under the ceiling: untouched.
    body2 = {"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 31999}}
    clamp_max_tokens_for_model(body2, "claude-haiku-4.5")
    assert body2["max_tokens"] == 32000 and body2["thinking"]["budget_tokens"] == 31999

    # No catalog ceiling (OpenRouter-served models): untouched.
    body3 = {"max_tokens": 128000}
    clamp_max_tokens_for_model(body3, "glm-5.2")
    assert body3["max_tokens"] == 128000


def test_hoist_trailing_system_becomes_positioned_user_reminder() -> None:
    # Trailing role:system reminders must NOT be hoisted to the top-level system field:
    # system precedes messages in the prompt, so a changing reminder would rewrite the
    # prompt head each turn and invalidate the whole prompt-cache prefix. Convert in
    # place to a user message instead; the top-level system stays byte-stable.
    body = {
        "system": [{"type": "text", "text": "main system"}],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "<system-reminder>be concise</system-reminder>"},
        ],
    }

    AnthropicProvider._hoist_system_messages(body)

    assert body["system"] == [{"type": "text", "text": "main system"}]  # head unchanged
    assert [m["role"] for m in body["messages"]] == ["user", "user"]
    reminder = body["messages"][1]["content"][0]["text"]
    assert "be concise" in reminder
    assert reminder.count("<system-reminder") == 1  # already-wrapped content not re-wrapped


def test_hoist_leading_system_message_moves_into_top_level_system() -> None:
    body = {
        "system": "main system",
        "messages": [
            {"role": "system", "content": "extra setup"},
            {"role": "user", "content": "hi"},
        ],
    }

    AnthropicProvider._hoist_system_messages(body)

    assert [m["role"] for m in body["messages"]] == ["user"]
    assert body["system"] == [
        {"type": "text", "text": "main system"},
        {"type": "text", "text": "extra setup"},
    ]


def test_hoist_system_messages_noop_when_no_system_role() -> None:
    import copy

    body = {"system": "s", "messages": [{"role": "user", "content": "hi"}]}
    before = copy.deepcopy(body)
    AnthropicProvider._hoist_system_messages(body)
    assert body == before


def test_strip_synthetic_thinking_removes_only_synthetic_blocks() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "from glm", "signature": SYNTHETIC_THINKING_SIGNATURE},
                    {"type": "text", "text": "hello"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "native", "signature": "real-anthropic-sig"},
                    {"type": "text", "text": "world"},
                ],
            },
        ]
    }

    AnthropicProvider._strip_synthetic_thinking(body)

    assert body["messages"][1]["content"] == [{"type": "text", "text": "hello"}]
    # a genuine Anthropic thinking block (real signature) is preserved
    assert body["messages"][2]["content"][0]["type"] == "thinking"


def test_strip_synthetic_thinking_drops_all_synthetic_message() -> None:
    # A reasoning-only GLM/Kimi turn (e.g. max_tokens exhausted while thinking) leaves an
    # assistant message that is ONLY a synthetic thinking block. It must be dropped
    # entirely: an empty-text placeholder is rejected by Anthropic ("text content blocks
    # must be non-empty"), and consecutive user turns are accepted.
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "x", "signature": SYNTHETIC_THINKING_SIGNATURE}
                ],
            },
            {"role": "user", "content": "continue"},
        ]
    }

    AnthropicProvider._strip_synthetic_thinking(body)

    assert [m["role"] for m in body["messages"]] == ["user", "user"]


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
