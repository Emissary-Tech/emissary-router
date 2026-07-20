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


def test_hoist_prompt_prefix_is_stable_across_turns_with_new_reminders() -> None:
    # Cache-prefix regression guard: the top-level system must stay byte-identical and
    # turn N's converted messages must be an exact prefix of turn N+1's, even when a
    # NEW per-turn reminder appears. Anthropic hashes tools -> system -> messages, so
    # any head change would invalidate every cache breakpoint in the conversation.
    def body(messages):
        return {"system": [{"type": "text", "text": "main system"}], "messages": list(messages)}

    turn1_msgs = [
        {"role": "user", "content": "fix the bug"},
        {"role": "system", "content": "<system-reminder>R1</system-reminder>"},
    ]
    turn2_msgs = turn1_msgs + [
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        {"role": "user", "content": "now add tests"},
        {"role": "system", "content": "<system-reminder>R2</system-reminder>"},
    ]
    b1, b2 = body(turn1_msgs), body(turn2_msgs)
    AnthropicProvider._hoist_system_messages(b1)
    AnthropicProvider._hoist_system_messages(b2)

    assert b1["system"] == b2["system"] == [{"type": "text", "text": "main system"}]
    assert b2["messages"][: len(b1["messages"])] == b1["messages"]


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


def test_strip_removes_gemini_thought_signature_from_tool_use() -> None:
    # Anthropic 400s "Extra inputs are not permitted" on unknown tool_use keys
    # (live-verified); the Gemini signature preserved for gemini->gemini replay must be
    # stripped when the turn routes back to Anthropic.
    body = {
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "get_weather",
                 "input": {"city": "Paris"}, "thought_signature": "sig-abc"},
            ]},
        ]
    }
    AnthropicProvider._strip_synthetic_thinking(body)
    block = body["messages"][0]["content"][0]
    assert "thought_signature" not in block
    assert block["name"] == "get_weather" and block["input"] == {"city": "Paris"}


def test_sanitize_tool_id_is_deterministic_and_pattern_valid() -> None:
    import re

    from emissary_router.providers.base import sanitize_tool_id

    # valid ids pass through untouched
    assert sanitize_tool_id("toolu_abc123") == "toolu_abc123"
    assert sanitize_tool_id(None) is None
    # kimi-style id becomes pattern-valid, deterministically
    bad = "functions.get_weather:0"
    fixed = sanitize_tool_id(bad)
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", fixed)
    assert fixed == sanitize_tool_id(bad)  # same input -> same output, always
    # near-collisions stay distinct (hash suffix)
    assert sanitize_tool_id("functions.get_weather.0") != fixed


def test_strip_sanitizes_foreign_tool_ids_and_keeps_pairing() -> None:
    # Kimi emits tool ids like "functions.get_weather:0"; Anthropic 400s on them
    # (String should match pattern '^[a-zA-Z0-9_-]+$'). Both the tool_use id and the
    # matching tool_result tool_use_id must map to the SAME sanitized id.
    from emissary_router.providers.base import sanitize_tool_id

    bad = "functions.get_weather:0"
    body = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": bad, "name": "get_weather", "input": {"city": "Paris"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": bad, "content": "sunny"},
            ]},
        ]
    }
    AnthropicProvider._strip_synthetic_thinking(body)
    fixed = sanitize_tool_id(bad)
    assert body["messages"][1]["content"][0]["id"] == fixed
    assert body["messages"][2]["content"][0]["tool_use_id"] == fixed  # pairing preserved


# --- Sonnet 5-era xhigh effort: served Claude models must get a level they accept ---


def test_xhigh_effort_clamps_to_max_for_sonnet_and_drops_for_haiku() -> None:
    from emissary_router.providers.thinking import normalize_anthropic_thinking_for_model

    def sonnet5_body() -> dict:
        # Exact Claude Code 2.1.198+ shape when the user selects xhigh on Sonnet 5.
        return {
            "model": "claude-sonnet-5",
            "max_tokens": 64000,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "xhigh"},
            "messages": [{"role": "user", "content": "hi"}],
        }

    # sonnet-4.6 rejects xhigh outright (live 400: "This model does not support
    # effort level 'xhigh'. Supported levels: high, low, max, medium") -> clamp to max.
    body = sonnet5_body()
    changes = normalize_anthropic_thinking_for_model(body, "claude-sonnet-4.6")
    assert body["output_config"]["effort"] == "max"
    assert body["thinking"] == {"type": "adaptive"}  # adaptive is supported; keep it
    assert "output_config.effort=xhigh->max" in changes

    # haiku accepts neither adaptive nor effort -> budget conversion + drop.
    body = sonnet5_body()
    normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5")
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 63999}
    assert "effort" not in body["output_config"]


def test_empty_signature_thinking_is_stripped_like_synthetic() -> None:
    # A thinking block with no/empty signature can never be a valid Anthropic replay
    # (native thinking is always signed): it means a foreign provider's block lost its
    # marker — e.g. Claude Code only collects signatures from signature_delta events,
    # so a provider that carried it on content_block_start yields signature "".
    # Live-observed 400: "Invalid `signature` in `thinking` block".
    from emissary_router.providers.anthropic import AnthropicProvider

    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "leaked", "signature": ""},
                {"type": "text", "text": "hello"},
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "leaked2"},  # signature key absent
                {"type": "text", "text": "again"},
            ]},
        ]
    }
    AnthropicProvider._strip_synthetic_thinking(body)
    for message in body["messages"][1:]:
        assert [b["type"] for b in message["content"]] == ["text"]
