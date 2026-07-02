from __future__ import annotations

from emissary_router.config import ProviderConfig
from emissary_router.providers.openrouter import OpenRouterProvider


def test_openrouter_request_maps_tools_reasoning_and_cache_control() -> None:
    body = {
        "system": "You are Claude Code.",
        "max_tokens": 2048,
        "thinking": {"effort": "max"},
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Open a file"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "main.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "print('hi')",
                    },
                    {"type": "text", "text": "Summarize it"},
                ],
            },
        ],
    }

    req = OpenRouterProvider.to_openai_request(
        body,
        "anthropic/claude-sonnet-4.6",
        model_name="claude-sonnet-4.6",
    )

    assert req["messages"][0] == {"role": "system", "content": "You are Claude Code."}
    assert req["messages"][2]["tool_calls"][0]["function"]["name"] == "Read"
    assert req["messages"][3]["role"] == "tool"
    assert req["messages"][4]["role"] == "user"
    assert req["tools"][0]["function"]["name"] == "Read"
    assert req["reasoning"]["effort"] == "xhigh"
    assert req["cache_control"] == {"type": "ephemeral"}


def test_openrouter_reasoning_effort_preserves_minimal() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"effort": "minimal"}},
        "google/gemini-3.1-flash-lite",
    )

    assert req["reasoning"]["effort"] == "minimal"


def test_openrouter_reasoning_effort_clamps_from_model_capability() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"effort": "xhigh"}},
        "google/gemini-3.1-flash-lite",
        model_name="gemini-3.1-flash-lite",
    )

    assert req["reasoning"]["effort"] == "high"


def test_openrouter_sonnet_max_effort_maps_to_xhigh() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"effort": "max"}},
        "anthropic/claude-sonnet-4.6",
        model_name="claude-sonnet-4.6",
    )

    assert req["reasoning"]["effort"] == "xhigh"


def test_openrouter_haiku_effort_maps_to_budget_from_output_max_tokens() -> None:
    req = OpenRouterProvider.to_openai_request(
        {
            "messages": [],
            "max_tokens": 32000,
            "thinking": {"effort": "max"},
        },
        "anthropic/claude-haiku-4.5",
        model_name="claude-haiku-4.5",
    )

    assert req["reasoning"]["max_tokens"] == 31999
    assert "effort" not in req["reasoning"]


def test_openrouter_glm_max_effort_maps_to_xhigh() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"effort": "max"}},
        "z-ai/glm-5.2",
        model_name="glm-5.2",
    )
    assert req["reasoning"]["effort"] == "xhigh"


def test_openrouter_glm_high_effort_preserved() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"effort": "high"}},
        "z-ai/glm-5.2",
        model_name="glm-5.2",
    )
    assert req["reasoning"]["effort"] == "high"


def test_openrouter_kimi_max_effort_maps_to_xhigh() -> None:
    # Kimi routes via the effort param; max maps to xhigh (full reasoning budget).
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "max_tokens": 32000, "thinking": {"effort": "max"}},
        "moonshotai/kimi-k2.7-code",
        model_name="kimi-k2.7-code",
    )
    assert req["reasoning"]["effort"] == "xhigh"
    assert "max_tokens" not in req["reasoning"]


def test_openrouter_kimi_disabled_thinking_omits_reasoning() -> None:
    # Kimi always reasons; OpenRouter 400s on effort:none ("Reasoning is mandatory ...
    # cannot be disabled"). So a disable request must omit reasoning entirely.
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"type": "disabled"}},
        "moonshotai/kimi-k2.7-code",
        model_name="kimi-k2.7-code",
    )
    assert "reasoning" not in req


def test_openrouter_disable_thinking_still_sends_none_for_normal_models() -> None:
    # GLM (and others that can disable) still send the valid effort:none.
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"type": "disabled"}},
        "z-ai/glm-5.2",
        model_name="glm-5.2",
    )
    assert req["reasoning"] == {"effort": "none"}


def test_openrouter_reasoning_budget_maps_to_max_tokens() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 4096}},
        "anthropic/claude-sonnet-4.6",
    )

    assert req["reasoning"]["max_tokens"] == 4096


def test_openrouter_gemini_budget_still_maps_to_max_tokens() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 4096}},
        "google/gemini-3.1-flash-lite",
        model_name="gemini-3.1-flash-lite",
    )

    assert req["reasoning"]["max_tokens"] == 4096
    assert "effort" not in req["reasoning"]


def test_openrouter_output_config_effort_takes_precedence_over_budget() -> None:
    req = OpenRouterProvider.to_openai_request(
        {
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "output_config": {"effort": "high"},
        },
        "google/gemini-3.1-flash-lite",
        model_name="gemini-3.1-flash-lite",
    )

    assert req["reasoning"]["effort"] == "high"
    assert "max_tokens" not in req["reasoning"]


def test_openrouter_haiku_output_config_effort_maps_to_budget() -> None:
    req = OpenRouterProvider.to_openai_request(
        {
            "messages": [],
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 100},
            "output_config": {"effort": "high"},
        },
        "anthropic/claude-haiku-4.5",
        model_name="claude-haiku-4.5",
    )

    assert req["reasoning"]["max_tokens"] == 4095
    assert "effort" not in req["reasoning"]


def test_openrouter_disabled_thinking_keeps_reasoning_none() -> None:
    req = OpenRouterProvider.to_openai_request(
        {
            "messages": [],
            "max_tokens": 4096,
            "thinking": {"type": "disabled"},
        },
        "anthropic/claude-haiku-4.5",
        model_name="claude-haiku-4.5",
    )

    assert req["reasoning"] == {"effort": "none"}


def test_openrouter_explicit_reasoning_config_is_preserved() -> None:
    req = OpenRouterProvider.to_openai_request(
        {
            "messages": [],
            "reasoning": {"effort": "xhigh", "exclude": True},
        },
        "google/gemini-3.1-flash-lite",
    )

    assert req["reasoning"] == {"effort": "high", "exclude": True}


def test_openrouter_response_maps_tool_call_and_cache_usage() -> None:
    payload = {
        "id": "chatcmpl_1",
        "model": "google/gemini-3.1-flash-lite",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Need to inspect.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "Read",
                                "arguments": '{"file_path":"main.py"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 25,
            "prompt_tokens_details": {
                "cached_tokens": 700,
                "cache_write_tokens": 100,
            },
            "output_tokens_details": {"reasoning_tokens": 9},
        },
    }

    usage = OpenRouterProvider(ProviderConfig(type="openrouter")).usage_from_response(payload)
    message = OpenRouterProvider.from_openai_response(payload, "gemini-3.1-flash-lite")

    assert usage.input_tokens == 200
    assert usage.cache_read_input_tokens == 700
    assert usage.cache_creation_input_tokens == 100
    assert usage.reasoning_output_tokens == 9
    assert message["usage"]["input_tokens"] == 200
    assert message["usage"]["cache_read_input_tokens"] == 700
    assert message["usage"]["cache_creation_input_tokens"] == 100
    assert message["stop_reason"] == "tool_use"
    assert message["content"][1]["type"] == "tool_use"


def test_reasoning_text_prefers_flat_then_details_no_dup() -> None:
    from emissary_router.providers.openrouter import _reasoning_text

    assert _reasoning_text({"reasoning": "flat"}) == "flat"
    assert _reasoning_text({"reasoning_details": [{"type": "reasoning.text", "text": "detailed"}]}) == "detailed"
    # both present -> flat wins, no duplication
    assert _reasoning_text({"reasoning": "flat", "reasoning_details": [{"text": "detailed"}]}) == "flat"
    # signature-only detail (no text) -> nothing
    assert _reasoning_text({"reasoning_details": [{"type": "reasoning.text", "signature": "sig"}]}) == ""
    assert _reasoning_text({}) == ""


def test_from_openai_response_surfaces_reasoning_as_thinking() -> None:
    payload = {
        "choices": [
            {"message": {"reasoning": "glm thought", "content": "the answer"}, "finish_reason": "stop"}
        ]
    }
    msg = OpenRouterProvider.from_openai_response(payload, "glm-5.2")
    assert [b["type"] for b in msg["content"]] == ["thinking", "text"]
    assert msg["content"][0]["thinking"] == "glm thought"
    assert msg["content"][0]["signature"] == "emissary:non-anthropic-reasoning"
    assert msg["content"][1]["text"] == "the answer"


def test_openrouter_trailing_system_becomes_positioned_user_reminder() -> None:
    # Mid/trailing role:system messages must NOT fold into the leading system message:
    # that would rewrite the prompt head each turn and bust the prefix-matched prompt
    # cache. They convert in place to a user message; content is preserved either way.
    body = {
        "system": "main system",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "<system-reminder>be concise</system-reminder>"},
        ],
    }
    msgs = OpenRouterProvider._messages(body)
    assert [m["role"] for m in msgs] == ["system", "user", "user"]
    assert msgs[0]["content"] == "main system"  # head unchanged -> cache prefix stable
    assert "be concise" in msgs[2]["content"]  # reminder preserved at its position
    assert msgs[2]["content"].count("<system-reminder") == 1  # no double-wrap


def test_openrouter_prompt_prefix_is_stable_across_turns_with_new_reminders() -> None:
    # Cache-prefix regression guard: turn N's converted messages must be an exact
    # prefix of turn N+1's, even when a NEW per-turn reminder appears — otherwise the
    # provider's prefix-matched prompt cache misses the whole conversation each turn.
    turn1 = {
        "system": "main system",
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {"role": "system", "content": "<system-reminder>R1</system-reminder>"},
        ],
    }
    turn2 = {
        "system": "main system",
        "messages": turn1["messages"] + [
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            {"role": "user", "content": "now add tests"},
            {"role": "system", "content": "<system-reminder>R2</system-reminder>"},
        ],
    }
    m1 = OpenRouterProvider._messages(turn1)
    m2 = OpenRouterProvider._messages(turn2)
    assert m2[: len(m1)] == m1  # byte-identical prefix; new content only appended


def test_openrouter_leading_system_message_folds_into_system() -> None:
    body = {
        "system": "main system",
        "messages": [
            {"role": "system", "content": "extra setup"},
            {"role": "user", "content": "hi"},
        ],
    }
    msgs = OpenRouterProvider._messages(body)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "main system" in msgs[0]["content"] and "extra setup" in msgs[0]["content"]


def test_openrouter_stream_translation_reasoning_text_and_tools() -> None:
    import asyncio
    import json as _json

    chunks = [
        {"choices": [{"delta": {"role": "assistant", "reasoning": "thinking..."}}]},
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "Read", "arguments": '{"p":'}}
        ]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "1}"}}
        ]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 7}},
    ]
    lines = ["data: " + _json.dumps(c) for c in chunks] + ["data: [DONE]"]

    async def aiter():
        for line in lines:
            yield line

    async def run():
        sink: dict = {}
        events = []
        async for ev in OpenRouterProvider._iter_anthropic_events(aiter(), "kimi-k2.7-code", sink):
            events.append(ev.decode())
        return "".join(events), sink

    text, sink = asyncio.run(run())

    assert "event: message_start" in text
    # reasoning surfaced as a thinking block (parity with native models), closed with the
    # synthetic signature the Anthropic provider strips on later turns
    assert '"type": "thinking"' in text
    assert '"thinking_delta"' in text and '"thinking": "thinking..."' in text
    assert '"signature_delta"' in text and "emissary:non-anthropic-reasoning" in text
    assert '"type": "text"' in text and '"text": "Hel"' in text and '"text": "lo"' in text
    assert '"type": "tool_use"' in text and '"name": "Read"' in text
    assert '"input_json_delta"' in text
    assert '"stop_reason": "tool_use"' in text
    assert "event: message_stop" in text
    assert sink["usage"]["completion_tokens"] == 7
    # the final message_delta backfills CUMULATIVE usage so the client sees real
    # input tokens (message_start had to carry zeros — usage arrives at stream end)
    delta_part = text.split('"type": "message_delta"', 1)[1]
    assert '"input_tokens": 12' in delta_part
    assert '"output_tokens": 7' in delta_part


def _run_translation(chunks):
    import asyncio
    import json as _json

    lines = ["data: " + _json.dumps(c) for c in chunks] + ["data: [DONE]"]

    async def aiter():
        for line in lines:
            yield line

    async def run():
        sink: dict = {}
        events = []
        async for ev in OpenRouterProvider._iter_anthropic_events(aiter(), "m", sink):
            events.append(_json.loads(ev.decode().split("data: ", 1)[1]))
        return events, sink

    return asyncio.run(run())


def test_interleaved_parallel_tool_calls_keep_their_blocks() -> None:
    # OpenAI deltas may interleave fragments of parallel tool calls (0/1/0/1);
    # continuations carry no id/name. Each fragment must route to ITS block — never
    # fabricate an "unknown_tool" block with split JSON.
    events, _ = _run_translation([
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "function": {"name": "Read", "arguments": '{"p":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "id": "call_b", "function": {"name": "Write", "arguments": '{"q":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '1}'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "function": {"arguments": '2}'}}]}, "finish_reason": "tool_calls"}]},
    ])

    starts = {e["index"]: e["content_block"] for e in events if e["type"] == "content_block_start"}
    names = [b["name"] for b in starts.values() if b["type"] == "tool_use"]
    assert sorted(names) == ["Read", "Write"]  # exactly two tools, no unknown_tool
    read_idx = next(i for i, b in starts.items() if b.get("name") == "Read")
    write_idx = next(i for i, b in starts.items() if b.get("name") == "Write")
    args = {read_idx: "", write_idx: ""}
    for e in events:
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
            args[e["index"]] += e["delta"]["partial_json"]
    assert args[read_idx] == '{"p":1}'  # JSON intact per tool
    assert args[write_idx] == '{"q":2}'


def test_reasoning_between_tool_fragments_does_not_orphan_the_tool() -> None:
    events, _ = _run_translation([
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_a", "function": {"name": "Read", "arguments": '{"p":'}}]}}]},
        {"choices": [{"delta": {"reasoning": "hmm"}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '1}'}}]}, "finish_reason": "tool_calls"}]},
    ])
    tool_starts = [e for e in events if e["type"] == "content_block_start"
                   and e["content_block"]["type"] == "tool_use"]
    assert len(tool_starts) == 1 and tool_starts[0]["content_block"]["name"] == "Read"
    joined = "".join(e["delta"]["partial_json"] for e in events
                     if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
    assert joined == '{"p":1}'


def test_openrouter_stream_records_usage_before_trailing_events() -> None:
    # A cancel/drop after the usage chunk but before message_stop must still record
    # real usage in sink (telemetry), not zeros.
    import asyncio
    import json as _json

    chunks = [
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 9}},
    ]
    lines = ["data: " + _json.dumps(c) for c in chunks]  # no [DONE]: stream severed early

    async def aiter():
        for line in lines:
            yield line
        raise RuntimeError("connection reset mid-stream")

    async def run():
        sink: dict = {}
        try:
            async for _ in OpenRouterProvider._iter_anthropic_events(aiter(), "glm-5.2", sink):
                pass
        except RuntimeError:
            pass
        return sink

    sink = asyncio.run(run())
    assert sink["usage"]["prompt_tokens"] == 40  # recorded the moment it arrived


def test_openrouter_strips_cch_attribution_from_system() -> None:
    # The dynamic cch= line changes per request; left in, it busts the provider's
    # prefix cache AND diverges from the ledger's prefix_hash (which is computed on the
    # stripped system). The OpenRouter path must strip it like the Anthropic path does.
    from emissary_router.routing.cache_cost import extract_request_cost_features

    body = {
        "system": "stable prefix\nClaude Code attribution cch=abc123\nmore prefix",
        "messages": [{"role": "user", "content": "hi"}],
    }
    msgs = OpenRouterProvider._messages(body)
    assert "cch=" not in msgs[0]["content"]
    assert "stable prefix" in msgs[0]["content"] and "more prefix" in msgs[0]["content"]

    # hash/sender consistency: two requests differing only in the cch value produce the
    # SAME prefix_hash and the SAME outbound system text
    body2 = {**body, "system": body["system"].replace("cch=abc123", "cch=zzz999")}
    f1 = extract_request_cost_features(body, {})
    f2 = extract_request_cost_features(body2, {})
    assert f1.prefix_hash == f2.prefix_hash
    assert OpenRouterProvider._messages(body2)[0]["content"] == msgs[0]["content"]


# --- context-overflow error translation (Claude Code self-recovery depends on it) ---


def test_overflow_error_translated_to_anthropic_shape() -> None:
    from emissary_router.providers.openrouter import anthropic_error_payload

    # Exact body OpenRouter returns when the request exceeds the endpoint's window
    # (captured live; identical shape for kimi and anthropic models via OpenRouter).
    payload = {
        "error": {
            "message": (
                "This endpoint's maximum context length is 262144 tokens. However, "
                "you requested about 297572 tokens (297508 of text input, 64 in the "
                "output). Please reduce the length of either one, or use the "
                "context-compression plugin to compress your prompt automatically."
            ),
            "code": 400,
            "metadata": {"provider_name": None},
        }
    }

    translated = anthropic_error_payload(payload)

    # Claude Code only triggers its automatic context recovery (truncate old tool
    # results + retry) on Anthropic's "prompt is too long" invalid_request_error.
    assert translated["type"] == "error"
    assert translated["error"]["type"] == "invalid_request_error"
    assert translated["error"]["message"] == "prompt is too long: 297572 tokens > 262144 maximum"


def test_overflow_translation_reads_anthropic_raw_in_metadata() -> None:
    from emissary_router.providers.openrouter import anthropic_error_payload

    payload = {
        "error": {
            "message": "Provider returned error",
            "code": 400,
            "metadata": {
                "raw": '{"type":"error","error":{"type":"invalid_request_error",'
                '"message":"prompt is too long: 228501 tokens > 200000 maximum"}}',
                "provider_name": "Anthropic",
            },
        }
    }

    translated = anthropic_error_payload(payload)
    assert translated["error"]["message"] == "prompt is too long: 228501 tokens > 200000 maximum"
    assert translated["error"]["type"] == "invalid_request_error"


def test_non_overflow_errors_pass_through_unchanged() -> None:
    from emissary_router.providers.openrouter import anthropic_error_payload

    payload = {
        "error": {
            "message": "Invalid schema for function 'Read': required is not of type array",
            "code": 400,
        }
    }

    assert anthropic_error_payload(payload) is payload
