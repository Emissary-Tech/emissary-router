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


def test_openrouter_anthropic_sse_contains_cache_usage() -> None:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "gemini-3.1-flash-lite",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 200,
            "output_tokens": 5,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 100,
        },
    }

    sse = b"".join(OpenRouterProvider.anthropic_sse(message)).decode()

    assert "message_start" in sse
    assert '"cache_read_input_tokens": 700' in sse
    assert '"cache_creation_input_tokens": 100' in sse
    assert '"output_tokens": 5' in sse


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
