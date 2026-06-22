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

    req = OpenRouterProvider.to_openai_request(body, "anthropic/claude-sonnet-4.6")

    assert req["messages"][0] == {"role": "system", "content": "You are Claude Code."}
    assert req["messages"][2]["tool_calls"][0]["function"]["name"] == "Read"
    assert req["messages"][3]["role"] == "tool"
    assert req["messages"][4]["role"] == "user"
    assert req["tools"][0]["function"]["name"] == "Read"
    assert req["reasoning"]["effort"] == "high"
    assert req["cache_control"] == {"type": "ephemeral"}


def test_openrouter_reasoning_effort_preserves_minimal() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"effort": "minimal"}},
        "google/gemini-3.1-flash-lite",
    )

    assert req["reasoning"]["effort"] == "minimal"


def test_openrouter_reasoning_budget_maps_to_max_tokens() -> None:
    req = OpenRouterProvider.to_openai_request(
        {"messages": [], "thinking": {"type": "enabled", "budget_tokens": 4096}},
        "anthropic/claude-sonnet-4.6",
    )

    assert req["reasoning"]["max_tokens"] == 4096


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
