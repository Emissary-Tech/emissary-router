from __future__ import annotations

from emissary_router.config import ProviderConfig
from emissary_router.providers.google import GoogleProvider


def test_google_request_maps_anthropic_tools_and_results() -> None:
    body = {
        "system": "You are Claude Code.",
        "max_tokens": 1024,
        "temperature": 0.1,
        "thinking": {"effort": "max"},
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                    "additionalProperties": False,
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
                    }
                ],
            },
        ],
    }

    req = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert req["systemInstruction"]["parts"][0]["text"] == "You are Claude Code."
    assert req["contents"][1]["role"] == "model"
    assert req["contents"][1]["parts"][0]["functionCall"]["name"] == "Read"
    assert req["contents"][2]["parts"][0]["functionResponse"]["name"] == "Read"
    assert req["tools"][0]["functionDeclarations"][0]["name"] == "Read"
    assert "additionalProperties" not in req["tools"][0]["functionDeclarations"][0]["parameters"]
    assert req["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"


def test_google_response_maps_tool_call_and_cache_usage() -> None:
    payload = {
        "responseId": "gemini-response-1",
        "modelVersion": "gemini-3.1-flash-lite",
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "Need to inspect."},
                        {"functionCall": {"name": "Read", "args": {"file_path": "main.py"}}},
                    ]
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1000,
            "cachedContentTokenCount": 700,
            "candidatesTokenCount": 33,
            "thoughtsTokenCount": 12,
        },
    }

    usage = GoogleProvider(ProviderConfig(type="google")).usage_from_response(payload)
    message = GoogleProvider.from_google_response(payload, "gemini-3.1-flash-lite")

    assert usage.input_tokens == 300
    assert usage.cache_read_input_tokens == 700
    assert usage.output_tokens == 45
    assert usage.reasoning_output_tokens == 12
    assert message["usage"]["input_tokens"] == 300
    assert message["usage"]["cache_read_input_tokens"] == 700
    assert message["usage"]["output_tokens"] == 45
    assert message["stop_reason"] == "tool_use"
    assert message["content"][1]["type"] == "tool_use"
    assert message["content"][1]["name"] == "Read"


def test_google_anthropic_sse_contains_cache_usage() -> None:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "gemini-3.1-flash-lite",
        "content": [{"type": "text", "text": "done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 300,
            "output_tokens": 5,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 0,
        },
    }

    sse = b"".join(GoogleProvider.anthropic_sse(message)).decode()

    assert "message_start" in sse
    assert '"cache_read_input_tokens": 700' in sse
    assert '"output_tokens": 5' in sse


def test_google_thinking_effort_mapping_preserves_minimal() -> None:
    body = {"thinking": {"effort": "minimal"}}

    gemini3 = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert gemini3["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"


def test_google_thinking_budget_maps_to_level() -> None:
    body = {"thinking": {"type": "enabled", "budget_tokens": 2048}}

    gemini3 = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert gemini3["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"


def test_google_disabled_thinking_maps_to_minimal() -> None:
    body = {"thinking": {"type": "disabled"}}

    gemini3 = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert gemini3["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"
