from __future__ import annotations

import json

from emissary_router.openai_compat import (
    chat_to_messages,
    messages_response_to_chat,
    _session_id,
)


def _agent_style_request() -> dict:
    """The exact shape the CPST harness agent sends (system + tool loop history)."""
    return {
        "model": "router",
        "messages": [
            {"role": "system", "content": "You are a terminal agent."},
            {"role": "user", "content": "Convert data.csv to parquet."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_terminal_cmd",
                            "arguments": '{"command": "ls"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "data.csv\n"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "run_terminal_cmd",
                    "description": "Run a bash command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }


def test_chat_to_messages_full_tool_loop() -> None:
    out = chat_to_messages(_agent_style_request())

    assert out["stream"] is False
    assert out["max_tokens"] > 0
    # system hoisted with a cache breakpoint
    assert out["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # roles: user, assistant(tool_use), user(tool_result)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "user"]
    tool_use = out["messages"][1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["input"] == {"command": "ls"}
    tool_result = out["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    # last block carries the rolling cache breakpoint
    assert out["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # empty assistant text must NOT become an empty text block (Anthropic rejects it)
    assert all(
        b.get("type") != "text" for b in out["messages"][1]["content"]
    )
    assert out["tools"][0]["input_schema"]["required"] == ["command"]
    assert out["tool_choice"] == {"type": "auto"}


def test_chat_to_messages_malformed_tool_arguments_round_trip() -> None:
    req = _agent_style_request()
    req["messages"][2]["tool_calls"][0]["function"]["arguments"] = '{"command": bro'
    out = chat_to_messages(req)
    assert out["messages"][1]["content"][0]["input"] == {
        "_malformed_json": '{"command": bro'
    }


def test_messages_response_to_chat_tool_use() -> None:
    resp = {
        "id": "msg_abc",
        "model": "deepseek-v4-flash",
        "stop_reason": "tool_use",
        "content": [
            {"type": "thinking", "thinking": "…"},
            {"type": "text", "text": "Listing files."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "run_terminal_cmd",
                "input": {"command": "ls"},
            },
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 50,
        },
    }
    out = messages_response_to_chat(resp, "router")

    assert out["choices"][0]["finish_reason"] == "tool_calls"
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Listing files."
    tc = msg["tool_calls"][0]
    assert tc["id"] == "toolu_1"
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}
    # OpenAI prompt_tokens = full input incl. cached (the provider view)
    assert out["usage"]["prompt_tokens"] == 550
    assert out["usage"]["completion_tokens"] == 20
    assert out["usage"]["total_tokens"] == 570


def test_messages_response_to_chat_plain_completion() -> None:
    resp = {
        "id": "msg_x",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "done"}],
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    out = messages_response_to_chat(resp, "router")
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["choices"][0]["message"]["content"] == "done"
    assert "tool_calls" not in out["choices"][0]["message"]


def test_session_id_stable_within_episode_and_header_wins() -> None:
    body = chat_to_messages(_agent_style_request())
    a = _session_id({}, body)
    b = _session_id({}, body)
    assert a == b and a.startswith("cpst-")
    assert _session_id({"X-Claude-Code-Session-Id": "explicit"}, body) == "explicit"
