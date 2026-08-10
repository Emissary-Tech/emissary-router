"""OpenAI Responses provider: anthropic <-> Responses translation."""
import asyncio
import json

from emissary_router.providers.openai import OpenAIProvider


def _req(**overrides):
    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 32000,
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Seoul"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"},
                    {"type": "text", "text": "and tomorrow?"},
                ],
            },
        ],
        "tools": [{"name": "get_weather", "description": "w", "input_schema": {"type": "object"}}],
        "temperature": 1,
    }
    body.update(overrides)
    return body


def test_request_translation_shape() -> None:
    out = OpenAIProvider.to_responses_request(_req(), "gpt-5.6-luna", "gpt-5.6-luna")
    assert out["model"] == "gpt-5.6-luna"
    assert out["instructions"] == "You are helpful."
    assert out["max_output_tokens"] == 32000
    assert out["store"] is False
    assert "temperature" not in out  # gpt-5.x rejects sampling params
    kinds = [i.get("type") or i.get("role") for i in out["input"]]
    assert kinds == ["user", "assistant", "function_call", "function_call_output", "user"]
    call = out["input"][2]
    assert call["call_id"] == "toolu_1" and call["name"] == "get_weather"
    assert json.loads(call["arguments"]) == {"city": "Seoul"}
    result = out["input"][3]
    assert result["call_id"] == "toolu_1" and result["output"] == "sunny"
    # thinking blocks are never replayed upstream
    assert not any("thinking" in json.dumps(i) for i in out["input"])


def test_tools_are_flat_responses_shape() -> None:
    out = OpenAIProvider.to_responses_request(_req(), "gpt-5.6-sol", "gpt-5.6-sol")
    tool = out["tools"][0]
    assert tool["type"] == "function" and tool["name"] == "get_weather"
    assert "function" not in tool  # no chat-completions nesting


def test_reasoning_effort_mapping() -> None:
    body = _req(output_config={"effort": "max"})
    out = OpenAIProvider.to_responses_request(body, "gpt-5.6-luna", "gpt-5.6-luna")
    assert out["reasoning"] == {"effort": "max", "summary": "auto"}  # max is native on gpt-5.6
    body_min = _req(output_config={"effort": "minimal"})
    out_min = OpenAIProvider.to_responses_request(body_min, "gpt-5.6-luna", "gpt-5.6-luna")
    assert out_min["reasoning"]["effort"] == "low"  # minimal doesn't exist -> snaps to low
    # no explicit intent -> omit (provider default reasoning)
    out2 = OpenAIProvider.to_responses_request(_req(), "gpt-5.6-luna", "gpt-5.6-luna")
    assert "reasoning" not in out2
    # gpt-5.6 supports effort "none" -> a disable request maps to it
    body3 = _req(thinking={"type": "disabled"})
    out3 = OpenAIProvider.to_responses_request(body3, "gpt-5.6-luna", "gpt-5.6-luna")
    assert out3["reasoning"]["effort"] == "none"


def test_response_translation() -> None:
    payload = {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "answer"}]},
            {"type": "function_call", "call_id": "call_9", "name": "get_weather",
             "arguments": "{\"city\": \"Seoul\"}"},
        ],
        "usage": {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 800},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 30},
        },
    }
    msg = OpenAIProvider.from_responses_payload(payload, "gpt-5.6-luna")
    types = [b["type"] for b in msg["content"]]
    assert types == ["thinking", "text", "tool_use"]
    assert msg["stop_reason"] == "tool_use"
    assert msg["usage"] == {
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_read_input_tokens": 800,
        "cache_creation_input_tokens": 0,
    }


def test_budget_exhaustion_maps_to_max_tokens() -> None:
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "partial"}]}],
        "usage": {},
    }
    msg = OpenAIProvider.from_responses_payload(payload, "gpt-5.6-sol")
    assert msg["stop_reason"] == "max_tokens"


def test_stream_translation() -> None:
    events = [
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"type": "reasoning"}},
        {"type": "response.reasoning_summary_text.delta", "delta": "th"},
        {"type": "response.output_text.delta", "delta": "hi"},
        {"type": "response.output_item.added", "output_index": 2,
         "item": {"type": "function_call", "call_id": "call_1", "name": "get_weather"}},
        {"type": "response.function_call_arguments.delta", "output_index": 2,
         "delta": "{\"city\":"},
        {"type": "response.function_call_arguments.delta", "output_index": 2,
         "delta": " \"Seoul\"}"},
        {"type": "response.completed",
         "response": {"status": "completed",
                      "usage": {"input_tokens": 10, "output_tokens": 5}}},
    ]

    async def lines():
        for e in events:
            yield "data: " + json.dumps(e)
        yield "data: [DONE]"

    async def collect():
        sink = {}
        out = []
        async for chunk in OpenAIProvider._iter_anthropic_events(lines(), "gpt-5.6-luna", sink):
            out.append(chunk.decode())
        return out, sink

    chunks, sink = asyncio.run(collect())
    text = "".join(chunks)
    assert "message_start" in text
    assert '"thinking_delta"' in text and '"text_delta"' in text
    assert '"input_json_delta"' in text
    assert '"stop_reason": "tool_use"' in text
    assert sink["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_usage_from_response_cached_split() -> None:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    usage = provider.usage_from_response(
        {"usage": {"input_tokens": 500, "input_tokens_details": {"cached_tokens": 400},
                   "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 15}}}
    )
    assert usage.input_tokens == 100
    assert usage.cache_read_input_tokens == 400
    assert usage.cache_creation_input_tokens == 0
    assert usage.reasoning_output_tokens == 15
