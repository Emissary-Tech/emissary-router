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


def test_google_stream_translation_backfills_cache_usage() -> None:
    text, sink = _run_google_translation([
        {"responseId": "r1",
         "candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 700,
                           "candidatesTokenCount": 5}},
    ])
    # message_start carries zeros (usage arrives at stream end); the final
    # message_delta backfills the cumulative numbers including cache reads.
    delta_part = text.split('"type": "message_delta"', 1)[1]
    assert '"input_tokens": 300' in delta_part
    assert '"cache_read_input_tokens": 700' in delta_part
    assert '"output_tokens": 5' in delta_part
    assert sink["usageMetadata"]["cachedContentTokenCount"] == 700


def test_google_thinking_effort_mapping_preserves_minimal() -> None:
    body = {"thinking": {"effort": "minimal"}}

    gemini3 = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert gemini3["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "minimal"


def test_google_thinking_effort_clamps_from_model_capability() -> None:
    body = {"thinking": {"effort": "xhigh"}}

    gemini3 = GoogleProvider.to_google_request(
        body,
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite",
    )

    assert gemini3["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"


def test_google_output_config_effort_drives_adaptive_thinking_level() -> None:
    body = {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }

    gemini3 = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert gemini3["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}


def test_google_budget_only_thinking_passes_thinking_budget() -> None:
    body = {"thinking": {"type": "enabled", "budget_tokens": 2048}}

    gemini3 = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert gemini3["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 2048}


def test_google_disabled_thinking_maps_to_zero_budget() -> None:
    body = {"thinking": {"type": "disabled"}}

    gemini3 = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert gemini3["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_google_adaptive_thinking_without_effort_uses_dynamic_budget() -> None:
    body = {"thinking": {"type": "adaptive"}}

    gemini3 = GoogleProvider.to_google_request(body, "gemini-3.1-flash-lite")

    assert gemini3["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}


def test_replayed_function_calls_carry_thought_signature() -> None:
    # Gemini 3 400s on replayed functionCall parts without a thoughtSignature
    # ("Function call is missing a thought_signature", live-verified). Foreign-model
    # tool calls get the documented cross-model bypass value; a preserved real
    # signature is sent back verbatim.
    from emissary_router.providers.google import DUMMY_THOUGHT_SIGNATURE

    body = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
                {"type": "tool_use", "id": "t2", "name": "get_time", "input": {},
                 "thought_signature": "real-gemini-sig"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "sunny"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "noon"},
            ]},
        ]
    }
    contents, _ = GoogleProvider._contents_and_system(body)
    calls = [p for c in contents for p in c["parts"] if "functionCall" in p]
    assert calls[0]["thoughtSignature"] == DUMMY_THOUGHT_SIGNATURE  # foreign turn
    assert calls[1]["thoughtSignature"] == "real-gemini-sig"  # preserved native turn


def test_from_google_response_preserves_signature_and_surfaces_thought() -> None:
    payload = {
        "candidates": [{"content": {"parts": [
            {"thought": True, "text": "planning the call"},
            {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
             "thoughtSignature": "sig-abc"},
        ]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }
    msg = GoogleProvider.from_google_response(payload, "gemini-3.1-flash-lite")
    types = [b["type"] for b in msg["content"]]
    assert types == ["thinking", "tool_use"]
    assert msg["content"][0]["signature"] == "emissary:non-anthropic-reasoning"
    assert msg["content"][1]["thought_signature"] == "sig-abc"


def test_google_trailing_system_stays_positioned_and_leading_folds() -> None:
    body = {
        "system": "main system",
        "messages": [
            {"role": "system", "content": "extra setup"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "<system-reminder>R1</system-reminder>"},
        ],
    }
    contents, system_instruction = GoogleProvider._contents_and_system(body)
    assert "main system" in system_instruction and "extra setup" in system_instruction
    assert [c["role"] for c in contents] == ["user", "user"]
    assert "R1" in contents[1]["parts"][0]["text"]
    assert contents[1]["parts"][0]["text"].count("<system-reminder") == 1


def _run_google_translation(chunks):
    import asyncio
    import json as _json

    lines = ["data: " + _json.dumps(c) for c in chunks]

    async def aiter():
        for line in lines:
            yield line

    async def run():
        sink: dict = {}
        events = []
        async for ev in GoogleProvider._iter_anthropic_events(aiter(), "gemini-3.1-flash-lite", sink):
            events.append(ev.decode())
        return "".join(events), sink

    return asyncio.run(run())


def test_google_stream_translation_thought_text_and_tools() -> None:
    text, sink = _run_google_translation([
        {"responseId": "r1", "modelVersion": "gemini-3.1-flash-lite",
         "candidates": [{"content": {"parts": [{"thought": True, "text": "planning"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
        {"candidates": [{"content": {"parts": [{"text": "lo"}]}}]},
        {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}},
             "thoughtSignature": "sig-abc"}]},
            "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 7}},
    ])

    assert "event: message_start" in text
    # thought parts stream as a thinking block closed with the synthetic signature
    assert '"thinking_delta"' in text and '"thinking": "planning"' in text
    assert '"signature_delta"' in text and "emissary:non-anthropic-reasoning" in text
    # text streams as deltas after the thinking block closes
    assert '"text": "Hel"' in text and '"text": "lo"' in text
    # functionCall arrives whole -> complete tool_use block, real signature preserved
    assert '"type": "tool_use"' in text and '"name": "get_weather"' in text
    assert '"thought_signature": "sig-abc"' in text
    assert '"input_json_delta"' in text  # args delivered as one complete json blob
    assert '"stop_reason": "tool_use"' in text
    assert "event: message_stop" in text
    assert sink["responseId"] == "r1"
