from __future__ import annotations

from router.providers.anthropic import AnthropicProvider


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
