from __future__ import annotations

import json

from emissary_router.config import AppConfig, ProviderConfig
from emissary_router.providers.registry import build_provider
from emissary_router.providers.thinking import SYNTHETIC_THINKING_SIGNATURE
from emissary_router.providers.zai import ZAI_DEFAULT_BASE_URL, ZaiProvider


def _provider() -> ZaiProvider:
    return ZaiProvider(ProviderConfig(type="zai", api_key="k"))


def test_registry_builds_zai_with_default_base_url() -> None:
    provider = build_provider("zai", ProviderConfig(type="zai", api_key="k"))
    assert isinstance(provider, ZaiProvider)
    assert provider.name == "zai"
    assert provider._base_url == ZAI_DEFAULT_BASE_URL

    custom = build_provider(
        "zai", ProviderConfig(type="zai", api_key="k", base_url="http://localhost:9999")
    )
    assert custom._base_url == "http://localhost:9999"


def test_glm_via_zai_is_selectable_in_config() -> None:
    config = AppConfig.model_validate(
        {
            "models": {"glm-5.2": {"enabled": True, "provider": "zai"}},
            "default": "glm-5.2",
        }
    )
    resolved = config.resolve_model("glm-5.2")
    assert resolved.provider == "zai"
    assert resolved.model_id == "glm-5.2"
    assert config.required_provider_env() == {"zai": "ZAI_API_KEY"}


def test_client_anthropic_credentials_never_reach_zai() -> None:
    # Claude Code sends its own Anthropic credentials; forwarding them to z.ai
    # would 401. They are dropped so the provider's ZAI_API_KEY is attached instead.
    forwarded = ZaiProvider._forward_headers(
        {
            "x-api-key": "sk-ant-real-user-key",
            "Authorization": "Bearer oauth-token",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "context-1m-2025-08-07",
            "content-type": "application/json",
        }
    )
    assert "x-api-key" not in {k.lower() for k in forwarded}
    assert "authorization" not in {k.lower() for k in forwarded}
    assert forwarded["anthropic-version"] == "2023-06-01"


def test_response_thinking_signature_is_restamped() -> None:
    # z.ai signs thinking blocks with its own opaque values; real Anthropic rejects
    # them on replay (live 400 "Invalid `signature` in `thinking` block"). Restamping
    # with our synthetic marker routes them through the existing sanitizing rules.
    payload = {
        "content": [
            {"type": "thinking", "thinking": "let me think", "signature": "ca7852b34d8b4726"},
            {"type": "text", "text": "4"},
        ]
    }
    out = _provider()._rewrite_response_json(payload)
    assert out["content"][0]["signature"] == SYNTHETIC_THINKING_SIGNATURE
    assert out["content"][1] == {"type": "text", "text": "4"}


def test_sse_thinking_signature_lines_are_restamped() -> None:
    provider = _provider()

    # z.ai carries the signature on the thinking content_block_start (live-observed),
    # but Claude Code only collects signatures from signature_delta events — so the
    # rewrite must BOTH restamp the start and inject a synthetic signature_delta
    # event (otherwise CC stores signature "" and native Claude 400s on the replay;
    # observed live).
    start = ('data: {"type": "content_block_start", "index": 0, "content_block": '
             '{"type": "thinking", "thinking": "", "signature": "0cffa860c00e4618"}}')
    out = provider._rewrite_sse_line(start)
    first, injected = out.split("\n\n", 1)
    assert json.loads(first[6:])["content_block"]["signature"] == SYNTHETIC_THINKING_SIGNATURE
    assert injected.startswith("event: content_block_delta\ndata: ")
    injected_event = json.loads(injected.split("data: ", 1)[1])
    assert injected_event["delta"] == {
        "type": "signature_delta",
        "signature": SYNTHETIC_THINKING_SIGNATURE,
    }
    assert injected_event["index"] == 0

    # Anthropic-style late signature handled too, in case z.ai changes shape
    delta = ('data: {"type": "content_block_delta", "index": 0, "delta": '
             '{"type": "signature_delta", "signature": "abc123"}}')
    out = provider._rewrite_sse_line(delta)
    assert json.loads(out[6:])["delta"]["signature"] == SYNTHETIC_THINKING_SIGNATURE

    # Everything else passes through byte-identical
    for line in (
        "event: content_block_start",
        'data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}',
        'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}}',
        "",
        "data: not-json signature",
    ):
        assert provider._rewrite_sse_line(line) == line


def test_sonnet5_era_thinking_passes_through_untouched_for_glm() -> None:
    # z.ai accepts the whole Sonnet 5-era surface natively (live-verified: adaptive +
    # every effort level minimal..max/xhigh, disabled, enabled+budget all 200), so the
    # anthropic-format normalization must pass it through without rewriting anything —
    # rewrites would be silent behavior changes and bust the prompt-shape stability.
    from emissary_router.providers.thinking import normalize_anthropic_thinking_for_model

    body = {
        "model": "claude-sonnet-5",
        "max_tokens": 64000,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh"},
        "messages": [{"role": "user", "content": "hi"}],
    }
    changes = normalize_anthropic_thinking_for_model(body, "glm-5.2")
    assert changes == []
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "xhigh"
