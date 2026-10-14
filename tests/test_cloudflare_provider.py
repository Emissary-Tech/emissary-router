"""Cloudflare AI Gateway Auto Router provider (bench-only competitor)."""
from __future__ import annotations

import json

import pytest

from emissary_router.config import ProviderConfig
from emissary_router.providers.cloudflare import CloudflareProvider, gateway_url
from emissary_router.schemas import RequestContext


def _ctx(session: str | None = "sess-1") -> RequestContext:
    return RequestContext(request_id="r1", conversation_id=session, classifier_input="", requested_model="m")


def test_gateway_url_needs_account_and_gateway(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_GATEWAY_ID", raising=False)
    with pytest.raises(ValueError):
        gateway_url()
    assert gateway_url("acct", "gw") == (
        "https://gateway.ai.cloudflare.com/v1/acct/gw/compat/chat/completions"
    )


def test_headers_carry_gateway_auth_session_and_skip_cache(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ALLOWED_MODELS", "anthropic/*, openai/gpt-5.6-luna")
    provider = CloudflareProvider(ProviderConfig(type="cloudflare", api_key="tok",
                                                 base_url="https://gw.example/compat/chat/completions"))
    headers = provider._request_headers(_ctx())
    assert headers["cf-aig-authorization"] == "Bearer tok"
    assert headers["cf-aig-session-id"] == "sess-1"
    assert headers["cf-aig-skip-cache"] == "true"
    assert headers["cf-aig-allowed-models"] == "anthropic/*,openai/gpt-5.6-luna"
    assert "Authorization" not in headers and "X-OpenRouter-Metadata" not in headers
    # the join key with the gateway's own per-request log (its `metadata` field)
    assert json.loads(headers["cf-aig-metadata"]) == {"request_id": "r1", "session": "sess-1"}
    # no session -> no affinity header, metadata keeps the request id
    headers = provider._request_headers(_ctx(None))
    assert "cf-aig-session-id" not in headers
    assert json.loads(headers["cf-aig-metadata"]) == {"request_id": "r1"}


def test_condition_pool_beats_the_env_pool(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ALLOWED_MODELS", "anthropic/*")
    cfg = ProviderConfig(type="cloudflare", api_key="tok", base_url="https://gw.example/compat/chat/completions")
    provider = CloudflareProvider(cfg, allowed_models=["openai/gpt-6-luna", " fireworks/glm-5.3 "])
    assert provider._request_headers(_ctx())["cf-aig-allowed-models"] == "openai/gpt-6-luna,fireworks/glm-5.3"
    # an explicit empty pool = the gateway's default pool, even with the env var set
    assert "cf-aig-allowed-models" not in CloudflareProvider(cfg, allowed_models=[])._request_headers(_ctx())
    # no condition pool -> the env fallback
    assert CloudflareProvider(cfg)._request_headers(_ctx())["cf-aig-allowed-models"] == "anthropic/*"


def test_system_and_tool_messages_are_plain_strings_without_cache_marks():
    body = {
        "system": [{"type": "text", "text": "A", "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": "B"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out",
                                          "cache_control": {"type": "ephemeral"}}]},
        ],
        "max_tokens": 10,
    }
    req = CloudflareProvider.to_openai_request(body, "cloudflare/auto", _ctx(), model_name="cloudflare-auto")
    msgs = req["messages"]
    assert msgs[0] == {"role": "system", "content": "A\n\nB"}       # same join as the no-cache path
    assert isinstance(msgs[1]["content"], list) and msgs[1]["content"][0]["text"] == "hi"
    assert msgs[2]["tool_calls"][0]["id"] == "t1"
    tool = next(m for m in msgs if m["role"] == "tool")
    assert tool["content"] == "out" and tool["tool_call_id"] == "t1"
    assert "cache_control" not in json.dumps(req)


def test_request_body_drops_openrouter_only_fields():
    req = CloudflareProvider.to_openai_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 100,
         "thinking": {"type": "enabled", "budget_tokens": 31999}},
        "cloudflare/auto", _ctx(), model_name="cloudflare-auto",
    )
    assert req["model"] == "cloudflare/auto"
    assert "session_id" not in req and "usage" not in req
    # dynamic router: no reasoning surface, nothing forwarded (set by the bench-gated
    # capability entry; here the name is unknown so the generic path runs)
    assert req["messages"][0]["content"] == "hi"


def test_routed_model_is_read_from_response_headers():
    provider = CloudflareProvider(ProviderConfig(type="cloudflare", api_key="tok",
                                                 base_url="https://gw.example/compat/chat/completions"))
    meta = provider._metadata_from_headers({
        "cf-aig-routed-model": "anthropic/claude-sonnet-5",
        "cf-aig-routing-reason": "cost_optimal_within_pool",
        "cf-aig-request-id": "req-9",
    })
    assert meta == {"routed_model": "anthropic/claude-sonnet-5",
                    "routing_reason": "cost_optimal_within_pool", "gateway_request_id": "req-9"}
    assert provider._metadata_from_headers({}) == {}


def test_raw_event_keeps_cloudflare_routing_fields():
    import json
    from emissary_router.pipeline import _routed_raw_event
    raw = _routed_raw_event(
        {"routed_model": "openai/gpt-5.6-luna", "routing_reason": "cost_optimal_within_pool", "id": "chatcmpl-1"},
        "cloudflare/auto",
    )
    data = json.loads(raw)
    assert data["routed_model"] == "openai/gpt-5.6-luna"
    assert data["routing_reason"] == "cost_optimal_within_pool"
    assert data["or_cost"] is None   # no billed-cost field at the gateway; priced by the ledger


def test_bench_extras_register_cloudflare_auto(monkeypatch):
    # Re-evaluate the gated blocks in a scratch namespace instead of reloading the
    # modules: a reload would replace the ReasoningSettings class object other
    # tests hold references to.
    import runpy
    monkeypatch.setenv("EMISSARY_ROUTER_BENCH_EXTRAS", "1")
    import emissary_router.catalog as catalog_mod
    import emissary_router.providers.thinking as thinking_mod
    catalog = runpy.run_path(catalog_mod.__file__)
    thinking = runpy.run_path(thinking_mod.__file__)
    assert catalog["CATALOG"]["cloudflare-auto"].providers == {"cloudflare": "cloudflare/auto"}
    assert catalog["CATALOG"]["cloudflare-auto"].pricing.input == 0.0
    assert catalog["CATALOG"]["openrouter-auto"].providers == {"openrouter": "openrouter/auto"}
    assert thinking["THINKING_CAPABILITIES"]["cloudflare-auto"].no_reasoning_surface
    assert catalog["PROVIDER_ENV"]["cloudflare"] == "CLOUDFLARE_API_TOKEN"
