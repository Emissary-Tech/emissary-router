"""Cloudflare AI Gateway Auto Router provider (bench-only competitor)."""
from __future__ import annotations

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
    # no session -> no affinity header
    assert "cf-aig-session-id" not in provider._request_headers(_ctx(None))


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
