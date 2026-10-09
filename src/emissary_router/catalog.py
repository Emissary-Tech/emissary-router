from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["anthropic", "openrouter", "google", "zai", "openai", "cloudflare"]


@dataclass(frozen=True)
class TokenPricing:
    input: float
    output: float
    cache_read: float
    cache_write_5m: float
    cache_write_1h: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    # provider -> upstream model id. The first entry is the recommended default.
    providers: dict[ProviderName, str]
    default_provider: ProviderName
    pricing: TokenPricing


def cost_score(spec: ModelSpec) -> float:
    """Blended $/Mtok used to order models cheap -> expensive for routing.

    The cost ranking is DERIVED from pricing — never from dict insertion order — so
    reordering CATALOG can't silently change which model is treated as "cheaper".
    """
    return spec.pricing.input + spec.pricing.output


def _price(input: float, output: float, cache_read: float, cache_write: float) -> TokenPricing:
    # One write rate: the platform catalog carries a single cache-write price per
    # model (Anthropic's 1h tier is not configured anywhere we route).
    return TokenPricing(
        input=input, output=output, cache_read=cache_read,
        cache_write_5m=cache_write, cache_write_1h=cache_write,
    )


# The model set, upstream ids and prices mirror the platform gateway's catalog
# (emissary-ai app/services/routing_gateway/catalog.py, 2026-10-08) so a benchmark
# condition here prices a call exactly as the platform would. Keep the two in step:
# a price edit there is a price edit here. Listed cheap -> expensive for readability
# only; routing derives order via cost_score.
CATALOG: dict[str, ModelSpec] = {
    "deepseek-v4-flash": ModelSpec(
        name="deepseek-v4-flash",
        providers={"openrouter": "deepseek/deepseek-v4-flash"},
        default_provider="openrouter",
        pricing=_price(0.14, 0.28, 0.028, 0.14),
    ),
    # Snapshot-named alias (the emissary-qwen-router classifier labels deepseek by
    # its snapshot name).
    "deepseek-v4-flash-0731": ModelSpec(
        name="deepseek-v4-flash-0731",
        providers={"openrouter": "deepseek/deepseek-v4-flash-0731"},
        default_provider="openrouter",
        pricing=_price(0.14, 0.28, 0.028, 0.14),
    ),
    "gpt-6-luna": ModelSpec(
        name="gpt-6-luna",
        # Native OpenAI Responses API by default (reasoning models run best there);
        # OpenRouter opt-in serves the same model through the chat translation.
        providers={"openai": "gpt-6-luna", "openrouter": "openai/gpt-6-luna"},
        default_provider="openai",
        pricing=_price(0.10, 0.50, 0.01, 0.125),
    ),
    "qwen3.8-omni-flash": ModelSpec(
        name="qwen3.8-omni-flash",
        providers={"openrouter": "qwen/qwen3.8-omni-flash"},
        default_provider="openrouter",
        pricing=_price(0.15, 0.47, 0.016, 0.15),
    ),
    # Same price and host as omni-flash; the router heads are trained on omni data.
    "qwen3.8-flash": ModelSpec(
        name="qwen3.8-flash",
        providers={"openrouter": "qwen/qwen3.8-flash"},
        default_provider="openrouter",
        pricing=_price(0.15, 0.47, 0.016, 0.20),
    ),
    "deepseek-v4.1-flash": ModelSpec(
        name="deepseek-v4.1-flash",
        providers={"openrouter": "deepseek/deepseek-v4.1-flash"},
        default_provider="openrouter",
        pricing=_price(0.15, 0.60, 0.003, 0.15),
    ),
    "gpt-5.6-luna": ModelSpec(
        name="gpt-5.6-luna",
        providers={"openai": "gpt-5.6-luna", "openrouter": "openai/gpt-5.6-luna"},
        default_provider="openai",
        pricing=_price(0.20, 1.20, 0.02, 0.25),
    ),
    "gemini-3.1-flash-lite": ModelSpec(
        name="gemini-3.1-flash-lite",
        # OpenRouter by default; native Google opt-in ({"provider": "google"}) keeps
        # thoughtSignature round-trips and live streaming (see providers/google.py).
        providers={"openrouter": "google/gemini-3.1-flash-lite", "google": "gemini-3.1-flash-lite"},
        default_provider="openrouter",
        pricing=_price(0.25, 1.50, 0.025, 0.25),
    ),
    "kimi-k2.7-code": ModelSpec(
        name="kimi-k2.7-code",
        # OpenRouter only. Always reasons — thinking cannot be disabled (THINKING_CAPABILITIES).
        providers={"openrouter": "moonshotai/kimi-k2.7-code"},
        default_provider="openrouter",
        pricing=_price(0.95, 4.00, 0.19, 0.95),
    ),
    "claude-haiku-4.5": ModelSpec(
        name="claude-haiku-4.5",
        providers={"anthropic": "claude-haiku-4-5", "openrouter": "anthropic/claude-haiku-4.5"},
        default_provider="anthropic",
        pricing=_price(1.00, 5.00, 0.10, 1.25),
    ),
    "glm-5.2": ModelSpec(
        name="glm-5.2",
        # OpenRouter by default; "zai" is Z.ai's native Anthropic-compatible endpoint
        # (GLM Coding Plan) — opt in per model with {"provider": "zai"}.
        providers={"openrouter": "z-ai/glm-5.2", "zai": "glm-5.2"},
        default_provider="openrouter",
        pricing=_price(1.40, 4.40, 0.26, 1.40),
    ),
    "glm-5.3": ModelSpec(
        name="glm-5.3",
        providers={"openrouter": "z-ai/glm-5.3"},
        default_provider="openrouter",
        pricing=_price(1.40, 4.40, 0.26, 1.40),
    ),
    "glm-5.3-flash": ModelSpec(
        name="glm-5.3-flash",
        providers={"openrouter": "z-ai/glm-5.3-flash"},
        default_provider="openrouter",
        pricing=_price(0.15, 0.50, 0.03, 0.15),
    ),
    "gpt-5.6-terra": ModelSpec(
        name="gpt-5.6-terra",
        providers={"openai": "gpt-5.6-terra", "openrouter": "openai/gpt-5.6-terra"},
        default_provider="openai",
        pricing=_price(2.00, 12.00, 0.20, 2.50),
    ),
    "claude-sonnet-5": ModelSpec(
        name="claude-sonnet-5",
        # Adaptive thinking is the provider default; temperature/top_p are stripped
        # for the claude-5 series (providers/thinking.py REJECTS_SAMPLING_PARAMS).
        providers={"anthropic": "claude-sonnet-5", "openrouter": "anthropic/claude-sonnet-5"},
        default_provider="anthropic",
        pricing=_price(2.00, 10.00, 0.20, 2.50),
    ),
    "claude-sonnet-5.5": ModelSpec(
        name="claude-sonnet-5.5",
        providers={"anthropic": "claude-sonnet-5-5", "openrouter": "anthropic/claude-sonnet-5.5"},
        default_provider="anthropic",
        pricing=_price(2.00, 10.00, 0.10, 2.50),
    ),
    "gpt-6-sol": ModelSpec(
        name="gpt-6-sol",
        providers={"openai": "gpt-6-sol", "openrouter": "openai/gpt-6-sol"},
        default_provider="openai",
        pricing=_price(2.00, 10.00, 0.20, 2.50),
    ),
    "gpt-6.1-sol": ModelSpec(
        name="gpt-6.1-sol",
        providers={"openai": "gpt-6.1-sol", "openrouter": "openai/gpt-6.1-sol"},
        default_provider="openai",
        pricing=_price(2.00, 10.00, 0.10, 2.50),
    ),
    "kimi-k3": ModelSpec(
        name="kimi-k3",
        providers={"openrouter": "moonshotai/kimi-k3"},
        default_provider="openrouter",
        pricing=_price(3.00, 15.00, 0.30, 3.00),
    ),
    "gpt-5.6-sol": ModelSpec(
        name="gpt-5.6-sol",
        providers={"openai": "gpt-5.6-sol", "openrouter": "openai/gpt-5.6-sol"},
        default_provider="openai",
        pricing=_price(4.00, 20.00, 0.40, 5.00),
    ),
    "claude-opus-5.5": ModelSpec(
        name="claude-opus-5.5",
        providers={"anthropic": "claude-opus-5-5", "openrouter": "anthropic/claude-opus-5.5"},
        default_provider="anthropic",
        pricing=_price(4.00, 20.00, 0.20, 5.00),
    ),
    "claude-opus-5": ModelSpec(
        name="claude-opus-5",
        providers={"anthropic": "claude-opus-5", "openrouter": "anthropic/claude-opus-5"},
        default_provider="anthropic",
        pricing=_price(5.00, 25.00, 0.50, 6.25),
    ),
}


PROVIDER_ENV: dict[ProviderName, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "zai": "ZAI_API_KEY",
    "openai": "OPENAI_API_KEY",
    # Cloudflare AI Gateway token (cf-aig-authorization); the gateway's account and
    # id come from CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_GATEWAY_ID (providers/cloudflare.py)
    "cloudflare": "CLOUDFLARE_API_TOKEN",
}

ROUTER_API_KEY_ENV = "EMISSARY_ROUTER_API_KEY"

# Benchmark-only competitor routers, gated behind an env var so they can never leak
# into production rosters (a zero-priced entry would sort "cheapest" and pollute any
# enable-everything config). The er_bench gateway sets the var; the live ER never
# does. Both are served through ER's protocol translation as single-model
# conditions (the classifier has no head for them). The routed model is read back
# from the response (OpenRouter: body `model`; Cloudflare: `cf-aig-routed-model`)
# and priced per routed model by the bench ledger — these entries' own zero prices
# are deliberate, never a cost.
if os.environ.get("EMISSARY_ROUTER_BENCH_EXTRAS"):
    _ZERO = _price(0.0, 0.0, 0.0, 0.0)
    CATALOG["openrouter-auto"] = ModelSpec(
        name="openrouter-auto",
        providers={"openrouter": "openrouter/auto"},
        default_provider="openrouter",
        pricing=_ZERO,
    )
    CATALOG["cloudflare-auto"] = ModelSpec(
        name="cloudflare-auto",
        providers={"cloudflare": "cloudflare/auto"},
        default_provider="cloudflare",
        pricing=_ZERO,
    )
