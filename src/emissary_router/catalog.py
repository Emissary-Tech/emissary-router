from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["anthropic", "openrouter", "google", "zai", "openai"]


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


# Listed cheap -> expensive for readability only; routing derives order via cost_score.
CATALOG: dict[str, ModelSpec] = {
    "deepseek-v4-flash": ModelSpec(
        name="deepseek-v4-flash",
        # The router's cost anchor (81% of picks in the v7 routing sim). Deliberately
        # the BASE id — NOT deepseek-v4-flash-latest / -0731 (newer snapshot; our
        # labels were made against the base id). DeepSeek official pricing
        # (2026-08-04): 0.14 miss / 0.028 cache hit (80% discount, automatic disk
        # prefix cache, no write premium) / 0.28 out.
        providers={"openrouter": "deepseek/deepseek-v4-flash"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.14,
            output=0.28,
            cache_read=0.028,
            cache_write_5m=0.14,
            cache_write_1h=0.14,
        ),
    ),
    "gpt-5.6-luna": ModelSpec(
        name="gpt-5.6-luna",
        # Default via the native OpenAI Responses API (reasoning models run best
        # there; reasoning is the provider default and bills as output tokens).
        # OpenRouter is available opt-in ({"provider": "openrouter"}) — it serves the
        # same model through the chat-completions translation, so reasoning behavior
        # differs slightly; use it when no native OpenAI key is available. Cheapest
        # model in the catalog; caching is automatic (no write premium).
        providers={"openai": "gpt-5.6-luna", "openrouter": "openai/gpt-5.6-luna"},
        default_provider="openai",
        pricing=TokenPricing(
            input=0.20,
            output=1.20,
            cache_read=0.02,
            cache_write_5m=0.20,
            cache_write_1h=0.20,
        ),
    ),
    "gemini-3.1-flash-lite": ModelSpec(
        name="gemini-3.1-flash-lite",
        # Default via OpenRouter. Native Google is available opt-in
        # ({"provider": "google"}): the thoughtSignature requirement on replayed tool
        # calls is handled by the provider (real signatures round-trip; cross-provider
        # histories get a bridge value; the Claude boundary strips them), and
        # responses stream live (Google SSE translated to Anthropic SSE as chunks
        # arrive) — all verified live.
        providers={"openrouter": "google/gemini-3.1-flash-lite", "google": "gemini-3.1-flash-lite"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.25,
            output=1.50,
            cache_read=0.025,
            cache_write_5m=0.25,
            cache_write_1h=0.25,
        ),
    ),
    "glm-5.2": ModelSpec(
        name="glm-5.2",
        # Caching is implicit (no cache-write premium), so cache_write == input
        # price; only cache reads are discounted. "zai" is Z.ai's native
        # Anthropic-compatible endpoint (GLM Coding Plan) — opt in per model with
        # {"provider": "zai"}; unlike OpenRouter's multi-host routing it serves from
        # one place, so implicit cache reads land reliably.
        providers={"openrouter": "z-ai/glm-5.2", "zai": "glm-5.2"},
        default_provider="openrouter",
        # Z.ai OFFICIAL pricing is canonical here (decision 2026-08-04): 1.40/4.40,
        # cached input 0.26, cache storage currently free (no write premium).
        # OpenRouter's marketplace price floats by host (0.76/2.42 observed) — we
        # price by the official sheet for stability and label-era consistency.
        pricing=TokenPricing(
            input=1.40,
            output=4.40,
            cache_read=0.26,
            cache_write_5m=1.40,
            cache_write_1h=1.40,
        ),
    ),
    "kimi-k2.7-code": ModelSpec(
        name="kimi-k2.7-code",
        # OpenRouter only. Implicit caching (no cache-write premium). Always reasons —
        # thinking cannot be disabled (see THINKING_CAPABILITIES).
        # Moonshot AI's own endpoint price (int4 serving) is canonical (2026-08-04):
        # 0.95/4.00, cache read 0.19. Cheaper third-party hosts float below it
        # (0.73/3.50 observed) — priced by the first party, same principle as glm.
        providers={"openrouter": "moonshotai/kimi-k2.7-code"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.95,
            output=4.00,
            cache_read=0.19,
            cache_write_5m=0.95,
            cache_write_1h=0.95,
        ),
    ),
    "claude-haiku-4.5": ModelSpec(
        name="claude-haiku-4.5",
        providers={
            "anthropic": "claude-haiku-4-5",
            "openrouter": "anthropic/claude-haiku-4.5",
        },
        default_provider="anthropic",
        pricing=TokenPricing(
            input=1.00,
            output=5.00,
            cache_read=0.10,
            cache_write_5m=1.25,
            cache_write_1h=2.00,
        ),
    ),
    "gpt-5.6-terra": ModelSpec(
        name="gpt-5.6-terra",
        # Mid-tier gpt-5.6 (Responses API). Disabled in the default roster — dominated
        # by luna below and sol/opus above in the router-level ablation.
        providers={"openai": "gpt-5.6-terra", "openrouter": "openai/gpt-5.6-terra"},
        default_provider="openai",
        pricing=TokenPricing(
            input=2.00,
            output=12.00,
            cache_read=0.20,
            cache_write_5m=2.00,
            cache_write_1h=2.00,
        ),
    ),
    "claude-sonnet-5": ModelSpec(
        name="claude-sonnet-5",
        # Replaces sonnet-4.6 outright (same price; 4.6 is retired from the catalog).
        # Adaptive thinking is the provider default; temperature/top_p are stripped
        # for the claude-5 series (see providers/thinking.py REJECTS_SAMPLING_PARAMS).
        providers={
            "anthropic": "claude-sonnet-5",
            "openrouter": "anthropic/claude-sonnet-5",
        },
        default_provider="anthropic",
        pricing=TokenPricing(
            input=3.00,
            output=15.00,
            cache_read=0.30,
            cache_write_5m=3.75,
            cache_write_1h=6.00,
        ),
    ),
    "kimi-k3": ModelSpec(
        name="kimi-k3",
        # OpenRouter only. Sonnet-priced premium reasoner; disabled in the default
        # roster (router-level marginal contribution ~0 — MODEL_LIFECYCLE.md) and
        # enabled per-config for benchmark baselines. Caching assumed implicit like
        # kimi-k2.7 (no cache-write premium; ~20% cache-read) — verify before serving
        # at scale.
        providers={"openrouter": "moonshotai/kimi-k3"},
        default_provider="openrouter",
        # 3.00/15.00, cache read 0.30 — live-verified 2026-08-04 (exactly
        # sonnet-tier pricing, including the cache-read rate).
        pricing=TokenPricing(
            input=3.00,
            output=15.00,
            cache_read=0.30,
            cache_write_5m=3.00,
            cache_write_1h=3.00,
        ),
    ),
    "gpt-5.6-sol": ModelSpec(
        name="gpt-5.6-sol",
        # OpenAI flagship reasoner (Responses API): record-setting hard reasoning
        # (AIME 1.000 / GPQA 0.934) but weak tool-calling. Disabled in the default
        # roster (router-level contribution <=0); enable for reasoning-heavy configs.
        providers={"openai": "gpt-5.6-sol", "openrouter": "openai/gpt-5.6-sol"},
        default_provider="openai",
        pricing=TokenPricing(
            input=5.00,
            output=30.00,
            cache_read=0.50,
            cache_write_5m=5.00,
            cache_write_1h=5.00,
        ),
    ),
    "claude-opus-5": ModelSpec(
        name="claude-opus-5",
        # Escalation endpoint: strongest measured premium (multi-turn SOTA). Same
        # claude-5 sampling-param constraints as sonnet-5.
        providers={
            "anthropic": "claude-opus-5",
            "openrouter": "anthropic/claude-opus-5",
        },
        default_provider="anthropic",
        pricing=TokenPricing(
            input=5.00,
            output=25.00,
            cache_read=0.50,
            cache_write_5m=6.25,
            cache_write_1h=10.00,
        ),
    ),
}


PROVIDER_ENV: dict[ProviderName, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "zai": "ZAI_API_KEY",
    "openai": "OPENAI_API_KEY",
}

ROUTER_API_KEY_ENV = "EMISSARY_ROUTER_API_KEY"
