from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["anthropic", "openrouter", "google", "zai", "openai", "vllm"]


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
        # Moved to the -0731 snapshot (2026-08-26, impact test: the
        # emissary-qwen-router classifier is trained against it). DeepSeek official
        # pricing: 0.22 in / 0.007 cache hit / 0.66 out (no write premium).
        providers={"openrouter": "deepseek/deepseek-v4-flash-0731"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.22,
            output=0.66,
            cache_read=0.007,
            cache_write_5m=0.22,
            cache_write_1h=0.22,
        ),
    ),
    "gpt-5.6-luna": ModelSpec(
        name="gpt-5.6-luna",
        # Default via the native OpenAI Responses API (reasoning models run best
        # there; reasoning is the provider default and bills as output tokens).
        # OpenRouter is available opt-in ({"provider": "openrouter"}) — it serves the
        # same model through the chat-completions translation, so reasoning behavior
        # differs slightly; use it when no native OpenAI key is available. Cheapest
        # model in the catalog. 2026-08 price sheet: short-context tier, with a
        # 1.25x cache-write line item (long-context tier is 2x across the board).
        providers={"openai": "gpt-5.6-luna", "openrouter": "openai/gpt-5.6-luna"},
        default_provider="openai",
        pricing=TokenPricing(
            input=0.2,
            output=1.2,
            cache_read=0.02,
            cache_write_5m=0.25,
            cache_write_1h=0.25,
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
            input=2.0,
            output=12.0,
            cache_read=0.2,
            cache_write_5m=2.5,
            cache_write_1h=2.5,
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
            # Anthropic price cut (user-reported 2026-08-15): 3.00/15.00 -> 2.00/10.00;
            # cache rates follow the standard multipliers (read 0.1x, write 1.25x/2x).
            input=2.00,
            output=10.00,
            cache_read=0.20,
            cache_write_5m=2.50,
            cache_write_1h=4.00,
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
            input=4.0,
            output=20.0,
            cache_read=0.4,
            cache_write_5m=5.0,
            cache_write_1h=5.0,
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
    # self-hosted OpenAI-compatible endpoint (vLLM); key optional on private networks
    "vllm": "VLLM_API_KEY",
}

ROUTER_API_KEY_ENV = "EMISSARY_ROUTER_API_KEY"

# Benchmark-only catalog extras, gated behind an env var so they can never leak
# into production rosters (a zero-priced entry would sort "cheapest" and pollute
# any enable-everything config). The er_bench gateway sets the var; the live ER
# never does. openrouter-auto = the competitor Auto Router served through ER's
# protocol translation; billed cost comes from the OR key's usage delta, never
# from ER telemetry (pricing zeros are deliberate).
if os.environ.get("EMISSARY_ROUTER_BENCH_EXTRAS"):
    # Candidate under evaluation: impact test serves it on our own vLLM at the
    # saturation-derived internal rates (A100 @ $3/h: 0.19 cold-in / 0.04 cache-hit
    # / 0.75 out).
    CATALOG["qwen3.8-27b"] = ModelSpec(
        name="qwen3.8-27b",
        providers={"openrouter": "qwen/qwen3.8-27b", "vllm": "qwen/qwen3.8-27b"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.19,
            output=0.75,
            cache_read=0.04,
            cache_write_5m=0.19,
            cache_write_1h=0.19,
        ),
    )
    # Candidate under evaluation (2026-08-22, boss request): same lifecycle as
    # qwen3.8-27b. OpenRouter listing pricing ($0.10/$0.15); cache read at the
    # same ~1/9 ratio used for the 27b entry.
    CATALOG["qwen3.5-9b"] = ModelSpec(
        name="qwen3.5-9b",
        providers={"openrouter": "qwen/qwen3.5-9b", "vllm": "qwen/qwen3.5-9b"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.10,
            output=0.15,
            cache_read=0.01,
            cache_write_5m=0.10,
            cache_write_1h=0.10,
        ),
    )
    # Impact-test alias: the emissary-qwen-router classifier labels deepseek by
    # its snapshot name. Same serving id/pricing as the base entry.
    CATALOG["deepseek-v4-flash-0731"] = ModelSpec(
        name="deepseek-v4-flash-0731",
        providers={"openrouter": "deepseek/deepseek-v4-flash-0731"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.22,
            output=0.66,
            cache_read=0.007,
            cache_write_5m=0.22,
            cache_write_1h=0.22,
        ),
    )
    CATALOG["openrouter-auto"] = ModelSpec(
        name="openrouter-auto",
        providers={"openrouter": "openrouter/auto"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.0,
            output=0.0,
            cache_read=0.0,
            cache_write_5m=0.0,
            cache_write_1h=0.0,
        ),
    )
