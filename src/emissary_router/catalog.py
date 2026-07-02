from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProviderName = Literal["anthropic", "openrouter", "google"]


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
    # Tokens the model can hold. Requests estimated above this are never routed here
    # (context-fit guard); the provider would reject them with a context-overflow 400.
    context_window: int = 200_000
    # Window when the client sends the `context-1m` anthropic-beta header (native
    # Anthropic only — the OpenRouter path does not forward that beta).
    context_window_1m_beta: int | None = None


def cost_score(spec: ModelSpec) -> float:
    """Blended $/Mtok used to order models cheap -> expensive for routing.

    The cost ranking is DERIVED from pricing — never from dict insertion order — so
    reordering CATALOG can't silently change which model is treated as "cheaper".
    """
    return spec.pricing.input + spec.pricing.output


# Listed cheap -> expensive for readability only; routing derives order via cost_score.
CATALOG: dict[str, ModelSpec] = {
    "gemini-3.1-flash-lite": ModelSpec(
        name="gemini-3.1-flash-lite",
        # Native Google Gemini 3 is unsafe for Claude Code tool loops; OpenRouter only.
        providers={"openrouter": "google/gemini-3.1-flash-lite"},
        default_provider="openrouter",
        context_window=1_048_576,
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
        # OpenRouter only. Caching is implicit (no cache-write premium), so
        # cache_write == input price; only cache reads are discounted.
        providers={"openrouter": "z-ai/glm-5.2"},
        default_provider="openrouter",
        context_window=1_048_576,
        pricing=TokenPricing(
            input=0.94,
            output=3.00,
            cache_read=0.18,
            cache_write_5m=0.94,
            cache_write_1h=0.94,
        ),
    ),
    "kimi-k2.7-code": ModelSpec(
        name="kimi-k2.7-code",
        # OpenRouter only. Implicit caching (no cache-write premium). Always reasons —
        # thinking cannot be disabled (see THINKING_CAPABILITIES).
        providers={"openrouter": "moonshotai/kimi-k2.7-code"},
        default_provider="openrouter",
        context_window=262_144,
        pricing=TokenPricing(
            input=0.74,
            output=3.50,
            cache_read=0.15,
            cache_write_5m=0.74,
            cache_write_1h=0.74,
        ),
    ),
    "claude-haiku-4.5": ModelSpec(
        name="claude-haiku-4.5",
        providers={
            "anthropic": "claude-haiku-4-5",
            "openrouter": "anthropic/claude-haiku-4.5",
        },
        default_provider="anthropic",
        context_window=200_000,
        pricing=TokenPricing(
            input=1.00,
            output=5.00,
            cache_read=0.10,
            cache_write_5m=1.25,
            cache_write_1h=2.00,
        ),
    ),
    "claude-sonnet-4.6": ModelSpec(
        name="claude-sonnet-4.6",
        providers={
            "anthropic": "claude-sonnet-4-6",
            "openrouter": "anthropic/claude-sonnet-4.6",
        },
        default_provider="anthropic",
        context_window=200_000,
        context_window_1m_beta=1_000_000,
        pricing=TokenPricing(
            input=3.00,
            output=15.00,
            cache_read=0.30,
            cache_write_5m=3.75,
            cache_write_1h=6.00,
        ),
    ),
}


PROVIDER_ENV: dict[ProviderName, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
}

ROUTER_API_KEY_ENV = "EMISSARY_ROUTER_API_KEY"
