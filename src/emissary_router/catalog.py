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


# Cheap-to-expensive order. Routing policy relies on this insertion order.
CATALOG: dict[str, ModelSpec] = {
    "gemini-3.1-flash-lite": ModelSpec(
        name="gemini-3.1-flash-lite",
        # Native Google Gemini 3 is unsafe for Claude Code tool loops; OpenRouter only.
        providers={"openrouter": "google/gemini-3.1-flash-lite"},
        default_provider="openrouter",
        pricing=TokenPricing(
            input=0.25,
            output=1.50,
            cache_read=0.025,
            cache_write_5m=0.25,
            cache_write_1h=0.25,
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
    "claude-sonnet-4.6": ModelSpec(
        name="claude-sonnet-4.6",
        providers={
            "anthropic": "claude-sonnet-4-6",
            "openrouter": "anthropic/claude-sonnet-4.6",
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
}


PROVIDER_ENV: dict[ProviderName, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
}

ROUTER_API_KEY_ENV = "EMISSARY_ROUTER_API_KEY"
