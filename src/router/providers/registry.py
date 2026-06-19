from __future__ import annotations

from router.config import ProviderConfig, infer_provider_type
from router.providers.anthropic import AnthropicProvider
from router.providers.base import Provider
from router.providers.google import GoogleProvider
from router.providers.openrouter import OpenRouterProvider


def build_provider(name: str, config: ProviderConfig) -> Provider:
    provider_type = infer_provider_type(name, config)
    if provider_type == "anthropic":
        return AnthropicProvider(config)
    if provider_type == "openrouter":
        return OpenRouterProvider(config)
    if provider_type == "google":
        return GoogleProvider(config)
    raise ValueError(f"unsupported provider {name}: {provider_type}")
