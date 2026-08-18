from __future__ import annotations

from emissary_router.config import ProviderConfig
from emissary_router.providers.anthropic import AnthropicProvider
from emissary_router.providers.base import Provider
from emissary_router.providers.google import GoogleProvider
from emissary_router.providers.openai import OpenAIProvider
from emissary_router.providers.openrouter import OpenRouterProvider
from emissary_router.providers.zai import ZaiProvider


def build_provider(name: str, config: ProviderConfig) -> Provider:
    provider_type = config.type or name
    if provider_type == "anthropic":
        return AnthropicProvider(config)
    if provider_type == "openrouter":
        return OpenRouterProvider(config)
    if provider_type == "google":
        return GoogleProvider(config)
    if provider_type == "openai":
        return OpenAIProvider(config)
    if provider_type == "zai":
        return ZaiProvider(config)
    if provider_type == "vllm":
        # A self-hosted OpenAI-compatible server. Reuses the OpenRouter
        # translation wholesale; VLLM_BASE_URL points at .../v1/chat/completions.
        import os
        cfg = config.model_copy(update={
            "base_url": config.base_url or os.environ.get("VLLM_BASE_URL"),
        })
        if not cfg.base_url:
            raise ValueError("vllm provider requires VLLM_BASE_URL")
        return OpenRouterProvider(cfg)
    raise ValueError(f"unsupported provider {name}: {provider_type}")
