"""Context-fit guard: never deviate a request to a model whose window can't hold it.

A confident candidate with a smaller context window than the request would answer with
a provider 400, normalized to "prompt is too long"; Claude Code then truncates tool
results or compacts even though the default could have served the turn. The guard takes
such models out of the candidate set before routing (they join `skip_models`).

The default is never excluded. The client chose it and manages its own context against
it, so a request that overflows the default still surfaces as the normalized 400 and the
client's recovery path, exactly as before the guard existed.
"""
from __future__ import annotations

import logging
from typing import Any

from emissary_router.catalog import CATALOG
from emissary_router.config import AppConfig
from emissary_router.routing.cache_cost import RequestCostFeatures

logger = logging.getLogger(__name__)

# Headroom on the fit check. The request size is an estimate; Charm saw requests over a
# model's limit by up to ~3% still being routed there, and 5% covers the typical error.
CONTEXT_MARGIN = 0.05


def request_context_tokens(features: RequestCostFeatures, cache_ledger=None) -> int:
    """Size of this request: our estimate, floored by the cache the provider reported
    for this session's latest turn (real tokenizer numbers)."""
    observed = cache_ledger.observed_context_tokens(features) if cache_ledger is not None else 0
    return max(features.estimated_input_tokens, observed)


def fits_context(window: int | None, context_tokens: int, reserve_tokens: int) -> bool:
    """Does input + reserved output fit the window, less CONTEXT_MARGIN? An unknown
    window (None) always fits: missing catalog data must not shrink the candidate set."""
    if window is None:
        return True
    return context_tokens + reserve_tokens <= window * (1 - CONTEXT_MARGIN)


def oversized_models(
    config: AppConfig,
    body: dict[str, Any],
    features: RequestCostFeatures,
    cache_ledger=None,
) -> frozenset[str]:
    """Enabled non-default models whose window cannot hold this request.

    Output is reserved the way each provider validates it: OpenRouter checks input plus
    the request's max_tokens against the window (measured), so the whole requested
    output must fit there; elsewhere the expected output size is reserved. Stateless per
    request: once the client compacts, excluded models are eligible again on the next
    request.
    """
    context_tokens = request_context_tokens(features, cache_ledger)
    max_tokens = _max_tokens(body)
    oversized: set[str] = set()
    for model_name in config.enabled_models():
        if model_name == config.default:
            continue
        spec = CATALOG.get(model_name)
        if spec is None:
            continue
        provider = config.resolve_model(model_name).provider
        if provider == "openrouter" and max_tokens:
            reserve = max_tokens
        else:
            reserve = features.expected_output_tokens
        if not fits_context(spec.context_window, context_tokens, reserve):
            oversized.add(model_name)
    if oversized:
        logger.info(
            "context-fit guard: ~%d tokens exclude %s", context_tokens, sorted(oversized)
        )
    return frozenset(oversized)


def _max_tokens(body: dict[str, Any]) -> int:
    try:
        return max(0, int(body.get("max_tokens") or 0))
    except (TypeError, ValueError):
        return 0
