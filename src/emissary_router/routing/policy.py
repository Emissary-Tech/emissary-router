from __future__ import annotations

from emissary_router.config import AppConfig
from emissary_router.routing.cache_cost import (
    RequestCostFeatures,
    estimate_cost,
    is_cheaper,
)
from emissary_router.schemas import RouteDecision


def choose_model(
    config: AppConfig,
    probabilities: dict[str, float],
    skip_models: frozenset[str] | set[str] = frozenset(),
    cost_features: RequestCostFeatures | None = None,
    cache_ledger=None,
) -> RouteDecision:
    """Confidence-gated, cache-aware routing.

    Cache awareness is not a mode: whenever the request's cost features and the cache
    ledger are available (every gateway request), candidates are compared by
    cache-adjusted per-request cost — a warm model is credited its observed cache
    reads, switching is priced at full input plus a cache write. Where there is no
    cache signal (cold start, providers without usable cache reporting) the estimates
    carry no discount and the comparison is a flat per-request price comparison for
    THIS request's input/output shape (which can order models differently than the
    blended catalog cost_score used by the price-ordered fallback).

    `skip_models` are never deviated to (e.g. always-on-reasoning models on a
    background call); if the default itself is skipped, the cheapest usable model
    serves instead.
    """
    if cost_features is not None and cache_ledger is not None:
        return _cache_aware(config, probabilities, skip_models, cost_features, cache_ledger)
    return _price_ordered(config, probabilities, skip_models)


def _price_ordered(
    config: AppConfig,
    probabilities: dict[str, float],
    skip_models: frozenset[str] | set[str] = frozenset(),
) -> RouteDecision:
    """Fallback when no cost features/ledger are available: scan cheap -> expensive and
    pick the first confident model; otherwise the default."""
    for model_name in config.enabled_models():
        if model_name in skip_models:
            continue
        if probabilities.get(model_name, 0.0) >= config.confidence:
            reason = (
                "default"
                if model_name == config.default
                else f"deviate_if_confident:p>={config.confidence}"
            )
            return RouteDecision(
                model_name=model_name,
                reason=reason,
                probabilities=probabilities,
            )

    if config.default in skip_models:
        # A user may legitimately set an always-on-reasoning model as default;
        # background calls fall back to the cheapest usable model instead of burning
        # reasoning tokens on every title/summary call.
        for model_name in config.enabled_models():  # cheap -> expensive
            if model_name not in skip_models:
                return RouteDecision(
                    model_name=model_name,
                    reason="default_unsuitable:cheapest_alternative",
                    probabilities=probabilities,
                )

    return RouteDecision(
        model_name=config.default,
        reason="default",
        probabilities=probabilities,
    )


def _cache_aware(
    config: AppConfig,
    probabilities: dict[str, float],
    skip_models: frozenset[str] | set[str],
    cost_features: RequestCostFeatures,
    cache_ledger,
) -> RouteDecision:
    candidates: list[str] = []
    if config.default not in skip_models:
        candidates.append(config.default)
    for model_name in config.enabled_models():
        if model_name == config.default or model_name in skip_models:
            continue
        if probabilities.get(model_name, 0.0) >= config.confidence:
            candidates.append(model_name)

    if not candidates:
        # Everything usable is skipped or nothing qualifies; the price path carries the
        # default / default-unsuitable fallbacks.
        return _price_ordered(config, probabilities, skip_models)

    estimates = {
        model_name: estimate_cost(config, model_name, cost_features, cache_ledger)
        for model_name in candidates
    }
    estimated_costs = {name: cost.to_dict() for name, cost in estimates.items()}
    best = min(estimates.values(), key=lambda estimate: estimate.total_usd)
    default_estimate = estimates.get(config.default)

    # `is_cheaper` is strict, so when best IS the default this is always a stay.
    if default_estimate is not None and not is_cheaper(best, default_estimate):
        # Stayed on default. Distinguish *why* so the cause is visible in telemetry:
        #   no_confident_candidate — no non-default model cleared `confidence`
        #   warm_default_cheaper   — the default's warm cache beat a confident candidate
        #   default_not_beaten     — no cache in play; the default was simply not more
        #                            expensive than the candidates for this request
        if len(candidates) == 1:
            stayed_reason = "cache_aware:no_confident_candidate"
        elif default_estimate.cache_prediction.warm:
            stayed_reason = "cache_aware:warm_default_cheaper"
        else:
            stayed_reason = "cache_aware:default_not_beaten"
        return RouteDecision(
            model_name=config.default,
            reason=stayed_reason,
            probabilities=probabilities,
            estimated_costs=estimated_costs,
            cache_prediction=default_estimate.cache_prediction.to_dict(),
        )

    return RouteDecision(
        model_name=best.model_name,
        reason="cache_aware:candidate_cheaper",
        probabilities=probabilities,
        estimated_costs=estimated_costs,
        cache_prediction=best.cache_prediction.to_dict(),
    )
