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
    cost_features: RequestCostFeatures | None = None,
    cache_ledger=None,
) -> RouteDecision:
    if config.policy == "deviate_if_confident":
        return _deviate_if_confident(config, probabilities)
    if config.policy == "cache_aware":
        if cost_features is None or cache_ledger is None:
            return _deviate_if_confident(config, probabilities)
        return _cache_aware(config, probabilities, cost_features, cache_ledger)
    raise ValueError(f"unknown policy: {config.policy}")


def _deviate_if_confident(config: AppConfig, probabilities: dict[str, float]) -> RouteDecision:
    """Default to the configured model, deviating only when confidence is high."""
    for model_name in config.enabled_models():
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

    return RouteDecision(
        model_name=config.default,
        reason="default",
        probabilities=probabilities,
    )


def _cache_aware(
    config: AppConfig,
    probabilities: dict[str, float],
    cost_features: RequestCostFeatures,
    cache_ledger,
) -> RouteDecision:
    eligible = [config.default]
    for model_name in config.enabled_models():
        if model_name == config.default:
            continue
        if probabilities.get(model_name, 0.0) >= config.confidence:
            eligible.append(model_name)

    estimates = {
        model_name: estimate_cost(config, model_name, cost_features, cache_ledger)
        for model_name in eligible
    }
    baseline = estimates[config.default]
    best = min(estimates.values(), key=lambda estimate: estimate.total_usd)
    estimated_costs = {name: cost.to_dict() for name, cost in estimates.items()}

    if best.model_name != config.default and is_cheaper(best, baseline):
        return RouteDecision(
            model_name=best.model_name,
            reason="cache_aware:candidate_cheaper",
            probabilities=probabilities,
            estimated_costs=estimated_costs,
            cache_prediction=best.cache_prediction.to_dict(),
        )

    # Stayed on default. Distinguish *why* so the cause is visible in telemetry:
    #   no_confident_candidate — no non-default model cleared `confidence`
    #   warm_default_cheaper   — a confident candidate existed but default won after cache
    stayed_reason = (
        "cache_aware:no_confident_candidate"
        if len(eligible) == 1
        else "cache_aware:warm_default_cheaper"
    )
    return RouteDecision(
        model_name=config.default,
        reason=stayed_reason,
        probabilities=probabilities,
        estimated_costs=estimated_costs,
        cache_prediction=baseline.cache_prediction.to_dict(),
    )
