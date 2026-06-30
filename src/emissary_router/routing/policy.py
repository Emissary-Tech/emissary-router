from __future__ import annotations

from emissary_router.config import AppConfig
from emissary_router.schemas import RouteDecision


def choose_model(
    config: AppConfig,
    probabilities: dict[str, float],
    skip_models: frozenset[str] | set[str] = frozenset(),
) -> RouteDecision:
    if config.policy == "deviate_if_confident":
        return _deviate_if_confident(config, probabilities, skip_models)
    raise ValueError(f"unknown policy: {config.policy}")


def _deviate_if_confident(
    config: AppConfig,
    probabilities: dict[str, float],
    skip_models: frozenset[str] | set[str] = frozenset(),
) -> RouteDecision:
    """Default to the configured model, deviating only when confidence is high.

    `skip_models` are never deviated to (e.g. always-on-reasoning models on a background
    call); the default model remains the fallback regardless.
    """
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

    return RouteDecision(
        model_name=config.default,
        reason="default",
        probabilities=probabilities,
    )
