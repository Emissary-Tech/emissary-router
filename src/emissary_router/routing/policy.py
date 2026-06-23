from __future__ import annotations

from emissary_router.config import AppConfig
from emissary_router.schemas import RouteDecision


def choose_model(config: AppConfig, probabilities: dict[str, float]) -> RouteDecision:
    if config.policy == "deviate_if_confident":
        return _deviate_if_confident(config, probabilities)
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
