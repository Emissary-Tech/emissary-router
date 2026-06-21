from __future__ import annotations

from router.config import AppConfig
from router.schemas import RouteDecision


def choose_model(config: AppConfig, probabilities: dict[str, float]) -> RouteDecision:
    policy = config.router.policy
    candidates = policy.candidates or config.router.enabled

    if policy.name == "argmax":
        model_name = max(
            candidates,
            key=lambda name: probabilities.get(name, 0.0),
        )
        return RouteDecision(
            model_name=model_name,
            reason="argmax",
            probabilities=probabilities,
        )

    if policy.name == "cheap_first":
        for model_name in candidates:
            if probabilities.get(model_name, 0.0) >= policy.tau:
                return RouteDecision(
                    model_name=model_name,
                    reason=f"{policy.name}:p>={policy.tau}",
                    probabilities=probabilities,
                )

    if policy.name not in {"cheap_first", "argmax"}:
        raise ValueError(f"routing policy '{policy.name}' is not implemented")

    return RouteDecision(
        model_name=config.router.default,
        reason="default",
        probabilities=probabilities,
    )
