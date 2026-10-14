from __future__ import annotations

from emissary_router.catalog import CATALOG
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

    Context guard: a model whose window the request's estimated input + the client's
    `max_tokens` (expected output when no cap was sent) exceeds is not a candidate. If
    the default itself cannot take the request, the priciest fitting model stands in
    as the anchor; if NOTHING fits, the default serves with reason
    `context_exceeds_config` and the provider's own "prompt is too long" error reaches
    the client untouched — that error is the agent's signal to compact, never masked.
    Stateless: after the agent compacts, excluded models are eligible again on the
    very next request. Excluded names ride along as `RouteDecision.context_excluded`.
    """
    if cost_features is not None and cache_ledger is not None:
        return _cache_aware(config, probabilities, skip_models, cost_features, cache_ledger)
    return _price_ordered(config, probabilities, skip_models)


def fits_context(spec, features) -> bool:
    """Per-request context eligibility: the estimated input plus the output room the
    client asked for (`max_tokens`) must fit the model's window. OpenRouter and OpenAI
    apply exactly that check ("you requested N tokens (A of input, B in the output)";
    OpenRouter measured 2026-10-14), so reserving less would admit a model the provider
    then rejects. Anthropic checks the input alone (measured the same day: 147K input +
    a 64K cap on a 200K model was served), so for an Anthropic-served model the full
    reserve is deliberately conservative — it only costs haiku-4.5 eligibility between
    136K and 200K of input. The expected-output estimate is the fallback for clients
    that sent no ceiling. An
    unknown window never excludes (missing catalog data must not shrink the candidate
    set), and features without an estimate never exclude either."""
    window = getattr(spec, "context_window", None)
    estimated = getattr(features, "estimated_input_tokens", None)
    if window is None or estimated is None:
        return True
    reserve = getattr(features, "output_ceiling_tokens", None)
    if not reserve:
        reserve = int(getattr(features, "expected_output_tokens", 0) or 0)
    return int(estimated) + int(reserve) <= window


def _price_ordered(
    config: AppConfig,
    probabilities: dict[str, float],
    skip_models: frozenset[str] | set[str] = frozenset(),
    models: list[str] | None = None,
    default_name: str | None = None,
    context_excluded: list[str] | None = None,
) -> RouteDecision:
    """Fallback when no cost features/ledger are available: scan cheap -> expensive and
    pick the first confident model; otherwise the default. `models`/`default_name`
    let the cache-aware path hand over its context-filtered roster and re-seated anchor."""
    models = list(models) if models is not None else config.enabled_models()
    default_name = default_name or config.default
    for model_name in models:
        if model_name in skip_models:
            continue
        if probabilities.get(model_name, 0.0) >= config.confidence:
            reason = (
                "default"
                if model_name == default_name
                else f"deviate_if_confident:p>={config.confidence}"
            )
            return RouteDecision(
                model_name=model_name,
                reason=reason,
                probabilities=probabilities,
                context_excluded=context_excluded,
            )

    if default_name in skip_models:
        # A user may legitimately set an always-on-reasoning model as default;
        # background calls fall back to the cheapest usable model instead of burning
        # reasoning tokens on every title/summary call.
        for model_name in models:  # cheap -> expensive
            if model_name not in skip_models:
                return RouteDecision(
                    model_name=model_name,
                    reason="default_unsuitable:cheapest_alternative",
                    probabilities=probabilities,
                    context_excluded=context_excluded,
                )

    return RouteDecision(
        model_name=default_name,
        reason="default",
        probabilities=probabilities,
        context_excluded=context_excluded,
    )


def _cache_aware(
    config: AppConfig,
    probabilities: dict[str, float],
    skip_models: frozenset[str] | set[str],
    cost_features: RequestCostFeatures,
    cache_ledger,
) -> RouteDecision:
    # Context eligibility runs before everything (see choose_model): the roster for
    # THIS request is the models whose window still fits it.
    enabled = config.enabled_models()  # cheap -> expensive
    fitting = [m for m in enabled if fits_context(CATALOG.get(m), cost_features)]
    excluded = [m for m in enabled if m not in fitting] or None
    if not fitting:
        # Nothing configured can take this request. Serve the default and let the
        # provider's error reach the agent untouched — it is the agent's own signal
        # to compact; masking it here would hide the problem.
        return RouteDecision(
            model_name=config.default,
            reason="context_exceeds_config",
            probabilities=probabilities,
            context_excluded=excluded,
        )
    # The configured default cannot take this request: the priciest fitting model
    # stands in as the anchor so "uncertain -> strongest available" keeps holding.
    default_name = config.default if config.default in fitting else fitting[-1]

    candidates: list[str] = []
    if default_name not in skip_models:
        candidates.append(default_name)
    for model_name in fitting:
        if model_name == default_name or model_name in skip_models:
            continue
        if probabilities.get(model_name, 0.0) >= config.confidence:
            candidates.append(model_name)

    if not candidates:
        # Everything usable is skipped or nothing qualifies; the price path carries the
        # default / default-unsuitable fallbacks (over the fitting roster).
        return _price_ordered(config, probabilities, skip_models, fitting, default_name, excluded)

    estimates = {
        model_name: estimate_cost(config, model_name, cost_features, cache_ledger)
        for model_name in candidates
    }
    estimated_costs = {name: cost.to_dict() for name, cost in estimates.items()}
    best = min(estimates.values(), key=lambda estimate: estimate.total_usd)
    default_estimate = estimates.get(default_name)

    # `is_cheaper` is strict, so when best IS the default this is always a stay.
    if default_estimate is not None and not is_cheaper(best, default_estimate):
        # Charm-style escalation (always on): every confident candidate priced above
        # the default only stays on the default when the default's own head clears
        # the gate. If the default is unconfident about this request, send it to
        # the cheapest confident candidate even though it costs more — "I say I
        # can't; they say they can." Inert when nothing is priced above the default.
        if (
            len(candidates) > 1
            and probabilities.get(default_name, 0.0) < config.confidence
        ):
            non_default = [e for n, e in estimates.items() if n != default_name]
            up = min(non_default, key=lambda estimate: estimate.total_usd)
            return RouteDecision(
                model_name=up.model_name,
                reason="cache_aware:escalate_default_unconfident",
                probabilities=probabilities,
                context_excluded=excluded,
                estimated_costs=estimated_costs,
                cache_prediction=up.cache_prediction.to_dict(),
            )
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
            model_name=default_name,
            reason=stayed_reason,
            probabilities=probabilities,
                context_excluded=excluded,
            estimated_costs=estimated_costs,
            cache_prediction=default_estimate.cache_prediction.to_dict(),
        )

    return RouteDecision(
        model_name=best.model_name,
        reason="cache_aware:candidate_cheaper",
        probabilities=probabilities,
                context_excluded=excluded,
        estimated_costs=estimated_costs,
        cache_prediction=best.cache_prediction.to_dict(),
    )
