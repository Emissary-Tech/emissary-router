"""Effort-suffixed classifier labels: ``<model>@<effort>``.

A classifier trained with effort arms emits one head per (model, effort) —
``claude-opus-5@low``, ``claude-opus-5@high`` — next to, or instead of, the plain
model head. The config and the catalog know only base models, so the heads are
collapsed to one probability per base model BEFORE the label gate and the policy
see them. Whether the router may decide a model's effort is a per-model config
choice (ModelEntry.effort_routing), the same rule the platform gateway applies:

- effort routing enabled for the model and the client did not turn thinking off
  or pin a budget: the model is confident when ANY effort clears the confidence
  gate, and it is served at the LOWEST effort that does — the cheapest effort the
  classifier vouches for. A client-sent effort is overridden by that choice.
  opus-5 passes 97% of its high-effort wins at low too, at a third of the output
  tokens (LLMRouterBench, 2026-10), so "highest-probability variant" would pick
  among near-identical heads at random; the lowest confident effort is what turns
  the arms into savings. No effort confident: the request goes out untouched (the
  client's effort, or the provider default) — the classifier had no opinion to impose.
- otherwise nothing is forced and the model is read at the head for the effort
  the request will actually run at: the client's effort (nearest head), the
  lowest head when thinking is off, else the plain head, else the best head.

A plain-label checkpoint passes through unchanged, so one serving path handles
both checkpoint generations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from emissary_router.providers.thinking import EFFORT_ORDER, ReasoningSettings

_EFFORT_SEP = "@"


def split_label(label: str) -> tuple[str, str | None]:
    """'claude-opus-5@low' -> ('claude-opus-5', 'low'); plain labels -> (label, None).

    Only a recognized effort level counts as a suffix, so a model name that happens
    to contain '@' for other reasons is left alone."""
    base, sep, effort = label.rpartition(_EFFORT_SEP)
    if sep and base and effort in EFFORT_ORDER:
        return base, effort
    return label, None


def has_effort_heads(probabilities: dict[str, float]) -> bool:
    return any(split_label(label)[1] is not None for label in probabilities)


def thinking_off(reasoning: ReasoningSettings) -> bool:
    """The client turned thinking off (effort none, reasoning.enabled false, or a
    thinking type that disables) — a contract effort routing never overrides."""
    return reasoning.effort == "none" or reasoning.enabled is False


def pins_budget(reasoning: ReasoningSettings) -> bool:
    """The client sized thinking with a token budget instead of a rung — also
    left alone: forcing an effort next to it would send two controls."""
    return reasoning.effort is None and reasoning.max_tokens is not None


@dataclass(frozen=True)
class HeadPick:
    """Which head stood for a base model, and what that means for the request."""
    label: str                  # the head read: 'claude-opus-5@low' or the plain name
    probability: float          # the probability the policy saw for the model
    forced_effort: str | None   # effort forced onto the request; None = untouched
    reason: str                 # how the head was chosen (telemetry)


def _rank(effort: str) -> int:
    return EFFORT_ORDER.index(effort)


def _nearest(arms: dict[str, float], target: str) -> str:
    """The arm closest to `target`; ties resolve upward (keep "think harder")."""
    rank = _rank(target)
    return min(arms, key=lambda e: (abs(_rank(e) - rank), -_rank(e)))


def collapse_effort_heads(
    probabilities: dict[str, float],
    *,
    routed: Iterable[str],
    confidence: float,
    reasoning: ReasoningSettings,
) -> tuple[dict[str, float], dict[str, HeadPick]]:
    """Fold ``base@effort`` heads into one probability per base model.

    `routed` names the models whose effort the router may decide; `reasoning` is
    what the client's request says about thinking. Returns (base_probabilities,
    pick_by_base); the probabilities carry the same keys a plain-label classifier
    would have emitted, so the label gate and the policy are none the wiser.
    """
    variants: dict[str, dict[str | None, float]] = {}
    for label, p in probabilities.items():
        base, effort = split_label(label)
        variants.setdefault(base, {})[effort] = p

    routed = set(routed)
    off = thinking_off(reasoning)
    pinned = off or pins_budget(reasoning)
    base_probs: dict[str, float] = {}
    picks: dict[str, HeadPick] = {}
    for base, by_effort in variants.items():
        plain = by_effort.get(None)
        arms = {e: p for e, p in by_effort.items() if e is not None}
        if not arms:
            pick = HeadPick(base, plain, None, "plain")
        elif base in routed and not pinned:
            pick = _route_effort(base, arms, plain, confidence)
        else:
            pick = _pin_effort(base, arms, plain, reasoning.effort, off)
        base_probs[base] = pick.probability
        picks[base] = pick
    return base_probs, picks


def _route_effort(
    base: str, arms: dict[str, float], plain: float | None, confidence: float
) -> HeadPick:
    # Confident at any effort -> confident; served at the lowest one that clears
    # the gate. The plain head (the provider's own default effort) only stands in
    # when no arm clears the gate — nothing is forced then, so the request runs at
    # that default anyway.
    best = max(list(arms.values()) + ([plain] if plain is not None else []))
    for effort in sorted(arms, key=_rank):
        if arms[effort] >= confidence:
            return HeadPick(f"{base}{_EFFORT_SEP}{effort}", best, effort, "lowest_confident_effort")
    if plain is not None and plain >= confidence:
        return HeadPick(base, best, None, "plain")
    label = max(arms, key=arms.get)
    return HeadPick(f"{base}{_EFFORT_SEP}{label}", best, None, "no_confident_effort")


def _pin_effort(
    base: str,
    arms: dict[str, float],
    plain: float | None,
    client_effort: str | None,
    off: bool,
) -> HeadPick:
    # The request goes out as it is; read the head for the effort it will run at.
    if off:
        effort, reason = min(arms, key=_rank), "lowest_effort"
    elif client_effort in EFFORT_ORDER:
        effort, reason = _nearest(arms, client_effort), "client_effort"
    elif plain is not None:
        # the plain head IS the measurement at the provider's default effort
        return HeadPick(base, plain, None, "plain")
    else:
        # arms only and no effort named: the provider default runs, which no head
        # names exactly — the best arm stands in (flagged for telemetry)
        effort, reason = max(arms, key=arms.get), "best_effort"
    return HeadPick(f"{base}{_EFFORT_SEP}{effort}", arms[effort], None, reason)
