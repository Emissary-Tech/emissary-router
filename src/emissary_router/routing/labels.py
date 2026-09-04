"""Effort-suffixed classifier labels: ``<model>@<effort>``.

Newer classifier deployments emit per-effort heads (``gpt-5.6-luna@low``,
``gpt-5.6-luna@high``, …) next to, or instead of, plain model labels. The catalog and
the config know only base models, so labels are collapsed at routing time: the policy
sees one probability per base model (the best variant), and once a model is chosen the
winning variant's effort is forced onto the request. A plain label winning means the
client's own effort stands.
"""
from __future__ import annotations

EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def split_label(label: str) -> tuple[str, str | None]:
    """'gpt-5.6-luna@low' -> ('gpt-5.6-luna', 'low'); plain labels -> (label, None).

    Only a recognized effort level counts as a suffix, so model names that happen to
    contain '@' for other reasons are left alone.
    """
    base, sep, effort = label.rpartition("@")
    if sep and base and effort in EFFORT_LEVELS:
        return base, effort
    return label, None


def collapse_effort_labels(
    probabilities: dict[str, float],
) -> tuple[dict[str, float], dict[str, str]]:
    """Fold ``base@effort`` heads into one probability per base model.

    Returns (base_probabilities, winning_label_by_base). The winning label is the
    variant with the highest probability for that base — possibly the plain label.
    """
    base_probs: dict[str, float] = {}
    winner: dict[str, str] = {}
    for label, p in probabilities.items():
        base, _ = split_label(label)
        if base not in base_probs or p > base_probs[base]:
            base_probs[base] = p
            winner[base] = label
    return base_probs, winner


def forced_effort_for(label: str | None) -> str | None:
    """Effort to force for a chosen base model given its winning label (None = keep)."""
    if not label:
        return None
    return split_label(label)[1]
