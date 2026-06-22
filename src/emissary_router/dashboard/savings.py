from __future__ import annotations

from typing import Any

from emissary_router.config import PricingConfig, TokenPricing


def _cost_for_tokens(price: TokenPricing, row: dict[str, Any]) -> float:
    cache_write = price.cache_write_5m
    return (
        (row.get("input_tokens") or 0) * price.input
        + (row.get("output_tokens") or 0) * price.output
        + (row.get("cache_read_tokens") or 0) * price.cache_read
        + (row.get("cache_creation_tokens") or 0) * cache_write
    ) / 1_000_000


def compute_summary(
    aggregates: list[dict[str, Any]],
    pricing: PricingConfig,
    baseline_model: str,
) -> dict[str, Any]:
    """Totals + estimated savings vs sending every call to the baseline model.

    Savings is an estimate: it applies the baseline model's prices to each call's
    actual token counts, which differ from what the baseline model would really emit.
    """
    total_cost = sum((row.get("cost_usd") or 0.0) for row in aggregates)
    total_events = sum((row.get("n") or 0) for row in aggregates)

    by_model = [
        {
            "served_model": row["served_model"],
            "n": row.get("n") or 0,
            "cost_usd": round(row.get("cost_usd") or 0.0, 6),
        }
        for row in aggregates
    ]

    baseline_price = pricing.pricing.get(baseline_model)
    if baseline_price is None:
        return {
            "total_events": total_events,
            "total_cost_usd": round(total_cost, 6),
            "baseline_model": baseline_model,
            "baseline_available": False,
            "baseline_cost_usd": None,
            "savings_usd": None,
            "savings_pct": None,
            "by_model": by_model,
        }

    baseline_cost = sum(_cost_for_tokens(baseline_price, row) for row in aggregates)
    savings = baseline_cost - total_cost
    savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
    return {
        "total_events": total_events,
        "total_cost_usd": round(total_cost, 6),
        "baseline_model": baseline_model,
        "baseline_available": True,
        "baseline_cost_usd": round(baseline_cost, 6),
        "savings_usd": round(savings, 6),
        "savings_pct": round(savings_pct, 1),
        "by_model": by_model,
    }
