from __future__ import annotations

from emissary_router.config import PricingConfig, TokenPricing
from emissary_router.dashboard.savings import compute_summary

PRICING = PricingConfig(
    pricing={
        "claude-sonnet-4.6": TokenPricing(
            input=3.0, output=15.0, cache_read=0.3, cache_write_5m=3.75
        ),
        "claude-haiku-4.5": TokenPricing(
            input=1.0, output=5.0, cache_read=0.1, cache_write_5m=1.25
        ),
    }
)


def test_savings_vs_baseline():
    # 1M input + 1M output served on haiku (actual) vs sonnet (baseline)
    aggregates = [
        {
            "served_model": "claude-haiku-4.5",
            "n": 1,
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cost_usd": 6.0,  # 1*1 + 1*5
        }
    ]
    summary = compute_summary(aggregates, PRICING, "claude-sonnet-4.6")
    assert summary["baseline_available"] is True
    assert summary["total_cost_usd"] == 6.0
    assert summary["baseline_cost_usd"] == 18.0  # 1*3 + 1*15
    assert summary["savings_usd"] == 12.0
    assert summary["savings_pct"] == round(12 / 18 * 100, 1)


def test_baseline_unavailable_when_no_price():
    aggregates = [
        {"served_model": "gemini-3.1-flash-lite", "n": 2, "input_tokens": 100,
         "output_tokens": 50, "cache_read_tokens": 0, "cache_creation_tokens": 0, "cost_usd": 0.001}
    ]
    summary = compute_summary(aggregates, PRICING, "no-such-model")
    assert summary["baseline_available"] is False
    assert summary["savings_usd"] is None
    assert summary["total_events"] == 2
