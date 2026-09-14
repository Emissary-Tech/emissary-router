"""Length heads ("<model>:len") and the kappa failure penalty in cost-aware routing."""
from __future__ import annotations

from emissary_router.caching.ledger import CacheLedger
from emissary_router.config import AppConfig
from emissary_router.routing.cache_cost import (
    LEN_CAP_TOKENS,
    LEN_FLOOR_TOKENS,
    RequestCostFeatures,
    len_to_tokens,
)
from emissary_router.routing.labels import (
    collapse_effort_labels,
    expected_output_by_model,
    split_len_labels,
)
from emissary_router.routing.policy import choose_model


def _config(**overrides) -> AppConfig:
    raw = {
        "models": {
            "claude-opus-5": True,
            "gpt-5.6-luna": True,
            "deepseek-v4-flash-0731": True,
        },
        "default": "claude-opus-5",
        "confidence": 0.65,
        "cost_aware": True,
    }
    raw.update(overrides)
    return AppConfig.model_validate(raw)


def _features() -> RequestCostFeatures:
    # no session -> cold cache for every model: a flat per-request price comparison
    return RequestCostFeatures(
        session_id=None,
        prefix_hash="prefix",
        estimated_input_tokens=2000,
        estimated_cacheable_prefix_tokens=1500,
        estimated_fresh_input_tokens=500,
        expected_output_tokens=1024,
    )


def test_split_len_labels_keeps_pass_heads_and_effort_suffixes() -> None:
    probs = {
        "gpt-5.6-luna": 0.9,
        "gpt-5.6-luna:len": 0.3,
        "claude-opus-5@high": 0.8,
        "claude-opus-5@high:len": 0.5,
    }
    pass_probs, len_by_label = split_len_labels(probs)
    assert pass_probs == {"gpt-5.6-luna": 0.9, "claude-opus-5@high": 0.8}
    assert len_by_label == {"gpt-5.6-luna": 0.3, "claude-opus-5@high": 0.5}


def test_len_to_tokens_inverts_the_training_normalization() -> None:
    assert len_to_tokens(0.0) == LEN_FLOOR_TOKENS
    assert len_to_tokens(1.0) == LEN_CAP_TOKENS
    assert len_to_tokens(0.5) < len_to_tokens(0.6)
    assert len_to_tokens(0.5, correction=2.0) == 2 * len_to_tokens(0.5)
    assert len_to_tokens(1.7) == LEN_CAP_TOKENS  # clipped, never above the cap


def test_expected_output_follows_the_winning_effort_variant() -> None:
    probs = {
        "gpt-5.6-luna@low": 0.6, "gpt-5.6-luna@low:len": 0.2,
        "gpt-5.6-luna@high": 0.9, "gpt-5.6-luna@high:len": 0.8,
        "claude-opus-5": 0.7, "claude-opus-5:len": 0.4,
    }
    pass_probs, len_by_label = split_len_labels(probs)
    base_probs, winner = collapse_effort_labels(pass_probs)
    expected = expected_output_by_model(len_by_label, winner)
    assert expected["gpt-5.6-luna"] == len_to_tokens(0.8)  # the high variant won
    assert expected["claude-opus-5"] == len_to_tokens(0.4)


def test_per_model_expected_output_reprices_candidates() -> None:
    config = _config()
    probs = {"claude-opus-5": 0.9, "gpt-5.6-luna": 0.9, "deepseek-v4-flash-0731": 0.9}
    # request-level estimate only: 0731 is the cheaper model per token
    flat = choose_model(config, probs, cost_features=_features(), cache_ledger=CacheLedger())
    assert flat.model_name == "deepseek-v4-flash-0731"
    # length heads say 0731 would write 20k tokens here and luna 200: luna is cheaper
    verbose = choose_model(
        config, probs, cost_features=_features(), cache_ledger=CacheLedger(),
        expected_output_by_model={"gpt-5.6-luna": 200, "deepseek-v4-flash-0731": 20000},
    )
    assert verbose.model_name == "gpt-5.6-luna"
    assert verbose.estimated_costs["deepseek-v4-flash-0731"]["expected_output_tokens"] == 20000
    assert verbose.estimated_costs["gpt-5.6-luna"]["expected_output_tokens"] == 200
    # a model without a length head keeps the rolling estimate
    assert verbose.estimated_costs["claude-opus-5"]["expected_output_tokens"] == 1024


def test_kappa_makes_a_borderline_cheap_candidate_lose_to_the_sure_default() -> None:
    probs = {"claude-opus-5": 0.99, "gpt-5.6-luna": 0.7, "deepseek-v4-flash-0731": 0.2}
    cheap = choose_model(_config(), probs, cost_features=_features(), cache_ledger=CacheLedger())
    assert cheap.model_name == "gpt-5.6-luna"
    assert cheap.reason == "cache_aware:candidate_cheaper"

    penalized = choose_model(
        _config(kappa_usd=1.0), probs, cost_features=_features(), cache_ledger=CacheLedger()
    )
    assert penalized.model_name == "claude-opus-5"
    assert penalized.reason == "cache_aware:default_not_beaten"
    luna = penalized.estimated_costs["gpt-5.6-luna"]
    assert luna["score_usd"] > luna["total_usd"]  # 1.0 * (1 - 0.7) added
    opus = penalized.estimated_costs["claude-opus-5"]
    assert abs(opus["score_usd"] - opus["total_usd"] - 0.01) < 1e-6


def test_kappa_zero_without_len_heads_is_the_previous_policy() -> None:
    config = _config()
    for probs in (
        {"claude-opus-5": 0.9, "gpt-5.6-luna": 0.9, "deepseek-v4-flash-0731": 0.9},
        {"claude-opus-5": 0.9, "gpt-5.6-luna": 0.7, "deepseek-v4-flash-0731": 0.1},
        {"claude-opus-5": 0.3, "gpt-5.6-luna": 0.1, "deepseek-v4-flash-0731": 0.1},
    ):
        before = choose_model(config, probs, cost_features=_features(), cache_ledger=CacheLedger())
        after = choose_model(
            config, probs, cost_features=_features(), cache_ledger=CacheLedger(),
            expected_output_by_model={},
        )
        assert (before.model_name, before.reason) == (after.model_name, after.reason)
        for name, cost in after.estimated_costs.items():
            assert cost["score_usd"] == cost["total_usd"]


def test_cost_aware_off_makes_kappa_inert() -> None:
    probs = {"claude-opus-5": 0.99, "gpt-5.6-luna": 0.7, "deepseek-v4-flash-0731": 0.2}
    off = choose_model(
        _config(kappa_usd=1.0, cost_aware=False), probs,
        cost_features=_features(), cache_ledger=CacheLedger(),
    )
    assert off.model_name == "gpt-5.6-luna"  # the plain cheapest-confident pick
    for cost in off.estimated_costs.values():
        assert cost["score_usd"] == cost["total_usd"]


def test_default_without_a_head_carries_no_failure_penalty() -> None:
    probs = {"gpt-5.6-luna": 0.7, "deepseek-v4-flash-0731": 0.2}  # anchorless classifier
    decision = choose_model(
        _config(kappa_usd=1.0), probs, cost_features=_features(), cache_ledger=CacheLedger()
    )
    opus = decision.estimated_costs["claude-opus-5"]
    assert opus["score_usd"] == opus["total_usd"]  # missing head != sure failure
    # The pick itself follows the pre-existing escalation rule: a default with no
    # head reads as unconfident, so a confident candidate is served even at a
    # worse score. kappa only reorders candidates; it does not override that rule.
    assert decision.model_name == "gpt-5.6-luna"
    assert decision.reason == "cache_aware:escalate_default_unconfident"
