from __future__ import annotations

import json

from emissary_router.pipeline import _routed_raw_event


def test_no_metadata_no_probs_stays_none():
    assert _routed_raw_event({}, "anthropic/claude-opus-5") is None


def test_or_cost_only_keeps_legacy_shape():
    raw = _routed_raw_event({"or_cost": 0.01, "id": "gen-1"}, "openrouter/auto")
    data = json.loads(raw)
    assert data["or_cost"] == 0.01
    assert "probs" not in data


def test_probs_persisted_with_tau_and_rounded():
    raw = _routed_raw_event(
        {}, "anthropic/claude-opus-5",
        probabilities={"claude-opus-5": 0.91237, "deepseek-v4-flash": 0.00042},
        tau=0.65,
    )
    data = json.loads(raw)
    assert data["probs"] == {"claude-opus-5": 0.9124, "deepseek-v4-flash": 0.0004}
    assert data["tau"] == 0.65
    assert "routed_model" not in data


def test_probs_merge_with_routed_metadata():
    raw = _routed_raw_event(
        {"openrouter_model": "deepseek/deepseek-v4-flash", "or_cost": 0.002},
        "anthropic/claude-opus-5",
        probabilities={"deepseek-v4-flash": 0.88},
        tau=0.65,
    )
    data = json.loads(raw)
    assert data["routed_model"] == "deepseek/deepseek-v4-flash"
    assert data["probs"]["deepseek-v4-flash"] == 0.88
