from __future__ import annotations

import json

from emissary_router.pipeline import _routed_raw_event
from emissary_router.providers.thinking import force_effort
from emissary_router.routing.labels import collapse_effort_labels, forced_effort_for, split_label


def test_split_label_only_recognizes_effort_suffixes():
    assert split_label("gpt-5.6-luna@low") == ("gpt-5.6-luna", "low")
    assert split_label("gpt-5.6-luna") == ("gpt-5.6-luna", None)
    assert split_label("weird@name") == ("weird@name", None)   # not an effort level


def test_collapse_keeps_best_variant_per_base():
    probs = {"gpt-5.6-luna@low": 0.9, "gpt-5.6-luna@high": 0.4, "gpt-5.6-luna": 0.2,
             "claude-opus-5": 0.7, "deepseek-v4-flash@medium": 0.3}
    base, winner = collapse_effort_labels(probs)
    assert base == {"gpt-5.6-luna": 0.9, "claude-opus-5": 0.7, "deepseek-v4-flash": 0.3}
    assert winner["gpt-5.6-luna"] == "gpt-5.6-luna@low"
    assert winner["claude-opus-5"] == "claude-opus-5"          # plain label wins -> no override
    assert forced_effort_for(winner["gpt-5.6-luna"]) == "low"
    assert forced_effort_for(winner["claude-opus-5"]) is None
    assert forced_effort_for(None) is None


def test_force_effort_overwrites_every_effort_location():
    body = {"output_config": {"effort": "high"}, "reasoning": {"effort": "high", "max_tokens": 5},
            "thinking": {"type": "adaptive", "effort": "high"}, "effort": "high"}
    changes = force_effort(body, "low")
    assert body["output_config"]["effort"] == "low"
    assert body["reasoning"]["effort"] == "low" and body["reasoning"]["max_tokens"] == 5
    assert body["thinking"]["effort"] == "low"
    assert body["effort"] == "low"
    assert len(changes) == 4


def test_force_effort_creates_output_config_when_absent():
    body = {"messages": []}
    force_effort(body, "xhigh")
    assert body["output_config"] == {"effort": "xhigh"}
    assert force_effort(body, "xhigh") == []   # idempotent


def test_raw_event_records_label_and_forced_effort():
    raw = _routed_raw_event({}, "openai/gpt-5.6-luna", probabilities={"gpt-5.6-luna@low": 0.9},
                            tau=0.65, label="gpt-5.6-luna@low", forced_effort="low")
    data = json.loads(raw)
    assert data["label"] == "gpt-5.6-luna@low" and data["forced_effort"] == "low"
    assert data["probs"] == {"gpt-5.6-luna@low": 0.9}
    # no suffix chosen -> keys absent (legacy shape preserved)
    raw2 = _routed_raw_event({}, "x", probabilities={"a": 0.5}, tau=0.65)
    assert "forced_effort" not in json.loads(raw2)


def test_suffix_only_classifier_routes_as_base_with_forced_effort():
    # the deployed classifier has no plain "gpt-5.6-luna" head — only the three variants
    probs = {"gpt-5.6-luna@low": 0.31, "gpt-5.6-luna@medium": 0.82, "gpt-5.6-luna@high": 0.55,
             "claude-opus-5": 0.40}
    base, winner = collapse_effort_labels(probs)
    assert base["gpt-5.6-luna"] == 0.82                 # max over variants gates the model
    assert "gpt-5.6-luna@medium" not in base              # policy/pricing/cache see the base name only
    assert winner["gpt-5.6-luna"] == "gpt-5.6-luna@medium"
    assert forced_effort_for(winner["gpt-5.6-luna"]) == "medium"
    body = {"output_config": {"effort": "high"}}          # what Claude Code sent
    force_effort(body, forced_effort_for(winner["gpt-5.6-luna"]))
    assert body["output_config"]["effort"] == "medium"    # only the request effort changes
