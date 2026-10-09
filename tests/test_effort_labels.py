"""Effort routing: per-effort classifier heads -> one probability per model, the
lowest confident effort forced onto the request (routing/labels.py, thinking.py).
Same rules as the platform gateway's effort routing."""
from __future__ import annotations

import json

from emissary_router.config import AppConfig
from emissary_router.pipeline import _routed_raw_event
from emissary_router.providers.thinking import (
    ReasoningSettings,
    extract_reasoning_settings,
    force_effort,
)
from emissary_router.routing.labels import (
    HeadPick,
    collapse_effort_heads,
    has_effort_heads,
    pins_budget,
    split_label,
    thinking_off,
)

OPUS = {"claude-opus-5@low": 0.93, "claude-opus-5@medium": 0.95, "claude-opus-5@high": 0.96}
FLASH = {"glm-5.3-flash": 0.6}
SILENT = ReasoningSettings()


def _collapse(probs, *, routed=("claude-opus-5",), confidence=0.9, reasoning=SILENT):
    return collapse_effort_heads(probs, routed=routed, confidence=confidence, reasoning=reasoning)


def test_split_label_recognizes_effort_suffix_only():
    assert split_label("claude-opus-5@low") == ("claude-opus-5", "low")
    assert split_label("claude-opus-5") == ("claude-opus-5", None)
    assert split_label("team@acme") == ("team@acme", None)
    assert has_effort_heads(OPUS) and not has_effort_heads(FLASH)


def test_plain_checkpoint_passes_through_unchanged():
    probs = {"glm-5.3-flash": 0.6, "claude-opus-5": 0.97}
    base, picks = _collapse(probs)
    assert base == probs
    assert picks["claude-opus-5"].forced_effort is None
    assert picks["claude-opus-5"].reason == "plain"


def test_enabled_model_forces_lowest_confident_effort():
    base, picks = _collapse({**OPUS, **FLASH})
    assert base == {"claude-opus-5": 0.96, "glm-5.3-flash": 0.6}   # confident at any effort
    pick = picks["claude-opus-5"]
    assert pick.forced_effort == "low" and pick.label == "claude-opus-5@low"
    assert pick.reason == "lowest_confident_effort"


def test_enabled_model_skips_efforts_below_the_gate():
    _, picks = _collapse({"claude-opus-5@low": 0.5, "claude-opus-5@medium": 0.92, "claude-opus-5@high": 0.95})
    assert picks["claude-opus-5"].forced_effort == "medium"


def test_enabled_model_with_no_confident_effort_forces_nothing():
    base, picks = _collapse({"claude-opus-5@low": 0.5, "claude-opus-5@medium": 0.6, "claude-opus-5@high": 0.7})
    assert base["claude-opus-5"] == 0.7
    assert picks["claude-opus-5"].forced_effort is None
    assert picks["claude-opus-5"].reason == "no_confident_effort"


def test_enabled_model_overrides_a_client_effort():
    _, picks = _collapse(OPUS, reasoning=ReasoningSettings(effort="high"))
    assert picks["claude-opus-5"].forced_effort == "low"


def test_thinking_off_is_never_overridden_and_reads_the_lowest_head():
    for reasoning in (ReasoningSettings(effort="none"), ReasoningSettings(enabled=False)):
        base, picks = _collapse(OPUS, reasoning=reasoning)
        pick = picks["claude-opus-5"]
        assert pick.forced_effort is None and pick.label == "claude-opus-5@low"
        assert base["claude-opus-5"] == 0.93 and pick.reason == "lowest_effort"


def test_budget_is_a_client_setting_nothing_is_forced():
    _, picks = _collapse(OPUS, reasoning=ReasoningSettings(max_tokens=2048))
    pick = picks["claude-opus-5"]
    assert pick.forced_effort is None and pick.reason == "best_effort"


def test_disabled_model_reads_the_clients_effort_snapped_to_a_head():
    base, picks = _collapse(OPUS, routed=(), reasoning=ReasoningSettings(effort="xhigh"))
    pick = picks["claude-opus-5"]
    assert pick.label == "claude-opus-5@high" and pick.forced_effort is None
    assert base["claude-opus-5"] == 0.96 and pick.reason == "client_effort"


def test_disabled_model_silent_request_reads_plain_else_best_head():
    base, picks = _collapse({**OPUS, "claude-opus-5": 0.5}, routed=())
    assert picks["claude-opus-5"].label == "claude-opus-5" and base["claude-opus-5"] == 0.5
    base, picks = _collapse({"gpt-5.6-luna@low": 0.4, "gpt-5.6-luna@high": 0.8}, routed=())
    assert base == {"gpt-5.6-luna": 0.8} and picks["gpt-5.6-luna"].reason == "best_effort"


def test_plain_head_stands_in_when_no_effort_is_confident():
    probs = {**{k: 0.5 for k in OPUS}, "claude-opus-5": 0.95}
    base, picks = _collapse(probs)
    assert picks["claude-opus-5"].forced_effort is None
    assert picks["claude-opus-5"].label == "claude-opus-5" and base["claude-opus-5"] == 0.95


def test_client_reasoning_predicates_per_dialect():
    assert thinking_off(extract_reasoning_settings({"reasoning_effort": "none"}))
    assert thinking_off(extract_reasoning_settings({"reasoning": {"enabled": False}}))
    assert thinking_off(extract_reasoning_settings({"thinking": {"type": "disabled"}}))
    assert not thinking_off(extract_reasoning_settings({"reasoning_effort": "low"}))
    assert not thinking_off(extract_reasoning_settings({"thinking": {"type": "adaptive"}}))
    assert pins_budget(extract_reasoning_settings({"thinking": {"type": "enabled", "budget_tokens": 2048}}))
    assert pins_budget(extract_reasoning_settings({"reasoning": {"max_tokens": 2048}}))
    assert not pins_budget(extract_reasoning_settings({"reasoning": {"effort": "low", "max_tokens": 2048}}))
    assert extract_reasoning_settings({"output_config": {"effort": "max"}}).effort == "max"
    assert extract_reasoning_settings({}) == ReasoningSettings()


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


def test_model_entry_effort_routing_flag_defaults_off():
    cfg = AppConfig.model_validate({
        "models": {"claude-opus-5": {"enabled": True, "effort_routing": True}, "glm-5.3-flash": True},
        "default": "claude-opus-5",
    })
    assert cfg.models["claude-opus-5"].effort_routing is True
    assert cfg.models["glm-5.3-flash"].effort_routing is False


def test_raw_event_records_head_and_forced_effort():
    pick = HeadPick("claude-opus-5@low", 0.96, "low", "lowest_confident_effort")
    raw = _routed_raw_event({}, "claude-opus-5", probabilities=OPUS, tau=0.9,
                            pick=pick, effort_changes=["output_config.effort=high->low"])
    data = json.loads(raw)
    assert data["effort_head"] == "claude-opus-5@low" and data["forced_effort"] == "low"
    assert data["effort_changes"] == ["output_config.effort=high->low"]
    assert data["probs"] == {k: round(v, 4) for k, v in OPUS.items()}   # raw heads, arms included
    # a plain head with nothing forced leaves no effort keys (legacy shape preserved)
    raw2 = _routed_raw_event({}, "x", probabilities={"a": 0.5}, tau=0.65,
                             pick=HeadPick("a", 0.5, None, "plain"))
    assert "effort_head" not in json.loads(raw2)
    # an arm read without forcing is still recorded, for the dashboard/replay
    raw3 = _routed_raw_event({}, "x", probabilities=OPUS, tau=0.9,
                             pick=HeadPick("claude-opus-5@high", 0.96, None, "client_effort"))
    d3 = json.loads(raw3)
    assert d3["effort_head"] == "claude-opus-5@high" and "forced_effort" not in d3


def test_default_needs_no_classifier_head(monkeypatch):
    from emissary_router.pipeline import RouterPipeline
    cfg = AppConfig.model_validate({
        "models": {"claude-opus-5": True, "claude-haiku-4.5": True, "gemini-3.1-flash-lite": True},
        "default": "claude-opus-5", "confidence": 0.8,
    })
    pipe = RouterPipeline.__new__(RouterPipeline); pipe._config = cfg
    # classifier has heads for the cheap models only — the anchor is gate-exempt
    assert pipe._missing_probability_labels({"claude-haiku-4.5": 0.2, "gemini-3.1-flash-lite": 0.9}) == []
    assert pipe._missing_probability_labels({"claude-haiku-4.5": 0.2}) == ["gemini-3.1-flash-lite"]
