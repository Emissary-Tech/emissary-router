from __future__ import annotations

import asyncio
import json
import math

from emissary_router.config import AppConfig, RouterConfig
from emissary_router.pipeline import _routed_raw_event
from emissary_router.routing.calibration import logit, shift_probability, sigmoid, to_probabilities
from emissary_router.routing.classifier import ClassifierClient
from emissary_router.routing.policy import choose_model


def _config(**over):
    return AppConfig(
        models={"glm-5.2": True, "kimi-k3": True, "deepseek-v4-flash": True},
        default="glm-5.2",
        confidence=0.65,
        **over,
    )


class _Est:
    def __init__(self, name, usd, warm=False):
        self.model_name = name
        self.total_usd = usd
        self.cache_prediction = type("CP", (), {"warm": warm, "to_dict": lambda s: {}})()

    def to_dict(self):
        return {"total_usd": self.total_usd}


def _patch_costs(monkeypatch, costs):
    import emissary_router.routing.policy as pol
    monkeypatch.setattr(pol, "estimate_cost",
                        lambda config, name, feats, ledger, *a: _Est(name, costs[name]))
    monkeypatch.setattr(pol, "is_cheaper", lambda a, b: a.total_usd < b.total_usd)


# --- the transform itself ---------------------------------------------------------

def test_probs_with_zero_bias_are_returned_untouched():
    raw = {"glm-5.2": 0.9, "kimi-k3": 0.0, "deepseek-v4-flash": 1.0, "glm-5.2:len": 0.37}
    assert to_probabilities(raw, "probs") == raw
    for p in (0.0, 0.3, 0.65, 1.0):
        assert shift_probability(p, 0.0) == p


def test_logits_become_sigmoid_and_len_heads_are_never_shifted():
    raw = {"glm-5.2": 0.0, "kimi-k3": 2.0, "glm-5.2:len": -0.5}
    out = to_probabilities(raw, "logits")
    assert out["glm-5.2"] == 0.5 and abs(out["kimi-k3"] - sigmoid(2.0)) < 1e-12
    shifted = to_probabilities(raw, "logits", bias=1.0)
    assert abs(shifted["glm-5.2"] - sigmoid(1.0)) < 1e-12          # pass head: z + bias
    assert shifted["glm-5.2:len"] == out["glm-5.2:len"] == sigmoid(-0.5)  # len head: plain sigmoid
    shifted_p = to_probabilities({"glm-5.2": 0.8, "glm-5.2:len": 0.37}, "probs", bias=1.0)
    assert shifted_p["glm-5.2:len"] == 0.37


def test_bias_is_a_logit_shift_and_never_saturates():
    b = logit(0.65) - logit(0.8)                   # maps the old 0.8 gate onto 0.65
    assert abs(shift_probability(0.8, b) - 0.65) < 1e-9
    assert 0.0 < shift_probability(0.0, 5.0) < 1.0 and 0.0 < shift_probability(1.0, -5.0) < 1.0
    assert abs(to_probabilities({"a": 0.8}, "probs", b)["a"] - to_probabilities({"a": logit(0.8)}, "logits", b)["a"]) < 1e-9


# --- the shifted probabilities drive every decision --------------------------------

def test_negative_bias_removes_a_borderline_candidate(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0, "deepseek-v4-flash": 0.5, "kimi-k3": 3.0})
    raw = {"glm-5.2": 0.9, "deepseek-v4-flash": 0.70, "kimi-k3": 0.2}
    d0 = choose_model(_config(), to_probabilities(raw, "probs", 0.0), cost_features=object(), cache_ledger=object())
    assert d0.model_name == "deepseek-v4-flash" and d0.reason == "cache_aware:candidate_cheaper"
    d = choose_model(_config(), to_probabilities(raw, "probs", -0.5), cost_features=object(), cache_ledger=object())
    assert d.model_name == "glm-5.2" and d.reason == "cache_aware:no_confident_candidate"


def test_positive_bias_admits_a_below_gate_candidate(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0, "deepseek-v4-flash": 0.5, "kimi-k3": 3.0})
    raw = {"glm-5.2": 0.9, "deepseek-v4-flash": 0.60, "kimi-k3": 0.2}
    assert choose_model(_config(), raw, cost_features=object(), cache_ledger=object()).model_name == "glm-5.2"
    d = choose_model(_config(), to_probabilities(raw, "probs", 0.5), cost_features=object(), cache_ledger=object())
    assert d.model_name == "deepseek-v4-flash"


def test_escalation_reads_the_shifted_default_probability(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0, "kimi-k3": 3.0, "deepseek-v4-flash": 0.5})
    raw = {"glm-5.2": 0.66, "kimi-k3": 0.9, "deepseek-v4-flash": 0.1}
    d0 = choose_model(_config(), raw, cost_features=object(), cache_ledger=object())
    assert d0.model_name == "glm-5.2" and d0.reason == "cache_aware:default_not_beaten"
    d = choose_model(_config(), to_probabilities(raw, "probs", -0.2), cost_features=object(), cache_ledger=object())
    assert d.model_name == "kimi-k3" and d.reason == "cache_aware:escalate_default_unconfident"


def test_kappa_score_uses_the_shifted_probability(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0, "deepseek-v4-flash": 0.5, "kimi-k3": 3.0})
    raw = {"glm-5.2": 0.9, "deepseek-v4-flash": 0.70, "kimi-k3": 0.2}
    shifted = to_probabilities(raw, "probs", 1.0)
    d = choose_model(_config(cost_aware=True, kappa_usd=1.0), shifted, cost_features=object(), cache_ledger=object())
    expected = 0.5 + 1.0 * (1 - shifted["deepseek-v4-flash"])
    assert abs(d.estimated_costs["deepseek-v4-flash"]["score_usd"] - expected) < 1e-8
    assert d.probabilities == shifted


# --- wiring: classifier request format and telemetry ---------------------------------

def test_classifier_client_requests_the_configured_format(monkeypatch):
    seen = {}

    async def fake_post(self, headers, payload):
        seen.update(payload)
        return {"data": [{payload["data_format"]: {"glm-5.2": 1.5, "glm-5.2:len": -0.3}}]}

    monkeypatch.setattr(ClassifierClient, "_post_with_retry", fake_post)
    out = asyncio.run(ClassifierClient(RouterConfig()).predict("x"))
    assert seen["data_format"] == "logits" and out == {"glm-5.2": 1.5, "glm-5.2:len": -0.3}
    out = asyncio.run(ClassifierClient(RouterConfig(data_format="probs")).predict("x"))
    assert seen["data_format"] == "probs"


def test_raw_event_records_unshifted_probs_and_the_bias():
    off = json.loads(_routed_raw_event({}, "m", {"glm-5.2": 0.5}, 0.65, bias=0.0))
    on = json.loads(_routed_raw_event({}, "m", {"glm-5.2": 0.5}, 0.65, bias=-0.3))
    assert "confidence_bias" not in off
    assert on["confidence_bias"] == -0.3 and on["probs"] == {"glm-5.2": 0.5} and on["tau"] == 0.65


def test_default_config_is_bias_free_and_asks_for_logits():
    cfg = _config()
    assert cfg.confidence_bias == 0.0 and cfg.router.data_format == "logits"
