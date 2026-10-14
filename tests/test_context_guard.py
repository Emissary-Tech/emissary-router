"""Context guard (platform parity): a model whose window the request exceeds is not a
candidate; the anchor re-seats onto the priciest fitting model; nothing fitting ->
default with `context_exceeds_config`."""
from __future__ import annotations

import json
from types import SimpleNamespace

from emissary_router.catalog import CATALOG
from emissary_router.config import AppConfig
from emissary_router.pipeline import _routed_raw_event
from emissary_router.routing.policy import choose_model, fits_context

PLATFORM_WINDOWS = {
    "deepseek-v4-flash": 1_000_000, "deepseek-v4-flash-0731": 1_000_000, "gpt-6-luna": 1_050_000,
    "qwen3.8-omni-flash": 1_000_000, "qwen3.8-flash": 1_000_000, "deepseek-v4.1-flash": 1_048_576,
    "gpt-5.6-luna": 1_050_000, "gemini-3.1-flash-lite": 1_000_000, "kimi-k2.7-code": 262_000,
    "claude-haiku-4.5": 200_000, "glm-5.2": 1_048_576, "glm-5.3": 1_048_576, "glm-5.3-flash": 1_048_576,
    "gpt-5.6-terra": 1_050_000, "claude-sonnet-5": 1_000_000, "claude-sonnet-5.5": 1_000_000,
    "gpt-6-sol": 1_050_000, "gpt-6.1-sol": 1_050_000, "kimi-k3": 1_048_576, "gpt-5.6-sol": 1_050_000,
    "claude-opus-5.5": 1_000_000, "claude-opus-5": 1_000_000,
}


def _config(default="claude-opus-5"):
    return AppConfig(models={"claude-haiku-4.5": True, "kimi-k2.7-code": True, "claude-opus-5": True},
                     default=default, confidence=0.65)


def _feats(input_tokens, expected_output=1024, ceiling=None):
    return SimpleNamespace(estimated_input_tokens=input_tokens, expected_output_tokens=expected_output,
                           output_ceiling_tokens=ceiling)


class _Est:
    def __init__(self, name, usd):
        self.model_name = name; self.total_usd = usd
        self.cache_prediction = type("CP", (), {"warm": False, "to_dict": lambda s: {}})()

    def to_dict(self):
        return {"total_usd": self.total_usd}


def _patch_costs(monkeypatch):
    import emissary_router.routing.policy as pol
    usd = {"claude-haiku-4.5": 0.2, "kimi-k2.7-code": 0.5, "claude-opus-5": 3.0}
    monkeypatch.setattr(pol, "estimate_cost", lambda config, name, feats, ledger, *a: _Est(name, usd[name]))
    monkeypatch.setattr(pol, "is_cheaper", lambda a, b: a.total_usd < b.total_usd)


def test_catalog_windows_match_the_platform_catalog():
    assert {n: CATALOG[n].context_window for n in PLATFORM_WINDOWS} == PLATFORM_WINDOWS


def test_fits_context_boundary_and_unknowns():
    haiku = CATALOG["claude-haiku-4.5"]
    assert fits_context(haiku, _feats(200_000 - 1024))
    assert not fits_context(haiku, _feats(200_000 - 1024 + 1))
    # the client's max_tokens is what the provider reserves: Claude Code's 64K cap keeps
    # haiku out from 136K of input on, however small the expected output is
    assert fits_context(haiku, _feats(136_000, expected_output=500, ceiling=64_000))
    assert not fits_context(haiku, _feats(136_001, expected_output=500, ceiling=64_000))
    # no cap sent -> the expected-output estimate stands in
    assert fits_context(haiku, _feats(198_000, expected_output=1_000, ceiling=None))
    # features from the real extractor carry the ceiling
    from emissary_router.routing.cache_cost import extract_request_cost_features
    feats = extract_request_cost_features({"max_tokens": 64_000, "messages": [{"role": "user", "content": "x" * 400}]}, {})
    assert feats.output_ceiling_tokens == 64_000
    assert extract_request_cost_features({"messages": [{"role": "user", "content": "hi"}]}, {}).output_ceiling_tokens is None
    assert fits_context(SimpleNamespace(context_window=None), _feats(5_000_000))   # unknown window
    assert fits_context(haiku, object())                                          # no estimate
    assert fits_context(None, _feats(5_000_000))                                  # model not in catalog


def test_confident_small_window_model_is_not_a_candidate(monkeypatch):
    _patch_costs(monkeypatch)
    # 250K prompt: haiku (200K) is out even though it is the confident cheap pick
    d = choose_model(_config(), {"claude-haiku-4.5": 0.99, "kimi-k2.7-code": 0.2},
                     cost_features=_feats(250_000), cache_ledger=object())
    assert d.model_name == "claude-opus-5"
    assert d.reason == "cache_aware:no_confident_candidate"
    assert d.context_excluded == ["claude-haiku-4.5"]


def test_fitting_confident_model_still_wins(monkeypatch):
    _patch_costs(monkeypatch)
    d = choose_model(_config(), {"claude-haiku-4.5": 0.99, "kimi-k2.7-code": 0.9},
                     cost_features=_feats(250_000), cache_ledger=object())
    assert d.model_name == "kimi-k2.7-code" and d.reason == "cache_aware:candidate_cheaper"
    assert d.context_excluded == ["claude-haiku-4.5"]


def test_default_that_cannot_take_the_request_reseats_the_anchor(monkeypatch):
    _patch_costs(monkeypatch)
    # default haiku is out at 250K; the priciest fitting model (opus) anchors instead
    d = choose_model(_config(default="claude-haiku-4.5"), {"kimi-k2.7-code": 0.1, "claude-opus-5": 0.1},
                     cost_features=_feats(250_000), cache_ledger=object())
    assert d.model_name == "claude-opus-5"
    assert d.reason == "cache_aware:no_confident_candidate"
    assert d.context_excluded == ["claude-haiku-4.5"]
    # and at 300K kimi (262K) drops too; only opus is left
    d = choose_model(_config(default="claude-haiku-4.5"), {"kimi-k2.7-code": 0.99},
                     cost_features=_feats(300_000), cache_ledger=object())
    assert d.model_name == "claude-opus-5"
    assert sorted(d.context_excluded) == ["claude-haiku-4.5", "kimi-k2.7-code"]


def test_nothing_fits_serves_the_default_and_says_so(monkeypatch):
    _patch_costs(monkeypatch)
    d = choose_model(_config(), {"claude-haiku-4.5": 0.99}, cost_features=_feats(2_000_000), cache_ledger=object())
    assert d.model_name == "claude-opus-5" and d.reason == "context_exceeds_config"
    assert sorted(d.context_excluded) == ["claude-haiku-4.5", "claude-opus-5", "kimi-k2.7-code"]


def test_small_requests_are_unaffected(monkeypatch):
    _patch_costs(monkeypatch)
    d = choose_model(_config(), {"claude-haiku-4.5": 0.99}, cost_features=_feats(20_000), cache_ledger=object())
    assert d.model_name == "claude-haiku-4.5" and d.context_excluded is None


def test_raw_event_records_the_exclusions():
    data = json.loads(_routed_raw_event({}, "claude-opus-5", probabilities={"a": 0.5}, tau=0.65,
                                        context_excluded=["claude-haiku-4.5"]))
    assert data["context_excluded"] == ["claude-haiku-4.5"]
    assert "context_excluded" not in json.loads(_routed_raw_event({}, "x", probabilities={"a": 0.5}, tau=0.65))
