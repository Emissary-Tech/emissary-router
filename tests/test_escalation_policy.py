from __future__ import annotations

from emissary_router.routing.policy import choose_model


def _config(escalate: bool, tmp_path_factory=None, **over):
    from emissary_router.config import AppConfig
    return AppConfig(
        models={"glm-5.2": True, "kimi-k3": True, "deepseek-v4-flash": True},
        default="glm-5.2",
        confidence=0.65,
        escalate_if_default_unconfident=escalate,
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
                        lambda config, name, feats, ledger: _Est(name, costs[name]))
    monkeypatch.setattr(pol, "is_cheaper",
                        lambda a, b: a.total_usd < b.total_usd)


def test_flag_off_pricier_confident_stays_on_default(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0, "kimi-k3": 3.0})
    d = choose_model(_config(False), {"glm-5.2": 0.2, "kimi-k3": 0.9},
                     cost_features=object(), cache_ledger=object())
    assert d.model_name == "glm-5.2"


def test_flag_on_default_unconfident_escalates_to_pricier(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0, "kimi-k3": 3.0})
    d = choose_model(_config(True), {"glm-5.2": 0.2, "kimi-k3": 0.9},
                     cost_features=object(), cache_ledger=object())
    assert d.model_name == "kimi-k3"
    assert d.reason == "cache_aware:escalate_default_unconfident"


def test_flag_on_default_confident_stays(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0, "kimi-k3": 3.0})
    d = choose_model(_config(True), {"glm-5.2": 0.8, "kimi-k3": 0.9},
                     cost_features=object(), cache_ledger=object())
    assert d.model_name == "glm-5.2"


def test_flag_on_cheaper_confident_keeps_normal_path(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0, "deepseek-v4-flash": 0.3})
    d = choose_model(_config(True), {"glm-5.2": 0.2, "deepseek-v4-flash": 0.9},
                     cost_features=object(), cache_ledger=object())
    assert d.model_name == "deepseek-v4-flash"
    assert d.reason == "cache_aware:candidate_cheaper"


def test_flag_on_nothing_confident_stays_default(monkeypatch):
    _patch_costs(monkeypatch, {"glm-5.2": 1.0})
    d = choose_model(_config(True), {"glm-5.2": 0.2, "kimi-k3": 0.3},
                     cost_features=object(), cache_ledger=object())
    assert d.model_name == "glm-5.2"
    assert d.reason == "cache_aware:no_confident_candidate"
