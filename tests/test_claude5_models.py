"""New-model registrations (2026-08: sonnet-5 default swap, opus-5, kimi-k3)."""
from emissary_router.catalog import CATALOG, cost_score
from emissary_router.providers.thinking import (
    THINKING_CAPABILITIES,
    strip_unsupported_sampling_params,
)


def test_claude5_thinking_capabilities_match_sonnet46_surface() -> None:
    for name in ("claude-sonnet-5", "claude-opus-5"):
        cap = THINKING_CAPABILITIES[name]
        assert cap.accepts_adaptive_thinking
        assert cap.accepts_effort_param
        assert cap.max_effort == "max"


def test_kimi_k3_is_always_on_reasoning_like_k27() -> None:
    cap = THINKING_CAPABILITIES["kimi-k3"]
    assert not cap.can_disable_thinking
    assert cap.accepts_effort_param
    assert not cap.accepts_adaptive_thinking


def test_claude5_sampling_params_are_stripped() -> None:
    body = {"temperature": 1, "top_p": 0.9, "max_tokens": 32000}
    changes = strip_unsupported_sampling_params(body, "claude-opus-5")
    assert "temperature" not in body and "top_p" not in body
    assert body["max_tokens"] == 32000  # only sampling params are touched
    assert len(changes) == 2


def test_sampling_params_untouched_for_other_models() -> None:
    body = {"temperature": 0.2}
    assert strip_unsupported_sampling_params(body, "claude-haiku-4.5") == []
    assert body["temperature"] == 0.2


def test_new_models_price_ladder() -> None:
    # 2026-08 sonnet price cut (3/15 -> 2/10): sonnet now undercuts kimi-k3
    assert cost_score(CATALOG["claude-sonnet-5"]) < cost_score(CATALOG["kimi-k3"])
    assert cost_score(CATALOG["claude-opus-5"]) > cost_score(CATALOG["claude-sonnet-5"])
    assert set(CATALOG["kimi-k3"].providers) == {"openrouter"}


def test_effort_vocabulary_snapping() -> None:
    from emissary_router.providers.thinking import resolve_effort_for_model

    # claude-5 supports the full ladder -> xhigh and max pass through unchanged
    assert resolve_effort_for_model("xhigh", "claude-sonnet-5") == "xhigh"
    assert resolve_effort_for_model("max", "claude-opus-5") == "max"
    # gpt-5.6 has no "minimal" -> snaps to low; none is native
    assert resolve_effort_for_model("minimal", "gpt-5.6-luna") == "low"
    assert resolve_effort_for_model("none", "gpt-5.6-luna") == "none"


def test_disabled_thinking_caps_effort_at_high() -> None:
    from emissary_router.providers.thinking import normalize_anthropic_thinking_for_model

    body = {
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "max"},
        "max_tokens": 32000,
    }
    changes = normalize_anthropic_thinking_for_model(body, "claude-sonnet-5")
    assert body["output_config"]["effort"] == "high"  # xhigh/max + disabled -> 400 upstream
    assert any("max->high" in c for c in changes)
