"""Budget-style Claude models (pre-4.6, e.g. haiku-4.5): adaptive thinking is rewritten to
an explicit budget, which Anthropic only accepts at >= 1024 tokens and below max_tokens."""
from emissary_router.providers.thinking import (
    ANTHROPIC_MIN_THINKING_BUDGET,
    normalize_anthropic_thinking_for_model,
)


def _body(max_tokens):
    return {"model": "claude-opus-5", "max_tokens": max_tokens, "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}]}


def test_small_max_tokens_turns_thinking_off_instead_of_an_illegal_budget():
    body = _body(600)
    changes = normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5")
    assert body["thinking"] == {"type": "disabled"}
    assert any("disabled" in c for c in changes)


def test_budget_is_max_tokens_minus_one_when_legal():
    body = _body(4000)
    normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5")
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 3999}
    assert 3999 >= ANTHROPIC_MIN_THINKING_BUDGET


def test_boundary_1025_tokens_is_the_smallest_thinking_request():
    body = _body(1025)
    normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5")
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    body = _body(1024)
    normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5")
    assert body["thinking"] == {"type": "disabled"}


def test_adaptive_models_are_untouched():
    body = _body(600)
    assert normalize_anthropic_thinking_for_model(body, "claude-opus-5") == []
    assert body["thinking"] == {"type": "adaptive"}
