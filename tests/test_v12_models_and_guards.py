"""v12 roster (2026-10-22: claude-haiku-5.5, claude-fable-5.1, gpt-6-astra,
deepseek-v4.1-flash-1017) and the guards ported with it from the platform gateway
(live probes 2026-10-01 .. 2026-10-22): budgeted `enabled` -> adaptive on Claude 5,
`display` dropped where unsupported, each model's own thinking-off switch, "off"
resolved into effort vocabularies, sampling strip, forced tool_choice 400, the
OpenRouter host preference, and the dedicated classifier key."""
from emissary_router.catalog import CATALOG, CLASSIFIER_KEY_ENV, ROUTER_API_KEY_ENV, cost_score
from emissary_router.providers.openai import OpenAIProvider
from emissary_router.providers.openrouter import OpenRouterProvider
from emissary_router.providers.thinking import (
    REJECTS_SAMPLING_PARAMS,
    THINKING_CAPABILITIES,
    ReasoningSettings,
    always_on_reasoning_models,
    normalize_anthropic_thinking_for_model,
    strip_unsupported_sampling_params,
    thinking_off_effort_for_model,
)
from emissary_router.tool_choice import (
    REJECTS_FORCED_TOOL_CHOICE,
    forces_tool_choice,
    unsupported_tool_choice_message,
)


def test_v12_catalog_entries_match_the_platform() -> None:
    h = CATALOG["claude-haiku-5.5"]
    assert h.providers == {"anthropic": "claude-haiku-5-5", "openrouter": "anthropic/claude-haiku-5.5"}
    assert (h.pricing.input, h.pricing.output, h.pricing.cache_read, h.pricing.cache_write_5m) == (0.10, 0.50, 0.01, 0.125)
    f = CATALOG["claude-fable-5.1"]
    assert f.providers["anthropic"] == "claude-fable-5-1" and f.default_provider == "anthropic"
    assert (f.pricing.input, f.pricing.output, f.pricing.cache_read, f.pricing.cache_write_5m) == (10.0, 50.0, 0.25, 12.50)
    a = CATALOG["gpt-6-astra"]
    assert a.providers == {"openai": "gpt-6-astra", "openrouter": "openai/gpt-6-astra"} and a.default_provider == "openai"
    assert (a.pricing.input, a.pricing.output, a.pricing.cache_read, a.pricing.cache_write_5m) == (10.0, 50.0, 1.0, 12.5)
    d = CATALOG["deepseek-v4.1-flash-1017"]
    assert d.providers == {"openrouter": "deepseek/deepseek-v4.1-flash-1017"}
    assert d.pricing == CATALOG["deepseek-v4.1-flash"].pricing
    assert d.openrouter_provider_order == ("fireworks",)
    assert all(CATALOG[n].openrouter_provider_order is None for n in CATALOG if n != "deepseek-v4.1-flash-1017")
    # price ladder: haiku-5.5 ties gpt-6-luna at the bottom; astra and fable top the board
    assert cost_score(h) == cost_score(CATALOG["gpt-6-luna"])
    assert cost_score(a) == cost_score(f) > cost_score(CATALOG["claude-opus-5"])
    for name in ("claude-haiku-5.5", "claude-fable-5.1", "gpt-6-astra", "deepseek-v4.1-flash-1017"):
        assert CATALOG[name].context_window >= 1_000_000
        assert name in THINKING_CAPABILITIES


def test_every_claude_5_entry_rejects_the_budget_shape_and_has_display() -> None:
    claude5 = {n for n in THINKING_CAPABILITIES if n.startswith("claude-") and n != "claude-haiku-4.5"}
    assert claude5 == {"claude-sonnet-5", "claude-opus-5", "claude-opus-5.5", "claude-sonnet-5.5",
                       "claude-haiku-5.5", "claude-fable-5.1"}
    for name in claude5:
        cap = THINKING_CAPABILITIES[name]
        assert not cap.accepts_thinking_budget and cap.supports_thinking_display
        assert cap.supported_efforts == ("low", "medium", "high", "xhigh", "max")
    assert THINKING_CAPABILITIES["claude-haiku-4.5"].accepts_thinking_budget
    assert not THINKING_CAPABILITIES["claude-haiku-4.5"].supports_thinking_display


def test_enabled_budget_becomes_adaptive_on_claude_5() -> None:
    body = {"model": "r", "max_tokens": 16000, "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 8000}}
    changes = normalize_anthropic_thinking_for_model(body, "claude-haiku-5.5")
    assert body["thinking"] == {"type": "adaptive"}
    assert "output_config" not in body  # no invented effort: the model default applies
    assert changes == ["thinking=enabled(budget=8000)->adaptive"]
    body = {"model": "r", "max_tokens": 16000, "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 8000}, "output_config": {"effort": "low"}}
    normalize_anthropic_thinking_for_model(body, "claude-opus-5.5")
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "low"}
    body = {"model": "r", "max_tokens": 16000, "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 8000}}
    normalize_anthropic_thinking_for_model(body, "claude-fable-5.1")
    assert body["thinking"] == {"type": "adaptive"}


def test_older_claude_keeps_the_budget_shape() -> None:
    body = {"model": "r", "max_tokens": 16000, "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 8000}}
    assert normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5") == []
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 8000}


def test_display_is_dropped_where_the_model_has_no_display_knob() -> None:
    body = {"model": "r", "max_tokens": 16000, "messages": [],
            "thinking": {"type": "adaptive", "display": "summarized"}}
    changes = normalize_anthropic_thinking_for_model(body, "glm-5.2")  # adaptive yes, display no
    assert body["thinking"] == {"type": "adaptive"} and changes == ["thinking.display=summarized dropped"]
    body = {"model": "r", "max_tokens": 16000, "messages": [],
            "thinking": {"type": "adaptive", "display": "summarized"}}
    assert normalize_anthropic_thinking_for_model(body, "claude-haiku-5.5") == []
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    # haiku-4.5: the knob goes, then adaptive becomes a budget as before
    body = {"model": "r", "max_tokens": 16000, "messages": [],
            "thinking": {"type": "adaptive", "display": "summarized"}}
    normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5")
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 15999}


def test_thinking_off_gets_the_served_models_own_switch() -> None:
    # sonnet-5.5: between_tools (disabled 400s); the off-effort cap at high still applies
    body = {"max_tokens": 32000, "thinking": {"type": "disabled"}, "output_config": {"effort": "max"}}
    changes = normalize_anthropic_thinking_for_model(body, "claude-sonnet-5.5")
    assert body["thinking"] == {"type": "between_tools"} and body["output_config"]["effort"] == "high"
    assert "thinking=disabled->between_tools" in changes
    # haiku-5.5 / older Claude: disabled is the switch; a between_tools request becomes disabled
    body = {"max_tokens": 32000, "thinking": {"type": "between_tools"}}
    normalize_anthropic_thinking_for_model(body, "claude-haiku-5.5")
    assert body["thinking"] == {"type": "disabled"}
    body = {"max_tokens": 32000, "thinking": {"type": "disabled"}}
    assert normalize_anthropic_thinking_for_model(body, "claude-haiku-5.5") == []
    assert body["thinking"] == {"type": "disabled"}
    body = {"max_tokens": 32000, "thinking": {"type": "between_tools"}}
    normalize_anthropic_thinking_for_model(body, "claude-haiku-4.5")
    assert body["thinking"] == {"type": "disabled"}
    # opus-5.5 / fable-5.1: no switch at all -> adaptive at the lowest effort, client effort overridden
    for name in ("claude-opus-5.5", "claude-fable-5.1"):
        body = {"max_tokens": 32000, "thinking": {"type": "disabled"}, "output_config": {"effort": "high"}}
        changes = normalize_anthropic_thinking_for_model(body, name)
        assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "low"}
        assert changes[0].startswith("thinking=disabled->adaptive+effort=low")
    body = {"max_tokens": 32000, "thinking": {"type": "between_tools"}}
    normalize_anthropic_thinking_for_model(body, "claude-opus-5.5")
    assert body["thinking"] == {"type": "adaptive"} and body["output_config"] == {"effort": "low"}


def test_off_resolves_into_each_models_effort_vocabulary() -> None:
    assert thinking_off_effort_for_model("gpt-6.1-sol") == "low"    # no "none" rung
    assert thinking_off_effort_for_model("gpt-6-astra") == "low"
    assert thinking_off_effort_for_model("gpt-6-sol") == "none"
    assert thinking_off_effort_for_model("gpt-5.6-luna") == "none"
    assert thinking_off_effort_for_model("glm-5.2") == "none"
    assert thinking_off_effort_for_model("glm-5.3") == "low"        # always-on, lowest stands in
    assert thinking_off_effort_for_model("kimi-k2.7-code") == "low"
    assert thinking_off_effort_for_model("kimi-k3") == "none"
    assert thinking_off_effort_for_model("claude-opus-5.5") == "low"
    assert thinking_off_effort_for_model("claude-sonnet-5.5") == "low"  # between_tools has no OpenRouter spelling
    assert thinking_off_effort_for_model("claude-fable-5.1") == "low"
    assert thinking_off_effort_for_model("claude-haiku-5.5") == "none"
    assert thinking_off_effort_for_model("deepseek-v4.1-flash-1017") == "none"
    assert thinking_off_effort_for_model("not-a-model") == "none"


def test_background_candidates_exclude_only_models_without_any_off_path() -> None:
    always_on = always_on_reasoning_models()
    assert {"glm-5.3", "glm-5.3-flash", "kimi-k2.7-code"} <= always_on
    # the 5.5 pair and fable are served at their lowest effort instead, so an anchor
    # on them can always serve
    assert not ({"claude-opus-5.5", "claude-sonnet-5.5", "claude-fable-5.1", "claude-haiku-5.5"} & always_on)


def test_sampling_params_are_stripped_for_haiku55_and_fable() -> None:
    assert {"claude-haiku-5.5", "claude-fable-5.1"} <= REJECTS_SAMPLING_PARAMS
    for name in ("claude-haiku-5.5", "claude-fable-5.1"):
        body = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1000}
        assert len(strip_unsupported_sampling_params(body, name)) == 2
        assert "temperature" not in body and "top_p" not in body and body["max_tokens"] == 1000


def test_openai_responses_off_uses_the_models_vocabulary() -> None:
    off = {"max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
    assert OpenAIProvider.to_responses_request(dict(off), "gpt-6.1-sol", "gpt-6.1-sol")["reasoning"] == {"effort": "low", "summary": "auto"}
    assert OpenAIProvider.to_responses_request(dict(off), "gpt-6-astra", "gpt-6-astra")["reasoning"] == {"effort": "low", "summary": "auto"}
    assert OpenAIProvider.to_responses_request(dict(off), "gpt-6-sol", "gpt-6-sol")["reasoning"] == {"effort": "none", "summary": "auto"}
    # astra: minimal snaps to low, max passes, sampling params never forwarded
    body = {"max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}], "temperature": 0.3, "top_p": 0.9,
            "thinking": {"type": "adaptive"}, "output_config": {"effort": "minimal"}}
    req = OpenAIProvider.to_responses_request(body, "gpt-6-astra", "gpt-6-astra")
    assert req["reasoning"]["effort"] == "low" and "temperature" not in req and "top_p" not in req
    body["output_config"] = {"effort": "max"}
    assert OpenAIProvider.to_responses_request(body, "gpt-6-astra", "gpt-6-astra")["reasoning"]["effort"] == "max"


def test_openrouter_off_and_host_preference() -> None:
    off = {"max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}}
    assert OpenRouterProvider.to_openai_request(dict(off), "anthropic/claude-opus-5.5", model_name="claude-opus-5.5")["reasoning"] == {"effort": "low"}
    assert OpenRouterProvider.to_openai_request(dict(off), "anthropic/claude-fable-5.1", model_name="claude-fable-5.1")["reasoning"] == {"effort": "low"}
    assert OpenRouterProvider.to_openai_request(dict(off), "anthropic/claude-haiku-5.5", model_name="claude-haiku-5.5")["reasoning"] == {"effort": "none"}
    assert OpenRouterProvider.to_openai_request(dict(off), "moonshotai/kimi-k2.7-code", model_name="kimi-k2.7-code")["reasoning"] == {"effort": "low"}
    assert OpenRouterProvider.to_openai_request(dict(off), "z-ai/glm-5.3-flash", model_name="glm-5.3-flash")["reasoning"] == {"effort": "low"}
    assert OpenRouterProvider.to_openai_request(dict(off), "openai/gpt-6.1-sol", model_name="gpt-6.1-sol")["reasoning"] == {"effort": "low"}
    plain = {"max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}]}
    req = OpenRouterProvider.to_openai_request(dict(plain), "deepseek/deepseek-v4.1-flash-1017", model_name="deepseek-v4.1-flash-1017")
    assert req["provider"] == {"order": ["fireworks"]}
    assert "provider" not in OpenRouterProvider.to_openai_request(dict(plain), "deepseek/deepseek-v4.1-flash", model_name="deepseek-v4.1-flash")


def test_forced_tool_choice_guard() -> None:
    assert REJECTS_FORCED_TOOL_CHOICE == {"claude-sonnet-5.5", "claude-opus-5.5", "claude-fable-5.1"}
    tools = [{"name": "t", "input_schema": {"type": "object"}}]
    assert forces_tool_choice({"tools": tools, "tool_choice": {"type": "any"}})
    assert forces_tool_choice({"tools": tools, "tool_choice": {"type": "tool", "name": "t"}})
    assert not forces_tool_choice({"tools": tools, "tool_choice": {"type": "auto"}})
    assert not forces_tool_choice({"tools": tools})
    assert not forces_tool_choice({"tool_choice": {"type": "any"}})  # no tools: the field is inert
    msg = unsupported_tool_choice_message("claude-fable-5.1")
    assert msg.startswith("The router served this request with claude-fable-5.1")
    assert "claude-fable-5.1, claude-opus-5.5 and claude-sonnet-5.5" in msg and '"auto"' in msg


def test_arms_only_heads_collapse_for_the_v12_checkpoint() -> None:
    # frontier-effort-test (prod, 2026-10-22): opus-5.5 / sonnet-5.5 / haiku-5.5 / 1017
    # carry @low/@medium/@high arms and no plain head; fable-5.1 / astra are plain.
    from emissary_router.routing.labels import collapse_effort_heads

    probs = {"claude-opus-5.5@low": 0.7, "claude-opus-5.5@medium": 0.8, "claude-opus-5.5@high": 0.85,
             "claude-fable-5.1": 0.9, "gpt-6-luna": 0.3}
    # effort routing off, client effort low (Claude Code --effort low): the @low head stands
    base, picks = collapse_effort_heads(probs, routed=set(), confidence=0.75, reasoning=ReasoningSettings(effort="low"))
    assert base["claude-opus-5.5"] == 0.7 and picks["claude-opus-5.5"].label == "claude-opus-5.5@low"
    assert picks["claude-opus-5.5"].forced_effort is None and picks["claude-opus-5.5"].reason == "client_effort"
    assert base["claude-fable-5.1"] == 0.9 and picks["claude-fable-5.1"].reason == "plain"
    # effort routing on: confident at any arm, served at the lowest arm that clears tau
    base, picks = collapse_effort_heads(probs, routed={"claude-opus-5.5"}, confidence=0.75, reasoning=ReasoningSettings(effort="low"))
    assert base["claude-opus-5.5"] == 0.85 and picks["claude-opus-5.5"].forced_effort == "medium"
    # no effort named and routing off: the best arm stands in (flagged)
    base, picks = collapse_effort_heads(probs, routed=set(), confidence=0.75, reasoning=ReasoningSettings())
    assert picks["claude-opus-5.5"].reason == "best_effort" and base["claude-opus-5.5"] == 0.85


def test_classifier_key_prefers_the_dedicated_variable(monkeypatch) -> None:
    from emissary_router.routing.classifier import classifier_api_key

    monkeypatch.delenv(CLASSIFIER_KEY_ENV, raising=False)
    monkeypatch.setenv(ROUTER_API_KEY_ENV, "router-key")
    assert classifier_api_key() == "router-key"
    monkeypatch.setenv(CLASSIFIER_KEY_ENV, "classifier-key")
    assert classifier_api_key() == "classifier-key"
    monkeypatch.setenv(CLASSIFIER_KEY_ENV, "")
    assert classifier_api_key() == "router-key"


def test_missing_runtime_env_accepts_either_classifier_key(monkeypatch) -> None:
    from emissary_router.config import AppConfig, missing_runtime_env

    cfg = AppConfig.model_validate({"models": {"claude-haiku-5.5": True}, "default": "claude-haiku-5.5"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv(CLASSIFIER_KEY_ENV, raising=False)
    monkeypatch.delenv(ROUTER_API_KEY_ENV, raising=False)
    assert missing_runtime_env(cfg) == [ROUTER_API_KEY_ENV]
    monkeypatch.setenv(CLASSIFIER_KEY_ENV, "c")
    assert missing_runtime_env(cfg) == []
