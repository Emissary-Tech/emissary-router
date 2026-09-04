from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


# Anthropic's documented ladder: max is the TOP tier ("absolute highest capability"),
# above xhigh. OpenRouter's own top remains xhigh — its sender maps max->xhigh.
EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
SUPPORTED_THINKING_TYPES = {"enabled", "disabled"}
DEFAULT_MAX_THINKING_BUDGET = 32000
EFFORT_KEYS = {"effort", "reasoning_effort", "thinking_effort"}
SKIP_REWRITE_KEYS = {"messages", "system", "tools"}

# Signature stamped on thinking blocks we synthesize from a non-Anthropic provider's
# reasoning (e.g. OpenRouter/GLM/Kimi). Claude Code surfaces them like any thinking
# block, but they carry no valid Anthropic signature — so the Anthropic provider strips
# any history block bearing this marker before forwarding, or Anthropic 400s with
# "Invalid `signature` in `thinking` block".
SYNTHETIC_THINKING_SIGNATURE = "emissary:non-anthropic-reasoning"


@dataclass(frozen=True)
class ModelThinkingCapabilities:
    accepts_effort_param: bool = False
    accepts_adaptive_thinking: bool = False
    max_effort: str | None = None
    # Some models always reason and reject a disable request (OpenRouter returns 400
    # "Reasoning is mandatory ... cannot be disabled"). For those, never emit effort:none.
    can_disable_thinking: bool = True
    # The exact effort vocabulary when it has holes a linear clamp can't express
    # (e.g. gpt-5.6 supports none but NOT minimal).
    # None = every rung up to max_effort is fine.
    supported_efforts: tuple[str, ...] | None = None
    # True = senders must forward NO reasoning field of any kind. Distinct from
    # accepts_effort=False (haiku via OpenRouter still takes a budget, mapped to
    # native anthropic thinking semantics downstream). Set for dynamic router
    # entries (openrouter-auto) whose downstream host converts reasoning.max_tokens
    # into a hard output reservation and starves the response.
    no_reasoning_surface: bool = False


THINKING_CAPABILITIES = {
    "claude-haiku-4.5": ModelThinkingCapabilities(
        accepts_effort_param=False,
        accepts_adaptive_thinking=False,
    ),
    "gemini-3.1-flash-lite": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="high",
    ),
    # GLM-5.2 supports reasoning effort high/xhigh (xhigh == max). Adaptive thinking
    # is accepted by z.ai's native Anthropic-compatible endpoint (live-verified;
    # only the anthropic-format paths consult this flag — the OpenRouter path
    # translates via the effort param regardless).
    "glm-5.2": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=True,
        max_effort="xhigh",
    ),
    # OpenRouter: Kimi K2.7 Code reasons via the effort param (OpenRouter maps the level
    # to a reasoning budget; xhigh ~= 95% of max_tokens). It always reasons — thinking
    # can't be disabled — so on a disable request the reasoning field is OMITTED
    # entirely (sending effort:none gets a 400 "Reasoning is mandatory ... cannot be
    # disabled"). xhigh is allowed so a max-effort request gets full reasoning budget.
    "kimi-k2.7-code": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="xhigh",
        can_disable_thinking=False,
    ),
    # claude-5 series: adaptive thinking IS the provider default (omitting `thinking`
    # runs adaptive) and effort is a separate knob defaulting to high. Same
    # claude-5: full ladder low/medium/high (default) /xhigh/max. NOTE: with thinking
    # DISABLED, effort is capped at high (xhigh/max + disabled -> 400) — the anthropic
    # normalizer enforces that cap.
    "claude-sonnet-5": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=True,
        max_effort="max",
        supported_efforts=("low", "medium", "high", "xhigh", "max"),
    ),
    "claude-opus-5": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=True,
        max_effort="max",
        supported_efforts=("low", "medium", "high", "xhigh", "max"),
    ),
    # DeepSeek V4 Flash (OpenRouter): hybrid thinking — reasons via OpenRouter's
    # effort->budget translation and accepts a disable (non-thinking mode). Verify
    # live before serving at scale.
    "deepseek-v4-flash": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="xhigh",
    ),
    # gpt-5.6 series (OpenAI Responses API): reasoning is the provider default;
    # effort vocabulary none/low/medium/high/xhigh/max — "minimal" does NOT exist
    # (snaps to low), and "none" does, so thinking can be disabled.
    # snapshot-named alias of deepseek-v4-flash (same contract)
    "deepseek-v4-flash-0731": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="xhigh",
    ),
    "gpt-5.6-sol": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="max",
        supported_efforts=("none", "low", "medium", "high", "xhigh", "max"),
    ),
    "gpt-5.6-terra": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="max",
        supported_efforts=("none", "low", "medium", "high", "xhigh", "max"),
    ),
    "gpt-5.6-luna": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="max",
        supported_efforts=("none", "low", "medium", "high", "xhigh", "max"),
    ),
    # Assumed to behave like kimi-k2.7 (always reasons via OpenRouter's effort param;
    # disable requests omit the reasoning field). Label runs used provider-default
    # reasoning; verify live before serving at scale.
    "kimi-k3": ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="xhigh",
        can_disable_thinking=False,
    ),
}

# Benchmark-only competitor entry, gated like its catalog twin (see catalog.py).
# OpenRouter's model metadata omits the reasoning field for dynamic router models
# (openrouter/auto) — never send effort/thinking params; the chosen model runs at
# its own defaults (the competitor's as-shipped behavior).
if os.environ.get("EMISSARY_ROUTER_BENCH_EXTRAS"):
    # qwen3.8-27b (candidate): reasoning default ON at xhigh; effort vocabulary has
    # NO "high" (low/medium/xhigh only — OR metadata), so effort=high snaps to xhigh.
    THINKING_CAPABILITIES["qwen3.8-27b"] = ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="xhigh",
        supported_efforts=("low", "medium", "xhigh"),
    )
    THINKING_CAPABILITIES["qwen3.5-9b"] = ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=False,
        max_effort="xhigh",
        supported_efforts=("low", "medium", "xhigh"),
    )
    # GLM 5.3 candidates — same reasoning contract as glm-5.2 (effort param, OpenRouter path)
    THINKING_CAPABILITIES["glm-5.3-flash"] = ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=True,
        max_effort="xhigh",
    )
    THINKING_CAPABILITIES["glm-5.3"] = ModelThinkingCapabilities(
        accepts_effort_param=True,
        accepts_adaptive_thinking=True,
        max_effort="xhigh",
    )
    THINKING_CAPABILITIES["openrouter-auto"] = ModelThinkingCapabilities(
        accepts_effort_param=False,
        accepts_adaptive_thinking=False,
        no_reasoning_surface=True,
    )



# claude-5 models reject client sampling params (opus-5: any temperature/top_p;
# sonnet-5: non-default values) — measured during the 2026-07-31 label runs. Dropping
# the keys entirely is intent-preserving: the provider default takes over, which is
# what Claude Code's temperature=1 means anyway.
REJECTS_SAMPLING_PARAMS = {"claude-sonnet-5", "claude-opus-5"}


def strip_unsupported_sampling_params(body: dict[str, Any], served_model: str) -> list[str]:
    if served_model not in REJECTS_SAMPLING_PARAMS:
        return []
    changes: list[str] = []
    for key in ("temperature", "top_p", "top_k"):
        if key in body:
            changes.append(f"{key}={body.pop(key)} dropped ({served_model})")
    return changes


@dataclass(frozen=True)
class ReasoningSettings:
    effort: str | None = None
    max_tokens: int | None = None
    enabled: bool | None = None
    exclude: bool | None = None


def extract_reasoning_settings(body: dict[str, Any]) -> ReasoningSettings:
    output_config_effort = _output_config_effort(body)
    explicit_reasoning = body.get("reasoning")
    if isinstance(explicit_reasoning, dict):
        return ReasoningSettings(
            effort=_first_string(explicit_reasoning.get("effort"), output_config_effort),
            max_tokens=_int_or_none(explicit_reasoning.get("max_tokens")),
            enabled=_bool_or_none(explicit_reasoning.get("enabled")),
            exclude=_bool_or_none(explicit_reasoning.get("exclude")),
        )

    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        thinking_type = _string_or_none(thinking.get("type"))
        if thinking_type == "disabled":
            return ReasoningSettings(effort="none", enabled=False)
        effort = _first_string(
            thinking.get("effort"),
            thinking.get("reasoning_effort"),
            output_config_effort,
            body.get("effort"),
            body.get("reasoning_effort"),
        )
        budget = _first_int(
            thinking.get("budget_tokens"),
            thinking.get("thinking_budget"),
            thinking.get("max_tokens"),
            body.get("thinking_budget"),
            body.get("reasoning_max_tokens"),
        )
        enabled = True if thinking_type in {"enabled", "adaptive"} else None
        return ReasoningSettings(effort=effort, max_tokens=budget, enabled=enabled)

    effort = _first_string(output_config_effort, _find_effort(body))
    budget = _first_int(body.get("thinking_budget"), body.get("reasoning_max_tokens"))
    return ReasoningSettings(effort=effort, max_tokens=budget)


def normalize_effort(effort: str | None, max_effort: str = "high") -> str | None:
    if effort is None:
        return None
    if effort not in EFFORT_ORDER:
        return max_effort
    return effort if EFFORT_ORDER.index(effort) <= EFFORT_ORDER.index(max_effort) else max_effort


def resolve_effort_for_model(effort: str | None, model_name: str, default_max: str = "high") -> str | None:
    """Clamp to the model's max, then snap into its exact vocabulary.

    Unsupported rungs move to the nearest supported one (ties resolve upward, so an
    an unsupported "minimal" on gpt-5.6 becomes low; ties resolve upward to preserve
    "think harder" intent rather than silently downgrading)."""
    if effort is None:
        return None
    capabilities = THINKING_CAPABILITIES.get(model_name)
    effort = normalize_effort(effort, max_effort_for_model(model_name, default_max))
    supported = capabilities.supported_efforts if capabilities else None
    if not supported or effort in supported:
        return effort
    rank = EFFORT_ORDER.index(effort)
    return min(supported, key=lambda e: (abs(EFFORT_ORDER.index(e) - rank), -EFFORT_ORDER.index(e)))


def max_effort_for_model(model_name: str, default: str = "high") -> str:
    capabilities = THINKING_CAPABILITIES.get(model_name)
    return capabilities.max_effort if capabilities and capabilities.max_effort else default


def can_disable_thinking_for_model(model_name: str, default: bool = True) -> bool:
    capabilities = THINKING_CAPABILITIES.get(model_name)
    return capabilities.can_disable_thinking if capabilities else default


def always_on_reasoning_models() -> set[str]:
    """Models that always reason and can't be sent a disable — not appropriate for
    non-thinking (background) calls."""
    return {
        name for name, cap in THINKING_CAPABILITIES.items() if not cap.can_disable_thinking
    }


def accepts_effort_for_model(model_name: str, default: bool = True) -> bool:
    capabilities = THINKING_CAPABILITIES.get(model_name)
    return capabilities.accepts_effort_param if capabilities else default


def translates_reasoning_for_model(model_name: str) -> bool:
    """False when the capability entry opts out of reasoning entirely
    (no_reasoning_surface) — senders forward NO reasoning field, never a budget
    fallback. Passing a Claude-style budget_tokens (max_tokens-1) through as
    OpenRouter reasoning.max_tokens on such entries reserves the whole output
    window for reasoning and the response dies at stop_reason=max_tokens after a
    few chars (measured with openrouter-auto, 2026-08-15)."""
    capabilities = THINKING_CAPABILITIES.get(model_name)
    return not (capabilities and capabilities.no_reasoning_surface)


def effort_reasoning_for_model(settings: ReasoningSettings, model_name: str) -> str | None:
    effort = settings.effort
    if effort is None:
        return None

    effort = normalize_effort(effort, max_effort_for_model(model_name))
    return "minimal" if effort == "none" else effort


def normalize_anthropic_thinking_for_model(
    body: dict[str, Any],
    served_model: str,
) -> list[str]:
    """Normalize Anthropic thinking fields to the served model's capabilities.

    Claude Code can send adaptive thinking or effort values based on the requested
    Claude model. Routing can serve a different Claude model, so this normalizes the
    already-copied provider body before sending it to Anthropic.
    """
    capabilities = THINKING_CAPABILITIES.get(served_model)
    if capabilities is None:
        return []

    changes: list[str] = []
    thinking_field = body.get("thinking")
    thinking_disabled = (
        isinstance(thinking_field, dict) and thinking_field.get("type") == "disabled"
    )

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for key in list(value.keys()):
                if key in SKIP_REWRITE_KEYS:
                    continue
                item = value[key]
                child_path = f"{path}.{key}" if path else key
                if (
                    key == "thinking"
                    and isinstance(item, dict)
                    and item.get("type") not in SUPPORTED_THINKING_TYPES
                ):
                    if capabilities.accepts_adaptive_thinking:
                        walk(item, child_path)
                    else:
                        original_type = item.get("type")
                        budget = thinking_budget_from_max_tokens(body)
                        value[key] = {"type": "enabled", "budget_tokens": budget}
                        changes.append(
                            f"{child_path}={original_type}->enabled(budget={budget})"
                        )
                elif key in EFFORT_KEYS and isinstance(item, str):
                    if not capabilities.accepts_effort_param:
                        del value[key]
                        changes.append(f"{child_path}={item} dropped")
                    else:
                        clamped = resolve_effort_for_model(item, served_model)
                        # With thinking disabled, Anthropic rejects xhigh/max
                        # (effort is capped at high).
                        if thinking_disabled and clamped in ("xhigh", "max"):
                            clamped = "high"
                        if clamped != item:
                            value[key] = clamped
                            changes.append(f"{child_path}={item}->{clamped}")
                else:
                    walk(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(body)
    return changes


def _find_effort(body: dict[str, Any]) -> str | None:
    def walk(value: Any) -> str | None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in SKIP_REWRITE_KEYS:
                    continue
                if key in EFFORT_KEYS and isinstance(item, str):
                    return item
                found = walk(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = walk(item)
                if found:
                    return found
        return None

    return walk(body)


def thinking_budget_from_max_tokens(body: dict[str, Any]) -> int:
    max_tokens = _int_or_none(body.get("max_tokens"))
    return max(max_tokens - 1, 1) if max_tokens else DEFAULT_MAX_THINKING_BUDGET


def _output_config_effort(body: dict[str, Any]) -> str | None:
    output_config = body.get("output_config")
    if not isinstance(output_config, dict):
        return None
    return _string_or_none(output_config.get("effort"))


def _first_string(*values: Any) -> str | None:
    for value in values:
        result = _string_or_none(value)
        if result is not None:
            return result
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        result = _int_or_none(value)
        if result is not None:
            return result
    return None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def force_effort(body: dict[str, Any], level: str) -> list[str]:
    """Override the request's effort with ``level`` everywhere a client may have put
    it (Anthropic ``output_config.effort``, OpenAI-style ``reasoning.effort``,
    ``thinking.effort``/``reasoning_effort``, top-level keys), so every provider path
    reads the forced value. Used when a classifier label carries an effort suffix."""
    changes: list[str] = []
    oc = body.get("output_config")
    if not isinstance(oc, dict):
        oc = {}
        body["output_config"] = oc
    if oc.get("effort") != level:
        changes.append(f"output_config.effort={oc.get('effort')}->{level}")
        oc["effort"] = level
    for container_key in ("reasoning", "thinking"):
        container = body.get(container_key)
        if isinstance(container, dict):
            for k in ("effort", "reasoning_effort"):
                if k in container and container[k] != level:
                    changes.append(f"{container_key}.{k}={container[k]}->{level}")
                    container[k] = level
    for k in ("effort", "reasoning_effort"):
        if k in body and body[k] != level:
            changes.append(f"{k}={body[k]}->{level}")
            body[k] = level
    return changes
