from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "max", "xhigh"]


@dataclass(frozen=True)
class ReasoningSettings:
    effort: str | None = None
    max_tokens: int | None = None
    enabled: bool | None = None
    exclude: bool | None = None


def extract_reasoning_settings(body: dict[str, Any]) -> ReasoningSettings:
    explicit_reasoning = body.get("reasoning")
    if isinstance(explicit_reasoning, dict):
        return ReasoningSettings(
            effort=_string_or_none(explicit_reasoning.get("effort")),
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

    effort = _find_effort(body)
    budget = _first_int(body.get("thinking_budget"), body.get("reasoning_max_tokens"))
    return ReasoningSettings(effort=effort, max_tokens=budget)


def normalize_effort(effort: str | None, max_effort: str = "high") -> str | None:
    if effort is None:
        return None
    if effort not in EFFORT_ORDER:
        return max_effort
    return effort if EFFORT_ORDER.index(effort) <= EFFORT_ORDER.index(max_effort) else max_effort


def effort_from_budget(max_tokens: int) -> str:
    if max_tokens <= 0:
        return "none"
    if max_tokens <= 1024:
        return "minimal"
    if max_tokens <= 4096:
        return "low"
    if max_tokens <= 8192:
        return "medium"
    return "high"


def _find_effort(body: dict[str, Any]) -> str | None:
    skip_keys = {"messages", "system", "tools"}
    effort_keys = {"effort", "reasoning_effort", "thinking_effort"}

    def walk(value: Any) -> str | None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in skip_keys:
                    continue
                if key in effort_keys and isinstance(item, str):
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
