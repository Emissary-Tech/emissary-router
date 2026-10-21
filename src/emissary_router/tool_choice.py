"""Forced tool choice vs the Claude models that reject it.

claude-sonnet-5.5, claude-opus-5.5 and claude-fable-5.1 return 400 on every request
whose `tool_choice` forces a tool call (Anthropic `any` / `tool`), regardless of
thinking settings, on the Anthropic API and through OpenRouter alike (platform live
2026-10-01 for the 5.5 pair, 2026-10-22 for fable-5.1: 'tool_choice: type "tool" and
"any" are not supported for this model'; haiku-5.5 accepts forced calls). The
router answers such a request with its own 400 before any provider call: the
client forced a tool for the model IT named, the router served one that cannot
honor it, and silently dropping the forcing would change the request's meaning.
Same contract as the platform gateway (`routing_gateway/tool_choice.py`).
"""
from __future__ import annotations

from typing import Any

REJECTS_FORCED_TOOL_CHOICE = frozenset({"claude-sonnet-5.5", "claude-opus-5.5", "claude-fable-5.1"})

_FORCING_TYPES = {"any", "tool"}


def forces_tool_choice(body: dict[str, Any]) -> bool:
    """True when this Messages request forces a tool call: a top-level
    `tool_choice` of type `any` or `tool`, with tools present (without tools the
    field is inert and providers ignore it)."""
    if not body.get("tools"):
        return False
    tool_choice = body.get("tool_choice")
    return isinstance(tool_choice, dict) and tool_choice.get("type") in _FORCING_TYPES


def unsupported_tool_choice_message(served_model: str) -> str:
    names = sorted(REJECTS_FORCED_TOOL_CHOICE)
    models = ", ".join(names[:-1]) + f" and {names[-1]}"
    return (
        f"The router served this request with {served_model}, which does not support "
        f'tool_choice type "any" or "tool" (Anthropic returns 400 for {models}). '
        'Send tool_choice type "auto" (or omit it), or disable these models in the '
        "router config for requests that must force a tool call."
    )


def unsupported_tool_choice_payload(message: str) -> dict[str, Any]:
    # Anthropic Messages error shape, so Claude Code and SDK clients surface it as-is.
    return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
