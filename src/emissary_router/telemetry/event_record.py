from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from emissary_router.caching.usage import Usage


@dataclass(frozen=True)
class EventRecord:
    """One routed API call, normalized to the lean columns the dashboard needs.

    Deliberately flat: this is an append-only fact log, not a relational entity.
    Large blobs (full request/response, classifier text) are never stored here;
    ``raw_event`` is populated only when telemetry.include_raw_event is enabled.
    """

    id: str
    ts: float
    session_id: str | None
    turn_id: int | None
    call_kind: str  # "main" | "background"
    requested_model: str | None
    served_model: str
    provider: str
    model_id: str
    route_reason: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cost_usd: float | None
    duration_ms: float
    http_status: int | None = None
    raw_event: str | None = None


COLUMNS: list[str] = [field.name for field in fields(EventRecord)]


def call_kind_from_body(body: dict[str, Any]) -> str:
    """Main calls carry tools or active reasoning; background utility calls (title/topic) do not."""
    if body.get("tools"):
        return "main"
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") != "disabled":
        return "main"
    output_config = body.get("output_config")
    if isinstance(output_config, dict) and output_config.get("effort"):
        return "main"
    return "background"


def _is_system_reminder(text: str) -> bool:
    return text.lstrip().startswith("<system-reminder>")


def latest_real_user_text(messages: Any) -> str | None:
    """The most recent genuine user input, used as the per-turn anchor.

    Skips tool_result continuations AND ``<system-reminder>`` blocks, which Claude Code
    injects mid-conversation as user text. Those reminders evolve over a session, so
    anchoring on them would split one user input into many turns. We anchor only on the
    real prompt text, which stays stable across an agent loop and changes per new input.
    """
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content.strip() and not _is_system_reminder(content):
                return content
            continue
        if isinstance(content, list):
            texts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and not _is_system_reminder(block.get("text", ""))
            ]
            if texts:
                return "\n".join(texts)
    return None


def usage_tokens(usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_input_tokens,
        "cache_creation_tokens": usage.cache_creation_input_tokens,
    }
