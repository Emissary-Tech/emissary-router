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


def usage_tokens(usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_input_tokens,
        "cache_creation_tokens": usage.cache_creation_input_tokens,
    }
