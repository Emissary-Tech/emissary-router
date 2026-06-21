from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnthropicRequest:
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    conversation_id: str | None
    router_view: str
    requested_model: str | None


@dataclass(frozen=True)
class RouteDecision:
    model_name: str
    reason: str
    probabilities: dict[str, float]
