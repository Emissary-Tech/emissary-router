from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any, Protocol

from starlette.responses import Response

from emissary_router.caching.usage import Usage
from emissary_router.config import ResolvedModel
from emissary_router.schemas import AnthropicRequest, RequestContext

ProviderComplete = Callable[[Usage, dict[str, Any]], None]

# Anthropic validates tool ids against this pattern. Some upstream models emit ids
# outside it (Kimi: "functions.get_weather:0"), which 400s when the conversation later
# routes to a native Anthropic model.
TOOL_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def sanitize_tool_id(tool_id: str | None) -> str | None:
    """Map a tool id to an Anthropic-valid one, deterministically.

    Valid ids pass through untouched. Invalid ones get their offending characters
    replaced plus a short hash of the original (collision-safe), so the same upstream
    id always maps to the same sanitized id — tool_use/tool_result pairing and the
    prompt-cache prefix stay stable across turns.
    """
    if not tool_id or TOOL_ID_RE.fullmatch(tool_id):
        return tool_id
    digest = hashlib.sha1(tool_id.encode("utf-8")).hexdigest()[:8]
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id)[:48]
    return f"{cleaned}_{digest}"


class Provider(Protocol):
    name: str

    async def messages(
        self,
        request: AnthropicRequest,
        model: ResolvedModel,
        context: RequestContext,
        on_complete: ProviderComplete | None = None,
    ) -> Response:
        ...

    def usage_from_response(self, payload: dict) -> Usage:
        ...
