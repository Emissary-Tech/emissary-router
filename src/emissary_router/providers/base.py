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

# Claude Code appends a dynamic `cch=` attribution line to the system prompt. It
# changes between requests, so anywhere it survives into a provider request it busts
# that provider's prefix-matched prompt cache — and the ledger's prefix_hash is
# computed WITHOUT it, so senders must strip it too or predictions and reality drift.
# Single definition: the hash (routing/cache_cost) and every provider strip with this.
CCH_ATTRIBUTION_LINE_RE = re.compile(r"(?m)^.*\bcch=[^\s<>\"]+.*(?:\n|$)")


def strip_cch_text(text: str) -> str:
    return CCH_ATTRIBUTION_LINE_RE.sub("", text)


def last_cache_control(content) -> dict | None:
    """The `cache_control` of the last marked block in a message's content, else None.

    Claude Code (2.1.299 measured; newer Claude models) puts its per-turn conversation
    cache breakpoint on a trailing role:"system" message. Any rewrite of that message
    must carry the mark over, or the whole conversation is billed at full price every
    call (only system+tools stay cached — the 2026-10-14 bench fingerprint)."""
    if isinstance(content, dict):
        cc = content.get("cache_control")
        return cc if isinstance(cc, dict) else None
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and isinstance(block.get("cache_control"), dict):
                return block["cache_control"]
    return None


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
