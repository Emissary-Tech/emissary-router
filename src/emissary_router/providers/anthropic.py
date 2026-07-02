from __future__ import annotations

import json
import logging
import re
from copy import deepcopy

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from emissary_router.caching.usage import Usage
from emissary_router.config import ProviderConfig, ResolvedModel
from emissary_router.schemas import AnthropicRequest, RequestContext
from emissary_router.providers.base import ProviderComplete
from emissary_router.providers.thinking import (
    SYNTHETIC_THINKING_SIGNATURE,
    normalize_anthropic_thinking_for_model,
)

CCH_ATTRIBUTION_LINE_RE = re.compile(r"(?m)^.*\bcch=[^\s<>\"]+.*(?:\n|$)")

logger = logging.getLogger(__name__)


def _system_blocks(content) -> list[dict]:
    """Normalize a system message's content (str / dict / list) to Anthropic blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return [
            block if isinstance(block, dict) else {"type": "text", "text": str(block)}
            for block in content
            if block
        ]
    return []


def _system_text(content) -> str:
    """Flatten a system message's content to plain text (for reminder conversion)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "") if content.get("type") == "text" else ""
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return ""


def _request_summary(body: dict) -> str:
    messages = body.get("messages") or []
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    system = body.get("system")
    system_kind = "list" if isinstance(system, list) else ("str" if isinstance(system, str) else "none")
    return (
        f"model={body.get('model')} stream={bool(body.get('stream'))} "
        f"msg_roles={roles} system={system_kind} thinking={body.get('thinking')} "
        f"top_keys={sorted(body.keys())}"
    )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._base_url = (config.base_url or "https://api.anthropic.com").rstrip("/")

    async def messages(
        self,
        request: AnthropicRequest,
        model: ResolvedModel,
        context: RequestContext,
        on_complete: ProviderComplete | None = None,
    ) -> Response:
        body = deepcopy(request.body)
        # Claude Code (targeting a newer model) can put a role:"system" message inside
        # `messages`; an older served model rejects it ("role 'system' is not supported
        # on this model"). Move that content into the top-level `system` field.
        self._hoist_system_messages(body)
        # Drop thinking blocks we synthesized from another provider's reasoning on a
        # prior turn — their signature is not valid to Anthropic and would 400.
        self._strip_synthetic_thinking(body)
        if self._config.cache.strip_dynamic_attribution:
            self._strip_cch_attribution(body)
        thinking_changes = normalize_anthropic_thinking_for_model(body, model.name)
        body["model"] = model.model_id
        headers = self._forward_headers(request.headers)
        if self._config.api_key and "x-api-key" not in {k.lower() for k in headers}:
            headers["x-api-key"] = self._config.api_key
        headers["accept-encoding"] = "identity"

        if body.get("stream"):
            async def gen():
                buf = bytearray()
                status: int | None = None
                error: str | None = None
                async with httpx.AsyncClient(timeout=None) as client:
                    try:
                        async with client.stream(
                            "POST",
                            f"{self._base_url}/v1/messages",
                            headers=headers,
                            json=body,
                        ) as response:
                            status = response.status_code
                            async for chunk in response.aiter_raw():
                                buf.extend(chunk)
                                yield chunk
                    except Exception as exc:
                        error = repr(exc)
                        raise
                    finally:
                        if status is not None and status >= 400:
                            logger.warning(
                                "anthropic upstream %s | %s | error_body=%s",
                                status,
                                _request_summary(body),
                                buf.decode("utf-8", "replace")[:800],
                            )
                        self._complete(
                            on_complete,
                            self._usage_from_sse(buf.decode("utf-8", "replace")),
                            {
                                "http_status": status,
                                "stream": True,
                                "error": error,
                                "thinking_normalization": thinking_changes,
                            },
                        )

            return StreamingResponse(gen(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self._base_url}/v1/messages", headers=headers, json=body)
        if response.status_code >= 400:
            logger.warning(
                "anthropic upstream %s | %s | error_body=%s",
                response.status_code,
                _request_summary(body),
                response.text[:800],
            )
        try:
            payload = response.json()
            self._complete(
                on_complete,
                self.usage_from_response(payload),
                {
                    "http_status": response.status_code,
                    "stream": False,
                    "thinking_normalization": thinking_changes,
                },
            )
            return JSONResponse(payload, status_code=response.status_code)
        except ValueError:
            self._complete(
                on_complete,
                Usage(),
                {
                    "http_status": response.status_code,
                    "stream": False,
                    "error": (response.text or "(empty body)")[:300],
                    "thinking_normalization": thinking_changes,
                },
            )
            return Response(
                response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/plain"),
            )

    def usage_from_response(self, payload: dict) -> Usage:
        usage = payload.get("usage", {}) or {}
        return Usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        )

    @staticmethod
    def _hoist_system_messages(body: dict) -> None:
        """Normalize role:"system" messages for models that reject them inline.

        LEADING system messages (before any user/assistant turn) are the initial system
        prompt — they belong in the top-level `system` field. MID/TRAILING ones are
        per-turn reminders: they are converted IN PLACE to user messages wrapped in
        <system-reminder> tags. Hoisting those to the head would rewrite the first
        bytes of the prompt whenever a reminder changes, invalidating the provider's
        prefix-matched prompt cache for the entire conversation.
        """
        messages = body.get("messages")
        if not isinstance(messages, list):
            return
        hoisted: list[dict] = []
        remaining: list = []
        leading = True
        changed = False
        for message in messages:
            if not (isinstance(message, dict) and message.get("role") == "system"):
                if isinstance(message, dict) and message.get("role") in ("user", "assistant"):
                    leading = False
                remaining.append(message)
                continue
            changed = True
            if leading:
                hoisted.extend(_system_blocks(message.get("content")))
            else:
                text = _system_text(message.get("content"))
                if text:
                    if "<system-reminder" not in text:
                        text = f"<system-reminder>\n{text}\n</system-reminder>"
                    remaining.append({"role": "user", "content": [{"type": "text", "text": text}]})
        if not changed:
            return
        body["messages"] = remaining
        if hoisted:
            existing = body.get("system")
            if isinstance(existing, str):
                existing = [{"type": "text", "text": existing}] if existing else []
            elif isinstance(existing, list):
                existing = list(existing)
            else:
                existing = []
            body["system"] = existing + hoisted

    @staticmethod
    def _strip_synthetic_thinking(body: dict) -> None:
        """Remove residue that other providers' turns leave in the history.

        - thinking blocks bearing our synthetic signature (Anthropic rejects the
          signature as invalid);
        - `thought_signature` keys on tool_use blocks (Gemini's signature, preserved
          for gemini->gemini replay; Anthropic 400s "Extra inputs are not permitted").
        """
        messages = body.get("messages")
        if not isinstance(messages, list):
            return
        cleaned: list = []
        changed = False
        for message in messages:
            if not (
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), list)
            ):
                cleaned.append(message)
                continue
            content = message["content"]
            kept = []
            message_changed = False
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "thinking"
                    and block.get("signature") == SYNTHETIC_THINKING_SIGNATURE
                ):
                    message_changed = True
                    continue
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and "thought_signature" in block
                ):
                    block = {k: v for k, v in block.items() if k != "thought_signature"}
                    message_changed = True
                kept.append(block)
            if not message_changed:
                cleaned.append(message)
                continue
            changed = True
            if kept:
                message["content"] = kept
                cleaned.append(message)
            # else: drop the message entirely. An empty-text placeholder is NOT an
            # option — Anthropic 400s with "text content blocks must be non-empty" —
            # and consecutive user turns are accepted (merged), so dropping is safe.
        if changed:
            body["messages"] = cleaned

    @staticmethod
    def _strip_cch_attribution(body: dict) -> None:
        system = body.get("system")
        if isinstance(system, str):
            body["system"] = CCH_ATTRIBUTION_LINE_RE.sub("", system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    block["text"] = CCH_ATTRIBUTION_LINE_RE.sub("", block["text"])

    @staticmethod
    def _usage_from_sse(text: str) -> Usage:
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        for line in text.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if event.get("type") == "message_start":
                event_usage = event.get("message", {}).get("usage", {}) or {}
                for key in usage:
                    if key in event_usage:
                        usage[key] = event_usage[key]
            elif event.get("type") == "message_delta":
                event_usage = event.get("usage", {}) or {}
                if "output_tokens" in event_usage:
                    usage["output_tokens"] = event_usage["output_tokens"]
        return Usage(**usage)

    @staticmethod
    def _complete(
        on_complete: ProviderComplete | None,
        usage: Usage,
        metadata: dict,
    ) -> None:
        if on_complete is None:
            return
        try:
            on_complete(usage, metadata)
        except Exception:
            # Telemetry must never break the user-facing model response.
            return

    @staticmethod
    def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
        allowed = {
            "x-api-key",
            "authorization",
            "anthropic-version",
            "anthropic-beta",
            "content-type",
        }
        return {k: v for k, v in headers.items() if k.lower() in allowed}
