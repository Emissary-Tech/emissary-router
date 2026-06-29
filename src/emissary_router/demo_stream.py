"""Demo-only real streaming, one passthrough per provider.

Each provider's upstream is read in its OWN native streaming format and converted to a
single normalized event stream:

    {"type": "text", "text": str}                        # assistant text delta (shown)
    {"type": "thinking", "text": str}                    # reasoning delta (captured, hidden)
    {"type": "signature", "text": str}                   # thinking-block signature (anthropic)
    {"type": "tool_call", "id", "name", "input": dict}   # one completed tool call
    {"type": "done", "usage": Usage, "stop_reason", "status"}

This deliberately bypasses the OpenAI-SSE -> Anthropic-SSE translation the Claude Code
integration needs. Claude Code only speaks the Anthropic format, so the production
OpenRouter provider buffers the whole completion and re-emits it as Anthropic SSE (no
real token streaming). The demo frontend is ours, so here we read OpenRouter's OpenAI
SSE directly and stream it for real. Anthropic already streams natively, so that side
just reuses the provider's streaming response.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import httpx

from emissary_router.caching.usage import Usage
from emissary_router.config import ResolvedModel
from emissary_router.providers.openrouter import OpenRouterProvider
from emissary_router.schemas import AnthropicRequest, RequestContext


async def stream_model(
    provider, model: ResolvedModel, body: dict, headers: dict[str, str]
) -> AsyncIterator[dict[str, Any]]:
    """Stream one model turn for real, dispatching to the provider-native passthrough."""
    if model.provider == "openrouter":
        async for event in _openrouter_passthrough(provider, model, body):
            yield event
    else:
        async for event in _anthropic_passthrough(provider, model, body, headers):
            yield event


async def _anthropic_passthrough(
    provider, model: ResolvedModel, body: dict, headers: dict[str, str]
) -> AsyncIterator[dict[str, Any]]:
    """Anthropic streams natively, so reuse the provider's real streaming response (it
    proxies the upstream SSE byte-for-byte) and normalize its Anthropic SSE events."""
    captured: dict = {}

    def on_complete(usage: Usage, metadata: dict) -> None:
        captured["usage"] = usage
        captured["status"] = metadata.get("http_status")

    response = await provider.messages(
        AnthropicRequest(body={**body, "stream": True}, headers=dict(headers)),
        model=model,
        context=RequestContext(
            request_id="demo", conversation_id=None, classifier_input="", requested_model=model.name
        ),
        on_complete=on_complete,
    )

    tool_blocks: dict[int, dict] = {}
    stop_reason: str | None = None
    buf = ""
    async for chunk in response.body_iterator:
        buf += chunk.decode("utf-8", "replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            event = _data_json(line)
            if event is None:
                continue
            etype = event.get("type")
            if etype == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "tool_use":
                    tool_blocks[event.get("index")] = {
                        "id": block.get("id"), "name": block.get("name"), "_json": ""
                    }
            elif etype == "content_block_delta":
                delta = event.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta" and delta.get("text"):
                    yield {"type": "text", "text": delta["text"]}
                elif dtype == "thinking_delta" and delta.get("thinking"):
                    yield {"type": "thinking", "text": delta["thinking"]}
                elif dtype == "signature_delta" and delta.get("signature"):
                    yield {"type": "signature", "text": delta["signature"]}
                elif dtype == "input_json_delta":
                    blk = tool_blocks.get(event.get("index"))
                    if blk is not None:
                        blk["_json"] += delta.get("partial_json", "")
            elif etype == "message_delta":
                stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason

    for index in sorted(tool_blocks):
        blk = tool_blocks[index]
        yield {
            "type": "tool_call", "id": blk.get("id"),
            "name": blk.get("name"), "input": _loads(blk.get("_json")),
        }
    yield {
        "type": "done", "usage": captured.get("usage") or Usage(),
        "stop_reason": stop_reason, "status": captured.get("status"),
    }


async def _openrouter_passthrough(
    provider, model: ResolvedModel, body: dict
) -> AsyncIterator[dict[str, Any]]:
    """Real streaming from OpenRouter: ask for a true stream and translate the OpenAI SSE
    deltas as they arrive (the production provider instead buffers the whole response)."""
    oai = OpenRouterProvider.to_openai_request(body, model.model_id, None, model.name)
    oai["stream"] = True
    oai["stream_options"] = {"include_usage": True}
    headers = {
        "Authorization": f"Bearer {provider._config.api_key or ''}",
        "Content-Type": "application/json",
        "X-OpenRouter-Metadata": "enabled",
    }

    tool_acc: dict[int, dict] = {}
    finish_reason: str | None = None
    usage_payload: dict | None = None
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", provider._base_url, headers=headers, json=oai) as response:
            status = response.status_code
            if status >= 400:
                body_text = (await response.aread()).decode("utf-8", "replace")
                yield {"type": "done", "usage": Usage(), "stop_reason": None,
                       "status": status, "error": body_text[:300]}
                return
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if event.get("usage"):
                    usage_payload = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield {"type": "text", "text": delta["content"]}
                    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                    if reasoning:
                        yield {"type": "thinking", "text": reasoning}
                    for call in delta.get("tool_calls") or []:
                        slot = tool_acc.setdefault(
                            call.get("index", 0), {"id": None, "name": "", "args": ""}
                        )
                        if call.get("id"):
                            slot["id"] = call["id"]
                        fn = call.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

    for index in sorted(tool_acc):
        slot = tool_acc[index]
        yield {
            "type": "tool_call",
            "id": slot["id"] or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": slot["name"] or "unknown_tool",
            "input": _loads(slot["args"]),
        }
    usage = provider.usage_from_response({"usage": usage_payload}) if usage_payload else Usage()
    yield {
        "type": "done", "usage": usage,
        "stop_reason": OpenRouterProvider._stop_reason(finish_reason), "status": status,
    }


def _data_json(line: str) -> dict | None:
    """Parse one Anthropic-format SSE `data:` line into its event dict."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        event = json.loads(payload)
    except (ValueError, TypeError):
        return None
    return event if isinstance(event, dict) else None


def _loads(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "{}")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}
