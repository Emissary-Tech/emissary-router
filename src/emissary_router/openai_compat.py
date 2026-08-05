"""OpenAI chat-completions compatibility endpoint.

Lets OpenAI-SDK agents (e.g. the Arize cost-per-successful-task harness) run through
the router unchanged: POST /v1/chat/completions is translated to the internal
Anthropic Messages shape, routed by the normal pipeline (classifier, confidence
policy, cache-aware costing, telemetry), and the response is translated back.
Non-streaming only — a stream request gets a 400 with a clear message.

Injections so the router's session features work for clients that never send them:

- ``x-claude-code-session-id``: passed through when the client sends it (one line via
  the OpenAI SDK's ``extra_headers``); otherwise derived from a fingerprint of the
  first user message, so every step of one agent episode lands in one session
  (cache ledger + telemetry grouping). Concurrent episodes with an identical first
  message would share a fingerprint — callers that care should send the header.
- ``cache_control`` ephemeral breakpoints on the system block and the final message
  block, so Anthropic-side prompt caching engages. The OpenAI-format providers
  already handle/strip these on translation (the Claude Code path sends them daily).
- ``anthropic-version``: OpenAI SDK clients don't send it; the anthropic provider
  forwards it upstream when present.

The harness's own $/Mtok cost math cannot price a multi-model router — take the
router row's cost from ER telemetry (cache-aware) instead.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid

from starlette.responses import JSONResponse, Response

SESSION_HEADER = "x-claude-code-session-id"
# max_tokens policy: OpenAI clients may omit it (the CPST harness does), and native
# provider calls without it run uncapped — so for parity this adapter does NOT invent
# one. Absence flows through: the OpenAI/OpenRouter providers OMIT the field upstream
# (their old hardcoded 4096 fallback was removed), and only the Anthropic provider
# fills it (the native Messages schema requires the field) with 64000 — safe for
# every roster Claude (opus-5/sonnet-5 max output 128K, haiku 64K). Client-sent max_tokens always passes through everywhere.

_FINISH_REASON = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _openai_error(message: str, status: int, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "api_error", "code": code}},
        status_code=status,
    )


def _tool_args(arguments: str | None) -> dict:
    """Tool-call arguments round-trip. Malformed JSON (models do emit it) is wrapped
    rather than dropped so the history the client replays stays consistent."""
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {"_malformed_json": arguments}


def chat_to_messages(body: dict) -> dict:
    """OpenAI chat-completions request -> Anthropic Messages request."""
    system_blocks: list[dict] = []
    messages: list[dict] = []

    def append_blocks(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "developer"):
            if content:
                system_blocks.append({"type": "text", "text": str(content)})
        elif role == "user":
            if content:
                append_blocks("user", [{"type": "text", "text": str(content)}])
        elif role == "assistant":
            blocks: list[dict] = []
            if content and str(content).strip():
                blocks.append({"type": "text", "text": str(content)})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                        "name": fn.get("name") or "unknown_tool",
                        "input": _tool_args(fn.get("arguments")),
                    }
                )
            append_blocks("assistant", blocks)
        elif role == "tool":
            append_blocks(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id") or "",
                        "content": "" if content is None else str(content),
                    }
                ],
            )

    if system_blocks:
        system_blocks[-1]["cache_control"] = {"type": "ephemeral"}
    if messages:
        last = messages[-1]["content"]
        if last:
            last[-1] = {**last[-1], "cache_control": {"type": "ephemeral"}}

    out: dict = {
        "model": body.get("model") or "router",
        "messages": messages,
        "stream": False,
    }
    client_max = body.get("max_tokens") or body.get("max_completion_tokens")
    if client_max:
        out["max_tokens"] = client_max
    if system_blocks:
        out["system"] = system_blocks
    for knob in ("temperature", "top_p", "stop"):
        if body.get(knob) is not None:
            out["stop_sequences" if knob == "stop" else knob] = (
                [body[knob]] if knob == "stop" and isinstance(body[knob], str) else body[knob]
            )

    tools = body.get("tools") or []
    tool_choice = body.get("tool_choice", "auto")
    if tools and tool_choice != "none":
        out["tools"] = [
            {
                "name": (t.get("function") or {}).get("name"),
                "description": (t.get("function") or {}).get("description", ""),
                "input_schema": (t.get("function") or {}).get("parameters")
                or {"type": "object", "properties": {}},
            }
            for t in tools
            if t.get("type") == "function"
        ]
        if tool_choice == "required":
            out["tool_choice"] = {"type": "any"}
        elif isinstance(tool_choice, dict):
            name = (tool_choice.get("function") or {}).get("name")
            if name:
                out["tool_choice"] = {"type": "tool", "name": name}
        else:
            out["tool_choice"] = {"type": "auto"}
    return out


def messages_response_to_chat(resp: dict, requested_model: str) -> dict:
    """Anthropic Messages response -> OpenAI chat-completion response."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in resp.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text") or "")
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
            )
        # thinking / redacted_thinking blocks are dropped; their tokens still show in usage.

    message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp.get("usage") or {}
    prompt_tokens = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
    )
    completion_tokens = usage.get("output_tokens") or 0

    return {
        "id": f"chatcmpl-{resp.get('id') or uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.get("model") or requested_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _FINISH_REASON.get(resp.get("stop_reason"), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            # OpenAI-standard cache detail so clients that log it (the CPST harness
            # does) can recompute cache-aware cost per row. cached = cache READS;
            # cache writes have no OpenAI equivalent and live in ER telemetry.
            "prompt_tokens_details": {
                "cached_tokens": usage.get("cache_read_input_tokens") or 0
            },
        },
    }


def _session_id(headers: dict[str, str], anthropic_body: dict) -> str:
    for k, v in headers.items():
        if k.lower() == SESSION_HEADER and v:
            return v
    for m in anthropic_body.get("messages") or []:
        if m.get("role") == "user":
            for block in m.get("content") or []:
                if block.get("type") == "text":
                    digest = hashlib.sha1(block["text"].encode()).hexdigest()[:16]
                    return f"cpst-{digest}"
    return f"cpst-{uuid.uuid4().hex[:16]}"


async def handle_chat_completions(pipeline, body: dict, headers: dict[str, str]) -> Response:
    if body.get("stream"):
        return _openai_error(
            "streaming is not supported on the compatibility endpoint; "
            "send stream=false (the Anthropic-format /v1/messages endpoint streams)",
            400,
            code="stream_not_supported",
        )
    anthropic_body = chat_to_messages(body)
    fwd = dict(headers)
    fwd[SESSION_HEADER] = _session_id(headers, anthropic_body)
    fwd.setdefault("anthropic-version", "2023-06-01")

    resp = await pipeline.handle_messages(anthropic_body, fwd)
    raw = bytes(resp.body) if hasattr(resp, "body") else b""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return _openai_error(
            f"upstream returned non-JSON (status {resp.status_code})", 502
        )
    if resp.status_code != 200:
        err = payload.get("error") or {}
        return _openai_error(
            err.get("message") or json.dumps(payload)[:500],
            resp.status_code,
            code=err.get("type"),
        )
    return JSONResponse(messages_response_to_chat(payload, body.get("model") or "router"))
