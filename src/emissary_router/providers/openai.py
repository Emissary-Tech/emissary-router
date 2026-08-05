from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from emissary_router.caching.usage import Usage
from emissary_router.config import ProviderConfig, ResolvedModel
from emissary_router.schemas import AnthropicRequest, RequestContext
from emissary_router.providers.base import ProviderComplete, sanitize_tool_id, strip_cch_text
from emissary_router.providers.openrouter import anthropic_error_payload
from emissary_router.providers.thinking import (
    SYNTHETIC_THINKING_SIGNATURE,
    can_disable_thinking_for_model,
    extract_reasoning_settings,
    resolve_effort_for_model,
)

logger = logging.getLogger(__name__)

# The Responses API is the native surface for OpenAI reasoning models (gpt-5.x):
# reasoning is ON by provider default, reasoning tokens bill as output tokens, and
# the gpt-5.x series REJECTS sampling params (temperature/top_p) — so this provider
# never forwards them. Requests are stateless (`store: false`): the full history is
# replayed each call like every other provider here; prior-turn reasoning items are
# not replayed (thinking blocks in history are synthetic and stripped), which is the
# same degradation the OpenRouter reasoning bridge accepts.
# gpt-5.6 effort vocabulary: none/low/medium/high/xhigh/max — no "minimal" (snaps to
# low via the capabilities table), and "none" exists so thinking CAN be disabled.
_RESPONSES_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


def _request_summary(body: dict[str, Any]) -> str:
    items = body.get("input") or []
    kinds = [i.get("type") or i.get("role") for i in items if isinstance(i, dict)]
    return (
        f"model={body.get('model')} stream={bool(body.get('stream'))} "
        f"input_kinds={kinds[:12]} tools={len(body.get('tools') or [])} "
        f"reasoning={body.get('reasoning')} top_keys={sorted(body.keys())}"
    )


class OpenAIProvider:
    """Anthropic /v1/messages <-> OpenAI Responses API translation."""

    name = "openai"

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._base_url = (config.base_url or "https://api.openai.com/v1/responses").rstrip("/")

    async def messages(
        self,
        request: AnthropicRequest,
        model: ResolvedModel,
        context: RequestContext,
        on_complete: ProviderComplete | None = None,
    ) -> Response:
        body = self.to_responses_request(request.body, model.model_id, model.name)
        headers = {
            "Authorization": f"Bearer {self._config.api_key or ''}",
            "Content-Type": "application/json",
        }

        if request.body.get("stream"):
            return await self._open_stream(body, headers, model.name, on_complete)

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(self._base_url, headers=headers, json=body)

        if response.status_code >= 400:
            logger.warning(
                "openai upstream %s | %s | error_body=%s",
                response.status_code,
                _request_summary(body),
                response.text[:800],
            )

        try:
            payload = response.json()
        except ValueError:
            self._complete(
                on_complete,
                Usage(),
                {
                    "http_status": response.status_code,
                    "stream": False,
                    "error": (response.text or "(empty body)")[:300],
                },
            )
            return Response(
                response.content,
                status_code=response.status_code,
                media_type=response.headers.get("content-type", "text/plain"),
            )

        usage = self.usage_from_response(payload)
        self._complete(
            on_complete,
            usage,
            {
                "http_status": response.status_code,
                "stream": False,
                "openai_status": payload.get("status"),
                "id": payload.get("id"),
            },
        )

        if response.status_code >= 400:
            if response.status_code == 400:
                payload = anthropic_error_payload(payload)
            return JSONResponse(payload, status_code=response.status_code)

        return JSONResponse(self.from_responses_payload(payload, model.name))

    # ------------------------------------------------------------------ request

    @classmethod
    def to_responses_request(
        cls,
        body: dict[str, Any],
        model_id: str,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model_id,
            "input": cls._input_items(body),
            # omit when absent — same parity rule as the OpenRouter sender.
            **({"max_output_tokens": body["max_tokens"]} if body.get("max_tokens") is not None else {}),
            "store": False,
            "stream": False,
        }
        instructions = cls._instructions(body)
        if instructions:
            request["instructions"] = instructions
        # gpt-5.x rejects temperature/top_p — never forwarded, whatever the client sent.

        tools = cls._tools(body.get("tools") or [])
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        reasoning = cls._reasoning(body, model_name or model_id)
        if reasoning:
            request["reasoning"] = reasoning
        return request

    @classmethod
    def _instructions(cls, body: dict[str, Any]) -> str:
        parts: list[str] = []
        top = strip_cch_text(cls._stringify(body.get("system")))
        if top:
            parts.append(top)
        for message in body.get("messages", []) or []:
            if message.get("role") != "system":
                break
            inline = strip_cch_text(cls._stringify(message.get("content")))
            if inline:
                parts.append(inline)
        return "\n\n".join(parts)

    @classmethod
    def _input_items(cls, body: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        leading = True
        for message in body.get("messages", []) or []:
            role = message.get("role")
            if role == "system":
                if leading:
                    continue  # folded into instructions
                inline = strip_cch_text(cls._stringify(message.get("content")))
                if inline:
                    if "<system-reminder" not in inline:
                        inline = f"<system-reminder>\n{inline}\n</system-reminder>"
                    items.append(cls._user_item([{"type": "input_text", "text": inline}]))
                continue
            leading = False

            blocks = cls._blocks(message.get("content"))
            if role == "assistant":
                content: list[dict[str, Any]] = []
                for block in blocks:
                    kind = block.get("type")
                    if kind == "text" and block.get("text"):
                        content.append({"type": "output_text", "text": block["text"]})
                    elif kind == "tool_use":
                        if content:
                            items.append({"role": "assistant", "content": content})
                            content = []
                        items.append(
                            {
                                "type": "function_call",
                                "call_id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                                "name": block.get("name") or "unknown_tool",
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            }
                        )
                    # thinking blocks (incl. synthetic) are never replayed upstream
                if content:
                    items.append({"role": "assistant", "content": content})
                continue

            if role == "user":
                content = []
                for block in blocks:
                    kind = block.get("type")
                    if kind == "tool_result":
                        if content:
                            items.append(cls._user_item(content))
                            content = []
                        items.append(
                            {
                                "type": "function_call_output",
                                "call_id": block.get("tool_use_id") or "",
                                "output": cls._stringify(block.get("content")),
                            }
                        )
                    elif kind == "image":
                        image = cls._image_item(block)
                        if image:
                            content.append(image)
                    elif kind == "text":
                        if block.get("text"):
                            content.append({"type": "input_text", "text": block["text"]})
                    else:
                        content.append({"type": "input_text", "text": cls._stringify(block)})
                if content:
                    items.append(cls._user_item(content))
        return items

    @staticmethod
    def _user_item(content: list[dict[str, Any]]) -> dict[str, Any]:
        return {"role": "user", "content": content}

    @staticmethod
    def _image_item(block: dict[str, Any]) -> dict[str, Any] | None:
        source = block.get("source") or {}
        if source.get("type") == "base64" and source.get("media_type") and source.get("data"):
            return {
                "type": "input_image",
                "image_url": f"data:{source['media_type']};base64,{source['data']}",
            }
        return None

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Responses tools are FLAT (no nested "function" wrapper, unlike chat completions).
        return [
            {
                "type": "function",
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            }
            for tool in tools
            if tool.get("name")
        ]

    @classmethod
    def _reasoning(cls, body: dict[str, Any], model_name: str) -> dict[str, Any] | None:
        """Map Claude Code's thinking intent onto Responses `reasoning`.

        Reasoning is the gpt-5.x provider DEFAULT, so "no explicit intent" sends
        nothing. A disable request on an always-reasoning model is omitted (the model
        reasons anyway — kimi precedent); an explicit effort is clamped into the
        Responses vocabulary. `summary: "auto"` rides along whenever we send the field
        so thinking can surface to the client.
        """
        settings = extract_reasoning_settings(body)
        if settings.effort == "none":
            if can_disable_thinking_for_model(model_name):
                return {"effort": "none", "summary": "auto"}
            return None
        if settings.effort is not None:
            effort = resolve_effort_for_model(settings.effort, model_name)
            if effort not in _RESPONSES_EFFORTS:
                effort = "high"
            return {"effort": effort, "summary": "auto"}
        # budget-only (Haiku shape) or bare enabled: provider default already reasons.
        return None

    # ----------------------------------------------------------------- response

    @classmethod
    def from_responses_payload(cls, payload: dict[str, Any], model_label: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        has_tool = False
        for item in payload.get("output") or []:
            kind = item.get("type")
            if kind == "reasoning":
                summary = "".join(
                    part.get("text", "")
                    for part in item.get("summary") or []
                    if isinstance(part, dict)
                )
                if summary:
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": summary,
                            "signature": SYNTHETIC_THINKING_SIGNATURE,
                        }
                    )
            elif kind == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text" and part.get("text"):
                        content.append({"type": "text", "text": part["text"]})
            elif kind == "function_call":
                has_tool = True
                try:
                    args = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": sanitize_tool_id(item.get("call_id")) or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": item.get("name") or "unknown_tool",
                        "input": args,
                    }
                )
        if not content:
            content.append({"type": "text", "text": ""})

        return {
            "id": payload.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": model_label,
            "content": content,
            "stop_reason": cls._stop_reason(payload, has_tool),
            "stop_sequence": None,
            "usage": cls._usage_payload(payload.get("usage") or {}),
        }

    @staticmethod
    def _stop_reason(payload: dict[str, Any], has_tool: bool) -> str:
        if has_tool:
            return "tool_use"
        if payload.get("status") == "incomplete":
            reason = (payload.get("incomplete_details") or {}).get("reason")
            if reason == "max_output_tokens":
                return "max_tokens"
        return "end_turn"

    def usage_from_response(self, payload: dict) -> Usage:
        usage = payload.get("usage", {}) or {}
        cached = int((usage.get("input_tokens_details", {}) or {}).get("cached_tokens", 0) or 0)
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        return Usage(
            input_tokens=max(input_tokens - cached, 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=0,  # Responses caching is automatic; no write premium
            reasoning_output_tokens=int(
                (usage.get("output_tokens_details", {}) or {}).get("reasoning_tokens", 0) or 0
            ),
        )

    @classmethod
    def _usage_payload(cls, usage: dict[str, Any]) -> dict[str, int]:
        cached = int((usage.get("input_tokens_details", {}) or {}).get("cached_tokens", 0) or 0)
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        return {
            "input_tokens": max(input_tokens - cached, 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 0,
        }

    # ---------------------------------------------------------------- streaming

    async def _open_stream(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        model_label: str,
        on_complete: ProviderComplete | None,
    ) -> Response:
        stream_body = {**body, "stream": True}
        client = httpx.AsyncClient(timeout=None)
        try:
            request = client.build_request("POST", self._base_url, headers=headers, json=stream_body)
            response = await client.send(request, stream=True)
        except Exception:
            await client.aclose()
            raise

        if response.status_code >= 400:
            status = response.status_code
            raw = await response.aread()
            await response.aclose()
            await client.aclose()
            text = raw.decode("utf-8", "replace")
            logger.warning(
                "openai upstream %s | %s | error_body=%s",
                status,
                _request_summary(stream_body),
                text[:800],
            )
            self._complete(
                on_complete,
                Usage(),
                {"http_status": status, "stream": True, "error": (text or "upstream error")[:300]},
            )
            try:
                payload = json.loads(text)
            except ValueError:
                payload = {"error": {"type": "api_error", "message": (text or "upstream error")[:300]}}
            if status == 400:
                payload = anthropic_error_payload(payload)
            return JSONResponse(payload, status_code=status)

        return StreamingResponse(
            self._translate_stream(client, response, stream_body, model_label, on_complete),
            media_type="text/event-stream",
        )

    async def _translate_stream(
        self,
        client: httpx.AsyncClient,
        response: httpx.Response,
        stream_body: dict[str, Any],
        model_label: str,
        on_complete: ProviderComplete | None,
    ):
        status = response.status_code
        error: str | None = None
        sink: dict[str, Any] = {}
        try:
            async for event in self._iter_anthropic_events(response.aiter_lines(), model_label, sink):
                yield event
        except Exception as exc:  # noqa: BLE001 - surface in-stream, then end cleanly
            error = repr(exc)
            logger.warning(
                "openai upstream stream failed | %s | error=%s",
                _request_summary(stream_body),
                error,
            )
            yield self._sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": f"upstream stream failed: {error}"},
                },
            )
        finally:
            try:
                await response.aclose()
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._complete(
                on_complete,
                self.usage_from_response({"usage": sink.get("usage") or {}}),
                {"http_status": status, "stream": True, "error": error},
            )

    @classmethod
    async def _iter_anthropic_events(cls, lines, model_label: str, sink: dict[str, Any]):
        """Translate Responses typed SSE events into Anthropic SSE byte events."""
        sse = cls._sse
        next_index = 0
        cur_stream: tuple[str, int] | None = None  # (kind: thinking|text, index)
        open_tools: dict[int, int] = {}  # Responses output_index -> Anthropic block index
        has_tool = False
        final: dict[str, Any] = {}

        def close_stream_block() -> list[bytes]:
            nonlocal cur_stream
            if cur_stream is None:
                return []
            kind, index = cur_stream
            out: list[bytes] = []
            if kind == "thinking":
                out.append(
                    sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "signature_delta",
                                "signature": SYNTHETIC_THINKING_SIGNATURE,
                            },
                        },
                    )
                )
            out.append(sse("content_block_stop", {"type": "content_block_stop", "index": index}))
            cur_stream = None
            return out

        def open_stream_block(kind: str) -> list[bytes]:
            nonlocal next_index, cur_stream
            if cur_stream is not None and cur_stream[0] == kind:
                return []
            out = close_stream_block()
            block = (
                {"type": "thinking", "thinking": ""}
                if kind == "thinking"
                else {"type": "text", "text": ""}
            )
            out.append(
                sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": next_index, "content_block": block},
                )
            )
            cur_stream = (kind, next_index)
            next_index += 1
            return out

        yield sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "model": model_label,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "output_tokens": 0,
                    },
                },
            },
        )

        async for line in lines:
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            kind = obj.get("type")

            if kind == "response.output_item.added":
                item = obj.get("item") or {}
                if item.get("type") == "function_call":
                    has_tool = True
                    for event in close_stream_block():
                        yield event
                    yield sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": next_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": sanitize_tool_id(item.get("call_id"))
                                or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": item.get("name") or "unknown_tool",
                                "input": {},
                            },
                        },
                    )
                    open_tools[obj.get("output_index", len(open_tools))] = next_index
                    next_index += 1
            elif kind == "response.reasoning_summary_text.delta":
                delta = obj.get("delta") or ""
                if delta:
                    for event in open_stream_block("thinking"):
                        yield event
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": cur_stream[1],
                            "delta": {"type": "thinking_delta", "thinking": delta},
                        },
                    )
            elif kind == "response.output_text.delta":
                delta = obj.get("delta") or ""
                if delta:
                    for event in open_stream_block("text"):
                        yield event
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": cur_stream[1],
                            "delta": {"type": "text_delta", "text": delta},
                        },
                    )
            elif kind == "response.function_call_arguments.delta":
                block_index = open_tools.get(obj.get("output_index"))
                delta = obj.get("delta") or ""
                if block_index is not None and delta:
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": delta},
                        },
                    )
            elif kind in ("response.completed", "response.incomplete", "response.failed"):
                final = obj.get("response") or {}
                if final.get("usage"):
                    sink["usage"] = final["usage"]

        for event in close_stream_block():
            yield event
        for index in sorted(open_tools.values()):
            yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
        yield sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": cls._stop_reason(final, has_tool),
                    "stop_sequence": None,
                },
                "usage": cls._usage_payload(final.get("usage") or {}),
            },
        )
        yield sse("message_stop", {"type": "message_stop"})

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _blocks(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, list):
            return [
                block if isinstance(block, dict) else {"type": "text", "text": str(block)}
                for block in content
            ]
        return [{"type": "text", "text": "" if content is None else str(content)}]

    @classmethod
    def _stringify(cls, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(
                        block.get("text") or block.get("content")
                        or json.dumps(block, ensure_ascii=False)
                    )
                else:
                    parts.append(str(block))
            return "\n".join(map(str, parts))
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    @staticmethod
    def _sse(event: str, data: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

    @staticmethod
    def _complete(
        on_complete: ProviderComplete | None,
        usage: Usage,
        metadata: dict[str, Any],
    ) -> None:
        if on_complete is None:
            return
        try:
            on_complete(usage, metadata)
        except Exception:
            return
