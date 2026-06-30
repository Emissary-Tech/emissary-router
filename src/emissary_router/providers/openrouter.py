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
from emissary_router.providers.base import ProviderComplete
from emissary_router.providers.thinking import (
    SYNTHETIC_THINKING_SIGNATURE,
    accepts_effort_for_model,
    can_disable_thinking_for_model,
    extract_reasoning_settings,
    max_effort_for_model,
    normalize_effort,
    thinking_budget_from_max_tokens,
)


logger = logging.getLogger(__name__)


def _request_summary(oai_body: dict[str, Any]) -> str:
    messages = oai_body.get("messages") or []
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return (
        f"model={oai_body.get('model')} stream={bool(oai_body.get('stream'))} "
        f"msg_roles={roles} tools={len(oai_body.get('tools') or [])} "
        f"reasoning={oai_body.get('reasoning')} top_keys={sorted(oai_body.keys())}"
    )


class OpenRouterProvider:
    name = "openrouter"

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._base_url = (
            config.base_url or "https://openrouter.ai/api/v1/chat/completions"
        ).rstrip("/")

    async def messages(
        self,
        request: AnthropicRequest,
        model: ResolvedModel,
        context: RequestContext,
        on_complete: ProviderComplete | None = None,
    ) -> Response:
        oai_body = self.to_openai_request(request.body, model.model_id, context, model.name)
        headers = {
            "Authorization": f"Bearer {self._config.api_key or ''}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        }

        if request.body.get("stream"):
            # True streaming: translate OpenRouter's OpenAI-style SSE into Anthropic SSE
            # as chunks arrive, so the client sees output live instead of waiting for the
            # whole (possibly multi-minute, reasoning-heavy) generation to finish.
            return StreamingResponse(
                self._translate_stream(oai_body, headers, model.name, on_complete),
                media_type="text/event-stream",
            )

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(self._base_url, headers=headers, json=oai_body)

        if response.status_code >= 400:
            logger.warning(
                "openrouter upstream %s | %s | error_body=%s",
                response.status_code,
                _request_summary(oai_body),
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
                    "stream": bool(request.body.get("stream")),
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
                "stream": bool(request.body.get("stream")),
                "openrouter_model": payload.get("model"),
                "openrouter_metadata": payload.get("openrouter_metadata"),
                "id": payload.get("id"),
            },
        )

        if response.status_code >= 400:
            return JSONResponse(payload, status_code=response.status_code)

        message = self.from_openai_response(payload, model.name)
        return JSONResponse(message)

    async def _translate_stream(
        self,
        oai_body: dict[str, Any],
        headers: dict[str, str],
        model_label: str,
        on_complete: ProviderComplete | None,
    ):
        """Stream from OpenRouter and translate its SSE into Anthropic SSE in real time."""
        stream_body = {**oai_body, "stream": True, "stream_options": {"include_usage": True}}
        status: int | None = None
        error: str | None = None
        sink: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST", self._base_url, headers=headers, json=stream_body
                ) as response:
                    status = response.status_code
                    if status >= 400:
                        raw = await response.aread()
                        try:
                            err = (json.loads(raw) or {}).get("error") or {}
                        except (ValueError, AttributeError):
                            err = {}
                        error = (
                            (err.get("message") if isinstance(err, dict) else None)
                            or raw.decode("utf-8", "replace")[:300]
                            or "upstream error"
                        )
                        logger.warning(
                            "openrouter upstream %s | %s | error_body=%s",
                            status,
                            _request_summary(stream_body),
                            raw.decode("utf-8", "replace")[:800],
                        )
                        yield self._sse(
                            "error",
                            {"type": "error", "error": {"type": "api_error", "message": error}},
                        )
                        return
                    async for event in self._iter_anthropic_events(
                        response.aiter_lines(), model_label, sink
                    ):
                        yield event
            except Exception as exc:  # noqa: BLE001 - record then re-raise to end the stream
                error = repr(exc)
                raise
            finally:
                self._complete(
                    on_complete,
                    self.usage_from_response({"usage": sink.get("usage") or {}}),
                    {"http_status": status, "stream": True, "error": error},
                )

    @classmethod
    async def _iter_anthropic_events(cls, lines, model_label: str, sink: dict[str, Any]):
        """Translate an async iterator of OpenAI SSE lines into Anthropic SSE byte events.

        Text, reasoning, and tool_use all stream live: reasoning deltas become a thinking
        block (so Claude Code surfaces them like a native model), closed with a synthetic
        signature that the Anthropic provider strips from later turns. The observed usage
        is stashed in `sink["usage"]` for telemetry.
        """
        sse = cls._sse
        next_index = 0
        cur: tuple[str, int, int | None] | None = None  # (kind, index, oai_tool_index)
        finish_reason: str | None = None
        final_usage: dict[str, Any] = {}

        def close_cur() -> list[bytes]:
            nonlocal cur
            if cur is None:
                return []
            out: list[bytes] = []
            if cur[0] == "thinking":
                # Close the thinking block with a signature so the client treats it like a
                # native thinking block. It's a synthetic marker, not a real Anthropic
                # signature; the Anthropic provider strips these from later turns.
                out.append(
                    sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": cur[1],
                            "delta": {
                                "type": "signature_delta",
                                "signature": SYNTHETIC_THINKING_SIGNATURE,
                            },
                        },
                    )
                )
            out.append(sse("content_block_stop", {"type": "content_block_stop", "index": cur[1]}))
            cur = None
            return out

        def open_thinking() -> list[bytes]:
            nonlocal next_index, cur
            if cur is not None and cur[0] == "thinking":
                return []
            out = close_cur()
            out.append(
                sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": next_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                )
            )
            cur = ("thinking", next_index, None)
            next_index += 1
            return out

        def open_text() -> list[bytes]:
            nonlocal next_index, cur
            if cur is not None and cur[0] == "text":
                return []
            out = close_cur()
            out.append(
                sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": next_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            cur = ("text", next_index, None)
            next_index += 1
            return out

        def open_tool(oai_idx: int, tool_id: str | None, name: str | None) -> list[bytes]:
            nonlocal next_index, cur
            out = close_cur()
            out.append(
                sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": next_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id or f"toolu_{uuid.uuid4().hex[:24]}",
                            "name": name or "unknown_tool",
                            "input": {},
                        },
                    },
                )
            )
            cur = ("tool", next_index, oai_idx)
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
            if obj.get("usage"):
                final_usage = obj["usage"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning")
            if reasoning:
                for event in open_thinking():
                    yield event
                yield sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": cur[1],
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )

            content = delta.get("content")
            if content:
                for event in open_text():
                    yield event
                yield sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": cur[1],
                        "delta": {"type": "text_delta", "text": content},
                    },
                )

            for tool_call in delta.get("tool_calls") or []:
                oai_idx = tool_call.get("index", 0)
                fn = tool_call.get("function") or {}
                if cur is None or cur[0] != "tool" or cur[2] != oai_idx:
                    for event in open_tool(oai_idx, tool_call.get("id"), fn.get("name")):
                        yield event
                args = fn.get("arguments")
                if args:
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": cur[1],
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        },
                    )

        for event in close_cur():
            yield event
        yield sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": cls._stop_reason(finish_reason), "stop_sequence": None},
                "usage": {"output_tokens": int(final_usage.get("completion_tokens") or 0)},
            },
        )
        yield sse("message_stop", {"type": "message_stop"})
        sink["usage"] = final_usage

    def usage_from_response(self, payload: dict) -> Usage:
        usage = payload.get("usage", {}) or {}
        details = usage.get("prompt_tokens_details", {}) or {}
        cached_tokens = int(details.get("cached_tokens", 0) or 0)
        cache_write_tokens = int(details.get("cache_write_tokens", 0) or 0)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        return Usage(
            input_tokens=max(prompt_tokens - cached_tokens - cache_write_tokens, 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            cache_read_input_tokens=cached_tokens,
            cache_creation_input_tokens=cache_write_tokens,
            reasoning_output_tokens=(usage.get("output_tokens_details", {}) or {}).get(
                "reasoning_tokens", 0
            )
            or 0,
        )

    @classmethod
    def to_openai_request(
        cls,
        body: dict[str, Any],
        model_id: str,
        context: RequestContext | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model_id,
            "messages": cls._messages(body),
            "max_tokens": body.get("max_tokens", 4096),
            "stream": False,
        }
        if body.get("temperature") is not None:
            request["temperature"] = body["temperature"]
        if body.get("stop_sequences"):
            request["stop"] = body["stop_sequences"]

        tools = cls._tools(body.get("tools") or [])
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        reasoning = cls._reasoning(body, model_name or model_id)
        if reasoning:
            request["reasoning"] = reasoning

        if context and context.conversation_id:
            request["session_id"] = context.conversation_id[:256]

        if _is_anthropic_model(model_id):
            request["cache_control"] = {"type": "ephemeral"}

        return request

    @classmethod
    def from_openai_response(cls, payload: dict[str, Any], model_label: str) -> dict[str, Any]:
        choice = (payload.get("choices") or [{}])[0]
        oai_message = choice.get("message", {}) or {}
        content: list[dict[str, Any]] = []

        # Surface reasoning as a thinking block (parity with the streaming path), stamped
        # with the synthetic signature the Anthropic provider strips on later turns.
        reasoning = oai_message.get("reasoning")
        if reasoning:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning,
                    "signature": SYNTHETIC_THINKING_SIGNATURE,
                }
            )

        if oai_message.get("content"):
            content.append({"type": "text", "text": oai_message["content"]})

        for tool_call in oai_message.get("tool_calls") or []:
            fn = tool_call.get("function", {}) or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": fn.get("name") or "unknown_tool",
                    "input": args,
                }
            )

        if not content:
            content.append({"type": "text", "text": ""})

        usage = cls._usage_payload(payload.get("usage") or {})
        return {
            "id": payload.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": model_label,
            "content": content,
            "stop_reason": cls._stop_reason(choice.get("finish_reason")),
            "stop_sequence": None,
            "usage": usage,
        }

    @classmethod
    def anthropic_sse(cls, message: dict[str, Any]):
        skeleton = {
            **message,
            "content": [],
            "stop_reason": None,
            "usage": {
                "input_tokens": message["usage"].get("input_tokens", 0),
                "cache_read_input_tokens": message["usage"].get("cache_read_input_tokens", 0),
                "cache_creation_input_tokens": message["usage"].get("cache_creation_input_tokens", 0),
                "output_tokens": 0,
            },
        }
        yield cls._sse("message_start", {"type": "message_start", "message": skeleton})

        for index, block in enumerate(message["content"]):
            if block["type"] == "text":
                yield cls._sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield cls._sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    },
                )
            else:
                yield cls._sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block["id"],
                            "name": block["name"],
                            "input": {},
                        },
                    },
                )
                yield cls._sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
                        },
                    },
                )
            yield cls._sse("content_block_stop", {"type": "content_block_stop", "index": index})

        yield cls._sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": message["stop_reason"],
                    "stop_sequence": message.get("stop_sequence"),
                },
                "usage": {"output_tokens": message["usage"].get("output_tokens", 0)},
            },
        )
        yield cls._sse("message_stop", {"type": "message_stop"})

    @classmethod
    def _messages(cls, body: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        # Fold the top-level system AND any role:"system" messages (Claude Code sends
        # those for newer models) into one leading system message. OpenAI-format
        # providers take system content this way, and dropping the inline ones would lose
        # instructions (the reminders CC injects).
        system_parts: list[str] = []
        top_system = cls._stringify(body.get("system"))
        if top_system:
            system_parts.append(top_system)
        for message in body.get("messages", []) or []:
            if isinstance(message, dict) and message.get("role") == "system":
                inline = cls._stringify(message.get("content"))
                if inline:
                    system_parts.append(inline)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for message in body.get("messages", []) or []:
            role = message.get("role")
            if role == "system":
                continue  # already folded into the leading system message above
            blocks = cls._blocks(message.get("content"))

            if role == "assistant":
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                for block in blocks:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                                "type": "function",
                                "function": {
                                    "name": block.get("name") or "unknown_tool",
                                    "arguments": json.dumps(
                                        block.get("input") or {}, ensure_ascii=False
                                    ),
                                },
                            }
                        )
                converted: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    converted["tool_calls"] = tool_calls
                messages.append(converted)
                continue

            if role == "user":
                tool_messages: list[dict[str, Any]] = []
                text_parts: list[Any] = []
                for block in blocks:
                    block_type = block.get("type")
                    if block_type == "tool_result":
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id"),
                                "content": cls._stringify(block.get("content")),
                            }
                        )
                    elif block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "image":
                        image = cls._image_content(block)
                        if image:
                            text_parts.append(image)
                    else:
                        text_parts.append(cls._stringify(block))
                messages.extend(tool_messages)
                if text_parts:
                    messages.append({"role": "user", "content": cls._openai_content(text_parts)})

        return messages

    @staticmethod
    def _tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
            if tool.get("name")
        ]

    @classmethod
    def _reasoning(cls, body: dict[str, Any], model_name: str) -> dict[str, Any] | None:
        settings = extract_reasoning_settings(body)
        reasoning: dict[str, Any] = {}
        if settings.effort == "none":
            # A model that always reasons (e.g. Kimi) rejects a disable with HTTP 400
            # ("Reasoning is mandatory ... cannot be disabled"). Omit the field instead
            # so the request succeeds; the model just reasons by default.
            if can_disable_thinking_for_model(model_name):
                reasoning["effort"] = "none"
        elif settings.effort is not None and not accepts_effort_for_model(model_name):
            reasoning["max_tokens"] = thinking_budget_from_max_tokens(body)
        elif settings.effort is not None:
            effort = normalize_effort(settings.effort, max_effort_for_model(model_name))
            reasoning["effort"] = "xhigh" if effort == "max" else effort
        elif settings.max_tokens is not None:
            reasoning["max_tokens"] = settings.max_tokens
        elif settings.enabled:
            reasoning["enabled"] = True
        if settings.exclude is not None:
            reasoning["exclude"] = settings.exclude
        if not reasoning:
            return None
        return reasoning

    @staticmethod
    def _blocks(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, list):
            return [block if isinstance(block, dict) else {"type": "text", "text": str(block)} for block in content]
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
                    parts.append(block.get("text") or block.get("content") or json.dumps(block, ensure_ascii=False))
                else:
                    parts.append(str(block))
            return "\n".join(map(str, parts))
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    @staticmethod
    def _image_content(block: dict[str, Any]) -> dict[str, Any] | None:
        source = block.get("source") or {}
        media_type = source.get("media_type")
        data = source.get("data")
        if source.get("type") == "base64" and media_type and data:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
        return None

    @staticmethod
    def _openai_content(parts: list[Any]) -> Any:
        if all(isinstance(part, str) for part in parts):
            return "\n".join(parts)
        content = []
        for part in parts:
            if isinstance(part, str):
                content.append({"type": "text", "text": part})
            else:
                content.append(part)
        return content

    @classmethod
    def _usage_payload(cls, usage: dict[str, Any]) -> dict[str, int]:
        details = usage.get("prompt_tokens_details", {}) or {}
        cached_tokens = int(details.get("cached_tokens", 0) or 0)
        cache_write_tokens = int(details.get("cache_write_tokens", 0) or 0)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        return {
            "input_tokens": max(prompt_tokens - cached_tokens - cache_write_tokens, 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
            "cache_read_input_tokens": cached_tokens,
            "cache_creation_input_tokens": cache_write_tokens,
        }

    @staticmethod
    def _stop_reason(finish_reason: str | None) -> str:
        return {
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "stop": "end_turn",
            "content_filter": "end_turn",
        }.get(finish_reason or "", "end_turn")

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


def _is_anthropic_model(model_id: str) -> bool:
    return "anthropic/" in model_id or "claude" in model_id
