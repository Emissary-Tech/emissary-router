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


def _reasoning_text(obj: dict[str, Any]) -> str:
    """Reasoning text from an OpenAI-style delta/message. Prefers the flat `reasoning`
    field (what GLM/Kimi send); falls back to text inside `reasoning_details` for models
    that only populate the structured form. Signature-only detail entries (no text) are
    skipped, so this never duplicates or emits empty thinking."""
    text = obj.get("reasoning")
    if text:
        return text
    details = obj.get("reasoning_details")
    if isinstance(details, list):
        return "".join(
            d["text"] for d in details if isinstance(d, dict) and d.get("text")
        )
    return ""


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
            # whole (possibly multi-minute, reasoning-heavy) generation to finish. The
            # upstream stream is opened BEFORE we commit our own response so a pre-body
            # upstream error (429/401/5xx) is returned with its real HTTP status —
            # status-based client handling (rate-limit backoff, auth errors) keeps working.
            return await self._open_stream(oai_body, headers, model.name, on_complete)

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

    async def _open_stream(
        self,
        oai_body: dict[str, Any],
        headers: dict[str, str],
        model_label: str,
        on_complete: ProviderComplete | None,
    ) -> Response:
        """Open the upstream stream, then decide the response shape.

        Pre-body upstream errors return a JSONResponse with the REAL status code so
        status-based client handling (429 backoff, 401 surfacing) still works; only a
        2xx upstream hands off to the SSE translator.
        """
        stream_body = {**oai_body, "stream": True, "stream_options": {"include_usage": True}}
        client = httpx.AsyncClient(timeout=None)
        try:
            request = client.build_request(
                "POST", self._base_url, headers=headers, json=stream_body
            )
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
                "openrouter upstream %s | %s | error_body=%s",
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
        """Translate an already-open (2xx) OpenRouter SSE stream into Anthropic SSE live."""
        status = response.status_code
        error: str | None = None
        sink: dict[str, Any] = {}
        try:
            async for event in self._iter_anthropic_events(
                response.aiter_lines(), model_label, sink
            ):
                yield event
        except Exception as exc:  # noqa: BLE001 - surface in-stream, then end cleanly
            # GeneratorExit/CancelledError are BaseException and bypass this branch, so
            # we only get here for real upstream/translation failures while the client
            # is still listening: send a terminal error event instead of severing the
            # socket mid-stream (which clients report as a cryptic parse failure).
            error = repr(exc)
            logger.warning(
                "openrouter upstream stream failed | %s | error=%s",
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
            except Exception:  # noqa: BLE001 - never mask the original outcome on cleanup
                pass
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
        # Active thinking/text block (they alternate); tool blocks are tracked
        # separately per OpenAI tool index and stay open until stream end, so
        # interleaved argument fragments (0/1/0) or reasoning arriving between two
        # fragments of one call can never orphan a tool and fabricate an
        # "unknown_tool" block with split JSON.
        cur_stream: tuple[str, int] | None = None  # (kind: thinking|text, index)
        open_tools: dict[int, int] = {}  # OpenAI tool index -> Anthropic block index
        finish_reason: str | None = None
        final_usage: dict[str, Any] = {}

        def close_stream_block() -> list[bytes]:
            nonlocal cur_stream
            if cur_stream is None:
                return []
            kind, index = cur_stream
            out: list[bytes] = []
            if kind == "thinking":
                # Close the thinking block with a signature so the client treats it like a
                # native thinking block. It's a synthetic marker, not a real Anthropic
                # signature; the Anthropic provider strips these from later turns.
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
            block = {"type": "thinking", "thinking": ""} if kind == "thinking" else {"type": "text", "text": ""}
            out.append(
                sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": next_index, "content_block": block},
                )
            )
            cur_stream = (kind, next_index)
            next_index += 1
            return out

        def open_tool(oai_idx: int, tool_id: str | None, name: str | None) -> list[bytes]:
            nonlocal next_index
            out = close_stream_block()
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
            open_tools[oai_idx] = next_index
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
                # Record the moment usage arrives: a client cancel or upstream drop
                # during the trailing events must not zero out telemetry for a
                # generation the provider already billed.
                final_usage = obj["usage"]
                sink["usage"] = final_usage
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}

            reasoning = _reasoning_text(delta)
            if reasoning:
                for event in open_stream_block("thinking"):
                    yield event
                yield sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": cur_stream[1],
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    },
                )

            content = delta.get("content")
            if content:
                for event in open_stream_block("text"):
                    yield event
                yield sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": cur_stream[1],
                        "delta": {"type": "text_delta", "text": content},
                    },
                )

            for tool_call in delta.get("tool_calls") or []:
                oai_idx = tool_call.get("index", 0)
                fn = tool_call.get("function") or {}
                if oai_idx not in open_tools:
                    for event in open_tool(oai_idx, tool_call.get("id"), fn.get("name")):
                        yield event
                args = fn.get("arguments")
                if args:
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": open_tools[oai_idx],
                            "delta": {"type": "input_json_delta", "partial_json": args},
                        },
                    )

        for event in close_stream_block():
            yield event
        for index in sorted(open_tools.values()):
            yield sse("content_block_stop", {"type": "content_block_stop", "index": index})
        # Cumulative usage in the final message_delta (the protocol-sanctioned backfill
        # point): OpenAI-format usage only arrives at stream end, so message_start had
        # to carry zeros — without this the client would never see input/cache tokens
        # (context-window tracking and cost display would read an empty prompt).
        yield sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": cls._stop_reason(finish_reason), "stop_sequence": None},
                "usage": cls._usage_payload(final_usage),
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
        reasoning = _reasoning_text(oai_message)
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
    def _messages(cls, body: dict[str, Any]) -> list[dict[str, Any]]:
        # Top-level system plus LEADING role:"system" messages fold into one leading
        # system message (OpenAI format). MID/TRAILING role:"system" messages are
        # per-turn reminders Claude Code injects: convert them in place to user
        # messages so the prompt head stays byte-stable — folding them to the front
        # would bust the provider's prefix-matched prompt cache every time a reminder
        # changes, and would misrepresent a per-turn note as a global preamble.
        messages: list[dict[str, Any]] = []
        body_messages: list[dict[str, Any]] = []
        system_parts: list[str] = []
        top_system = cls._stringify(body.get("system"))
        if top_system:
            system_parts.append(top_system)

        leading = True
        for message in body.get("messages", []) or []:
            role = message.get("role")
            if role == "system":
                inline = cls._stringify(message.get("content"))
                if not inline:
                    continue
                if leading:
                    system_parts.append(inline)
                else:
                    if "<system-reminder" not in inline:
                        inline = f"<system-reminder>\n{inline}\n</system-reminder>"
                    body_messages.append({"role": "user", "content": inline})
                continue
            leading = False
            body_messages.append(message)

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for message in body_messages:
            role = message.get("role")
            if role == "user" and isinstance(message.get("content"), str):
                messages.append(message)
                continue
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
