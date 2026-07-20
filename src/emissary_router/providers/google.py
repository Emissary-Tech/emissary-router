from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from typing import Any

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from emissary_router.caching.usage import Usage
from emissary_router.config import ProviderConfig, ResolvedModel
from emissary_router.schemas import AnthropicRequest, RequestContext
from emissary_router.providers.base import ProviderComplete, sanitize_tool_id, strip_cch_text
from emissary_router.providers.thinking import (
    SYNTHETIC_THINKING_SIGNATURE,
    effort_reasoning_for_model,
    extract_reasoning_settings,
)

logger = logging.getLogger(__name__)

# Gemini 3 validates that functionCall parts in replayed history carry the
# thoughtSignature it originally emitted ("Function call is missing a
# thought_signature in functionCall parts" -> 400, live-verified). Tool calls made by
# OTHER models (sonnet/haiku/glm/kimi) never had one, and Claude Code's
# Anthropic-format history can't carry Gemini's. Google documents this exact dummy
# value as the sanctioned bypass for history imported from other models.
DUMMY_THOUGHT_SIGNATURE = "context_engineering_is_the_way_to_go"


def _request_summary(body: dict[str, Any]) -> str:
    contents = body.get("contents") or []
    roles = [c.get("role") for c in contents if isinstance(c, dict)]
    return (
        f"contents_roles={roles} tools={len(body.get('tools') or [])} "
        f"generationConfig={body.get('generationConfig')} top_keys={sorted(body.keys())}"
    )


class GoogleProvider:
    name = "google"

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._base_url = (config.base_url or "https://generativelanguage.googleapis.com").rstrip("/")

    async def messages(
        self,
        request: AnthropicRequest,
        model: ResolvedModel,
        context: RequestContext,
        on_complete: ProviderComplete | None = None,
    ) -> Response:
        gemini_body = self.to_google_request(request.body, model.model_id, model.name)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._config.api_key or "",
        }

        if request.body.get("stream"):
            # True streaming: translate Google's SSE into Anthropic SSE as chunks
            # arrive (thought parts as a live thinking block, text deltas, complete
            # functionCall parts as tool_use blocks) instead of buffering the whole
            # generation and replaying it.
            return await self._open_stream(gemini_body, model, headers, on_complete)

        endpoint = f"{self._base_url}/v1beta/models/{model.model_id}:generateContent"
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(endpoint, headers=headers, json=gemini_body)

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
                "model_version": payload.get("modelVersion"),
                "response_id": payload.get("responseId"),
            },
        )

        if response.status_code >= 400:
            logger.warning(
                "google upstream %s | %s | error_body=%s",
                response.status_code,
                _request_summary(gemini_body),
                response.text[:800],
            )
            return JSONResponse(payload, status_code=response.status_code)

        message = self.from_google_response(payload, model.name)
        return JSONResponse(message)

    async def _open_stream(
        self,
        gemini_body: dict[str, Any],
        model: ResolvedModel,
        headers: dict[str, str],
        on_complete: ProviderComplete | None,
    ) -> Response:
        """Open the upstream Google stream, then decide the response shape.

        Pre-body upstream errors return a JSONResponse with the REAL status code;
        only a 2xx upstream hands off to the SSE translator.
        """
        endpoint = (
            f"{self._base_url}/v1beta/models/{model.model_id}:streamGenerateContent?alt=sse"
        )
        client = httpx.AsyncClient(timeout=None)
        try:
            http_request = client.build_request(
                "POST", endpoint, headers=headers, json=gemini_body
            )
            response = await client.send(http_request, stream=True)
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
                "google upstream %s | %s | error_body=%s",
                status,
                _request_summary(gemini_body),
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
            self._translate_stream(client, response, gemini_body, model.name, on_complete),
            media_type="text/event-stream",
        )

    async def _translate_stream(
        self,
        client: httpx.AsyncClient,
        response: httpx.Response,
        gemini_body: dict[str, Any],
        model_label: str,
        on_complete: ProviderComplete | None,
    ):
        """Translate an already-open (2xx) Google SSE stream into Anthropic SSE live."""
        status = response.status_code
        error: str | None = None
        sink: dict[str, Any] = {}
        try:
            async for event in self._iter_anthropic_events(
                response.aiter_lines(), model_label, sink
            ):
                yield event
        except Exception as exc:  # noqa: BLE001 - surface in-stream, then end cleanly
            error = repr(exc)
            logger.warning(
                "google upstream stream failed | %s | error=%s",
                _request_summary(gemini_body),
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
                self.usage_from_response({"usageMetadata": sink.get("usageMetadata") or {}}),
                {
                    "http_status": status,
                    "stream": True,
                    "error": error,
                    "model_version": sink.get("modelVersion"),
                    "response_id": sink.get("responseId"),
                },
            )

    @classmethod
    async def _iter_anthropic_events(cls, lines, model_label: str, sink: dict[str, Any]):
        """Translate an async iterator of Google SSE lines into Anthropic SSE events.

        Thought parts stream as a live thinking block (closed with the synthetic
        signature the Anthropic provider strips on later turns), text parts as text
        deltas, and functionCall parts — which Google delivers whole — as complete
        tool_use blocks, preserving the part's real thoughtSignature as
        `thought_signature` for a gemini->gemini replay. Cumulative usageMetadata is
        stashed in `sink` for telemetry and backfilled in the final message_delta.
        """
        sse = cls._sse
        next_index = 0
        cur_stream: tuple[str, int] | None = None  # (kind: thinking|text, index)
        saw_tool_use = False
        started = False
        finish_reason: str | None = None

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
            index = next_index
            next_index += 1
            cur_stream = (kind, index)
            block = (
                {"type": "thinking", "thinking": ""}
                if kind == "thinking"
                else {"type": "text", "text": ""}
            )
            return [
                sse(
                    "content_block_start",
                    {"type": "content_block_start", "index": index, "content_block": block},
                )
            ]

        def ensure_started(chunk: dict[str, Any]) -> list[bytes]:
            nonlocal started
            if started:
                return []
            started = True
            message_id = chunk.get("responseId") or f"msg_{uuid.uuid4().hex[:24]}"
            return [
                sse(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
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
            ]

        async for line in lines:
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue

            if chunk.get("responseId"):
                sink["responseId"] = chunk["responseId"]
            if chunk.get("modelVersion"):
                sink["modelVersion"] = chunk["modelVersion"]
            if chunk.get("usageMetadata"):
                sink["usageMetadata"] = chunk["usageMetadata"]

            for event in ensure_started(chunk):
                yield event

            candidate = (chunk.get("candidates") or [{}])[0]
            if candidate.get("finishReason"):
                finish_reason = candidate["finishReason"]
            parts = ((candidate.get("content") or {}).get("parts") or [])
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("thought"):
                    text = part.get("text") or ""
                    if not text:
                        continue
                    if cur_stream is None or cur_stream[0] != "thinking":
                        for event in close_stream_block():
                            yield event
                        for event in open_stream_block("thinking"):
                            yield event
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": cur_stream[1],
                            "delta": {"type": "thinking_delta", "thinking": text},
                        },
                    )
                    continue
                if "functionCall" in part:
                    for event in close_stream_block():
                        yield event
                    call = part.get("functionCall") or {}
                    index = next_index
                    next_index += 1
                    saw_tool_use = True
                    block: dict[str, Any] = {
                        "type": "tool_use",
                        "id": sanitize_tool_id(call.get("id")) or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": call.get("name") or "unknown_tool",
                        "input": {},
                    }
                    # Keep Gemini's real signature with the call so a gemini->gemini
                    # replay can send it back verbatim (the dummy bypass covers loss).
                    if part.get("thoughtSignature"):
                        block["thought_signature"] = part["thoughtSignature"]
                    yield sse(
                        "content_block_start",
                        {"type": "content_block_start", "index": index, "content_block": block},
                    )
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(
                                    call.get("args") or {}, ensure_ascii=False
                                ),
                            },
                        },
                    )
                    yield sse(
                        "content_block_stop", {"type": "content_block_stop", "index": index}
                    )
                    continue
                if "text" in part:
                    text = part.get("text") or ""
                    if not text:
                        continue
                    if cur_stream is None or cur_stream[0] != "text":
                        for event in close_stream_block():
                            yield event
                        for event in open_stream_block("text"):
                            yield event
                    yield sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": cur_stream[1],
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )

        for event in ensure_started({}):
            yield event
        for event in close_stream_block():
            yield event

        usage = cls._usage_payload_from_metadata(sink.get("usageMetadata") or {})
        yield sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use" if saw_tool_use else cls._stop_reason(finish_reason),
                    "stop_sequence": None,
                },
                # Backfill CUMULATIVE usage: message_start had to carry zeros because
                # Google reports usage at stream end.
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            },
        )
        yield sse("message_stop", {"type": "message_stop"})

    def usage_from_response(self, payload: dict) -> Usage:
        metadata = payload.get("usageMetadata", {}) or {}
        cached_tokens = int(metadata.get("cachedContentTokenCount", 0) or 0)
        prompt_tokens = int(metadata.get("promptTokenCount", 0) or 0)
        output_tokens = int(metadata.get("candidatesTokenCount", 0) or 0)
        reasoning_tokens = int(metadata.get("thoughtsTokenCount", 0) or 0)
        return Usage(
            input_tokens=max(prompt_tokens - cached_tokens, 0),
            output_tokens=output_tokens + reasoning_tokens,
            cache_read_input_tokens=cached_tokens,
            reasoning_output_tokens=reasoning_tokens,
        )

    @classmethod
    def to_google_request(
        cls,
        body: dict,
        model_id: str,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        contents, system_instruction = cls._contents_and_system(body)
        request: dict[str, Any] = {"contents": contents or [{"role": "user", "parts": [{"text": ""}]}]}

        if system_instruction:
            request["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        tools = cls._tools(body.get("tools") or [])
        if tools:
            request["tools"] = tools
            request["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        generation_config = cls._generation_config(body, model_id, model_name)
        if generation_config:
            request["generationConfig"] = generation_config

        return request

    @classmethod
    def from_google_response(cls, payload: dict, model_label: str) -> dict[str, Any]:
        candidate = (payload.get("candidates") or [{}])[0]
        parts = ((candidate.get("content") or {}).get("parts") or [])
        content: list[dict[str, Any]] = []

        thought_text: list[str] = []
        for part in parts:
            if part.get("thought"):
                # Surface Gemini's thought summary like the other providers do, stamped
                # with the synthetic signature the Anthropic provider strips later.
                if part.get("text"):
                    thought_text.append(part["text"])
                continue
            if "text" in part:
                text = part.get("text") or ""
                if text:
                    content.append({"type": "text", "text": text})
            if "functionCall" in part:
                call = part.get("functionCall") or {}
                block: dict[str, Any] = {
                    "type": "tool_use",
                    "id": sanitize_tool_id(call.get("id")) or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": call.get("name") or "unknown_tool",
                    "input": call.get("args") or {},
                }
                # Keep Gemini's real signature with the call so a gemini->gemini replay
                # can send it back verbatim (best-effort: the dummy bypass covers loss).
                if part.get("thoughtSignature"):
                    block["thought_signature"] = part["thoughtSignature"]
                content.append(block)
        if thought_text:
            content.insert(
                0,
                {
                    "type": "thinking",
                    "thinking": "\n".join(thought_text),
                    "signature": SYNTHETIC_THINKING_SIGNATURE,
                },
            )

        if not content:
            content.append({"type": "text", "text": ""})

        usage = cls._usage_payload_from_metadata(payload.get("usageMetadata") or {})
        return {
            "id": payload.get("responseId") or f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": model_label,
            "content": content,
            "stop_reason": "tool_use"
            if any(block.get("type") == "tool_use" for block in content)
            else cls._stop_reason(candidate.get("finishReason")),
            "stop_sequence": None,
            "usage": usage,
        }

    @classmethod
    def _contents_and_system(cls, body: dict) -> tuple[list[dict[str, Any]], str]:
        system_parts: list[str] = []
        top_system = strip_cch_text(cls._stringify(body.get("system", "")))
        if top_system:
            system_parts.append(top_system)
        tool_names_by_id: dict[str, str] = {}
        contents: list[dict[str, Any]] = []
        leading = True

        for message in body.get("messages", []) or []:
            role = message.get("role")

            if role == "system":
                # Same normalization as the other providers: leading system messages
                # join the system instruction; mid/trailing ones are per-turn reminders
                # and stay at their position (as user text) so the prompt head — and
                # with it the provider's prefix cache — stays byte-stable.
                inline = strip_cch_text(cls._stringify(message.get("content")))
                if not inline:
                    continue
                if leading:
                    system_parts.append(inline)
                else:
                    if "<system-reminder" not in inline:
                        inline = f"<system-reminder>\n{inline}\n</system-reminder>"
                    contents.append({"role": "user", "parts": [{"text": inline}]})
                continue
            leading = False

            blocks = cls._blocks(message.get("content"))
            parts: list[dict[str, Any]] = []

            if role == "assistant":
                for block in blocks:
                    if block.get("type") == "text":
                        parts.append({"text": block.get("text", "")})
                    elif block.get("type") == "tool_use":
                        tool_id = block.get("id")
                        name = block.get("name") or "unknown_tool"
                        if tool_id:
                            tool_names_by_id[tool_id] = name
                        parts.append(
                            {
                                "functionCall": {
                                    "name": name,
                                    "args": block.get("input") or {},
                                },
                                # Gemini 3 rejects replayed functionCall parts without a
                                # thoughtSignature. Use the real one when the block
                                # carries it (a prior native-Gemini turn); otherwise the
                                # documented cross-model bypass value.
                                "thoughtSignature": block.get("thought_signature")
                                or DUMMY_THOUGHT_SIGNATURE,
                            }
                        )
                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue

            if role == "user":
                for block in blocks:
                    block_type = block.get("type")
                    if block_type == "text":
                        parts.append({"text": block.get("text", "")})
                    elif block_type == "tool_result":
                        tool_use_id = block.get("tool_use_id")
                        name = tool_names_by_id.get(tool_use_id) or tool_use_id or "tool_result"
                        parts.append(
                            {
                                "functionResponse": {
                                    "name": name,
                                    "response": {"result": cls._stringify(block.get("content", ""))},
                                }
                            }
                        )
                    elif block_type == "image":
                        image_part = cls._image_part(block)
                        if image_part:
                            parts.append(image_part)
                    else:
                        parts.append({"text": cls._stringify(block)})
                if parts:
                    contents.append({"role": "user", "parts": parts})

        return contents, "\n\n".join(system_parts)

    @classmethod
    def _tools(cls, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        declarations = []
        for tool in tools:
            name = tool.get("name")
            if not name:
                continue
            declarations.append(
                {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": cls._clean_schema(
                        tool.get("input_schema") or {"type": "object", "properties": {}}
                    ),
                }
            )
        return [{"functionDeclarations": declarations}] if declarations else []

    @classmethod
    def _generation_config(
        cls,
        body: dict,
        model_id: str,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if body.get("max_tokens") is not None:
            config["maxOutputTokens"] = body["max_tokens"]
        if body.get("temperature") is not None:
            config["temperature"] = body["temperature"]
        if body.get("stop_sequences"):
            config["stopSequences"] = body["stop_sequences"]

        thinking_config = cls._thinking_config(body, model_id, model_name)
        if thinking_config:
            config["thinkingConfig"] = thinking_config
        return config

    @classmethod
    def _thinking_config(
        cls,
        body: dict,
        model_id: str,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        settings = extract_reasoning_settings(body)
        if settings.max_tokens is None and settings.effort is None and not settings.enabled:
            return {}

        if settings.enabled is False or settings.effort == "none":
            return {"thinkingBudget": 0}
        if settings.effort is None and settings.max_tokens is not None:
            return {"thinkingBudget": settings.max_tokens}
        if settings.effort is None and settings.enabled:
            return {"thinkingBudget": -1}

        effort = effort_reasoning_for_model(settings, model_name or model_id)
        return {"thinkingLevel": effort or "medium"}

    @classmethod
    def _blocks(cls, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, list):
            blocks = []
            for block in content:
                blocks.append(block if isinstance(block, dict) else {"type": "text", "text": str(block)})
            return blocks
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
    def _image_part(block: dict[str, Any]) -> dict[str, Any] | None:
        source = block.get("source") or {}
        media_type = source.get("media_type")
        if source.get("type") == "base64" and media_type and source.get("data"):
            return {"inlineData": {"mimeType": media_type, "data": source["data"]}}
        return None

    @classmethod
    def _clean_schema(cls, schema: Any) -> Any:
        schema = deepcopy(schema)
        return cls._clean_schema_in_place(schema)

    @classmethod
    def _clean_schema_in_place(cls, value: Any, parent_key: str | None = None) -> Any:
        valid_keys = {
            "type",
            "format",
            "title",
            "description",
            "nullable",
            "enum",
            "maxItems",
            "minItems",
            "properties",
            "required",
            "minProperties",
            "maxProperties",
            "minLength",
            "maxLength",
            "pattern",
            "example",
            "anyOf",
            "propertyOrdering",
            "default",
            "items",
            "minimum",
            "maximum",
        }
        if isinstance(value, list):
            return [cls._clean_schema_in_place(item, parent_key) for item in value]
        if not isinstance(value, dict):
            return value

        if parent_key != "properties":
            for key in list(value.keys()):
                if key not in valid_keys:
                    del value[key]
        if isinstance(value.get("type"), list):
            types = [item for item in value["type"] if item != "null"]
            if "null" in value["type"]:
                value["nullable"] = True
            if len(types) == 1:
                value["type"] = types[0]
            elif types:
                value.pop("type", None)
                value["anyOf"] = [{"type": item} for item in types]
        for key, item in list(value.items()):
            value[key] = cls._clean_schema_in_place(item, key)
        return value

    @classmethod
    def _usage_payload_from_metadata(cls, metadata: dict[str, Any]) -> dict[str, int]:
        cached_tokens = int(metadata.get("cachedContentTokenCount", 0) or 0)
        prompt_tokens = int(metadata.get("promptTokenCount", 0) or 0)
        output_tokens = int(metadata.get("candidatesTokenCount", 0) or 0)
        reasoning_tokens = int(metadata.get("thoughtsTokenCount", 0) or 0)
        return {
            "input_tokens": max(prompt_tokens - cached_tokens, 0),
            "output_tokens": output_tokens + reasoning_tokens,
            "cache_read_input_tokens": cached_tokens,
            "cache_creation_input_tokens": 0,
        }

    @staticmethod
    def _stop_reason(finish_reason: str | None) -> str:
        return {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
            "SAFETY": "end_turn",
            "RECITATION": "end_turn",
            "LANGUAGE": "end_turn",
            "MALFORMED_FUNCTION_CALL": "end_turn",
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
