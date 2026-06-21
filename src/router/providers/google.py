from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from router.caching.usage import Usage
from router.config import ProviderConfig, ResolvedModel
from router.schemas import AnthropicRequest, RequestContext
from router.providers.base import ProviderComplete
from router.providers.thinking import (
    effort_from_budget,
    extract_reasoning_settings,
    normalize_effort,
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
        gemini_body = self.to_google_request(request.body, model.model_id)
        endpoint = f"{self._base_url}/v1beta/models/{model.model_id}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._config.api_key or "",
        }

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
            return JSONResponse(payload, status_code=response.status_code)

        message = self.from_google_response(payload, model.name)
        if request.body.get("stream"):
            return StreamingResponse(self.anthropic_sse(message), media_type="text/event-stream")
        return JSONResponse(message)

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
    def to_google_request(cls, body: dict, model_id: str) -> dict[str, Any]:
        contents, system_instruction = cls._contents_and_system(body)
        request: dict[str, Any] = {"contents": contents or [{"role": "user", "parts": [{"text": ""}]}]}

        if system_instruction:
            request["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        tools = cls._tools(body.get("tools") or [])
        if tools:
            request["tools"] = tools
            request["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        generation_config = cls._generation_config(body, model_id)
        if generation_config:
            request["generationConfig"] = generation_config

        return request

    @classmethod
    def from_google_response(cls, payload: dict, model_label: str) -> dict[str, Any]:
        candidate = (payload.get("candidates") or [{}])[0]
        parts = ((candidate.get("content") or {}).get("parts") or [])
        content: list[dict[str, Any]] = []

        for part in parts:
            if part.get("thought"):
                continue
            if "text" in part:
                text = part.get("text") or ""
                if text:
                    content.append({"type": "text", "text": text})
            if "functionCall" in part:
                call = part.get("functionCall") or {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": call.get("name") or "unknown_tool",
                        "input": call.get("args") or {},
                    }
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
    def _contents_and_system(cls, body: dict) -> tuple[list[dict[str, Any]], str]:
        system_instruction = cls._stringify(body.get("system", ""))
        tool_names_by_id: dict[str, str] = {}
        contents: list[dict[str, Any]] = []

        for message in body.get("messages", []) or []:
            role = message.get("role")
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
                                }
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

        return contents, system_instruction

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
    def _generation_config(cls, body: dict, model_id: str) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if body.get("max_tokens") is not None:
            config["maxOutputTokens"] = body["max_tokens"]
        if body.get("temperature") is not None:
            config["temperature"] = body["temperature"]
        if body.get("stop_sequences"):
            config["stopSequences"] = body["stop_sequences"]

        thinking_config = cls._thinking_config(body, model_id)
        if thinking_config:
            config["thinkingConfig"] = thinking_config
        return config

    @classmethod
    def _thinking_config(cls, body: dict, model_id: str) -> dict[str, Any]:
        settings = extract_reasoning_settings(body)
        if settings.max_tokens is None and settings.effort is None and not settings.enabled:
            return {}

        effort = normalize_effort(settings.effort, "high")
        if effort is None and settings.max_tokens is not None:
            effort = effort_from_budget(settings.max_tokens)
        if effort is None and settings.enabled:
            effort = "medium"

        if effort == "none":
            effort = "minimal"
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
