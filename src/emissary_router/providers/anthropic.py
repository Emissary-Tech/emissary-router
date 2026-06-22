from __future__ import annotations

import json
import re
from copy import deepcopy

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from emissary_router.caching.usage import Usage
from emissary_router.config import ProviderConfig, ResolvedModel
from emissary_router.schemas import AnthropicRequest, RequestContext
from emissary_router.providers.base import ProviderComplete

CCH_ATTRIBUTION_LINE_RE = re.compile(r"(?m)^.*\bcch=[^\s<>\"]+.*(?:\n|$)")


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
        if self._config.cache.strip_dynamic_attribution:
            self._strip_cch_attribution(body)
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
                        self._complete(
                            on_complete,
                            self._usage_from_sse(buf.decode("utf-8", "replace")),
                            {"http_status": status, "stream": True, "error": error},
                        )

            return StreamingResponse(gen(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(f"{self._base_url}/v1/messages", headers=headers, json=body)
        try:
            payload = response.json()
            self._complete(
                on_complete,
                self.usage_from_response(payload),
                {"http_status": response.status_code, "stream": False},
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
