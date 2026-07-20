from __future__ import annotations

import json

from emissary_router.config import ProviderConfig
from emissary_router.providers.anthropic import AnthropicProvider
from emissary_router.providers.thinking import SYNTHETIC_THINKING_SIGNATURE

ZAI_DEFAULT_BASE_URL = "https://api.z.ai/api/anthropic"


class ZaiProvider(AnthropicProvider):
    """Z.ai (Zhipu) native Anthropic-compatible endpoint for GLM models.

    The endpoint speaks the Anthropic Messages protocol (live-verified: streaming
    SSE event shapes, tool_use round trips, adaptive thinking + effort params, and
    implicit-cache reads reported as `cache_read_input_tokens`), so the whole
    request side is inherited from AnthropicProvider unchanged.

    The one incompatibility is response THINKING SIGNATURES: z.ai signs thinking
    blocks with its own opaque values, and real Anthropic rejects those on replay
    (live 400: "Invalid `signature` in `thinking` block"). Responses are therefore
    restamped with SYNTHETIC_THINKING_SIGNATURE — the same marker the OpenRouter
    path uses — so the existing history-sanitizing rules cover every next-turn
    route: the Anthropic provider strips the marker, and z.ai itself tolerates it
    in history (live-verified).
    """

    name = "zai"
    REWRITES_RESPONSES = True

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.base_url:
            self._base_url = ZAI_DEFAULT_BASE_URL

    @staticmethod
    def _forward_headers(headers: dict[str, str]) -> dict[str, str]:
        # The client's Anthropic credentials (Claude Code sends x-api-key or an
        # OAuth Authorization header) must never reach z.ai — z.ai would 401 on
        # them. Dropping them here makes messages() attach the provider's own
        # ZAI_API_KEY instead.
        forwarded = AnthropicProvider._forward_headers(headers)
        for key in list(forwarded):
            if key.lower() in ("x-api-key", "authorization"):
                del forwarded[key]
        return forwarded

    def _rewrite_response_json(self, payload: dict) -> dict:
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "thinking" and block.get("signature"):
                block["signature"] = SYNTHETIC_THINKING_SIGNATURE
        return payload

    def _rewrite_sse_line(self, line: str) -> str:
        # z.ai carries the signature on the thinking content_block_start
        # (live-observed); signature_delta is handled too in case that changes.
        stripped = line.rstrip("\r")
        if not stripped.startswith("data: ") or '"signature"' not in stripped:
            return line
        try:
            event = json.loads(stripped[6:])
        except ValueError:
            return line
        changed = False
        block = event.get("content_block")
        if isinstance(block, dict) and block.get("type") == "thinking" and block.get("signature"):
            block["signature"] = SYNTHETIC_THINKING_SIGNATURE
            changed = True
        delta = event.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "signature_delta" and delta.get("signature"):
            delta["signature"] = SYNTHETIC_THINKING_SIGNATURE
            changed = True
        if not changed:
            return line
        return "data: " + json.dumps(event, ensure_ascii=False)
