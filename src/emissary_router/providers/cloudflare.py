"""Cloudflare AI Gateway's Auto Router (``cloudflare/auto``) as a provider.

A benchmark competitor, like ``openrouter/auto``: the gateway picks a model per
request from its own pool. It speaks the OpenAI chat-completions wire format at

    https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/compat/chat/completions

so the whole request/response translation is inherited from OpenRouterProvider; only
the gateway's conventions differ:

- auth is ``cf-aig-authorization: Bearer <token>`` (CLOUDFLARE_API_TOKEN); the model
  keys are the ones stored in the gateway (BYOK) or Cloudflare's unified billing.
- session affinity is a header, ``cf-aig-session-id`` — Auto Router keeps one model
  for a turn only when it sees it, which is what lets prompt caching work (docs:
  "requests in the same conversation may go to different models" otherwise). The
  router sends the client's session id there, the same value OpenRouter gets as the
  body ``session_id``.
- the served model comes back in RESPONSE headers, not the body ``model`` field:
  ``cf-aig-routed-model`` (e.g. ``anthropic/claude-sonnet-5``), with
  ``cf-aig-routing-reason`` saying why. Both land in telemetry's raw_event
  (``routed_model`` / ``routing_reason``) for the bench pick tables.
- there is no billed-cost field, and the compat ``usage`` is NOT enough to price a
  call: for Anthropic-served calls ``prompt_tokens`` is only the uncached remainder
  (measured 2026-10-14: 2 tokens for a 14,548-token prompt, cache write/read split
  absent), and OpenAI-served calls hide ``cached_tokens``. The gateway's own log
  (``GET /accounts/{acct}/ai-gateway/gateways/{gw}/logs``) has an exact per-request
  ``cost`` (reconstructs to the cent: cache write 1.25x, read 0.1x), so the request
  is tagged for that join: ``cf-aig-metadata`` carries our request id + session (the
  log's filterable ``metadata``) and the response's ``cf-aig-request-id`` (= the
  log's ``event_id`` filter key) is kept as ``gateway_request_id`` in raw_event.
- the gateway applies its own Anthropic cache breakpoint at the end of the prompt:
  an appended turn reads the whole previous prompt (measured $0.0030 vs $0.0375
  fresh), a replaced last message misses entirely. Explicit ``cache_control`` marks
  therefore carry nothing here and are stripped; and a system or tool message must
  be a plain string — a content-part array is dropped on the floor (measured: the
  model answered without its system prompt, 24 input tokens billed).
- the gateway's own response cache is bypassed per request (``cf-aig-skip-cache``):
  a benchmark must measure model calls, never a replayed answer.
- the pool is restricted via ``cf-aig-allowed-models`` from the condition
  (``ModelEntry.allowed_models``, so each benchmark condition names its own pool),
  else from ``CLOUDFLARE_ALLOWED_MODELS`` (comma-separated ``provider/model``).

Thinking/effort is never forwarded (THINKING_CAPABILITIES marks the entry
``no_reasoning_surface``): the Auto Router does not support thinking controls yet
(Cloudflare, 2026-09-30), so the chosen model runs at its own defaults.
"""
from __future__ import annotations

import json
import os
from typing import Any

from emissary_router.config import ProviderConfig
from emissary_router.providers.openrouter import OpenRouterProvider
from emissary_router.schemas import RequestContext

ACCOUNT_ENV = "CLOUDFLARE_ACCOUNT_ID"
GATEWAY_ENV = "CLOUDFLARE_GATEWAY_ID"
ALLOWED_MODELS_ENV = "CLOUDFLARE_ALLOWED_MODELS"
ROUTED_MODEL_HEADER = "cf-aig-routed-model"
ROUTING_REASON_HEADER = "cf-aig-routing-reason"
REQUEST_ID_HEADER = "cf-aig-request-id"


def gateway_url(account_id: str | None = None, gateway_id: str | None = None) -> str:
    account_id = account_id or os.environ.get(ACCOUNT_ENV)
    gateway_id = gateway_id or os.environ.get(GATEWAY_ENV)
    if not account_id or not gateway_id:
        raise ValueError(
            f"cloudflare provider requires {ACCOUNT_ENV} and {GATEWAY_ENV} "
            "(the AI Gateway account and gateway ids)"
        )
    return f"https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat/chat/completions"


class CloudflareProvider(OpenRouterProvider):
    name = "cloudflare"

    def __init__(self, config: ProviderConfig, allowed_models: list[str] | None = None):
        if not config.base_url:
            config = config.model_copy(update={"base_url": gateway_url()})
        super().__init__(config)
        # the condition's pool (ModelEntry.allowed_models) wins; the env var is the fallback
        pool = allowed_models if allowed_models is not None else os.environ.get(ALLOWED_MODELS_ENV, "").split(",")
        self._allowed_models = ",".join(m.strip() for m in pool if m and m.strip())

    def _request_headers(self, context: RequestContext) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "cf-aig-skip-cache": "true"}
        if self._config.api_key:
            headers["cf-aig-authorization"] = f"Bearer {self._config.api_key}"
        if context.conversation_id:
            headers["cf-aig-session-id"] = context.conversation_id[:256]
        if self._allowed_models:
            headers["cf-aig-allowed-models"] = self._allowed_models
        # lands in the gateway log's `metadata` (filterable by key/value): the join
        # between our telemetry row and Cloudflare's own `cost` for the same call
        meta = {k: v for k, v in (("request_id", context.request_id),
                                  ("session", context.conversation_id)) if v}
        if meta:
            headers["cf-aig-metadata"] = json.dumps(meta, separators=(",", ":"))
        return headers

    def _metadata_from_headers(self, headers: Any) -> dict[str, Any]:
        get = headers.get if hasattr(headers, "get") else (lambda k, d=None: d)
        out: dict[str, Any] = {}
        routed = get(ROUTED_MODEL_HEADER)
        if routed:
            out["routed_model"] = routed
        reason = get(ROUTING_REASON_HEADER)
        if reason:
            out["routing_reason"] = reason
        request_id = get(REQUEST_ID_HEADER)
        if request_id:
            out["gateway_request_id"] = request_id
        return out

    @classmethod
    def to_openai_request(
        cls,
        body: dict[str, Any],
        model_id: str,
        context: RequestContext | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        request = super().to_openai_request(body, model_id, context, model_name)
        # OpenRouter-only body fields: session stickiness travels as a header here,
        # usage accounting is always on at the gateway (no `usage` toggle), and the
        # top-level cache hint is meaningless (the gateway places its own breakpoint).
        request.pop("session_id", None)
        request.pop("usage", None)
        request.pop("cache_control", None)
        request["messages"] = [cls._gateway_message(m) for m in request.get("messages") or []]
        return request

    @staticmethod
    def _gateway_message(message: dict[str, Any]) -> dict[str, Any]:
        """System and tool messages as plain strings, cache marks off the parts that
        stay parts (see the module docstring for the measurements behind both)."""
        content = message.get("content")
        if not isinstance(content, list):
            return message
        if message.get("role") in ("system", "tool"):
            text = "\n\n".join(
                (p.get("text") or "") if isinstance(p, dict) else str(p) for p in content
            )
            return {**message, "content": text}
        parts = [
            {k: v for k, v in p.items() if k != "cache_control"} if isinstance(p, dict) else p
            for p in content
        ]
        return {**message, "content": parts}
