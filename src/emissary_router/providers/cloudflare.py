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
- there is no billed-cost field; the bench ledger prices calls per routed model.
- the gateway's own response cache is bypassed per request (``cf-aig-skip-cache``):
  a benchmark must measure model calls, never a replayed answer.
- optional ``CLOUDFLARE_ALLOWED_MODELS`` (comma-separated ``provider/model``) restricts
  the pool via ``cf-aig-allowed-models``.

Thinking/effort is never forwarded (THINKING_CAPABILITIES marks the entry
``no_reasoning_surface``): the Auto Router does not support thinking controls yet
(Cloudflare, 2026-09-30), so the chosen model runs at its own defaults.
"""
from __future__ import annotations

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

    def __init__(self, config: ProviderConfig):
        if not config.base_url:
            config = config.model_copy(update={"base_url": gateway_url()})
        super().__init__(config)
        allowed = os.environ.get(ALLOWED_MODELS_ENV, "")
        self._allowed_models = ",".join(m.strip() for m in allowed.split(",") if m.strip())

    def _request_headers(self, context: RequestContext) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "cf-aig-skip-cache": "true"}
        if self._config.api_key:
            headers["cf-aig-authorization"] = f"Bearer {self._config.api_key}"
        if context.conversation_id:
            headers["cf-aig-session-id"] = context.conversation_id[:256]
        if self._allowed_models:
            headers["cf-aig-allowed-models"] = self._allowed_models
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
        # and usage accounting is always on at the gateway (no `usage` toggle).
        request.pop("session_id", None)
        request.pop("usage", None)
        return request
