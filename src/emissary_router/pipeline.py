from __future__ import annotations

import logging
import os
import time
import uuid

import httpx
from starlette.responses import JSONResponse, Response

from emissary_router.caching.usage import Usage
from emissary_router.catalog import CATALOG, PROVIDER_ENV, TokenPricing
from emissary_router.config import AppConfig, ProviderConfig
from emissary_router.schemas import AnthropicRequest, RequestContext, RouteDecision
from emissary_router.providers.registry import build_provider
from emissary_router.routing.classifier import ClassifierClient
from emissary_router.routing.policy import choose_model
from emissary_router.routing.request_to_classifier_input import request_to_classifier_input
from emissary_router.telemetry import (
    EventRecord,
    SqliteStore,
    call_kind_from_body,
    usage_tokens,
)

logger = logging.getLogger(__name__)

SESSION_HEADER = "x-claude-code-session-id"


class RouterPipeline:
    def __init__(
        self,
        config: AppConfig,
        store: SqliteStore | None = None,
    ):
        self._config = config
        self._classifier = ClassifierClient(config.router)
        self._providers = self._build_providers()
        self._store = store

    def _build_providers(self):
        provider_names = {
            self._config.resolve_model(model_name).provider
            for model_name in self._config.enabled_models()
        }
        return {
            name: build_provider(
                name,
                ProviderConfig(type=name, api_key=os.environ.get(PROVIDER_ENV[name])),
            )
            for name in provider_names
        }

    async def handle_messages(self, body: dict, headers: dict[str, str]) -> Response:
        request_id = str(uuid.uuid4())
        started_at = time.time()
        session_id = _header(headers, SESSION_HEADER)
        call_kind = call_kind_from_body(body)
        classifier_input, classifier_input_metadata = request_to_classifier_input(body)

        # When the router classifier is unreachable (retries already exhausted in
        # ClassifierClient) or returns an unparseable response, fall back to the
        # configured default model rather than failing the request.
        try:
            probabilities = await self._classifier.predict(classifier_input)
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning("classifier failed; routing to default model: %s", exc)
            decision = self._default_decision(reason="fallback: router_issue")
        else:
            missing_labels = self._missing_probability_labels(probabilities)
            if missing_labels:
                self._record_failure(
                    request_id, started_at, body, session_id, call_kind,
                    "(routing error)", 502,
                )
                return JSONResponse(
                    {
                        "error": {
                            "type": "classifier_label_mismatch",
                            "message": "classifier response is missing labels required by config",
                            "missing_labels": missing_labels,
                        }
                    },
                    status_code=502,
                )
            decision = choose_model(self._config, probabilities)
        model = self._config.resolve_model(decision.model_name)
        provider = self._providers[model.provider]

        context = RequestContext(
            request_id=request_id,
            conversation_id=session_id,
            classifier_input=classifier_input,
            requested_model=body.get("model"),
        )

        def on_complete(usage: Usage, provider_metadata: dict) -> None:
            record = EventRecord(
                id=request_id,
                ts=time.time(),
                session_id=session_id,
                call_kind=call_kind,
                requested_model=body.get("model"),
                served_model=decision.model_name,
                provider=model.provider,
                model_id=model.model_id,
                route_reason=decision.reason,
                cost_usd=self._cost_usd(decision.model_name, usage),
                duration_ms=round((time.time() - started_at) * 1000, 3),
                http_status=_int_or_none(provider_metadata.get("http_status")),
                raw_event=None,
                **usage_tokens(usage),
            )
            self._write(record)

        return await provider.messages(
            AnthropicRequest(body=body, headers=headers),
            model=model,
            context=context,
            on_complete=on_complete,
        )

    def _default_decision(self, reason: str) -> RouteDecision:
        return RouteDecision(
            model_name=self._config.default,
            reason=reason,
            probabilities={},
        )

    def _record_failure(
        self,
        request_id: str,
        started_at: float,
        body: dict,
        session_id: str | None,
        call_kind: str,
        served_model: str,
        http_status: int | None,
    ) -> None:
        self._write(
            EventRecord(
                id=request_id,
                ts=time.time(),
                session_id=session_id,
                call_kind=call_kind,
                requested_model=body.get("model"),
                served_model=served_model,
                provider="-",
                model_id="-",
                route_reason="error",
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_creation_tokens=0,
                cost_usd=None,
                duration_ms=round((time.time() - started_at) * 1000, 3),
                http_status=http_status,
                raw_event=None,
            )
        )

    def _missing_probability_labels(self, probabilities: dict[str, float]) -> list[str]:
        expected = set(self._config.enabled_models())
        expected.add(self._config.default)
        return sorted(label for label in expected if label not in probabilities)

    def _cost_usd(self, model_name: str, usage: Usage) -> float | None:
        return _cost_usd(CATALOG[model_name].pricing, usage)

    def _write(self, record: EventRecord) -> None:
        if self._store is None:
            return
        try:
            self._store.write(record)
        except Exception:
            return


def _header(headers: dict[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def _int_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _cost_usd(price: TokenPricing, usage: Usage) -> float:
    cache_write_price = price.cache_write_5m
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * cache_write_price
    ) / 1_000_000
