from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
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
from emissary_router.telemetry import JsonlTelemetry


class RouterPipeline:
    def __init__(self, config: AppConfig):
        self._config = config
        self._classifier = ClassifierClient(config.router)
        self._providers = self._build_providers()
        self._telemetry = (
            JsonlTelemetry(
                Path(config.telemetry.log_path).expanduser(),
                retention_days=config.telemetry.retention_days,
                max_events=config.telemetry.max_events,
            )
            if config.telemetry.enabled
            else None
        )

    def _build_providers(self):
        provider_names = {
            self._config.resolve_model(model_name).provider
            for model_name in self._config.enabled_models()
        }
        return {
            name: build_provider(
                name,
                ProviderConfig(
                    type=name,
                    api_key=os.environ.get(PROVIDER_ENV[name]),
                ),
            )
            for name in provider_names
        }

    async def handle_messages(self, body: dict, headers: dict[str, str]) -> Response:
        request_id = str(uuid.uuid4())
        started_at = time.time()
        classifier_input, classifier_input_metadata = request_to_classifier_input(body)

        # When the router classifier is unreachable (retries already exhausted in
        # ClassifierClient) or returns an unusable response, fall back to the
        # configured default model rather than failing the request.
        try:
            probabilities = await self._classifier.predict(classifier_input)
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            decision = self._default_decision(reason="router_issue")
        else:
            missing_labels = self._missing_probability_labels(probabilities)
            if missing_labels:
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
            else:
                decision = choose_model(self._config, probabilities)
        model = self._config.resolve_model(decision.model_name)
        provider = self._providers[model.provider]
        context = RequestContext(
            request_id=request_id,
            conversation_id=None,
            classifier_input=classifier_input,
            requested_model=body.get("model"),
        )

        def on_complete(usage: Usage, provider_metadata: dict) -> None:
            self._write_telemetry(
                {
                    "ts": time.time(),
                    "duration_ms": round((time.time() - started_at) * 1000, 3),
                    "request_id": request_id,
                    "requested_model": body.get("model"),
                    "served_model": decision.model_name,
                    "provider": model.provider,
                    "model_id": model.model_id,
                    "pricing_model": decision.model_name,
                    "route_reason": decision.reason,
                    "probabilities": decision.probabilities,
                    "classifier_input": classifier_input_metadata,
                    "usage": asdict(usage),
                    "cost_usd": self._cost_usd(decision.model_name, usage),
                    "cache": {
                        "cache_read_input_tokens": usage.cache_read_input_tokens,
                        "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                        "cache_hit": usage.cache_read_input_tokens > 0,
                    },
                    "provider_metadata": provider_metadata,
                }
            )

        return await provider.messages(
            AnthropicRequest(body=body, headers=headers),
            model=model,
            context=context,
            on_complete=on_complete,
        )

    def _default_decision(
        self, reason: str
    ) -> RouteDecision:
        return RouteDecision(
            model_name=self._config.default,
            reason=reason,
            probabilities={},
        )

    def _missing_probability_labels(self, probabilities: dict[str, float]) -> list[str]:
        expected = set(self._config.enabled_models())
        expected.add(self._config.default)
        return sorted(label for label in expected if label not in probabilities)

    def _cost_usd(self, model_name: str, usage: Usage) -> float | None:
        return _cost_usd(CATALOG[model_name].pricing, usage)

    def _write_telemetry(self, row: dict) -> None:
        if self._telemetry is None:
            return
        try:
            self._telemetry.write(row)
        except OSError:
            return


def _cost_usd(price: TokenPricing, usage: Usage) -> float:
    cache_write_price = price.cache_write_5m
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * cache_write_price
    ) / 1_000_000
