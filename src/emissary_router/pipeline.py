from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
import uuid

from starlette.responses import JSONResponse, Response

from emissary_router.caching.usage import Usage
from emissary_router.config import AppConfig, PricingConfig, TokenPricing
from emissary_router.schemas import AnthropicRequest, RequestContext
from emissary_router.providers.registry import build_provider
from emissary_router.routing.classifier import ClassifierClient
from emissary_router.routing.policy import choose_model
from emissary_router.routing.request_to_classifier_input import request_to_classifier_input
from emissary_router.telemetry import JsonlTelemetry


class RouterPipeline:
    def __init__(self, config: AppConfig, pricing: PricingConfig):
        self._config = config
        self._pricing = pricing
        self._classifier = ClassifierClient(config.router)
        self._providers = {
            name: build_provider(name, provider_config)
            for name, provider_config in config.providers.items()
        }
        self._telemetry = (
            JsonlTelemetry(Path(config.telemetry.log_path).expanduser())
            if config.telemetry.enabled
            else None
        )

    async def handle_messages(self, body: dict, headers: dict[str, str]) -> Response:
        request_id = str(uuid.uuid4())
        started_at = time.time()
        classifier_input, classifier_input_metadata = request_to_classifier_input(body)
        probabilities = await self._classifier.predict(classifier_input)
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
                    "classifier_input": {
                        **classifier_input_metadata,
                        **(
                            {"text": classifier_input}
                            if self._config.telemetry.include_classifier_input
                            else {}
                        ),
                    },
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

    def _missing_probability_labels(self, probabilities: dict[str, float]) -> list[str]:
        expected = set(self._config.router.enabled)
        expected.add(self._config.router.default)
        expected.update(self._config.router.policy.candidates)
        return sorted(label for label in expected if label not in probabilities)

    def _cost_usd(self, model_name: str, usage: Usage) -> float | None:
        price = self._pricing.pricing.get(model_name)
        if price is None:
            return None
        return _cost_usd(price, usage)

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
