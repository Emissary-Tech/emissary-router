from __future__ import annotations

import json
import logging
import os
import time
import uuid

import httpx
from starlette.responses import JSONResponse, Response

from emissary_router.caching.ledger import CacheLedger
from emissary_router.caching.usage import Usage
from emissary_router.catalog import CATALOG, PROVIDER_ENV, TokenPricing
from emissary_router.config import AppConfig, ProviderConfig
from emissary_router.schemas import AnthropicRequest, RequestContext, RouteDecision
from emissary_router.providers.registry import build_provider
from emissary_router.routing.classifier import ClassifierClient
from emissary_router.routing.labels import collapse_effort_labels, select_forced_effort
from emissary_router.providers.thinking import always_on_reasoning_models, force_effort
from emissary_router.routing.cache_cost import extract_request_cost_features
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
        cache_ledger: CacheLedger | None = None,
    ):
        self._config = config
        self._classifier = ClassifierClient(config.router)
        self._providers = self._build_providers()
        # Reuse an existing ledger across hot-reloads so dashboard config edits don't
        # wipe warm-cache state. Entries are keyed by (session, provider, model_id,
        # prefix) and TTL-expire, so carrying them over is always safe: entries for a
        # model/provider that just changed simply never match and age out.
        self._cache_ledger = cache_ledger or CacheLedger()
        self._store = store

    @property
    def cache_ledger(self) -> CacheLedger:
        return self._cache_ledger

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
        cost_features = extract_request_cost_features(
            body, headers, self._cache_ledger.expected_output_tokens()
        )

        # Classifier probabilities for this call, persisted into raw_event so
        # offline analysis (tau sweeps, calibration) can replay decisions; stays
        # None on the single-model and classifier-fallback paths.
        probs: dict[str, float] | None = None
        label_winner: dict[str, str] = {}
        base_probs: dict[str, float] = {}
        # A single-model config forces the decision — skip classification entirely.
        # This is also what lets a config serve models the classifier has no head
        # for (e.g. the benchmark-only openrouter-auto passthrough entry).
        enabled = self._config.enabled_models()
        if len(enabled) == 1 and self._config.default == enabled[0]:
            decision = self._default_decision(reason="single_model")
        else:
            # When the router classifier is unreachable (retries already exhausted
            # in ClassifierClient) or returns an unparseable response, fall back to
            # the configured default model rather than failing the request.
            try:
                probabilities = await self._classifier.predict(classifier_input)
            except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
                logger.warning("classifier failed; routing to default model: %s", exc)
                decision = self._default_decision(reason="fallback: router_issue")
            else:
                # effort-suffixed heads (model@low ...) collapse to base models for
                # routing; the winning variant decides the forced effort below
                labeled_probs = probabilities
                probabilities, label_winner = collapse_effort_labels(probabilities)
                probs = labeled_probs
                base_probs = probabilities
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
                # Background (title/summary) calls run with thinking disabled; don't
                # route them to always-on-reasoning models that can't honor that (and
                # that reason on a utility call, wasting cost/latency).
                skip = always_on_reasoning_models() if call_kind == "background" else frozenset()
                decision = choose_model(
                    self._config,
                    probabilities,
                    skip_models=skip,
                    cost_features=cost_features,
                    cache_ledger=self._cache_ledger,
                )
        model = self._config.resolve_model(decision.model_name)
        provider = self._providers[model.provider]
        # The winning variant's effort is forced only when the served model's own
        # head cleared the gate — i.e. the classifier actually selected that
        # (model, effort). A default served as the fallback (nothing confident,
        # default gate-exempt) keeps the client's effort: the classifier had no
        # confident opinion to impose, and the fallback is meant to be the safe path.
        chosen_label = label_winner.get(decision.model_name)
        forced_effort = select_forced_effort(
            label_winner, base_probs, decision.model_name, self._config.confidence
        )
        if forced_effort:
            force_effort(body, forced_effort)

        context = RequestContext(
            request_id=request_id,
            conversation_id=session_id,
            classifier_input=classifier_input,
            requested_model=body.get("model"),
        )

        def on_complete(usage: Usage, provider_metadata: dict) -> None:
            # observed_at = request start: the provider refreshed its cache TTL when it
            # READ the cache (start of processing), not when the stream finished.
            self._cache_ledger.observe(
                model,
                cost_features,
                usage,
                is_main=(call_kind == "main"),
                observed_at=started_at,
            )
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
                raw_event=_routed_raw_event(
                    provider_metadata, model.model_id, probs, self._config.confidence,
                    label=chosen_label if forced_effort else None, forced_effort=forced_effort,
                ),
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
        # The default is gate-exempt (it serves whenever nothing else clears tau), so
        # it needs no head: a classifier trained without the anchor model (e.g. an
        # open-roster or effort-variant deployment) still routes correctly.
        expected = set(self._config.enabled_models()) - {self._config.default}
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


def _routed_raw_event(
    provider_metadata: dict,
    requested_model_id: str,
    probabilities: dict[str, float] | None = None,
    tau: float | None = None,
    label: str | None = None,
    forced_effort: str | None = None,
) -> str | None:
    """For dynamic-router calls (openrouter/auto), keep the actually-routed model
    and the provider's own credit cost — the response is the only place they exist,
    and the bench pick-distribution/cost tables are built from these rows. When the
    classifier ran, also keep its per-head probabilities and the serving tau so
    decisions can be replayed offline (tau sweeps, calibration) without re-running.
    Additive keys only — consumers .get() specific fields."""
    routed = provider_metadata.get("openrouter_model")
    or_cost = provider_metadata.get("or_cost")
    payload: dict = {}
    if (routed and routed != requested_model_id) or or_cost is not None:
        payload.update({
            "routed_model": routed,
            "or_cost": or_cost,
            "gen_id": provider_metadata.get("id"),
        })
    if probabilities:
        payload["probs"] = {k: round(v, 4) for k, v in probabilities.items()}
        if tau is not None:
            payload["tau"] = tau
    if forced_effort:
        payload["label"] = label
        payload["forced_effort"] = forced_effort
    return json.dumps(payload) if payload else None


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
