from __future__ import annotations

import asyncio
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
BASELINE_MODEL = "claude-sonnet-4.6"  # the "default Sonnet" side of the demo comparison


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
            decision = choose_model(
                self._config,
                probabilities,
                cost_features=cost_features,
                cache_ledger=self._cache_ledger,
            )
        model = self._config.resolve_model(decision.model_name)
        provider = self._providers[model.provider]

        context = RequestContext(
            request_id=request_id,
            conversation_id=session_id,
            classifier_input=classifier_input,
            requested_model=body.get("model"),
        )

        def on_complete(usage: Usage, provider_metadata: dict) -> None:
            self._cache_ledger.observe(
                model, cost_features, usage, is_main=(call_kind == "main")
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

    # ----- conference demo: default Sonnet vs routed, side by side -----

    async def chat(
        self,
        baseline_messages: list[dict],
        routed_messages: list[dict],
        session_id: str | None = None,
        max_tokens: int = 32000,
        effort: str | None = None,
        policy: str | None = None,
    ) -> dict:
        """One turn of two parallel conversations — straight Sonnet vs the routed
        system. Each side carries its own history (the answers diverge), and the routed
        side is re-routed every turn. The routed latency is split into router (classifier)
        time and model time. `effort` is the Sonnet-native reasoning level, applied to
        both sides; each provider converts it for its own model. Cost is from actual
        output.

        `policy` overrides the routing policy for this turn (so the page can toggle
        deviate_if_confident vs cache_aware live), and `session_id` scopes the cache
        ledger to this chat. The conversation prefix is sent with a cache breakpoint, so
        staying on a model reuses its cache and switching busts it — which is exactly the
        cost difference cache_aware weighs. Demo calls are not written to telemetry."""
        baseline, routed = await asyncio.gather(
            self._run_side(BASELINE_MODEL, baseline_messages, max_tokens, effort),
            self._route_and_run(routed_messages, max_tokens, effort, session_id, policy),
        )
        b_cost = baseline.get("cost_usd") or 0.0
        r_cost = routed.get("cost_usd") or 0.0
        savings_pct = round((b_cost - r_cost) / b_cost * 100) if b_cost > 0 else 0
        return {
            "baseline_model": BASELINE_MODEL,
            "baseline": baseline,
            "routed": routed,
            "savings_pct": savings_pct,
        }

    def _demo_body(self, messages: list[dict], max_tokens: int, effort: str | None) -> dict:
        body: dict = {"messages": messages, "max_tokens": max_tokens}
        if effort:
            body["output_config"] = {"effort": effort}
        return body

    async def _run_side(
        self, model_name: str, messages: list[dict], max_tokens: int, effort: str | None
    ) -> dict:
        res, _usage = await self._complete_once(model_name, self._demo_body(messages, max_tokens, effort))
        res["router_ms"] = 0.0
        res["model_ms"] = res["latency_ms"]
        res["total_ms"] = res["latency_ms"]
        return res

    async def _route_and_run(
        self,
        messages: list[dict],
        max_tokens: int,
        effort: str | None,
        session_id: str | None,
        policy: str | None,
    ) -> dict:
        body = self._demo_body(messages, max_tokens, effort)
        headers = {SESSION_HEADER: session_id} if session_id else {}
        cost_features = extract_request_cost_features(
            body, headers, self._cache_ledger.expected_output_tokens()
        )
        started = time.monotonic()
        decision = await self._route_decision(body, cost_features=cost_features, policy=policy)
        router_ms = round((time.monotonic() - started) * 1000, 1)
        res, usage = await self._complete_once(decision.model_name, body)
        # Record what the served model actually cached so the next turn's routing knows
        # this conversation is warm on it.
        self._cache_ledger.observe(self._config.resolve_model(decision.model_name), cost_features, usage)
        res["route_reason"] = decision.reason
        res["router_ms"] = router_ms
        res["model_ms"] = res["latency_ms"]
        res["total_ms"] = round(router_ms + res["latency_ms"], 1)
        return res

    async def _route_decision(
        self, body: dict, cost_features=None, policy: str | None = None
    ) -> RouteDecision:
        classifier_input, _ = request_to_classifier_input(body)
        try:
            probabilities = await self._classifier.predict(classifier_input)
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
            logger.warning("demo: classifier failed; using default: %s", exc)
            return self._default_decision("fallback: router_issue")
        if self._missing_probability_labels(probabilities):
            return self._default_decision("fallback: classifier_label_mismatch")
        config = self._config if policy is None else self._config.model_copy(update={"policy": policy})
        return choose_model(
            config, probabilities, cost_features=cost_features, cache_ledger=self._cache_ledger
        )

    async def _complete_once(self, model_name: str, body: dict) -> dict:
        model = self._config.resolve_model(model_name)
        provider = self._providers.get(model.provider) or build_provider(
            model.provider,
            ProviderConfig(type=model.provider, api_key=os.environ.get(PROVIDER_ENV[model.provider])),
        )
        captured: dict = {}

        def on_complete(usage: Usage, provider_metadata: dict) -> None:
            captured["usage"] = usage
            captured["http_status"] = provider_metadata.get("http_status")

        send_body = dict(body)
        send_body["messages"] = _with_cache_breakpoint(send_body.get("messages") or [])
        started = time.time()
        response = await provider.messages(
            AnthropicRequest(body=send_body, headers={}),
            model=model,
            context=RequestContext(
                request_id="demo",
                conversation_id=None,
                classifier_input="",
                requested_model=model_name,
            ),
            on_complete=on_complete,
        )
        latency_ms = round((time.time() - started) * 1000, 1)
        usage = captured.get("usage") or Usage()
        status = captured.get("http_status")
        return {
            "model": model_name,
            "provider": model.provider,
            "answer": _response_text(response),
            "error": None if (status is None or status < 400) else f"upstream {status}",
            "cost_usd": round(_cost_usd(CATALOG[model_name].pricing, usage), 6),
            "prompt_tokens": usage.input_tokens
            + usage.cache_read_input_tokens
            + usage.cache_creation_input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "cache_creation_tokens": usage.cache_creation_input_tokens,
            "latency_ms": latency_ms,
            "http_status": status,
        }, usage

    async def stream_chat(
        self,
        baseline_messages: list[dict],
        routed_messages: list[dict],
        session_id: str | None = None,
        max_tokens: int = 32000,
        effort: str | None = None,
        policy: str | None = None,
    ):
        """Stream both sides of one turn as one merged event stream. Yields dicts:
        `{"side","type":"delta"/"meta"/"done", ...}`. Same routing + cache wiring as
        chat(); both sides run concurrently and their events interleave."""
        queue: asyncio.Queue = asyncio.Queue()

        async def pump(side: str, gen):
            try:
                async for ev in gen:
                    await queue.put(ev)
            except Exception as exc:  # surface as a done so the merge still completes
                await queue.put({
                    "side": side, "type": "done", "model": "-", "error": str(exc),
                    "cost_usd": 0.0, "router_ms": 0.0, "model_ms": 0.0, "total_ms": 0.0,
                })

        tasks = [
            asyncio.create_task(pump("baseline", self._stream_side("baseline", BASELINE_MODEL, baseline_messages, max_tokens, effort))),
            asyncio.create_task(pump("routed", self._stream_routed(routed_messages, max_tokens, effort, session_id, policy))),
        ]
        remaining = 2
        try:
            while remaining:
                ev = await queue.get()
                yield ev
                if ev.get("type") == "done":
                    remaining -= 1
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _stream_routed(self, messages, max_tokens, effort, session_id, policy):
        body = self._demo_body(messages, max_tokens, effort)
        headers = {SESSION_HEADER: session_id} if session_id else {}
        cost_features = extract_request_cost_features(
            body, headers, self._cache_ledger.expected_output_tokens()
        )
        started = time.monotonic()
        decision = await self._route_decision(body, cost_features=cost_features, policy=policy)
        router_ms = round((time.monotonic() - started) * 1000, 1)
        yield {"side": "routed", "type": "meta", "model": decision.model_name,
               "route_reason": decision.reason, "router_ms": router_ms}
        captured: dict = {}
        async for ev in self._stream_side("routed", decision.model_name, messages, max_tokens, effort, captured):
            if ev.get("type") == "done":
                ev["router_ms"] = router_ms
                ev["total_ms"] = round(router_ms + ev["model_ms"], 1)
                ev["route_reason"] = decision.reason
            yield ev
        self._cache_ledger.observe(
            self._config.resolve_model(decision.model_name), cost_features, captured.get("usage") or Usage()
        )

    async def _stream_side(self, side, model_name, messages, max_tokens, effort, captured=None):
        model = self._config.resolve_model(model_name)
        provider = self._providers.get(model.provider) or build_provider(
            model.provider,
            ProviderConfig(type=model.provider, api_key=os.environ.get(PROVIDER_ENV[model.provider])),
        )
        send_body = self._demo_body(messages, max_tokens, effort)
        send_body["messages"] = _with_cache_breakpoint(send_body["messages"])
        send_body["stream"] = True
        cap: dict = {}

        def on_complete(usage: Usage, provider_metadata: dict) -> None:
            cap["usage"] = usage
            cap["status"] = provider_metadata.get("http_status")

        started = time.time()
        response = await provider.messages(
            AnthropicRequest(body=send_body, headers={}),
            model=model,
            context=RequestContext(
                request_id="demo", conversation_id=None, classifier_input="", requested_model=model_name
            ),
            on_complete=on_complete,
        )
        buf = ""
        async for chunk in response.body_iterator:
            text = chunk.decode("utf-8", "replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            buf += text
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                delta = _sse_text_delta(line)
                if delta:
                    yield {"side": side, "type": "delta", "text": delta}
        latency_ms = round((time.time() - started) * 1000, 1)
        usage = cap.get("usage") or Usage()
        if captured is not None:
            captured["usage"] = usage
        status = cap.get("status")
        yield {
            "side": side, "type": "done", "model": model_name, "provider": model.provider,
            "error": None if (status is None or status < 400) else f"upstream {status}",
            "cost_usd": round(_cost_usd(CATALOG[model_name].pricing, usage), 6),
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_input_tokens,
            "cache_creation_tokens": usage.cache_creation_input_tokens,
            "router_ms": 0.0, "model_ms": latency_ms, "total_ms": latency_ms,
        }

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


def _sse_text_delta(line: str) -> str | None:
    """Pull assistant text out of one Anthropic-format SSE line (both demo providers emit
    this shape). Returns the text of a content_block_delta/text_delta, else None."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        event = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if event.get("type") == "content_block_delta":
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            return delta.get("text") or None
    return None


def _with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Mark the last message as an ephemeral cache breakpoint so multi-turn chats reuse
    the conversation prefix. Staying on a model is then a cache read; switching is a cold
    re-read — the cost difference cache_aware is there to weigh. Applied only on the way
    to the provider, so routing and the classifier still see the plain messages."""
    if not messages:
        return messages
    out = [dict(m) for m in messages]
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in content]
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
        last["content"] = blocks
    else:
        return out
    out[-1] = last
    return out


def _response_text(response: Response) -> str:
    """Pull the assistant text out of a provider response (Anthropic message shape)."""
    raw = getattr(response, "body", b"")
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return ""
    content = payload.get("content") if isinstance(payload, dict) else None
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts).strip()
    return ""


def _cost_usd(price: TokenPricing, usage: Usage) -> float:
    cache_write_price = price.cache_write_5m
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * cache_write_price
    ) / 1_000_000
