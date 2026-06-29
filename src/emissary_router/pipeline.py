from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime

import httpx
from starlette.responses import JSONResponse, Response

from emissary_router import demo_search, demo_stream
from emissary_router.caching.ledger import CacheLedger
from emissary_router.caching.usage import Usage
from emissary_router.catalog import CATALOG, PROVIDER_ENV, TokenPricing
from emissary_router.config import AppConfig, ProviderConfig
from emissary_router.schemas import AnthropicRequest, RequestContext, RouteDecision
from emissary_router.providers.registry import build_provider
from emissary_router.providers.thinking import accepts_effort_for_model
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
_AGENT_MAX_ROUNDS = 16  # runaway guard for the tool loop, not a functional limit
# Anthropic requires this header on every request; Claude Code sends it, but the demo
# builds its own requests, so it must supply it (harmless for other providers).
_DEMO_HEADERS = {"anthropic-version": "2023-06-01"}


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

    async def chat_side(
        self,
        side: str,
        baseline_messages: list[dict],
        routed_messages: list[dict],
        session_id: str | None = None,
        max_tokens: int = 32000,
        effort: str | None = None,
        policy: str | None = None,
    ) -> dict:
        """Run just one side of a turn, so the page can fire both independently and render
        whichever finishes first. `side` is "baseline" (straight Sonnet) or "routed"."""
        if side == "baseline":
            return await self._run_side(BASELINE_MODEL, baseline_messages, max_tokens, effort)
        return await self._route_and_run(routed_messages, max_tokens, effort, session_id, policy)

    def _demo_body(
        self, messages: list[dict], max_tokens: int, effort: str | None, with_search: bool = False
    ) -> dict:
        body: dict = {
            "system": _demo_system(with_search),
            "messages": messages,
            "max_tokens": max_tokens,
        }
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
        _demo_reasoning_for_model(send_body, model_name)
        send_body["messages"] = _with_cache_breakpoint(send_body.get("messages") or [])
        started = time.time()
        response = await provider.messages(
            AnthropicRequest(body=send_body, headers=dict(_DEMO_HEADERS)),
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
        search: bool = False,
    ):
        """Stream both sides of one turn as one merged event stream. Yields dicts:
        `{"side","type":"delta"/"tool"/"meta"/"done", ...}`. Same routing + cache wiring
        as chat(); both sides run concurrently and their events interleave. When `search`
        is on, each side is a web-search agent (it may search before answering, emitting
        `tool` events between bursts of text)."""
        tools = [demo_search.WEB_SEARCH_TOOL] if search else None
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
            asyncio.create_task(pump("baseline", self._stream_side("baseline", BASELINE_MODEL, baseline_messages, max_tokens, effort, tools=tools))),
            asyncio.create_task(pump("routed", self._stream_routed(routed_messages, max_tokens, effort, session_id, policy, tools))),
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

    async def _stream_routed(self, messages, max_tokens, effort, session_id, policy, tools=None):
        body = self._demo_body(messages, max_tokens, effort, with_search=bool(tools))
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
        async for ev in self._stream_side("routed", decision.model_name, messages, max_tokens, effort, captured, tools):
            if ev.get("type") == "done":
                ev["router_ms"] = router_ms
                ev["total_ms"] = round(router_ms + ev["model_ms"], 1)
                ev["route_reason"] = decision.reason
            yield ev
        self._cache_ledger.observe(
            self._config.resolve_model(decision.model_name), cost_features, captured.get("usage") or Usage()
        )

    async def _stream_side(self, side, model_name, messages, max_tokens, effort, captured=None, tools=None):
        """Stream one side as a (possibly tool-using) agent. Streams text deltas; if the
        model calls tools, runs them and continues — looping until it answers without a
        tool call. The round cap is a runaway guard, not a functional limit."""
        model = self._config.resolve_model(model_name)
        provider = self._providers.get(model.provider) or build_provider(
            model.provider,
            ProviderConfig(type=model.provider, api_key=os.environ.get(PROVIDER_ENV[model.provider])),
        )
        convo = list(messages)
        agg = Usage()
        searches = 0
        status = None
        started = time.time()

        for _ in range(_AGENT_MAX_ROUNDS):
            send_body = self._demo_body(convo, max_tokens, effort, with_search=bool(tools))
            _demo_reasoning_for_model(send_body, model_name)
            send_body["messages"] = _with_cache_breakpoint(send_body["messages"])
            if tools:
                send_body["tools"] = tools

            text = {"text": ""}
            thinking = {"thinking": "", "signature": ""}
            tool_uses: list[dict] = []
            stop_reason = None
            # Real per-provider streaming: Anthropic native, OpenRouter passthrough.
            async for nev in demo_stream.stream_model(provider, model, send_body, _DEMO_HEADERS):
                ntype = nev["type"]
                if ntype == "text":
                    text["text"] += nev["text"]
                    yield {"side": side, "type": "delta", "text": nev["text"]}
                elif ntype == "thinking":
                    thinking["thinking"] += nev["text"]  # captured, not shown
                elif ntype == "signature":
                    thinking["signature"] += nev["text"]
                elif ntype == "tool_call":
                    tool_uses.append({"type": "tool_use", "id": nev["id"],
                                      "name": nev["name"], "input": nev.get("input") or {}})
                elif ntype == "done":
                    agg = _add_usage(agg, nev.get("usage") or Usage())
                    status = nev.get("status")
                    stop_reason = nev.get("stop_reason")

            if stop_reason != "tool_use" or not tool_uses:
                break

            # Replay the assistant turn (thinking + text + tool calls), run the tools,
            # feed the results back, then loop.
            convo.append({"role": "assistant",
                          "content": _assistant_turn_content(text["text"], thinking, tool_uses)})
            results = []
            for block in tool_uses:
                if block.get("name") == "web_search":
                    query = (block.get("input") or {}).get("query", "")
                    searches += 1
                    yield {"side": side, "type": "tool", "name": "web_search", "query": query}
                    results.append({
                        "type": "tool_result", "tool_use_id": block.get("id"),
                        "content": await demo_search.web_search(query),
                    })
            if not results:
                break
            convo.append({"role": "user", "content": results})

        latency_ms = round((time.time() - started) * 1000, 1)
        if captured is not None:
            captured["usage"] = agg
        yield {
            "side": side, "type": "done", "model": model_name, "provider": model.provider,
            "error": None if (status is None or status < 400) else f"upstream {status}",
            "cost_usd": round(_cost_usd(CATALOG[model_name].pricing, agg), 6),
            "output_tokens": agg.output_tokens,
            "cache_read_tokens": agg.cache_read_input_tokens,
            "cache_creation_tokens": agg.cache_creation_input_tokens,
            "searches": searches,
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


def _demo_reasoning_for_model(body: dict, model_name: str) -> None:
    """Demo-only: a model that doesn't take an effort param (e.g. Haiku) gets a thinking
    budget of half its max_tokens instead, so reasoning still happens when requested and
    the answer keeps room. Models that do take effort keep `output_config.effort`. The
    Claude Code integration never calls this, so its behavior is unchanged."""
    output_config = body.get("output_config")
    effort = output_config.get("effort") if isinstance(output_config, dict) else None
    if effort and not accepts_effort_for_model(model_name):
        body.pop("output_config", None)
        budget = max(1024, (body.get("max_tokens") or 32000) // 2)
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}


def _demo_system(with_search: bool) -> str:
    """Demo system prompt — applied identically to both sides. Honest (no impersonation),
    minimal, with today's real date (so models don't treat the present as the future) and
    an anti-fabrication rule. The web-search block is included only when search is on."""
    lines = [
        "You are a helpful, knowledgeable assistant.",
        "",
        f"Current date: {datetime.now().strftime('%Y-%m-%d')}",
        "",
        "# Guidelines",
        "- Be accurate, clear, and concise.",
        "- If you are unsure or lack the information, say so plainly — never fabricate "
        "specifics (names, numbers, dates, results).",
        "- Use Markdown (headings, lists, tables, code blocks) to structure longer "
        "answers when it improves clarity.",
    ]
    if with_search:
        lines += [
            "",
            "# Tools",
            "## web_search",
            "Use the `web_search` tool whenever the question involves current events, "
            "recent or time-sensitive facts, or specifics you cannot verify from memory. "
            "Base your answer on what the search returns rather than prior assumptions — "
            "if the results conflict with what you remember, trust the results, since "
            "your training data may be out of date.",
        ]
    return "\n".join(lines)


def _assistant_turn_content(text: str, thinking: dict, tool_uses: list[dict]) -> list[dict]:
    """Rebuild the assistant `content` for a tool round so it can be appended to the
    conversation and replayed. A signed thinking block is kept first (Anthropic requires
    it preserved when continuing after a tool call with extended thinking on); then any
    text, then the tool calls."""
    content: list[dict] = []
    if thinking.get("signature"):
        content.append({
            "type": "thinking",
            "thinking": thinking.get("thinking", ""),
            "signature": thinking["signature"],
        })
    if text:
        content.append({"type": "text", "text": text})
    content.extend(tool_uses)
    return content


def _add_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        cache_read_input_tokens=a.cache_read_input_tokens + b.cache_read_input_tokens,
        cache_creation_input_tokens=a.cache_creation_input_tokens + b.cache_creation_input_tokens,
    )


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
