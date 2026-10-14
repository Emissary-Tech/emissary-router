# Providers and caching

Each catalog model is bound to a provider. Provider and upstream model id are owned
by the catalog, not your config.

| Model | Provider | Upstream model id | Context window |
|---|---|---|---|
| `claude-sonnet-5` | Anthropic | `claude-sonnet-5` | 200K (1M with the `context-1m` beta) |
| `claude-haiku-4.5` | Anthropic | `claude-haiku-4-5` | 200K |
| `gemini-3.1-flash-lite` | OpenRouter (default) or Google native | `google/gemini-3.1-flash-lite` / `gemini-3.1-flash-lite` | 1M |
| `glm-5.2` | OpenRouter (default) or Z.ai native | `z-ai/glm-5.2` / `glm-5.2` | 1M |
| `kimi-k2.7-code` | OpenRouter | `moonshotai/kimi-k2.7-code` | 256K |

Provider API keys come from the environment (`ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` / `ZAI_API_KEY` for the native
Gemini / GLM opt-ins, `CLOUDFLARE_API_TOKEN` for the benchmark-only Auto Router entry),
loaded from `~/.emissary-router/.env` if present. Only the providers used by enabled
models need keys. The full model table lives in [pricing](pricing.md).

### Z.ai native for GLM

Z.ai's coding endpoint speaks the Anthropic Messages protocol, so the router serves
it through the same code path as Anthropic (verified live: streaming SSE shapes,
tool_use round trips, adaptive thinking + effort params, and implicit cache reads
reported as `cache_read_input_tokens`). One incompatibility is handled in the
provider: z.ai signs thinking blocks with its own opaque signatures, which real
Anthropic rejects on replay — the router restamps them with the same synthetic
marker the OpenRouter path uses, so switching models mid-session stays safe in
both directions (both verified live).

### Cloudflare AI Gateway Auto Router (benchmark only)

`cloudflare-auto` (upstream `cloudflare/auto`) is a competitor router, like
`openrouter-auto`: Cloudflare's gateway picks a model per request from its own pool.
It is gated behind `EMISSARY_ROUTER_BENCH_EXTRAS` and only makes sense as a
single-model condition (the classifier has no head for it). It speaks the
chat-completions wire format at
`https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat/chat/completions`,
so the request/response translation is the OpenRouter provider's; the gateway's own
conventions differ in these places, all handled in `providers/cloudflare.py`:

- auth is `cf-aig-authorization: Bearer $CLOUDFLARE_API_TOKEN`; model keys are the
  ones stored in the gateway or Cloudflare's unified billing (Auto Router needs the
  latter: no credits, 402);
- the client's session id goes out as `cf-aig-session-id` — Auto Router keeps one
  model per turn only when it sees it;
- the candidate pool is the condition's `allowed_models` (`cf-aig-allowed-models`),
  else `CLOUDFLARE_ALLOWED_MODELS`, else the gateway's default pool;
- the served model comes back in response headers (`cf-aig-routed-model`,
  `cf-aig-routing-reason`), recorded in telemetry's `raw_event` as `routed_model` /
  `routing_reason`, with `cf-aig-request-id` as `gateway_request_id`;
- **the compat `usage` cannot price a call.** Measured 2026-10-14: for Anthropic-served
  calls `prompt_tokens` is only the uncached remainder (2 tokens for a 14,548-token
  prompt; no cache write/read split), and OpenAI-served calls hide `cached_tokens`.
  Cloudflare's own per-request log (`GET /accounts/{acct}/ai-gateway/gateways/{gw}/logs`,
  token with AI Gateway Read) carries an exact `cost` (reconstructs to the cent:
  cache write 1.25x, read 0.1x) and the full split in `usage_metadata`
  (`input_tokens`, `input_cached_tokens`, `input_cache_creation_tokens`,
  `output_tokens`, `neurons` for Workers AI). The request is tagged for that join:
  `cf-aig-metadata` carries `{"request_id", "session"}` (the log's filterable
  `metadata`; the `event_id` filter does not match the response header). The bench
  ledger pulls those rows after a run (`er-bench cf-costs`);
- the gateway places its own Anthropic cache breakpoint at the end of the prompt
  (an appended turn reads the whole previous prompt: $0.0030 vs $0.0375 fresh; a
  replaced last message misses entirely), so ER's explicit `cache_control` marks are
  stripped, and a system or tool message is sent as a plain string — the gateway drops
  a content-part array on the floor (measured: the model answered without its system
  prompt, 24 input tokens billed);
- the gateway's response cache is bypassed per request (`cf-aig-skip-cache`).

Thinking/effort is never forwarded: Auto Router ignores `reasoning_effort` (measured:
no direction between low and high), so the chosen model runs at its own defaults.

## Gemini: OpenRouter by default, Google native opt-in

Native Google Gemini 3 function calling requires `thoughtSignature` values to be
returned on the tool-call history — something Claude Code's Anthropic-format
requests don't carry, and that cross-provider routing would otherwise break. The
native provider handles this: real signatures round-trip on `tool_use` blocks
(as `thought_signature`), histories minted by other providers get a bridge value,
and the Claude boundary strips the field before it can 400 ("Extra inputs are not
permitted"). Tool loops, cross-provider switches in both directions, and CC-shaped
thinking/effort requests are live-verified.

The native path is at feature parity with the OpenRouter one: responses stream for
real (Google SSE translated live into Anthropic SSE — thought parts as a thinking
block, text deltas, whole functionCall parts as tool_use blocks), and implicit cache
reads are reported natively (`cachedContentTokenCount`; measured — Google caches
prefix *fragments*, so a warm turn credits part of the prefix rather than all of
it, and short prefixes below the cache minimum report nothing). OpenRouter simply
remains the recommended default; pick `"provider": "google"` (with
`GOOGLE_API_KEY`) for first-party serving and single-host caching.

## Streaming

OpenRouter requests stream for real: the provider's OpenAI-format SSE is translated
to Anthropic SSE as chunks arrive — reasoning as a live thinking block, text deltas,
and incremental tool calls (per-index, safe under interleaving). Since the Claude
Code update thinking is on by default, so generations routinely reason at length
before the first output token; live translation keeps the client fed the whole time
instead of buffering until the end. Upstream errors before any bytes are sent return
their real HTTP status; mid-stream failures emit a terminal SSE `error` event.
Native Google streams the same way (Google SSE translated live), z.ai streams the
upstream's Anthropic SSE with only thinking signatures rewritten in-flight, and
native Anthropic requests were always a raw streaming passthrough.

## Context windows and long conversations

Claude Code decides when to auto-compact from the window of the model it *believes*
it is talking to (the one in `/model`), using the usage numbers the router passes
through. With the default 200K budget this is safe with the catalog above: every
routable model has a 200K+ window, so Claude Code compacts before any of them can
overflow.

⚠️ The case to know about is **1M mode** (`context-1m` beta on Sonnet). Claude Code
then lets the conversation grow far past 200K — beyond what `claude-haiku-4.5` (200K)
or `kimi-k2.7-code` (256K) can hold. The router deliberately does **not** reroute
around this: the conversation belongs to the client, and shrinking it is the
client's decision, not something the router should do silently from its own size
estimates. A request routed to a model that can't hold it comes back as that
provider's 400 (visible as a 400 row in the dashboard), normalized so the client's
own context management takes over.

What that looks like for the user (verified against a real Claude Code 2.1.198
session with an injected overflow 400):

- **Usually invisible.** Claude Code handles the 400 on its own: it truncates old
  tool results (microcompact), re-reads files it still needs, and retries — the turn
  completed a few seconds later with no banner and no lost input. The retry enters
  routing as a normal request.
- **Worst case** (recovery attempts keep failing): Claude Code stops with
  `Context limit reached · /compact or /clear to continue`. Running `/compact`
  summarizes and the session continues; the message you had queued is preserved.
  After any compact the conversation is small again and every model is back in play.

This self-recovery only triggers on Anthropic's `prompt is too long` error shape.
OpenRouter reports the same condition in an OpenAI shape ("This endpoint's maximum
context length is ..."), which Claude Code treats as a generic API error and
abandons the turn (verified live). The router therefore normalizes OpenRouter
context-overflow 400s into the Anthropic shape before returning them, so the
recovery works the same no matter which provider or model served the request; the
original OpenRouter error is still logged.

If you run 1M mode routinely, expect occasional overflow 400 rows for the sub-1M
models (`claude-haiku-4.5` / `kimi-k2.7-code`) as Claude Code trims and retries — or
disable those models for such sessions.

## Caching

There is no generic cache switch — caching works differently per provider, and the
router preserves each provider's native mechanism rather than managing caches itself.

- Anthropic: Claude Code's `cache_control` blocks pass through unchanged. The router
  strips the legacy dynamic `cch=` attribution line on every provider path so the
  prompt prefix stays byte-stable (current Claude Code no longer sends it; the strip
  protects sessions from older clients).
- OpenRouter: Claude models use OpenRouter/Anthropic cache accounting where
  available; Gemini, GLM, and Kimi use implicit (automatic) caching.
- Native Google: explicit `CachedContent` lifecycle management is out of scope in V1.

Cache reads/writes are recorded in [telemetry](telemetry.md) when the provider
reports them.

### Cache support by provider and model

| Model | anthropic | openrouter | zai | google |
|---|---|---|---|---|
| `claude-sonnet-5` | ✅ explicit prompt cache | ✅ Anthropic cache accounting via OpenRouter | — | — |
| `claude-haiku-4.5` | ✅ explicit prompt cache | ✅ Anthropic cache accounting via OpenRouter | — | — |
| `gemini-3.1-flash-lite` | — | ⚠️ implicit, per-host | — | ⚠️ implicit, single host — reads land (measured; partial-fragment credits) |
| `glm-5.2` | — | ⚠️ implicit, per-host | ⚠️ implicit, single host — reads land reliably (measured) | — |
| `kimi-k2.7-code` | — | ⚠️ implicit, per-host | — | — |

- ✅ — deterministic caching: `cache_control` breakpoints pass through and cache
  reads/writes are reported reliably, so the router can trust them for its cost
  estimates (`predictable` tier).
- ⚠️ — implicit (automatic) caching with **per-host** caches behind OpenRouter's load
  balancer: a hit needs the request to re-land on a host that recently served the same
  prefix, so hits are opportunistic rather than guaranteed. The router only credits
  these caches after an *observed* hit (`best_effort` tier) and never assumes one.
- Universally: a cache never carries across providers or models — switching the served
  model always starts cold on the new one. The router prices exactly that when it
  decides whether switching is worth it.

### Implicit caching on OpenRouter (gemini / glm / kimi)

Implicit caching has no write premium, so `cache_creation_tokens` is always 0 for
these models **by design**: the first (cold) call bills the whole prompt at the plain
input rate, which is exactly what telemetry records. Discounted `cache_read_tokens`
appear only when the serving host reports a hit.

Hits are best-effort. OpenRouter load-balances each request across multiple serving
hosts and implicit prefix caches are **per host**, so a hit needs the request to
re-land on a host that served the same prefix recently (measured: a repeated
~4k-token prefix on `glm-5.2` bounced across five hosts and hit only on a same-host
re-land; pinned to a single host, both glm and kimi hit on every call after the cold
one — though one identical pinned run on kimi's first-party host reported no hits at
all, so warm-up is not fully deterministic even then). OpenRouter's own billed `cost`
tracks `cached_tokens` exactly (cold ~4.2x the warm price on the same host, measured
on kimi), so there is no silent discounting: rows showing `cached 0` really were
billed at the input rate, and telemetry's conservative recording matches the actual
bill in both directions. Pinning OpenRouter provider order would make hits mostly
deterministic but changes which host (and price) serves the model, so it is
deliberately not configured in V1.
