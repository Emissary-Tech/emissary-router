# Providers and caching

Each catalog model is bound to a provider. Provider and upstream model id are owned
by the catalog, not your config.

| Model | Provider | Upstream model id | Context window |
|---|---|---|---|
| `claude-sonnet-4.6` | Anthropic | `claude-sonnet-4-6` | 200K (1M with the `context-1m` beta) |
| `claude-haiku-4.5` | Anthropic | `claude-haiku-4-5` | 200K |
| `gemini-3.1-flash-lite` | OpenRouter | `google/gemini-3.1-flash-lite` | 1M |
| `glm-5.2` | OpenRouter | `z-ai/glm-5.2` | 1M |
| `kimi-k2.7-code` | OpenRouter | `moonshotai/kimi-k2.7-code` | 256K |

Provider API keys come from the environment (`ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`), loaded from `~/.emissary-router/.env` if present. Only the
providers used by enabled models need keys.

## Why Gemini goes through OpenRouter

Native Google Gemini 3 function calling requires `thoughtSignature` values to be
returned on the tool-call history. Claude Code's Anthropic-format requests don't
carry those, and routing can switch providers between turns, so native Gemini is not
safe for Claude Code tool loops. The catalog therefore serves Gemini through
OpenRouter, which handles this for Claude Code workloads.

## Streaming

OpenRouter requests stream for real: the provider's OpenAI-format SSE is translated
to Anthropic SSE as chunks arrive — reasoning as a live thinking block, text deltas,
and incremental tool calls (per-index, safe under interleaving). Since the Claude
Code update thinking is on by default, so generations routinely reason at length
before the first output token; live translation keeps the client fed the whole time
instead of buffering until the end. Upstream errors before any bytes are sent return
their real HTTP status; mid-stream failures emit a terminal SSE `error` event.
Anthropic requests were always a raw streaming passthrough.

## Context windows and long conversations

Claude Code decides when to auto-compact from the window of the model it *believes*
it is talking to (the one in `/model`), using the usage numbers the router passes
through. With the default 200K budget this is safe with the catalog above: every
routable model has a 200K+ window, so Claude Code compacts before any of them can
overflow.

⚠️ The case that needs care is **1M mode** (`context-1m` beta on Sonnet). Claude Code
then lets the conversation grow far past 200K — beyond what `claude-haiku-4.5` (200K)
or `kimi-k2.7-code` (256K) can hold. The router handles this deterministically with a
**context-fit guard**: models whose window can't hold the request are excluded from
routing, and the normal price comparison picks the cheapest enabled model that fits.
Request size is taken from the router's own estimate *and* the provider-reported
cache size of the previous turn (real tokenizer numbers), the requested output is
reserved on the OpenRouter side (which validates input + output against the window —
measured), and Sonnet's 1M beta is honored per request. Oversized turns, retries, and
the compact summarization call all take the same path — nothing depends on routing
luck.

A provider overflow 400 therefore only remains possible when **no enabled model fits**
(e.g. only sub-1M models enabled in a 1M-mode session). What that looks like for the
user (verified against a real Claude Code 2.1.198 session with an injected overflow
400):

- **Usually invisible.** Claude Code handles the 400 on its own: it truncates old
  tool results (microcompact), re-reads files it still needs, and retries — the turn
  completed a few seconds later with no banner and no lost input.
- **Worst case** (recovery attempts keep failing): Claude Code stops with
  `Context limit reached · /compact or /clear to continue`. Running `/compact`
  summarizes and the session continues; the message you had queued is preserved.

This self-recovery only triggers on Anthropic's `prompt is too long` error shape.
OpenRouter reports the same condition in an OpenAI shape ("This endpoint's maximum
context length is ..."), which Claude Code treats as a generic API error and
abandons the turn (verified live). The router therefore normalizes OpenRouter
context-overflow 400s into the Anthropic shape before returning them, so the
recovery works the same no matter which provider or model served the request; the
original OpenRouter error is still logged.

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

| Model | anthropic | openrouter |
|---|---|---|
| `claude-sonnet-4.6` | ✅ explicit prompt cache | ✅ Anthropic cache accounting via OpenRouter |
| `claude-haiku-4.5` | ✅ explicit prompt cache | ✅ Anthropic cache accounting via OpenRouter |
| `gemini-3.1-flash-lite` | — | ⚠️ implicit, per-host |
| `glm-5.2` | — | ⚠️ implicit, per-host |
| `kimi-k2.7-code` | — | ⚠️ implicit, per-host |

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
