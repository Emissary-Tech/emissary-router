# Providers and caching

Each catalog model is bound to a provider. Provider and upstream model id are owned
by the catalog, not your config.

| Model | Provider | Upstream model id |
|---|---|---|
| `claude-sonnet-4.6` | Anthropic | `claude-sonnet-4-6` |
| `claude-haiku-4.5` | Anthropic | `claude-haiku-4-5` |
| `gemini-3.1-flash-lite` | OpenRouter | `google/gemini-3.1-flash-lite` |
| `glm-5.2` | OpenRouter | `z-ai/glm-5.2` |
| `kimi-k2.7-code` | OpenRouter | `moonshotai/kimi-k2.7-code` |

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

## Caching

There is no generic cache switch — caching works differently per provider, and the
router preserves each provider's native mechanism rather than managing caches itself.

- Anthropic: Claude Code's `cache_control` blocks pass through unchanged. The router
  removes the dynamic `cch=` attribution line by default
  (`strip_dynamic_attribution`) so the prompt prefix stays byte-stable and the
  provider's prefix cache keeps hitting.
- OpenRouter: Claude models use OpenRouter/Anthropic cache accounting where
  available; Gemini, GLM, and Kimi use implicit (automatic) caching.
- Native Google: explicit `CachedContent` lifecycle management is out of scope in V1.

Cache reads/writes are recorded in [telemetry](telemetry.md) when the provider
reports them.

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
