# Providers and caching

Each catalog model is bound to a provider. Provider and upstream model id are owned
by the catalog, not your config.

| Model | Provider | Upstream model id |
|---|---|---|
| `claude-sonnet-4.6` | Anthropic | `claude-sonnet-4-6` |
| `claude-haiku-4.5` | Anthropic | `claude-haiku-4-5` |
| `gemini-3.1-flash-lite` | OpenRouter | `google/gemini-3.1-flash-lite` |

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

For OpenRouter, Emissary Router calls the provider non-streaming and synthesizes
Anthropic SSE when Claude Code asked for a stream. This avoids fragile streaming
tool-call translation while still presenting a normal streaming response to Claude
Code.

## Caching

There is no generic cache switch — caching works differently per provider, and the
router preserves each provider's native mechanism rather than managing caches itself.

- Anthropic: Claude Code's `cache_control` blocks pass through unchanged. The router
  removes the dynamic `cch=` attribution line by default
  (`strip_dynamic_attribution`) so the prompt prefix stays byte-stable and the
  provider's prefix cache keeps hitting.
- OpenRouter: Claude models use OpenRouter/Anthropic cache accounting where
  available; Gemini uses provider-native implicit caching behind OpenRouter.
- Native Google: explicit `CachedContent` lifecycle management is out of scope in V1.

Cache reads/writes are recorded in [telemetry](telemetry.md) when the provider
reports them.

`cache_aware` routing treats cache predictions with two confidence levels:

- `predictable` — Anthropic direct cache accounting. The router can trust observed
  cache creation/read signals enough to use them in the next cost estimate.
- `best_effort` — implicit/provider-mediated cache behavior such as Gemini through
  OpenRouter. Matching prefix + TTL is only a hint, so the router does not discount it
  as warm until it has observed an actual cache read.
