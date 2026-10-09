# Configuration

Emissary Router reads one config file. Default path:

```text
~/.emissary-router/config.json
```

`er init` writes a working copy there, so you usually do not need to edit it to get
started — just add keys to `.env` (see [API keys](#api-keys)). JSON is the config
format. If you ran an older build that created a `config.yaml`, `er init` backs it up
to `config.yaml.bak` and writes a fresh `config.json`. (A YAML file is still loadable
if you point `--config` / `EMISSARY_ROUTER_CONFIG` at it explicitly.)

## Minimal config

This is the full shipped config. Everything not shown uses defaults.

```json
{
  "models": {
    "claude-sonnet-5": { "enabled": true, "provider": "anthropic" },
    "claude-haiku-4.5": { "enabled": true, "provider": "anthropic" },
    "gemini-3.1-flash-lite": { "enabled": true, "provider": "openrouter" },
    "glm-5.2": { "enabled": true, "provider": "openrouter" },
    "kimi-k2.7-code": { "enabled": true, "provider": "openrouter" }
  },
  "default": "claude-sonnet-5",
  "confidence": 0.8,
  "router": { "router_model": "emissary-model-router-shared" },
  "server": { "port": 8788 },
  "telemetry": { "enabled": true, "retention_days": 30, "max_events": 50000 }
}
```

The schema is strict: unknown keys are rejected so typos fail fast at load time
instead of being silently ignored.

## Field reference

### `models` (required)

A map over the built-in catalog. You can only toggle models Emissary supports — the
upstream model id and pricing are owned by the catalog, not the config. Each entry is
an object:

- `enabled` (bool, default `true`) — whether to route to this model
- `provider` (optional) — which provider serves it; omit to use the recommended one
- `effort_routing` (bool, default `false`) — let the router pick this model's
  reasoning effort from the classifier's per-effort heads; see
  [effort routing](#effort-routing)

```json
"claude-haiku-4.5": { "enabled": true, "provider": "openrouter" }
```

Shorthands are also accepted: `true`/`false` means `{ "enabled": ... }`, and a bare
provider name means `{ "provider": ... }`. So `"claude-haiku-4.5": false` disables it
and `"claude-haiku-4.5": "openrouter"` enables it on OpenRouter.

Run `er models` to see the catalog, which entries are enabled, each model's supported
providers, and its `cost_score`. Routing scans enabled models cheapest-first, where
"cheapest" is **derived from each model's price** (not the catalog's listing order), so
reordering the catalog can't change routing. Cheapest → most expensive today:

1. `deepseek-v4-flash` / `deepseek-v4-flash-0731`
2. `gpt-6-luna`
3. `qwen3.8-flash` / `qwen3.8-omni-flash`
4. `glm-5.3-flash`
5. `deepseek-v4.1-flash`
6. `gpt-5.6-luna`
7. `gemini-3.1-flash-lite`
8. `kimi-k2.7-code`
9. `glm-5.2` / `glm-5.3`
10. `claude-haiku-4.5`
11. `claude-sonnet-5` / `claude-sonnet-5.5` / `gpt-6-sol` / `gpt-6.1-sol`
12. `gpt-5.6-terra`
13. `kimi-k3`
14. `claude-opus-5.5` / `gpt-5.6-sol`
15. `claude-opus-5`

Prices (and so this order) mirror the platform gateway's catalog; see
[pricing](pricing.md).

#### Choosing a provider

Each model is reachable through one or more providers. Omitting `provider` uses the
recommended one. This is a transport choice — the model (and routing) is the same;
only how the request is delivered changes.

| Model                                              | Providers (recommended first) |
| -------------------------------------------------- | ----------------------------- |
| `claude-opus-5`, `claude-opus-5.5`, `claude-sonnet-5`, `claude-sonnet-5.5`, `claude-haiku-4.5` | `anthropic`, `openrouter` |
| `gpt-5.6-luna`, `gpt-5.6-terra`, `gpt-5.6-sol`, `gpt-6-luna`, `gpt-6-sol`, `gpt-6.1-sol` | `openai`, `openrouter` |
| `gemini-3.1-flash-lite`                            | `openrouter`, `google`        |
| `glm-5.2`                                          | `openrouter`, `zai`           |
| `glm-5.3`, `glm-5.3-flash`, `kimi-k2.7-code`, `kimi-k3`, `deepseek-v4-flash`, `deepseek-v4-flash-0731`, `deepseek-v4.1-flash`, `qwen3.8-flash`, `qwen3.8-omni-flash` | `openrouter` |

`zai` is Z.ai's native Anthropic-compatible endpoint (`ZAI_API_KEY`, e.g. a GLM
Coding Plan key). Unlike OpenRouter's multi-host routing it serves from one place,
so GLM's implicit cache reads land reliably turn over turn:

```json
"models": {
  "glm-5.2": { "enabled": true, "provider": "zai" }
}
```

Note: GLM Coding Plan keys are quota-based subscriptions — telemetry still prices
zai-served requests at the catalog's per-token GLM rates, so treat those cost rows
as reference numbers, not spend.

`google` is native Gemini (`GOOGLE_API_KEY`) — live streaming and native implicit
cache reporting, single-host; see
[providers and caching](providers-caching.md#gemini-openrouter-by-default-google-native-opt-in)
for details.

Common reasons to override: you only hold one provider's key, or you want to
consolidate billing. For example, if you only have an OpenRouter key, route the Claude
models through it too:

```json
"models": {
  "claude-sonnet-5": { "enabled": true, "provider": "openrouter" },
  "claude-haiku-4.5": { "enabled": true, "provider": "openrouter" },
  "gemini-3.1-flash-lite": { "enabled": true, "provider": "openrouter" }
}
```

Only the keys for the providers you actually resolve to are required — the example
above needs only `OPENROUTER_API_KEY`. Picking a provider a model does not support
(e.g. Gemini on `anthropic`) fails validation. Gemini is OpenRouter-only in V1 because
native Google Gemini 3 is unsafe for Claude Code tool loops (see
[providers and caching](providers-caching.md)). `glm-5.2` and `kimi-k2.7-code` are
OpenRouter-only as well; `kimi-k2.7-code` always reasons (its thinking can't be
disabled), so it keeps reasoning even on background calls.

> Routing to a model requires the configured `router.router_model` to be trained on it.
> The default shared router must recognize `glm-5.2` / `kimi-k2.7-code` for them to be
> routable — otherwise enabling them makes the classifier return a label mismatch. See
> [`router`](#router).

### `default` (required)

The model used when the router is not confident enough to route elsewhere. Must be one
of the enabled models. Set this to your most capable model — it is the quality
fallback used whenever no cheaper enabled model is confident enough, so deviations only
ever save cost.

### `confidence`

Float in `[0, 1]`, default `0.8`. Non-default models must meet this classifier
probability before the router is allowed to consider them. Higher `confidence` = more
conservative (stays on `default` more often). See [routing](#routing).

### `policy` (deprecated)

Older configs may contain a `policy` field; it is accepted and ignored. Routing is
cache-aware by default — see [Routing](#routing) — so there is no policy to choose.

### `router`

```json
"router": { "router_model": "emissary-model-router-shared" }
```

`router_model` selects which trained Emissary router to use — change it when you want a
different routing model. The classifier endpoint (`url`) and `timeout_seconds` have
working defaults and are not shown in the shipped config; override them only for a
private/staging deployment.

### `server`

```json
"server": { "host": "127.0.0.1", "port": 8788, "auth_key": null }
```

Binding to a non-loopback host (e.g. `0.0.0.0`) without `auth_key` fails validation.
When `auth_key` is set, `er code` passes it to Claude Code automatically, and the
[dashboard](dashboard.md) requires the same key.

### `telemetry`

```json
"telemetry": { "enabled": true, "db_path": "~/.emissary-router/events.sqlite3",
               "retention_days": 30, "max_events": 50000 }
```

See [telemetry](telemetry.md). `db_path` overrides the SQLite location;
`retention_days`/`max_events` accept `null` to keep everything. Disabling telemetry
also disables the [dashboard](dashboard.md).

## Routing

Routing is confidence-gated and cache-aware by default:

1. Non-default models must clear `confidence` to be considered at all.
2. The default plus every confident candidate are compared by **cache-adjusted
   per-request cost**: a model that is still warm for the session is credited its
   observed cache reads (the cheap cache-read rate), while switching to a cold model is
   priced at full input plus a cache write. The cheapest wins; the default stays unless
   a candidate is strictly cheaper _after_ cache effects.
3. **Escalation.** When the default's own head is below `confidence` and every
   confident candidate is pricier than the default, the cheapest confident one
   serves anyway ("I say I can't; they say they can"). Fires only when no
   cheaper-or-equal model is confident.
4. The classifier's per-head probabilities and the serving `confidence` are kept in
   each event's `raw_event` for offline replay.

Context limits are deliberately not a routing input: a request that exceeds the
served model's window surfaces as a normalized `prompt is too long` 400 and the
client's own context management (compaction) takes over — see
[providers and caching](providers-caching.md#context-windows-and-long-conversations).

Cache awareness is not a mode. Wherever there is no cache signal — cold start, or a
provider whose caching is opportunistic (see
[providers and caching](providers-caching.md)) — the estimates simply carry no
discount and the comparison is a flat per-request price comparison for the request's
input/output shape. Switching models always starts cold on the new model, and that
cost is exactly what the comparison accounts for.

The cache ledger behind this lives in memory in the gateway process; it is not shared
or persisted, and resets on restart (it re-warms within a turn). The default
`er start` / `er code` launch runs a single process, which is what the ledger expects.
Anthropic cache behavior is tracked directly (`predictable`); OpenRouter implicit
caching is credited only after an observed cache read (`best_effort`).

### Effort routing

A classifier trained with effort arms emits one head per (model, effort) —
`claude-opus-5@low`, `claude-opus-5@medium`, `claude-opus-5@high` — next to, or
instead of, the plain model head. The router folds them into one probability per
model before the gate and the policy run; what it does with the arms is a per-model
choice, `effort_routing` (same rule as the platform gateway's `effort_routing_enabled`):

- **on** — the model is confident when *any* effort clears `confidence`, and it is
  served at the **lowest** effort that does: the cheapest effort the classifier
  vouches for. A client-sent effort is overridden by that choice. Two client settings
  are never overridden: thinking turned off (`effort: none`, `thinking.type:
  disabled`, `reasoning.enabled: false`) and an explicit token budget
  (`thinking.budget_tokens`, `reasoning.max_tokens` without an effort). If no effort
  clears the gate nothing is forced — the request goes out as sent.
- **off** (default) — nothing is forced; the model is read at the head for the effort
  the request will actually run at (the client's effort snapped to the nearest arm;
  the lowest arm when thinking is off; the plain head when the request names no
  effort).

Why the lowest confident effort, not the most probable one: on LLMRouterBench
(2026-10) opus-5 passed 97% of its high-effort wins at low effort too, at a third of
the output tokens, so the arms' probabilities are near-identical and "most probable"
would pick among them at random. The lowest confident rung is where the savings are.

A plain-label classifier is unaffected: with no `@effort` heads, `effort_routing` is
inert. Telemetry records the head read for the served model (`effort_head`), the
effort forced (`forced_effort`) and the fields rewritten (`effort_changes`) in
`raw_event`.

## API keys

Keys live in environment variables or `~/.emissary-router/.env` — never in the config
file. Keeping keys out of the config means forking the repo can't leak them.

```dotenv
EMISSARY_ROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

Where to get each key:

- `EMISSARY_ROUTER_API_KEY` — the Emissary classifier key that powers routing. Sign up
  at https://withemissary.com and create one (Dashboard > Settings > Credentials).
- `ANTHROPIC_API_KEY` — the [Anthropic Console](https://console.anthropic.com/settings/keys),
  for models served directly by Anthropic.
- `OPENROUTER_API_KEY` — [OpenRouter](https://openrouter.ai/keys), for models served via
  OpenRouter.
- `GOOGLE_API_KEY` — [Google AI Studio](https://aistudio.google.com/apikey), only if you
  serve Gemini natively through the `google` provider.
- `OPENAI_API_KEY` — for the gpt-5.6 / gpt-6 models on their native Responses API.
- `CLOUDFLARE_API_TOKEN` (+ `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_GATEWAY_ID`) — only
  for the benchmark-only `cloudflare-auto` entry (Cloudflare AI Gateway's Auto Router).

The easiest way to set them is `er init`, which prompts for each key (skipping any
already in your environment) and writes them to `~/.emissary-router/.env` with `chmod
600`. At any prompt, press Enter to skip a key and set it later (or, when re-running, to
keep the current value). Only the keys for providers you actually use are required —
`er init` won't ask for the rest. Check with `er validate-config`.

Precedence: variables already exported in your shell win; then
`~/.emissary-router/.env`; then a `.env` in the current directory. The loader never
overrides an already-set variable.

### Changing a key

Re-run `er init` (enter to keep each current value, or type a new one), or edit
`~/.emissary-router/.env` directly. Then run `er restart` — the running gateway loaded
the old key at startup and won't pick up the change until it restarts. If a key is
exported in your shell, that value wins over `.env`, so change it where you exported it.

## Alternate locations

```bash
export EMISSARY_ROUTER_HOME=/path/to/dir       # parent for config.json, .env, logs, pid
export EMISSARY_ROUTER_CONFIG=/path/to/config.json
er code --config ./config.json -- [claude args]
```
