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

1. `gemini-3.1-flash-lite`
2. `glm-5.2`
3. `kimi-k2.7-code`
4. `claude-haiku-4.5`
5. `claude-sonnet-5`

#### Choosing a provider

Each model is reachable through one or more providers. Omitting `provider` uses the
recommended one. This is a transport choice — the model (and routing) is the same;
only how the request is delivered changes.

| Model                   | Providers (recommended first) |
| ----------------------- | ----------------------------- |
| `claude-sonnet-5`       | `anthropic`, `openrouter`     |
| `claude-haiku-4.5`      | `anthropic`, `openrouter`     |
| `gemini-3.1-flash-lite` | `openrouter`, `google`        |
| `glm-5.2`               | `openrouter`, `zai`           |
| `kimi-k2.7-code`        | `openrouter`                  |

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

### `cost_aware`

Boolean, default `false`. Master switch for the cost-aware extension: length-head
output pricing and `kappa_usd`. While `false`, a classifier's `<model>:len` heads are
logged but ignored and `kappa_usd` / `len_correction` are inert, so routing is exactly
the pre-extension behaviour. Turn it on per deployment (benchmark conditions set it
explicitly); turning it off again is a config edit.

### `kappa_usd`

Float `>= 0`, default `0`. Dollar penalty charged to one expected failure when the
confident candidates are compared: `score = estimated cost + kappa_usd × (1 − P(pass))`.
`0` keeps the plain cheapest-confident-candidate comparison; raise it to make a cheap
but only borderline-confident model lose to a pricier, surer one. `confidence` stays the
hard floor either way. See [routing](#routing).

### `len_correction`

Float `> 0`, default `1.0`. Only meaningful with a classifier that has length heads
(`<model>:len`, see [routing](#routing)): a global multiplier applied to the expected
output tokens read from those heads, because a mean-of-log estimate runs short on
heavy-tailed outputs. Fit it offline; leave at `1.0` otherwise.

### `len_cap_tokens` / `len_floor_tokens`

Integers, defaults `32000` / `100`. The normalization anchors of the classifier's
length heads: a head emits `y = (log tokens − log floor) / (log cap − log floor)`
clipped to `[0, 1]`, and the gateway inverts it with the same two numbers. They must be
the values the classifier was trained with (routerbench builder `ROUTER_LEN_CAP` /
`ROUTER_LEN_FLOOR`, recorded under `len_heads` in that dataset's `summary.json`). The
defaults are the 32K-contract classifiers' values; a classifier trained on the uncapped
label family is built with `len_cap_tokens = 131072`. Set them per deployment, never
by editing the code constants: changing the anchors under a running classifier
rescales every expected-output estimate.

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
3. **Length heads.** A classifier trained with length heads emits `<model>:len` next to
   each pass head — that model's predicted output size for this request (normalized
   log tokens: 100 → 0, 32k → 1). When present, each candidate's output is priced at
   its own predicted size instead of the request-level rolling estimate, so a
   per-token-cheap model that would write 20k tokens loses to a terse pricier one.
   Without such heads nothing changes.
4. **Failure penalty.** With `kappa_usd > 0` candidates are ranked by
   `cost + kappa_usd × (1 − P(pass))` rather than cost alone (step 2 is the
   `kappa_usd = 0` case). Both the pass heads' values, the length heads' values and
   the serving `tau` / `kappa_usd` are kept in each event's `raw_event` for offline
   replay.

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
