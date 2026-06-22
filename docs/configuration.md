# Configuration

Emissary Router reads one config file. Default path:

```text
~/.emissary-router/config.yaml
```

`install.sh` writes a working copy there, so you usually do not need to edit it to
get started — just add keys to `.env` (see [API keys](#api-keys)).

## Minimal config

This is the full shipped config. Everything not shown uses defaults.

```yaml
models:
  claude-sonnet-4.6: true
  claude-haiku-4.5: true
  gemini-3.1-flash-lite: true

default: claude-sonnet-4.6
confidence: 0.8

server:
  port: 8788

telemetry:
  enabled: true
  retention_days: 30
  max_events: 50000
```

The config schema is strict: unknown keys are rejected so typos fail fast at load
time instead of being silently ignored.

## Field reference

### `models` (required)

A map over the built-in catalog. You can only toggle models Emissary supports —
upstream model id and pricing are owned by the catalog, not the config. Each value
is one of:

- `false` — disabled
- `true` — enabled, using the model's recommended provider
- a provider name (e.g. `openrouter`) — enabled, served through that provider

```yaml
models:
  claude-sonnet-4.6: true
  claude-haiku-4.5: true
  gemini-3.1-flash-lite: false   # disabled
```

Run `er models` to see the catalog, which entries are enabled, and each model's
supported providers. Catalog order is cheap → expensive, and routing relies on it:

1. `gemini-3.1-flash-lite`
2. `claude-haiku-4.5`
3. `claude-sonnet-4.6`

#### Choosing a provider

Each model is reachable through one or more providers. `true` uses the recommended
one; set a provider name to choose explicitly. This is a transport choice — the model
(and routing) is the same; only how the request is delivered changes.

| Model | Providers (recommended first) |
|---|---|
| `claude-sonnet-4.6` | `anthropic`, `openrouter` |
| `claude-haiku-4.5` | `anthropic`, `openrouter` |
| `gemini-3.1-flash-lite` | `openrouter` |

Common reasons to override: you only hold one provider's key, or you want to
consolidate billing. For example, if you only have an OpenRouter key, route the Claude
models through it too:

```yaml
models:
  claude-sonnet-4.6: openrouter
  claude-haiku-4.5: openrouter
  gemini-3.1-flash-lite: true     # already OpenRouter
```

Only the keys for the providers you actually resolve to are required — the example
above needs only `OPENROUTER_API_KEY`. Picking a provider a model does not support
(e.g. `gemini-3.1-flash-lite: anthropic`) fails validation. Gemini is OpenRouter-only
in V1 because native Google Gemini 3 is unsafe for Claude Code tool loops (see
[providers and caching](providers-caching.md)).

### `default` (required)

The model used when the router is not confident enough to route elsewhere. Must be
one of the enabled models. Set this to your most capable model — it is the quality
fallback used whenever no cheaper enabled model is confident enough, so deviations
only ever save cost.

### `confidence`

Float in `[0, 1]`, default `0.8`. The router scans enabled models cheap → expensive
and picks the first whose classifier probability is `>= confidence`; if none
qualify, it uses `default`. Higher `confidence` = more conservative (stays on
`default` more often). See [routing](#routing).

### `server`

```yaml
server:
  host: 127.0.0.1     # default; loopback only
  port: 8788          # default
  auth_key: null      # required if host is not loopback
```

Binding to a non-loopback host (e.g. `0.0.0.0`) without `auth_key` fails validation.
When `auth_key` is set, `er code` passes it to Claude Code automatically.

### `telemetry`

```yaml
telemetry:
  enabled: true
  log_path: ~/.emissary-router/events.jsonl   # optional override
  retention_days: 30                           # null = keep forever
  max_events: 50000                            # null = no cap
```

See [telemetry](telemetry.md).

### `router` (advanced)

Only needed to point at a staging/private classifier deployment:

```yaml
router:
  url: https://api.withemissary.com/v1/classification   # default
  router_model: emissary-model-router-shared            # default
  timeout_seconds: 30
```

## Routing

The single policy is `deviate_if_confident`: stay on `default`, and drop to the
cheapest enabled model the classifier is confident about.

- All enabled models confident below `confidence` → `default`.
- A cheaper enabled model is confident → that model (a "deviation").

## API keys

Keys live in environment variables or `~/.emissary-router/.env` — never in
`config.yaml`. Keeping keys out of the config means forking the repo can't leak
them.

```dotenv
EMISSARY_ROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

Only the keys for providers you actually use are required (e.g. disabling all
OpenRouter-backed models drops the need for `OPENROUTER_API_KEY`). Check with
`er validate-config`.

Precedence: variables already exported in your shell win; then
`~/.emissary-router/.env`; then a `.env` in the current directory. The loader never
overrides an already-set variable.

## Alternate locations

```bash
export EMISSARY_ROUTER_HOME=/path/to/dir      # parent for config.yaml, .env, logs, pid
export EMISSARY_ROUTER_CONFIG=/path/to/config.yaml
er code --config ./config.yaml -- [claude args]
```
