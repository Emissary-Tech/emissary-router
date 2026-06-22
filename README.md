# Emissary Router

Emissary Router is a local Claude Code gateway that routes each request to a
supported model, keeps provider-specific caching intact, and records lightweight
cost/cache telemetry.

## Install

```bash
pip install emissary-router
```

With uv: `uv pip install emissary-router`. Before the package is published, install
from git: `pip install "git+<repo-url>"`.

This installs the `er` command. If a global install is blocked (an
"externally-managed environment"), install inside a virtualenv — or use an isolated
installer like `pipx install emissary-router`.

Then set up config and API keys:

```bash
er init
```

The installer performs an editable install and creates default files in one place:

```text
~/.emissary-router/
  config.yaml
  pricing.yaml
  events.sqlite3
  server.log
  server.pid
```

Then edit:

```bash
export EMISSARY_ROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...
```

Then run Claude Code through the router:

```bash
er code -- [claude args]
```

`er code` starts the local gateway automatically if it is not already running. The
gateway keeps running after Claude Code exits; stop it with:

```bash
er stop
```

Installing from a clone instead? Run `bash install.sh` (editable install), then
`er init`.

## Supported Models

Toggle models in `~/.emissary-router/config.json`:

```json
{
  "models": {
    "claude-sonnet-4.6": { "enabled": true, "provider": "anthropic" },
    "claude-haiku-4.5": { "enabled": true, "provider": "anthropic" },
    "gemini-3.1-flash-lite": { "enabled": true, "provider": "openrouter" }
  },
  "default": "claude-sonnet-4.6",
  "confidence": 0.8
}
```

Built-in models:

- `claude-sonnet-4.6` — Anthropic or OpenRouter
- `claude-haiku-4.5` — Anthropic or OpenRouter
- `gemini-3.1-flash-lite` — OpenRouter

Set `enabled: false` to drop a model, and `provider` to choose how it's served.
Users cannot add arbitrary upstream models in V1; model id and pricing are owned by
the built-in catalog. See [Configuration](docs/configuration.md) for details.

## Docs

Use `start`, `status`, `restart`, and `stop` when you want to manage the local
gateway process manually.

Override config paths with flags:

```bash
emissary-router start --config ./config.yaml --pricing ./pricing.yaml
emissary-router code --config ./config.yaml --pricing ./pricing.yaml -- [claude args]
```

Or environment variables:

```bash
export EMISSARY_ROUTER_CONFIG=./config.yaml
export EMISSARY_ROUTER_PRICING=./pricing.yaml
```

To keep config, pricing, logs, and pid files under a different parent directory:

```bash
export EMISSARY_ROUTER_HOME=/path/to/.emissary-router
```

## Config

The config is model-first: pick model names, then choose which provider serves each
model.

Most users do not need to set `router.url`; it defaults to:

```text
https://api.withemissary.com/v1/classification
```

Example:

```yaml
router:
  api_key: ${EMISSARY_ROUTER_API_KEY}
  router_model: emissary-model-router-shared
  default: claude-sonnet-4.6
  enabled:
    - claude-sonnet-4.6
    - claude-haiku-4.5
    - gemini-3.1-flash-lite
  policy:
    name: cheap_first
    tau: 0.8
    candidates:
      - gemini-3.1-flash-lite
      - claude-haiku-4.5
      - claude-sonnet-4.6

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

  openrouter:
    api_key: ${OPENROUTER_API_KEY}

models:
  claude-sonnet-4.6:
    provider: anthropic
    model_id: claude-sonnet-4-6

  claude-haiku-4.5:
    provider: anthropic
    model_id: claude-haiku-4-5

  gemini-3.1-flash-lite:
    provider: openrouter
    model_id: google/gemini-3.1-flash-lite
```

The model key, such as `gemini-3.1-flash-lite`, is the name returned by the
Emissary router classifier. Provider type is inferred for the built-in provider
names `anthropic`, `google`, and `openrouter`.

`validate-config` reports unresolved environment variables as warnings so example
files can be checked without real secrets. `start` and `code` are strict and fail
before provider calls if secrets are missing.

## API Keys

The simplest setup is to put keys directly in `~/.emissary-router/config.yaml`:

```yaml
router:
  api_key: your-emissary-router-api-key
  router_model: emissary-model-router-shared

providers:
  anthropic:
    api_key: your-anthropic-api-key

  openrouter:
    api_key: your-openrouter-api-key
```

You can also keep keys outside the file by using environment-variable placeholders:

```yaml
router:
  api_key: ${EMISSARY_ROUTER_API_KEY}

providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

If a config value still contains `${...}`, `emissary-router start` and
`emissary-router code` require that environment variable to be set.
`emissary-router validate-config` only reports unresolved variables as warnings.

## Advanced Router Settings

Override the Emissary classification endpoint only for staging, development, or a
private deployment:

```yaml
router:
  url: https://staging-api.withemissary.com/v1/classification
  api_key: ${EMISSARY_ROUTER_API_KEY}
  router_model: emissary-model-router-shared
  timeout_seconds: 30
```

## Advanced Provider Settings

Most users only need `api_key`. Advanced fields are available when you want to point
a built-in provider at a proxy or gateway:

```yaml
providers:
  google:
    api_key: ${GOOGLE_API_KEY}
    base_url: https://your-google-proxy.example.com
```

`base_url` is only required when you intentionally want to override the default
provider endpoint.

If you use a custom provider name, add `type` so Emissary Router knows which adapter
to use:

```yaml
providers:
  google_proxy:
    type: google
    api_key: ${GOOGLE_API_KEY}
    base_url: https://your-google-proxy.example.com

models:
  gemini-3.1-flash-lite:
    provider: google_proxy
    model_id: gemini-3.1-flash-lite
```

Built-in provider names infer their type automatically: `anthropic`, `google`, and
`openrouter`.

The only advanced cache switch in V1 is for Anthropic prefix stability:

```yaml
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    cache:
      strip_dynamic_attribution: true
```

Leave this enabled unless you are debugging raw Claude Code requests. It removes a
dynamic Claude Code attribution line before forwarding to Anthropic so stable prompt
prefixes remain cacheable. This is not a general cache control.

## Pricing

Pricing is keyed by model name. Keep this file up to date when providers change
prices.

```yaml
pricing:
  claude-sonnet-4.6:
    input: 3.00
    output: 15.00
    cache_read: 0.30
    cache_write_5m: 3.75
    cache_write_1h: 6.00

  gemini-3.1-flash-lite:
    input: 0.25
    output: 1.50
    cache_read: 0.025
    cache_write_5m: 0.25
```

Units are USD per 1M tokens.

## Running Claude Code

After editing `~/.emissary-router/config.yaml`, run:

```bash
emissary-router validate-config
emissary-router code -- [claude args]
```

`emissary-router code` launches Claude Code with the required local gateway
environment. If the local gateway is not already running, this command starts it in
the background first, then execs Claude Code. The gateway is not stopped
automatically when Claude Code exits; use `emissary-router stop` when you want to
shut it down. The injected environment includes:

```text
ANTHROPIC_BASE_URL=http://127.0.0.1:<port>
ENABLE_TOOL_SEARCH=true
```

## Debugging

For normal use, run Claude Code with `emissary-router code`. To start the gateway
without Claude Code:

```bash
emissary-router start
emissary-router status
emissary-router stop
```

To keep the gateway in the foreground and see server logs directly:

```bash
emissary-router debug
```

## Routing Policies

Implemented:

- `cheap_first`: scan `candidates` in order and choose the first model with
  `p(success) >= tau`; otherwise use `default`.
- `argmax`: choose the enabled candidate with the highest classifier probability.

## Providers

Implemented:

- `anthropic`: native Anthropic Messages pass-through.
- `google`: Google Gemini `generateContent`.
- `openrouter`: OpenAI-compatible Chat Completions.

For OpenRouter and Google, Emissary Router calls the provider non-streaming and
synthesizes Anthropic SSE when Claude Code requested streaming. This avoids fragile
streaming tool-call translation.

### Google Native Gemini 3

Use OpenRouter for Gemini 3 Claude Code workloads that involve tool calls.

Gemini 3 native function calling requires Google `thoughtSignature` values to be
returned exactly in follow-up tool-result turns. Claude Code speaks the Anthropic
Messages API and does not provide those Google signatures in its request history. A
router can only attach them if the same Google-native response created the tool call,
the router retained the signature, and the next tool-result request returns before
that state is lost.

Because model routing can switch providers between turns, native Google Gemini 3 is
not a safe general-purpose target for Claude Code tool loops in V1. Use one of:

- `openrouter` with `model_id: google/gemini-3.1-flash-lite`
- `google` native Gemini 3 only for non-tool or explicitly experimental sessions

## Thinking

Emissary Router preserves Claude Code thinking settings where each provider supports
them:

- Anthropic: normalizes Claude Code thinking fields to the served Claude model. For
  example, Haiku receives `adaptive` thinking as `enabled` plus a token budget because
  it does not accept the adaptive thinking form.
- Google: maps explicit effort, including Claude Code `output_config.effort`, to
  Gemini `thinkingConfig.thinkingLevel`. Budget-only requests use `thinkingBudget`,
  disabled thinking maps to `thinkingBudget: 0`, and adaptive requests without an
  effort or budget use Gemini dynamic thinking with `thinkingBudget: -1`.
- OpenRouter: maps effort settings to `reasoning.effort` for effort-capable models;
  Claude Sonnet `max` maps to OpenRouter `xhigh`. Budget-only requests map to
  `reasoning.max_tokens`. Claude Haiku is treated as budget-only, so effort requests
  map to `reasoning.max_tokens` using `max_tokens - 1`, matching native Anthropic
  behavior.

Provider-specific limits still apply. For example, Gemini 3 tool loops require
thought signatures on native Google responses, so OpenRouter remains the recommended
Gemini path for Claude Code in V1.

## Caching

Users do not need to configure caching.

Emissary Router handles provider cache compatibility internally:

- Anthropic: passes Claude Code cache controls through and defensively removes dynamic
  Claude Code attribution text that can break prefix caching.
- Google: relies on Gemini implicit caching and records
  `usageMetadata.cachedContentTokenCount`.
- OpenRouter: enables automatic prompt caching for Claude models where supported, and
  records `prompt_tokens_details.cached_tokens` and `cache_write_tokens`.

There is intentionally no single portable cache setting. The providers expose
different cache mechanisms:

- Anthropic supports prompt caching through `cache_control`; Claude Code already
  sends cache controls, so Emissary Router preserves them.
- OpenRouter supports top-level automatic caching for Claude models and reports cache
  read/write metrics in the usage object. Other models may use provider-native
  implicit caching behind OpenRouter.
- Google has Gemini implicit caching by default for stable prefixes. Gemini also has
  explicit `CachedContent` resources, but Emissary Router does not manage those in V1
  because they require create/update/delete lifecycle handling and storage billing.

Telemetry rows include normalized token usage, cache read/write tokens, routing
decision, provider, model id, session/turn grouping, and estimated cost. Rows are
stored in a local SQLite database (lean columns only — no request/response bodies or
prompt text unless you explicitly opt in). Old rows are pruned by `retention_days` and
`max_events`.

Default telemetry database:

```text
~/.emissary-router/events.sqlite3
```

## Dashboard

A local dashboard reads the telemetry database and shows where requests went and how
much routing saved versus sending everything to one model.

There is no separate command to launch it. `emissary-router start` and
`emissary-router code` print the dashboard URL, and open it in your browser the first
time the gateway starts (a cold start). Repeated runs that reuse an already-running
gateway just print the URL, so you don't get a new browser tab every time.

```text
Dashboard: http://127.0.0.1:8788/dashboard
```

To get back in after closing the tab, just visit that URL again. If you lost it,
`emissary-router status` prints it too. Pass `--no-open` to `start`/`code` to suppress
the automatic browser launch. The dashboard has three tabs:

- **Savings**: total spend, the all-`baseline_model` estimate, estimated savings, and
  per-model call distribution.
- **Requests**: every routed call (served model, provider, cost, cache hit, tokens),
  with per-row delete.
- **Per-input**: calls grouped by one user input (a turn) — how many main and
  background calls went to which models — with per-session delete.

### Storage

The dashboard is backed by the local SQLite database at
`~/.emissary-router/events.sqlite3` (the same telemetry store the gateway writes to).
Because it is a file, records persist across restarts — history is available the next
time you start the gateway. The store keeps only lean, queryable columns (no request or
response bodies, and no prompt text unless you opt in via `include_classifier_input` /
`include_raw_event`). Old rows are pruned by `retention_days` and `max_events`, so the
file stays bounded. Savings is an estimate: it applies the baseline model's prices to
each call's actual token counts.

### Dashboard auth

When `server.auth_key` is set, the dashboard, its JSON API, and the delete endpoints all
require that key (it is not enough to protect only `/v1/messages`). `emissary-router
dashboard` opens the URL with the key attached (`/dashboard?key=...`), and the page sends
it on every request.

For a remote gateway (`host: 0.0.0.0`), note that this puts the key in the URL. Prefer
keeping the gateway on loopback and reaching the dashboard through an SSH tunnel, e.g.:

```bash
ssh -N -L 8788:127.0.0.1:8788 user@remote-host
# then open http://127.0.0.1:8788/dashboard locally
```

## Server Settings

By default, Emissary Router listens on `127.0.0.1:8788`.

To change the local port:

```yaml
server:
  port: 8790
```

If you intentionally expose the gateway outside loopback, set `server.auth_key`:

```yaml
server:
  host: 0.0.0.0
  port: 8788
  auth_key: ${EMISSARY_ROUTER_AUTH_KEY}
```

Binding to `0.0.0.0` without `server.auth_key` fails validation. When `auth_key` is
set, `emissary-router code` automatically passes that key to Claude Code for gateway
auth, and the dashboard requires it too (see "Dashboard auth" above).
