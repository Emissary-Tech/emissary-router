# Emissary Router

Emissary Router is a local Claude Code gateway that routes each request to a
supported model, keeps provider-specific caching intact, and records lightweight
cost/cache telemetry.

## Install

```bash
pip install emissary-router
```

With uv: `uv pip install emissary-router`.

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

## Dashboard

`er code` and `er start` open a local dashboard in your browser showing cost savings,
recent requests, and per-session usage — plus a Settings tab to toggle models live:

```text
http://127.0.0.1:8788/dashboard
```

It stays up while the gateway runs, so reopen the URL any time. See
[Dashboard](docs/dashboard.md).

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

- [Configuration](docs/configuration.md)
- [Commands](docs/commands.md)
- [Dashboard](docs/dashboard.md)
- [Providers and Caching](docs/providers-caching.md)
- [Pricing](docs/pricing.md)
- [Thinking](docs/thinking.md)
- [Telemetry](docs/telemetry.md)
- [Troubleshooting](docs/troubleshooting.md)
