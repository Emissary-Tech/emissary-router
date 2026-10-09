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

`er init` creates `~/.emissary-router/config.json` and prompts for your keys (it skips
any already in your environment), writing them to `~/.emissary-router/.env`. At any
prompt press Enter to skip that key and set it later — or, when re-running, to keep the
current value. Run it again any time to change a key. You can also just export the keys
instead:

```bash
export EMISSARY_ROUTER_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...
```

Don't have an `EMISSARY_ROUTER_API_KEY` yet? Sign up at
[withemissary.com](https://withemissary.com) and create one (Dashboard > Settings > Credentials). See
[API keys](docs/configuration.md#api-keys) for where each provider key comes from — you
only need the ones your enabled models use.

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
    "deepseek-v4-flash": { "enabled": true, "provider": "openrouter" },
    "gpt-5.6-luna": { "enabled": true, "provider": "openai" },
    "gemini-3.1-flash-lite": { "enabled": true, "provider": "openrouter" },
    "glm-5.2": { "enabled": true, "provider": "openrouter" },
    "kimi-k2.7-code": { "enabled": true, "provider": "openrouter" },
    "claude-haiku-4.5": { "enabled": true, "provider": "anthropic" },
    "claude-sonnet-5": { "enabled": true, "provider": "anthropic" },
    "claude-opus-5": { "enabled": true, "provider": "anthropic" },
    "kimi-k3": { "enabled": false, "provider": "openrouter" }
  },
  "default": "claude-sonnet-5",
  "confidence": 0.8
}
```

Built-in models — the same set, upstream ids and prices as the Emissary platform
gateway's catalog (see [pricing](docs/pricing.md) for the table):

- Anthropic (or OpenRouter): `claude-opus-5`, `claude-opus-5.5`, `claude-sonnet-5`,
  `claude-sonnet-5.5`, `claude-haiku-4.5` — adaptive thinking is the claude-5 default;
  the router strips client `temperature`/`top_p`, which the claude-5 series rejects.
- OpenAI (native Responses API, or OpenRouter): `gpt-5.6-luna`, `gpt-5.6-terra`,
  `gpt-5.6-sol`, `gpt-6-luna`, `gpt-6-sol`, `gpt-6.1-sol`.
- OpenRouter: `deepseek-v4-flash` (and its `-0731` snapshot alias),
  `deepseek-v4.1-flash`, `qwen3.8-flash`, `qwen3.8-omni-flash`, `glm-5.3`,
  `glm-5.3-flash`, `kimi-k2.7-code` (always reasons), `kimi-k3`.
- `gemini-3.1-flash-lite` — OpenRouter, or native Google (`provider: "google"`).
- `glm-5.2` — OpenRouter, or native Z.ai (`provider: "zai"`, e.g. a GLM Coding Plan key).

Set `enabled: false` to drop a model, and `provider` to choose how it's served.
Users cannot add arbitrary upstream models in V1; model id and pricing are owned by
the built-in catalog.

Routing is confidence-gated and **cache-aware by default**: candidates the classifier
is confident about are compared by cache-adjusted cost, so the router only switches
models when it is genuinely cheaper after accounting for the prompt cache it would
give up (naive per-request switching can cost *more* than not routing at all). Where a
provider has no reliable cache signal the estimates simply carry no discount and the
comparison is a flat per-request price comparison.
See [Configuration](docs/configuration.md) for details.

## Docs

- [Configuration](docs/configuration.md)
- [Commands](docs/commands.md)
- [Dashboard](docs/dashboard.md)
- [Providers and Caching](docs/providers-caching.md)
- [Pricing](docs/pricing.md)
- [Thinking](docs/thinking.md)
- [Telemetry](docs/telemetry.md)
- [Troubleshooting](docs/troubleshooting.md)
