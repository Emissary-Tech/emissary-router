# Emissary Router

Emissary Router is a local Claude Code gateway that routes each request to a
supported model, keeps provider-specific caching intact, and records lightweight
cost/cache telemetry.

## Install

```bash
git clone <repo-url>
cd emissary-router
bash install.sh
```

Add keys to `~/.emissary-router/.env`:

```dotenv
EMISSARY_ROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

Then run Claude Code through the router:

```bash
er validate-config
er code -- [claude args]
```

`er code` starts the local gateway automatically if it is not already running. The
gateway keeps running after Claude Code exits; stop it with:

```bash
er stop
```

## Supported Models

Toggle models in `~/.emissary-router/config.yaml`:

```yaml
models:
  claude-sonnet-4.6: true
  claude-haiku-4.5: true
  gemini-3.1-flash-lite: true
default: claude-sonnet-4.6
confidence: 0.8
```

Built-in models:

- `claude-sonnet-4.6`
- `claude-haiku-4.5`
- `gemini-3.1-flash-lite` via OpenRouter

Users cannot add arbitrary upstream models in V1; provider, model id, and pricing
are owned by the built-in catalog.

## Docs

- [Configuration](docs/configuration.md)
- [Commands](docs/commands.md)
- [Providers and Caching](docs/providers-caching.md)
- [Pricing](docs/pricing.md)
- [Thinking](docs/thinking.md)
- [Telemetry](docs/telemetry.md)
- [Troubleshooting](docs/troubleshooting.md)
