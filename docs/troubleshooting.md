# Troubleshooting

Server logs are at `~/.emissary-router/server.log`. For live logs, run the gateway in
the foreground with `er debug`.

## `er validate-config` reports missing env

A required provider key isn't set. Add keys to `~/.emissary-router/.env` or export
them in your shell:

```dotenv
EMISSARY_ROUTER_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

Only the providers used by your enabled models are required. `er validate-config`
lists exactly which keys are missing under `missing_env`.

## "unknown models" or "default model must be enabled" on load

You can only use models in the built-in catalog, and `default` must be enabled. Run
`er models` to see valid names and current toggles, then fix `config.json`.

## Changed a key but nothing changed

The background gateway loads keys once at startup, so a `.env` edit (or `er init`)
doesn't reach an already-running gateway. Run `er restart` after changing a key.

## Keys in `.env` aren't picked up

The loader does not override variables already exported in your shell. If you also
have `ANTHROPIC_API_KEY` exported elsewhere, that value wins. Precedence: shell env →
`~/.emissary-router/.env` → `./.env`. Unset the stale shell variable, or put the
right value there.

## Gateway didn't become healthy

`er code`/`er start` waits for the gateway and then errors with a pointer to the log.
Check `~/.emissary-router/server.log`. Common causes: a missing key (run
`er validate-config`) or the port already in use.

```bash
er status        # is something already running?
er restart
```

## Port already in use

```json
"server": { "port": 8790 }
```

Then `er restart`.

## `could not find Claude Code command 'claude'`

`er code` needs the `claude` CLI on your `PATH`. Install Claude Code, or point at it:

```bash
er code --claude-command /path/to/claude -- [claude args]
```

## Gateway is still running after Claude Code exits

That's intentional — it's reused by later sessions. Stop it explicitly:

```bash
er stop
```

## Exposing outside loopback

Binding to a non-loopback host requires an auth key (validation enforces it):

```json
"server": { "host": "0.0.0.0", "auth_key": "choose-a-local-router-key" }
```

`er code` then passes that key to Claude Code automatically. Prefer an SSH tunnel
over exposing the gateway directly.
