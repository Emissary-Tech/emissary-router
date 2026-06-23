# Commands

The command is `er` (`emissary-router` is a compatibility alias). All commands
accept `--config PATH` to use a non-default config file.

## Everyday use

```bash
er code -- [claude args]
```

Launches Claude Code through the router. Starts the local gateway automatically if
it is not already running, then execs `claude`. The gateway keeps running after
Claude Code exits so later sessions reuse it — stop it with `er stop`.

`er code` injects this environment into Claude Code:

```text
ANTHROPIC_BASE_URL=http://127.0.0.1:<port>
ENABLE_TOOL_SEARCH=true
CLAUDE_CODE_ATTRIBUTION_HEADER=0
ANTHROPIC_API_KEY=<server.auth_key>   # only when auth_key is set
```

Flags:

- `--claude-command NAME` — binary to exec (default `claude`, or `$CLAUDE_COMMAND`).
- `--dry-run` — print the launch command and injected env, then exit without exec.
- everything after `--` is passed straight to Claude Code.

## Setup / inspection

```bash
er init              # create config and set/update API keys (re-run to change keys)
er validate-config   # check config + that required provider keys are present
er models            # list the catalog: enabled, chosen + supported providers, default
er config-path       # print the resolved config path
```

`er init` prompts for each required key, skipping any already exported in your
environment, and writes the rest to `~/.emissary-router/.env` (chmod 600). Re-running
it shows the current value masked — press enter to keep, or type a new one to replace.
With `--no-prompt` (or no terminal) it just writes a template `.env`. After changing a
key, run `er restart` so the running gateway reloads it.

`er validate-config` exits non-zero and lists `missing_env` if a required key is
absent, so it is safe to use in setup scripts. It does not make any network calls.

## Gateway lifecycle

```bash
er start     # start the background gateway
er status    # show health, pid, and url
er restart   # stop then start
er stop      # stop the background gateway
```

You normally do not need `start` — `er code` handles it. Use these when you want to
manage the process manually.

## Debugging

```bash
er debug     # run the gateway in the foreground with logs on the terminal
```

See [troubleshooting](troubleshooting.md).

## Environment overrides

```bash
export EMISSARY_ROUTER_HOME=/path/to/dir       # parent of config.json, .env, logs, pid
export EMISSARY_ROUTER_CONFIG=/path/to/config.json
export CLAUDE_COMMAND=/path/to/claude
```
