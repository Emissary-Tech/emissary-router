# Demo

A split-screen comparison page for showing routing live: you type a query, and the same
query is answered two ways side by side — straight Claude Sonnet on the left, the routed
system on the right (served by whichever model the router picks), with each side's model,
cost, and latency, plus a running savings tally.

It is single-turn and tool-free, so the classifier sees the same shape it was trained on
and routes typed questions out of the box. Each comparison makes two real model calls, so
the page sits behind the same auth as the [dashboard](dashboard.md). Demo calls are not
written to [telemetry](telemetry.md).

## Running it

```bash
er demo
```

This starts the gateway if needed and opens the demo in your browser:

```text
http://127.0.0.1:8788/demo
```

`er demo` requires `demo.enabled` (on by default). If an `auth_key` is set, the URL
includes it as `?key=...`, the same as the dashboard.

## Controls

Both sides always use the **same** settings so the comparison is fair; each provider
converts them for its own model internally.

- **Reasoning** — `off` / `low` / `medium` / `high`. Sent as the Sonnet-native
  `output_config.effort`; the gateway maps it per provider (see
  [thinking](thinking.md)). Default `off`.
- **Max tokens** — `32k` / `64k`, default `32k`. This is only a ceiling to avoid
  truncation; cost is computed from the *actual* output, so a higher ceiling does not
  raise cost on its own.
- **Presets** — example queries you can click; mix in a genuinely hard one to show the
  router escalate to Sonnet (both sides then match, with no savings — which is the point).

## What you see

Each side shows the served model, the answer, its cost, and latency. The routed side adds
a badge for the chosen model and either a savings percentage (it deviated to a cheaper
model) or an "escalated → sonnet" note (the classifier was not confident enough in a
cheaper model, so it stayed on the default). A strip at the bottom accumulates the session
total: all-Sonnet vs routed, and the percentage saved.

## Config

```json
"demo": { "enabled": true, "streaming": false }
```

- `enabled` — mount the demo routes (`/demo`, `/api/demo/compare`). Default `true`.
- `streaming` — default mode for the page's streaming toggle. Non-streaming returns both
  answers when ready; streaming renders both token by token.

## Notes

- The routed side uses the configured routing policy (default `deviate_if_confident`).
  For a single-turn demo this just picks the cheapest model the classifier is confident
  about, escalating to the default on hard queries.
- Because each comparison calls two models, keep `er demo` behind `auth_key` if the
  gateway is reachable beyond your machine.
