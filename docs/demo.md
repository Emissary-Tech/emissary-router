# Demo

A split-screen comparison page for showing routing live: you chat, and the same
conversation runs two ways side by side — straight Claude Sonnet on the left, the routed
system on the right (each turn served by whichever model the router picks). Each side
keeps its own history, and a running tally shows cumulative cost and latency, with
per-turn cost/latency on every message.

Each turn makes two real model calls, so the page sits behind the same auth as the
[dashboard](dashboard.md). No conversation is stored server-side — the browser holds the
in-progress chat, and **New chat** (or a refresh) starts a fresh one, so anyone can walk
up and begin. Demo calls are not written to [telemetry](telemetry.md).

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

## What you see

- Two chat transcripts, left = Sonnet, right = routed. The same user turns appear in both;
  the answers diverge.
- Each assistant message shows its cost and latency. The routed side also shows the chosen
  model and splits latency into **router** (classifier) time and **model** time —
  `router 40ms + model 1100ms = 1140ms` — so the router's own overhead is visible.
- The header shows the cumulative **running cost** (all-Sonnet vs routed, percent saved)
  and **total latency** across the chat.
- On a hard turn the routed side may **escalate to Sonnet** (the classifier was not
  confident in a cheaper model); both sides then match with no savings — which is the
  point: it only deviates when it pays off.

## Controls

Both sides always use the **same** settings so the comparison is fair; each provider
converts them for its own model internally.

- **Reasoning** — `off` / `low` / `medium` / `high`. Sent as the Sonnet-native
  `output_config.effort`; the gateway maps it per provider (see [thinking](thinking.md)).
- **Max tokens** — `32k` / `64k`. Only a ceiling to avoid truncation; cost is from the
  *actual* output, so a higher ceiling does not raise cost on its own.
- **Stream** — render both answers token by token (default from `demo.streaming`).
- **Web search** — give both sides a `web_search` tool and run a tool loop: the model can
  search before answering (a `🔎 searching` indicator shows mid-stream), looping until it
  answers. Search uses [Tavily](https://tavily.com) when `TAVILY_API_KEY` is set, and a
  mock result otherwise so the demo still runs offline. Enabling search forces streaming.
- **Presets** — example queries you can click to start; mix in a genuinely hard one to
  show the router escalate to Sonnet.

## Config

```json
"demo": { "enabled": true, "streaming": false }
```

- `enabled` — mount the demo routes (`/demo`, `/api/demo/chat`). Default `true`.
- `streaming` — default mode for the page's streaming toggle (token-by-token rendering).

## Notes

- The routed side uses the configured routing policy. With `cache_aware`, the router also
  accounts for prompt-cache effects across the multi-turn chat (see
  [configuration](configuration.md#policy)).
- Because each turn calls two models, keep `er demo` behind `auth_key` if the gateway is
  reachable beyond your machine.
