# Demo

A split-screen page for showing routing live. You chat, and the same conversation runs
two ways side by side — straight Claude Sonnet on the left, the routed system on the
right (each turn served by whichever model the router picks). A running tally shows
cumulative **cost** and **latency**, with per-turn numbers on every message.

It is meant to be driven in front of an audience: type a question, both sides answer,
and the difference (same/comparable answer, lower cost, the chosen model, the latency)
is visible at a glance.

Each turn makes two real model calls, so the page sits behind the same auth as the
[dashboard](dashboard.md). Nothing is stored server-side — the browser holds the
in-progress chat, and **New chat** (or a refresh) starts a fresh one, so anyone can walk
up and begin. Demo calls are not written to [telemetry](telemetry.md).

## Running it

**1. First-time setup** — create the config and enter your keys:

```bash
er init
```

`er init` writes `~/.emissary-router/config.json` and prompts for your API keys, skipping
any already in your environment: `EMISSARY_ROUTER_API_KEY` plus at least one provider key
(`ANTHROPIC_API_KEY` and/or `OPENROUTER_API_KEY`). For web search add `TAVILY_API_KEY`, or
paste it into the page later (see Controls). Re-run `er init` any time to change a key.

**2. Start the demo:**

```bash
er demo
```

This starts the gateway **in the background** (if it isn't already running) and opens the
demo:

```text
http://127.0.0.1:8788/demo
```

It requires `demo.enabled` (on by default). If an `auth_key` is set, the URL carries it as
`?key=...`, the same as the dashboard.

**3. Stop the gateway** when you're done — it keeps running after you close the browser:

```bash
er stop
```

## The screen

- **Two transcripts** — left = Sonnet, right = routed. The same user turns appear in
  both; the answers diverge. Answers render Markdown (tables, lists, code) and LaTeX math.
- **Per message** — cost and latency. The routed side also shows the **chosen model** and
  splits latency into **router** (classifier) + **model** time, e.g.
  `haiku · $0.0007 · router 40ms + model 900ms = 940ms`, so the router's own overhead is
  visible. A `🔎 N` mark shows how many web searches that turn ran.
- **Header** — cumulative **running cost** (all-Sonnet vs routed, percent saved) and
  **total latency** across the chat.
- **Escalation** — on a hard turn the routed side may show `escalated → sonnet`: the
  classifier wasn't confident in a cheaper model, so it stayed on the default. Both sides
  then match with no savings — which is the point: it only deviates when it pays off.

## Controls

Both sides always use the **same** settings, so the comparison is fair; each provider
converts them for its own model internally.

- **Routing** — `cache_aware` / `deviate` (`deviate_if_confident`). Per-turn override of
  the routing policy, so you can show the difference live. See [policy](configuration.md#policy).
- **Reasoning** — `off` / `low` / `medium` / `high`. Sent as `output_config.effort`; the
  gateway maps it per provider (see [thinking](thinking.md)). For a model that doesn't take
  an effort param (Haiku), the demo gives it a thinking budget of half `max_tokens` instead.
- **Max tokens** — `32k` / `64k`. Only a ceiling to avoid truncation; cost is from the
  *actual* output, so a higher ceiling does not raise cost on its own.
- **Stream** — on by default: each side streams its answer token-by-token, so whichever
  finishes first is read live (and you see the answer build). It uses a per-provider
  passthrough — Anthropic streams natively, and OpenRouter is read in its own OpenAI SSE
  format directly rather than buffered. Off falls back to two whole-answer requests, still
  rendering the faster side first.
- **Web search** — give both sides a `web_search` tool and run a tool loop: the model can
  search before answering (a `🔎 searching` indicator shows), looping until it answers.
  Uses [Tavily](https://tavily.com) when a key is set, a mock result otherwise so the demo
  still runs offline.
- **Web search key (Tavily)** — paste a Tavily key to enable real search without an env
  var. It's written to `~/.emissary-router/.env` (never the config) and applied
  immediately; the page only ever shows a masked hint.
- **Presets** — example queries across domains; click to start. A `New chat` button resets
  the conversation (and the cache scope).
- **⚙ Settings** (bottom-right) — jumps to the dashboard Settings tab to change models,
  providers, default, confidence, and policy. Saving hot-reloads with no restart.

## Presenting it (suggested flow)

1. **Easy question** (e.g. "What's the difference between TCP and UDP?") → routed goes to a
   cheap model. Point out: comparable answer, the model badge, the cost a fraction of
   Sonnet's. The header's "saved %" starts climbing.
2. **Hard question** (e.g. the logic puzzle, or a graduate physics question) → routed
   escalates to Sonnet. Point out: it didn't cheap out — when a small model isn't trusted,
   it stays on Sonnet. Both answers match.
3. **Across the chat** — keep going; the header shows cumulative cost + latency. The story:
   "right model per turn — cheap when it's enough, Sonnet when it isn't."
4. **Toggle Routing** to compare `cache_aware` vs `deviate` (start a New chat each time for
   a clean comparison). On longer chats, `cache_aware` keeps cost down by not switching in a
   way that throws away the prompt cache.
5. **Web search** on → ask something current ("latest ..."): both sides search (🔎), and the
   routed side answers from results at a lower cost.
6. **⚙ Settings** → flip a model's provider or the policy and save; it applies live.

Mix preset domains so the routing varies — some questions go cheap, some escalate. Use a
genuinely hard one (logic / graduate science) when you want the escalation to be reliable;
borderline questions can route either way run to run.

## How it works (under the hood)

- Each turn, the request is shown to the Emissary classifier, which scores how likely each
  model is to handle it. The policy then picks: `deviate_if_confident` takes the cheapest
  model that clears the confidence bar (else the default); `cache_aware` additionally prices
  in the prompt cache so it doesn't switch in a way that costs more than it saves.
- Both sides get the same small system prompt: today's real date (so the present isn't
  treated as the future), an anti-fabrication rule, and — when search is on — an instruction
  to trust search results over stale prior knowledge. It does not impersonate any product
  and does not affect routing.
- Cost is each model's catalog price applied to actual token usage (input, cache read/write,
  output), the same price table the gateway uses for billing — so the numbers are real.

## Config

```json
"demo": { "enabled": true }
```

- `enabled` — mount the demo routes (`/demo`, `/api/demo/chat`, `/api/demo/stream`).
  This block is optional and on by default; add it only to turn the demo off
  (`"demo": { "enabled": false }`).

Because each turn calls two models, keep `er demo` behind `auth_key` if the gateway is
reachable beyond your machine.
