# Dashboard

The router serves a local web dashboard showing what was routed, what it cost, and
how much you saved versus running everything on your default model. It reads the same
local telemetry database the gateway writes (see [telemetry](telemetry.md)) — nothing
leaves your machine.

## Opening it

`er code` and `er start` print the dashboard URL and open it in your browser
automatically whenever the gateway is healthy and telemetry is enabled:

```text
http://127.0.0.1:8788/dashboard
```

It stays available as long as the gateway runs — close the tab and reopen the URL any
time. `er status` also prints the URL. If telemetry is disabled
(`telemetry.enabled: false`), there is no dashboard.

If `server.auth_key` is set, the dashboard requires that key. The auto-opened URL
includes it as `?key=<auth_key>`; if you open the URL by hand, append the same query —
or send the key as an `x-api-key` or `Authorization: Bearer` header.

## Tabs

- **Savings** — total spend, the baseline cost (your `default` model priced against the
  same tokens), and the estimated savings (absolute and %), plus a per-model breakdown
  of calls and cost.
- **Requests** — one row per call: time, served model, status, requested model,
  provider, kind, cost, cache, and prompt/output tokens. "Prompt" is the full input
  **including cached tokens**, so it matches the provider's own reported input; the
  Cached column shows how much was a cache read vs a cache write. Delete a row with its
  button.
- **Sessions** — calls grouped by Claude Code session (one `er code` run ≈ one
  session): last activity, short session id, call counts (main / background), models
  used, and total cost. Delete a whole session with its button.
- **Settings** — toggle models on/off, choose a provider per model, set the default
  model and confidence, and pick the routing [policy](configuration.md#policy). Saving
  writes your config and the running gateway reloads it live — no restart needed.
  (Switching policy rebuilds the router, which resets the in-memory cache ledger that
  `cache_aware` uses; it re-warms over the next few turns.)

Data does not auto-refresh; use the **↻ Refresh** button to pull the latest.

## How savings are computed

Savings is an estimate: for every recorded call the baseline (`default` model) catalog
price is applied to that call's actual token counts, then compared to what the served
model actually cost. It is not a separate billing source — see [pricing](pricing.md)
and [telemetry](telemetry.md).

## HTTP API

The dashboard is backed by a small JSON API under the same auth as the page:

- `GET /dashboard` — the page
- `GET /api/summary` — savings totals and per-model aggregates
- `GET /api/events?limit=&session=` — recent rows
- `GET /api/sessions?limit=` — per-session rollup
- `GET /api/config`, `PUT /api/config` — read / update model toggles, default,
  confidence, and policy
- `DELETE /api/events/{id}` — delete one row
- `DELETE /api/sessions/{id}` — delete a whole session
