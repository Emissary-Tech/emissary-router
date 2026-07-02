# Telemetry

Telemetry is enabled by default and written to a local SQLite database:

```text
~/.emissary-router/events.sqlite3
```

It records what was routed where and what it cost — one row per call. Full
request/response bodies and prompt text are never stored, only the metadata below. The
[dashboard](dashboard.md) reads from this database.

## Row columns

- `id` — unique id for the call
- `ts`, `duration_ms` — request time (unix seconds) and wall-clock duration
- `session_id` — the Claude Code session the call belongs to (from the
  `X-Claude-Code-Session-Id` header)
- `call_kind` — `main` (an interactive turn: tools / thinking / effort present) or
  `background` (e.g. title and summary calls)
- `requested_model` — the model Claude Code asked for
- `served_model` / `provider` / `model_id` — what it was actually routed to
- `route_reason` — why the model was chosen. Gateway requests record one of the
  `cache_aware:*` reasons — `no_confident_candidate` (no cheaper model cleared
  `confidence`), `warm_default_cheaper` (the default's warm cache beat a confident
  candidate), `default_not_beaten` (no cache in play; the default was not more
  expensive for this request), `candidate_cheaper` (deviated). `default` and
  `deviate_if_confident:p>=0.8` come from the price-ordered fallback path (no cost
  features/ledger — not the normal gateway flow). `default_unsuitable:cheapest_alternative` means the default couldn't
  serve this call — always-on reasoning on a background call, or the request exceeds
  the default's context window — so the cheapest usable model served. Failures record
  `fallback: router_issue` (classifier unreachable) or `error` (upstream call failed)
- `input_tokens` / `output_tokens` — uncached input and output token counts
- `cache_read_tokens` / `cache_creation_tokens` — cache-read (hit) and cache-write tokens
- `cost_usd` — estimated cost from the catalog price of the served model
- `http_status` — upstream HTTP status (set on failures so errors stay visible)

`input_tokens` is the **uncached** input only. On a cache hit it is tiny (often 1)
while `cache_read_tokens` holds the bulk of the prompt — so the dashboard's "Prompt"
column sums `input + cache_read + cache_creation` to match the provider's reported
input. See [providers and caching](providers-caching.md).

## Bounding growth

`retention_days` and `max_events` keep the database bounded; it is pruned periodically
as new rows are written.

```json
"telemetry": {
  "enabled": true,
  "db_path": "~/.emissary-router/events.sqlite3",
  "retention_days": 30,
  "max_events": 50000
}
```

`retention_days` drops rows older than N days (`null` = keep all); `max_events` keeps
the newest N (`null` = no cap); `db_path` overrides the database location.

Set `enabled: false` to turn telemetry off entirely — this also disables the
[dashboard](dashboard.md).

## Cost

`cost_usd` is an estimate based on the served model's catalog price applied to that
call's actual token usage. See [pricing](pricing.md).
