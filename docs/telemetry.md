# Telemetry

Telemetry is enabled by default and written locally as JSON Lines (one JSON object
per line):

```text
~/.emissary-router/events.jsonl
```

It records what was routed where and what it cost. Full request/response bodies and
prompt text are never stored — only the metadata below.

## Row fields

- `ts`, `duration_ms` — request time and wall-clock duration
- `request_id` — unique id for the call
- `requested_model` — the model Claude Code asked for
- `served_model` / `provider` / `model_id` — what it was actually routed to
- `route_reason` — why the model was chosen, e.g. `cache_aware:default_after_cache`
  or `cache_aware:p>=0.8:cheaper_after_cache`
- `probabilities` — the classifier's per-model scores for the routing decision
- `estimated_costs` — per-candidate cache-adjusted cost estimates used by
  `cache_aware`
- `cache_prediction` — whether the selected model was predicted warm, and with what
  confidence
- `request_cost_features` — token estimates used for the cache-aware decision
- `classifier_input` — lightweight metadata about the classifier input (not the
  prompt text)
- `usage` — normalized token counts (input, output, cache read, cache creation)
- `cost_usd` — estimated cost from the catalog price of the served model
- `cache` — `cache_read_input_tokens`, `cache_creation_input_tokens`, `cache_hit`
- `provider_metadata` — small provider-specific extras (e.g. http status)

Example:

```json
{"ts": 1782100000.12, "duration_ms": 2310.4, "request_id": "…",
 "requested_model": "claude-sonnet-4-6", "served_model": "claude-haiku-4.5",
 "provider": "anthropic", "model_id": "claude-haiku-4-5",
 "route_reason": "cache_aware:p>=0.8:cheaper_after_cache",
 "cache_prediction": {"warm": true, "cached_tokens": 18400, "confidence": "predictable"},
 "usage": {"input_tokens": 21000, "output_tokens": 640, "cache_read_input_tokens": 18400},
 "cost_usd": 0.0061, "cache": {"cache_hit": true}}
```

## Bounding log growth

`retention_days` and `max_events` keep the file bounded. The log is pruned in place
(rewritten) periodically as new rows are written.

```json
"telemetry": { "enabled": true, "retention_days": 30, "max_events": 50000 }
```

`retention_days` drops rows older than N days (`null` = keep all); `max_events` keeps
the newest N (`null` = no cap). `log_path` is an optional override for the JSONL path.

Set `enabled: false` to turn telemetry off entirely.

## Cost

`cost_usd` is an estimate based on the served model's catalog price applied to that
call's actual token usage. See [pricing](pricing.md).
