# Pricing

Pricing is embedded in the built-in catalog (`src/emissary_router/catalog.py`). There
is no user-maintained `pricing.yaml` in V1 — since the catalog also owns the model
set, prices ship with it.

Prices are USD per 1M tokens.

| Model | input | output | cache read | cache write (5m) |
|---|---|---|---|---|
| `gemini-3.1-flash-lite` | 0.25 | 1.50 | 0.025 | 0.25 |
| `claude-haiku-4.5` | 1.00 | 5.00 | 0.10 | 1.25 |
| `claude-sonnet-4.6` | 3.00 | 15.00 | 0.30 | 3.75 |

These prices are used for the `cost_usd` estimate in [telemetry](telemetry.md),
computed from each call's actual token usage:

```text
cost = (input·in + output·out + cache_read·cr + cache_creation·cw) / 1e6
```

`cost_usd` is an estimate, not a billed amount — your provider invoice is the source
of truth. If provider prices change, update `catalog.py` and release a new version.
