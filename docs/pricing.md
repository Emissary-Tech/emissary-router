# Pricing

Pricing is embedded in the built-in catalog (`src/emissary_router/catalog.py`). There
is no user-maintained `pricing.yaml` in V1 — since the catalog also owns the model
set, prices ship with it.

Prices are USD per 1M tokens.

| Model | input | output | cache read | cache write (5m) |
|---|---|---|---|---|
| `gemini-3.1-flash-lite` | 0.25 | 1.50 | 0.025 | 0.25 |
| `kimi-k2.7-code` | 0.74 | 3.50 | 0.15 | 0.74 |
| `glm-5.2` | 0.94 | 3.00 | 0.18 | 0.94 |
| `claude-haiku-4.5` | 1.00 | 5.00 | 0.10 | 1.25 |
| `claude-sonnet-4.6` | 3.00 | 15.00 | 0.30 | 3.75 |

For glm / kimi / gemini the cache-write price equals the input price: their caching
is implicit, so there is no write premium — only cache reads are discounted.

Prices are per model, whichever provider transport serves it. Two consequences:
provider invoices can differ slightly from these list rates, and requests served
through a subscription key (e.g. GLM Coding Plan via the `zai` provider) draw down
quota rather than per-token spend — `cost_usd` still prices them at these rates as a
reference number.

These prices are used for the `cost_usd` estimate in [telemetry](telemetry.md),
computed from each call's actual token usage:

```text
cost = (input·in + output·out + cache_read·cr + cache_creation·cw) / 1e6
```

`cost_usd` is an estimate, not a billed amount — your provider invoice is the source
of truth. If provider prices change, update `catalog.py` and release a new version.
