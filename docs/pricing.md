# Pricing

Pricing is embedded in the built-in catalog (`src/emissary_router/catalog.py`). There
is no user-maintained `pricing.yaml` in V1 — since the catalog also owns the model
set, prices ship with it.

Prices are USD per 1M tokens.

| Model | input | output | cache read | cache write |
|---|---|---|---|---|
| `deepseek-v4-flash` / `-0731` | 0.14 | 0.28 | 0.028 | 0.14 |
| `gpt-6-luna` | 0.10 | 0.50 | 0.01 | 0.125 |
| `qwen3.8-omni-flash` | 0.15 | 0.47 | 0.016 | 0.15 |
| `qwen3.8-flash` | 0.15 | 0.47 | 0.016 | 0.20 |
| `deepseek-v4.1-flash` | 0.15 | 0.60 | 0.003 | 0.15 |
| `gpt-5.6-luna` | 0.20 | 1.20 | 0.02 | 0.25 |
| `gemini-3.1-flash-lite` | 0.25 | 1.50 | 0.025 | 0.25 |
| `kimi-k2.7-code` | 0.95 | 4.00 | 0.19 | 0.95 |
| `claude-haiku-4.5` | 1.00 | 5.00 | 0.10 | 1.25 |
| `glm-5.2` / `glm-5.3` | 1.40 | 4.40 | 0.26 | 1.40 |
| `glm-5.3-flash` | 0.15 | 0.50 | 0.03 | 0.15 |
| `gpt-5.6-terra` | 2.00 | 12.00 | 0.20 | 2.50 |
| `claude-sonnet-5` | 2.00 | 10.00 | 0.20 | 2.50 |
| `claude-sonnet-5.5` | 2.00 | 10.00 | 0.10 | 2.50 |
| `gpt-6-sol` | 2.00 | 10.00 | 0.20 | 2.50 |
| `gpt-6.1-sol` | 2.00 | 10.00 | 0.10 | 2.50 |
| `kimi-k3` | 3.00 | 15.00 | 0.30 | 3.00 |
| `gpt-5.6-sol` | 4.00 | 20.00 | 0.40 | 5.00 |
| `claude-opus-5.5` | 4.00 | 20.00 | 0.20 | 5.00 |
| `claude-opus-5` | 5.00 | 25.00 | 0.50 | 6.25 |

These are the platform gateway's catalog prices (emissary-ai `routing_gateway/catalog.py`,
2026-10-08), kept in step so a benchmark condition here prices a call exactly as the
platform would. First-party prices (Anthropic, OpenAI) are list prices; OpenRouter-served
open models are priced at the first party's own sheet rather than OpenRouter's floating
per-host rates.

For the OpenRouter-served open models the cache-write price equals the input price: their caching
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
