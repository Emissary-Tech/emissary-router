# Thinking

Claude Code sets thinking parameters based on the model it thinks it is talking to.
Because the router can serve a different (cheaper) model than Claude Code targeted,
Emissary Router re-maps thinking to the **served** model's capabilities before
forwarding. The goal is to preserve Claude Code's intent (how hard to think) while
staying within what each model/provider accepts.

Claude Code expresses thinking depth two ways:

- adaptive thinking + an effort level (`output_config.effort`) for Sonnet/Opus
- an explicit token budget (`thinking.budget_tokens`) for Haiku

The router normalizes per served model (all mappings live-verified):

- Anthropic Sonnet: accepts effort/adaptive; effort is clamped to Sonnet's ladder —
  Sonnet 5-era `xhigh` becomes `max` (sonnet-4.6 rejects `xhigh` outright).
- Anthropic Haiku: budget-only (rejects effort/adaptive). Adaptive/effort requests
  become `thinking: {type: enabled, budget_tokens: max_tokens - 1}`.
- OpenRouter Claude: same intent, expressed in OpenRouter's `reasoning` shape.
  Effort resolves through the served model's own vocabulary first (each model's
  `supported_efforts`), then snaps to OpenRouter's ceiling: its enum tops out at
  `xhigh`. (`max` is a real tier on native surfaces — Anthropic claude-5 and
  OpenAI gpt-5.6 both define it — but OpenRouter's unified enum does not, so it is
  invalid there.) Budget-only models receive `reasoning.max_tokens`.
- Gemini via OpenRouter: uses OpenRouter's reasoning shape.
  Anthropic's `max` effort maps to OpenRouter's `xhigh` (the vocabularies differ).
  Budget-only models receive `reasoning.max_tokens`.
- GLM / Kimi via OpenRouter: effort passes through up to `xhigh` (`max` → `xhigh`).
  Kimi always reasons — on a thinking-disable request the reasoning field is omitted
  entirely (sending a disable gets a 400), and background calls are never routed to
  always-on-reasoning models in the first place.
- Gemini via OpenRouter: OpenRouter's reasoning shape, effort capped at `high`.
- GLM via Z.ai (native): no rewriting at all — the endpoint accepts the whole
  Sonnet 5-era surface as-is (adaptive, every effort level including `max`/`xhigh`,
  disabled, enabled+budget). Response thinking is re-signed with the router's
  synthetic marker so later turns can route to any provider.
- Gemini native: effort maps to Google's `thinkingLevel`; thought summaries come
  back as a thinking block bearing the synthetic marker.

## Guards for shapes the served model rejects

A client sets thinking for the model *it* named; the router may serve one with a
different contract. These rewrites mirror the platform gateway (live probes,
2026-10-01 .. 2026-10-22) and are recorded per call in telemetry's
`thinking_normalization`:

- **Budgeted `enabled` on Claude 5.** Every Claude 5 (`claude-sonnet-5`, `-opus-5`,
  `-opus-5.5`, `-sonnet-5.5`, `-haiku-5.5`, `-fable-5.1`) 400s on
  `thinking: {type: enabled, budget_tokens}` ("use adaptive and output_config.effort").
  The shape becomes `{type: adaptive}`; the budget is dropped (adaptive has no slot
  for it and tokens do not map onto effort rungs), a client `output_config.effort`
  still applies, otherwise the model default. `claude-haiku-4.5` keeps the budget shape.
- **Thinking off.** `thinking.type` `disabled` and `between_tools` both read as "off",
  and the served model gets its own switch: `claude-sonnet-5.5` takes `between_tools`
  (`disabled` 400s), `claude-haiku-5.5` and the older Claudes take `disabled`.
  `claude-opus-5.5` and `claude-fable-5.1` have no switch at all (adaptive is the only
  type they accept) — an off request runs adaptive at their lowest effort (`low`),
  whatever effort the client named. On effort-shaped surfaces (OpenRouter and OpenAI
  Responses `reasoning.effort`) "off" resolves the same way: `none` where the model
  can switch reasoning off there, the lowest rung where it cannot (claude 5.5 /
  fable-5.1, `glm-5.3`, `glm-5.3-flash`, `kimi-k2.7-code` — OpenRouter 400s
  "Reasoning is mandatory" on none — and `gpt-6.1-sol` / `gpt-6-astra`, whose
  vocabulary has no `none`). `glm-5.3`, `glm-5.3-flash` and `kimi-k2.7-code` are still
  excluded as candidates for background (thinking-off) calls; `claude-opus-5.5` and
  `claude-sonnet-5.5` are not, so an anchor on them can always serve.
- **`thinking.display`.** A Claude 5 knob; dropped for models without it (older
  Claude, GLM on the Anthropic-compatible path).
- **Sampling params.** `temperature` / `top_p` / `top_k` are stripped for every
  Claude 5 (`claude-haiku-5.5` and `claude-fable-5.1` 400 on either key alone).
- **Forced tool choice.** `claude-sonnet-5.5`, `claude-opus-5.5` and `claude-fable-5.1`
  400 on `tool_choice` type `any` / `tool`, directly and through OpenRouter. When
  routing lands such a request on one of them the router answers its own 400
  (Anthropic error shape, `invalid_request_error`) before any provider call, naming
  the two ways out: send `auto` / omit `tool_choice`, or disable those models for
  requests that must force a tool.

This normalization only touches thinking fields; it never rewrites the cacheable
prefix (tools/system/messages), so prompt caching is unaffected.

See also: [providers and caching](providers-caching.md) for how each provider is
served and how cross-provider thinking signatures are kept safe.
