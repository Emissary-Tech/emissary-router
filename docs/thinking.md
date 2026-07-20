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

This normalization only touches thinking fields; it never rewrites the cacheable
prefix (tools/system/messages), so prompt caching is unaffected.

See also: [providers and caching](providers-caching.md) for how each provider is
served and how cross-provider thinking signatures are kept safe.
