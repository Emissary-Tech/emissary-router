# Thinking

Claude Code sets thinking parameters based on the model it thinks it is talking to.
Because the router can serve a different (cheaper) model than Claude Code targeted,
Emissary Router re-maps thinking to the **served** model's capabilities before
forwarding. The goal is to preserve Claude Code's intent (how hard to think) while
staying within what each model/provider accepts.

Claude Code expresses thinking depth two ways:

- adaptive thinking + an effort level (`output_config.effort`) for Sonnet/Opus
- an explicit token budget (`thinking.budget_tokens`) for Haiku

The router normalizes per served model:

- Anthropic Sonnet: accepts effort/adaptive; effort is clamped to Sonnet's max.
- Anthropic Haiku: budget-only (rejects effort/adaptive). Adaptive/effort requests
  become `thinking: {type: enabled, budget_tokens: max_tokens - 1}`.
- OpenRouter Claude: same intent, expressed in OpenRouter's `reasoning` shape.
  Anthropic's `max` effort maps to OpenRouter's `xhigh` (the vocabularies differ —
  Anthropic's top is `max`, OpenRouter's is `xhigh`). Budget-only models receive
  `reasoning.max_tokens`.
- Gemini via OpenRouter: uses OpenRouter's reasoning shape.

This normalization only touches thinking fields; it never rewrites the cacheable
prefix (tools/system/messages), so prompt caching is unaffected.

See also: Gemini is served through OpenRouter for Claude Code tool loops — see
[providers and caching](providers-caching.md).
