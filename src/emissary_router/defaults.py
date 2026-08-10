"""Default files written into the user's home on first setup.

These are embedded in the package (not copied from a checkout) so `er init` works
the same whether installed via pip/uv/pipx or from a clone.
"""

from __future__ import annotations

CONFIG_TEMPLATE = """{
  "models": {
    "deepseek-v4-flash": { "enabled": true, "provider": "openrouter" },
    "gpt-5.6-luna": { "enabled": true, "provider": "openai" },
    "gemini-3.1-flash-lite": { "enabled": true, "provider": "openrouter" },
    "glm-5.2": { "enabled": true, "provider": "openrouter" },
    "kimi-k2.7-code": { "enabled": true, "provider": "openrouter" },
    "claude-haiku-4.5": { "enabled": true, "provider": "anthropic" },
    "claude-sonnet-5": { "enabled": true, "provider": "anthropic" },
    "claude-opus-5": { "enabled": true, "provider": "anthropic" },
    "kimi-k3": { "enabled": false, "provider": "openrouter" },
    "gpt-5.6-terra": { "enabled": false, "provider": "openai" },
    "gpt-5.6-sol": { "enabled": false, "provider": "openai" }
  },
  "default": "claude-sonnet-5",
  "confidence": 0.8,
  "router": { "router_model": "emissary-model-router-shared" },
  "server": { "port": 8788 },
  "telemetry": { "enabled": true, "retention_days": 30, "max_events": 50000 }
}
"""
