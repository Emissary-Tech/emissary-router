"""Default files written into the user's home on first setup.

These are embedded in the package (not copied from a checkout) so `er init` works
the same whether installed via pip/uv/pipx or from a clone.
"""

from __future__ import annotations

CONFIG_TEMPLATE = """{
  "models": {
    "claude-sonnet-4.6": { "enabled": true, "provider": "anthropic" },
    "claude-haiku-4.5": { "enabled": true, "provider": "anthropic" },
    "gemini-3.1-flash-lite": { "enabled": true, "provider": "openrouter" }
  },
  "default": "claude-sonnet-4.6",
  "confidence": 0.8,
  "policy": "cache_aware",
  "router": { "router_model": "emissary-model-router-shared" },
  "server": { "port": 8788 },
  "telemetry": { "enabled": true, "retention_days": 30, "max_events": 50000 }
}
"""
