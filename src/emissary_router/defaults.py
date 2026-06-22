"""Default files written into the user's home on first setup.

These are embedded in the package (not copied from a checkout) so `er init` works
the same whether installed via pip/uv/pipx or from a clone.
"""

from __future__ import annotations

CONFIG_TEMPLATE = """models:
  claude-sonnet-4.6: true
  claude-haiku-4.5: true
  gemini-3.1-flash-lite: true

default: claude-sonnet-4.6
confidence: 0.8

server:
  port: 8788

telemetry:
  enabled: true
  retention_days: 30
  max_events: 50000
"""

ENV_TEMPLATE = """# Emissary Router secrets. Keep this file private (chmod 600).
EMISSARY_ROUTER_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
# GOOGLE_API_KEY=
"""
