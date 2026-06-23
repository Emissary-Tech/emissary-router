from __future__ import annotations

from emissary_router.server import _authorized, _strip_router_auth


def test_router_auth_key_accepts_x_api_key() -> None:
    assert _authorized({"x-api-key": "router-key"}, "router-key")


def test_router_auth_key_strips_router_key_before_provider_forward() -> None:
    headers = {
        "x-api-key": "router-key",
        "anthropic-version": "2023-06-01",
    }

    stripped = _strip_router_auth(headers, "router-key")

    assert "x-api-key" not in stripped
    assert stripped["anthropic-version"] == "2023-06-01"
