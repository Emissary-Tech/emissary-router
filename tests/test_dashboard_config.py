from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from emissary_router.dashboard import build_dashboard_router
from emissary_router.telemetry import SqliteStore

CONFIG = {
    "models": {
        "claude-sonnet-4.6": True,
        "claude-haiku-4.5": True,
        "gemini-3.1-flash-lite": True,
    },
    "default": "claude-sonnet-4.6",
    "confidence": 0.8,
}


def _client(tmp_path, on_config_change=None):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CONFIG))
    store = SqliteStore(tmp_path / "e.sqlite3")
    app = FastAPI()
    app.include_router(
        build_dashboard_router(
            store, "claude-sonnet-4.6", config_path=cfg, on_config_change=on_config_change
        )
    )
    return TestClient(app), cfg


def test_get_config_lists_catalog_cheap_first(tmp_path):
    client, _ = _client(tmp_path)
    body = client.get("/api/config").json()
    assert body["default"] == "claude-sonnet-4.6"
    assert body["confidence"] == 0.8
    assert [m["name"] for m in body["models"]][0] == "gemini-3.1-flash-lite"
    enabled = {m["name"]: m["enabled"] for m in body["models"]}
    # The three configured models are enabled; catalog-only models (not in this config)
    # show up disabled and toggleable.
    assert enabled["claude-sonnet-4.6"] is True
    assert enabled["claude-haiku-4.5"] is True
    assert enabled["gemini-3.1-flash-lite"] is True
    assert enabled["glm-5.2"] is False
    assert enabled["kimi-k2.7-code"] is False


def test_put_config_toggles_and_persists(tmp_path):
    client, cfg = _client(tmp_path)
    resp = client.put(
        "/api/config",
        json={
            "models": {
                "claude-sonnet-4.6": True,
                "claude-haiku-4.5": False,
                "gemini-3.1-flash-lite": True,
            },
            "default": "claude-sonnet-4.6",
            "confidence": 0.6,
        },
    )
    assert resp.status_code == 200
    assert resp.json() == {"saved": True, "restart_required": True}

    saved = json.loads(cfg.read_text())
    assert saved["models"]["claude-haiku-4.5"] is False
    assert saved["confidence"] == 0.6
    reread = {m["name"]: m["enabled"] for m in client.get("/api/config").json()["models"]}
    assert reread["claude-haiku-4.5"] is False


def test_get_config_exposes_provider_choices(tmp_path):
    client, _ = _client(tmp_path)
    models = {m["name"]: m for m in client.get("/api/config").json()["models"]}
    haiku = models["claude-haiku-4.5"]
    assert set(haiku["providers"]) == {"anthropic", "openrouter"}
    assert haiku["provider"] == "anthropic"  # catalog default
    gemini = models["gemini-3.1-flash-lite"]
    assert set(gemini["providers"]) == {"openrouter", "google"}


def test_put_provider_override_persists(tmp_path):
    client, cfg = _client(tmp_path)
    resp = client.put(
        "/api/config",
        json={
            "models": {
                "claude-sonnet-4.6": {"enabled": True},
                "claude-haiku-4.5": {"enabled": True, "provider": "openrouter"},
                "gemini-3.1-flash-lite": {"enabled": True},
            },
            "default": "claude-sonnet-4.6",
        },
    )
    assert resp.status_code == 200
    saved = json.loads(cfg.read_text())["models"]
    assert saved["claude-haiku-4.5"] == {"enabled": True, "provider": "openrouter"}
    assert {m["name"]: m["provider"] for m in client.get("/api/config").json()["models"]}[
        "claude-haiku-4.5"
    ] == "openrouter"


def test_put_rejects_unavailable_provider(tmp_path):
    client, _ = _client(tmp_path)
    # gemini is OpenRouter-only; anthropic is not available
    resp = client.put(
        "/api/config",
        json={"models": {"gemini-3.1-flash-lite": {"enabled": True, "provider": "anthropic"}}},
    )
    assert resp.status_code == 400


def test_save_triggers_reload_and_no_restart_required(tmp_path):
    calls = []
    client, _ = _client(tmp_path, on_config_change=lambda: calls.append(1))
    resp = client.put("/api/config", json={"models": {"claude-haiku-4.5": False}})
    assert resp.status_code == 200
    assert resp.json()["restart_required"] is False
    assert calls == [1]


def test_put_config_rejects_disabling_default(tmp_path):
    client, cfg = _client(tmp_path)
    resp = client.put(
        "/api/config",
        json={
            "models": {
                "claude-sonnet-4.6": False,
                "claude-haiku-4.5": True,
                "gemini-3.1-flash-lite": True,
            },
            "default": "claude-sonnet-4.6",
        },
    )
    assert resp.status_code == 400
    assert "error" in resp.json()
    # file left untouched on invalid input
    assert json.loads(cfg.read_text())["models"]["claude-sonnet-4.6"] is True
