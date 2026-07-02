from __future__ import annotations

import json

from fastapi.testclient import TestClient


def test_health_reflects_live_default_after_config_change(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "models": {
                    "claude-sonnet-4.6": True,
                    "claude-haiku-4.5": True,
                    "gemini-3.1-flash-lite": True,
                },
                "default": "claude-sonnet-4.6",
                "confidence": 0.8,
                "telemetry": {"enabled": True, "db_path": str(tmp_path / "e.sqlite3")},
            }
        )
    )
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    monkeypatch.setenv("EMISSARY_ROUTER_CONFIG", str(cfg))

    from emissary_router.server import create_app

    client = TestClient(create_app())
    assert client.get("/").json()["default"] == "claude-sonnet-4.6"

    # change the default via the dashboard -> reload -> health must reflect it (not stale)
    resp = client.put("/api/config", json={"default": "claude-haiku-4.5"})
    assert resp.status_code == 200
    assert client.get("/").json()["default"] == "claude-haiku-4.5"


def test_cache_ledger_survives_config_reload(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "models": {
                    "claude-sonnet-4.6": True,
                    "claude-haiku-4.5": True,
                    "gemini-3.1-flash-lite": True,
                },
                "default": "claude-sonnet-4.6",
                "confidence": 0.8,
                "telemetry": {"enabled": True, "db_path": str(tmp_path / "e.sqlite3")},
            }
        )
    )
    monkeypatch.setenv("EMISSARY_ROUTER_HOME", str(tmp_path))
    monkeypatch.setenv("EMISSARY_ROUTER_CONFIG", str(cfg))

    from emissary_router.caching.usage import Usage
    from emissary_router.routing.cache_cost import RequestCostFeatures
    from emissary_router.server import create_app

    app = create_app()
    client = TestClient(app)

    ledger_before = app.state.pipeline.cache_ledger
    model = app.state.config.resolve_model("claude-sonnet-4.6")
    feats = RequestCostFeatures("s1", "h", 12000, 10000, 2000, 1024)
    ledger_before.observe(model, feats, Usage(output_tokens=2000, cache_creation_input_tokens=10000))
    assert ledger_before.predict(model, feats).warm is True

    # a config save rebuilds the pipeline; the warm-cache state must NOT reset
    resp = client.put("/api/config", json={"confidence": 0.6})
    assert resp.status_code == 200

    ledger_after = app.state.pipeline.cache_ledger
    assert ledger_after is ledger_before
    assert ledger_after.predict(model, feats).warm is True
