from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from emissary_router.dashboard import build_dashboard_router
from emissary_router.telemetry import EventRecord, SqliteStore


def _store(tmp_path) -> SqliteStore:
    store = SqliteStore(tmp_path / "e.sqlite3")
    store.write(
        EventRecord(
            id="x", ts=time.time(), session_id="s1", call_kind="main",
            requested_model="claude-sonnet-4-6", served_model="claude-haiku-4.5",
            provider="anthropic", model_id="claude-haiku-4-5", route_reason="deviate_if_confident",
            input_tokens=10, output_tokens=2, cache_read_tokens=0, cache_creation_tokens=0,
            cost_usd=0.001, duration_ms=1.0, raw_event=None,
        )
    )
    return store


def _client(store, auth_key=None) -> TestClient:
    app = FastAPI()
    app.include_router(
        build_dashboard_router(store, "claude-sonnet-4.6", auth_key=auth_key)
    )
    return TestClient(app)


def test_no_auth_key_dashboard_is_open(tmp_path):
    c = _client(_store(tmp_path))
    assert c.get("/dashboard").status_code == 200
    assert c.get("/api/events").status_code == 200


def test_auth_key_blocks_dashboard_and_api_and_delete(tmp_path):
    c = _client(_store(tmp_path), auth_key="secret")
    # blocked without key
    assert c.get("/dashboard").status_code == 401
    assert c.get("/api/events").status_code == 401
    assert c.get("/api/summary").status_code == 401
    assert c.delete("/api/events/x").status_code == 401
    assert c.delete("/api/sessions/s1").status_code == 401


def test_auth_key_accepts_header_and_query(tmp_path):
    c = _client(_store(tmp_path), auth_key="secret")
    assert c.get("/api/events", headers={"x-api-key": "secret"}).status_code == 200
    assert c.get("/api/events", headers={"authorization": "Bearer secret"}).status_code == 200
    assert c.get("/dashboard?key=secret").status_code == 200
    assert c.get("/api/events?key=secret").status_code == 200
    assert c.get("/api/events?key=wrong").status_code == 401


def test_delete_session_removes_rows(tmp_path):
    store = _store(tmp_path)
    c = _client(store)
    assert c.delete("/api/sessions/s1").json()["deleted"] == 1
    assert store.total_events() == 0
