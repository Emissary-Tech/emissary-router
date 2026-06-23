from __future__ import annotations

import time

from emissary_router.telemetry import EventRecord, SqliteStore
from emissary_router.telemetry.event_record import call_kind_from_body


def _rec(store_id: str, **over) -> EventRecord:
    base = dict(
        id=store_id,
        ts=time.time(),
        session_id="s1",
        call_kind="main",
        requested_model="claude-sonnet-4-6",
        served_model="claude-haiku-4.5",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        route_reason="deviate_if_confident:p>=0.8",
        input_tokens=1000,
        output_tokens=200,
        cache_read_tokens=500,
        cache_creation_tokens=0,
        cost_usd=0.0012,
        duration_ms=42.0,
        raw_event=None,
    )
    base.update(over)
    return EventRecord(**base)


def test_write_list_and_delete(tmp_path):
    store = SqliteStore(tmp_path / "e.sqlite3")
    store.write(_rec("a"))
    store.write(_rec("b", session_id="s2"))
    assert store.total_events() == 2
    assert len(store.list_events(session_id="s1")) == 1

    assert store.delete_event("a") == 1
    assert store.total_events() == 1
    assert store.delete_session("s2") == 1
    assert store.total_events() == 0


def test_list_events_omits_raw_event(tmp_path):
    store = SqliteStore(tmp_path / "e.sqlite3")
    store.write(_rec("a", raw_event='{"big":"blob"}'))
    row = store.list_events()[0]
    assert "raw_event" not in row


def test_sessions_group_by_session(tmp_path):
    store = SqliteStore(tmp_path / "e.sqlite3")
    store.write(_rec("a", session_id="s1", call_kind="main", served_model="claude-haiku-4.5", cost_usd=0.001))
    store.write(_rec("b", session_id="s1", call_kind="background", served_model="gemini-3.1-flash-lite", cost_usd=0.0))
    store.write(_rec("c", session_id="s2", call_kind="main", served_model="claude-sonnet-4.6", cost_usd=0.01))

    sessions = store.sessions()
    assert len(sessions) == 2
    s1 = next(s for s in sessions if s["session_id"] == "s1")
    assert s1["n_calls"] == 2 and s1["n_main"] == 1 and s1["n_background"] == 1
    assert s1["models"]["claude-haiku-4.5"] == 1
    assert "last_ts" in s1


def test_prune_by_retention_and_max(tmp_path):
    store = SqliteStore(tmp_path / "e.sqlite3", retention_days=1, max_events=2)
    old = time.time() - 5 * 86400
    store.write(_rec("old", ts=old))
    store.write(_rec("a"))
    store.write(_rec("b"))
    store.write(_rec("c"))
    store.prune()
    ids = {row["id"] for row in store.list_events()}
    assert "old" not in ids  # retention dropped it
    assert len(ids) <= 2  # max_events cap


def test_call_kind_main_when_thinking_or_effort_without_tools():
    assert call_kind_from_body({"tools": [{"name": "x"}]}) == "main"
    assert call_kind_from_body({}) == "background"
    assert call_kind_from_body({"thinking": {"type": "disabled"}}) == "background"
    # no tools but reasoning active -> main (matches what Claude Code sends)
    assert call_kind_from_body({"thinking": {"type": "adaptive"}}) == "main"
    assert call_kind_from_body({"output_config": {"effort": "high"}}) == "main"
