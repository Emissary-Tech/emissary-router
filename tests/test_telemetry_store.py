from __future__ import annotations

import time

from emissary_router.telemetry import EventRecord, SqliteStore, TurnTracker
from emissary_router.telemetry.event_record import call_kind_from_body, latest_real_user_text


def _rec(store_id: str, **over) -> EventRecord:
    base = dict(
        id=store_id,
        ts=time.time(),
        session_id="s1",
        turn_id=1,
        call_kind="main",
        requested_model="claude-sonnet-4-6",
        served_model="claude-haiku-4.5",
        provider="anthropic",
        model_id="claude-haiku-4-5",
        route_reason="cheap_first:p>=0.8",
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


def test_turns_group_main_and_background(tmp_path):
    store = SqliteStore(tmp_path / "e.sqlite3")
    # one input -> 2 main (haiku, gemini) + 1 background, all turn 1
    store.write(_rec("a", turn_id=1, call_kind="main", served_model="claude-haiku-4.5", cost_usd=0.001))
    store.write(_rec("b", turn_id=1, call_kind="main", served_model="gemini-3.1-flash-lite", cost_usd=0.0002))
    store.write(_rec("c", turn_id=1, call_kind="background", served_model="claude-haiku-4.5", cost_usd=0.0))
    store.write(_rec("d", turn_id=2, call_kind="main", served_model="claude-sonnet-4.6", cost_usd=0.01))

    turns = store.turns(session_id="s1")
    assert len(turns) == 2
    turn1 = next(t for t in turns if t["turn_id"] == 1)
    assert turn1["n_calls"] == 3
    assert turn1["n_main"] == 2 and turn1["n_background"] == 1
    assert turn1["models"]["claude-haiku-4.5"] == 2


def test_max_turn_id_seeds_tracker(tmp_path):
    store = SqliteStore(tmp_path / "e.sqlite3")
    store.write(_rec("a", session_id="s9", turn_id=3))
    assert store.max_turn_id("s9") == 3
    tracker = TurnTracker(store)
    # first sight after "restart" seeds from max -> 4
    assert tracker.turn_id("s9", "new input") == 4


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


def test_turn_tracker_stable_within_input_increments_on_change():
    tracker = TurnTracker(store=None)
    # same input across the agent loop -> same turn
    assert tracker.turn_id("s1", "fix the bug") == 1
    assert tracker.turn_id("s1", "fix the bug") == 1
    # new input -> +1
    assert tracker.turn_id("s1", "add tests") == 2
    # different session independent
    assert tracker.turn_id("s2", "hello") == 1


def test_background_attaches_to_current_turn_without_advancing():
    tracker = TurnTracker(store=None)
    assert tracker.turn_id("s1", "fix the bug", "main") == 1
    # background (title gen) has its own prompt text but must not create a new turn
    assert tracker.turn_id("s1", "Generate a title for: fix the bug", "background") == 1
    assert tracker.turn_id("s1", "another bg prompt", "background") == 1
    # next real input still advances normally
    assert tracker.turn_id("s1", "add tests", "main") == 2
    assert tracker.turn_id("s1", "title", "background") == 2


def test_forget_resets_session_turns():
    tracker = TurnTracker(store=None)
    assert tracker.turn_id("s1", "a") == 1
    assert tracker.turn_id("s1", "b") == 2
    tracker.forget("s1")
    # after a session delete the counter restarts (store is empty -> seeds 0)
    assert tracker.turn_id("s1", "c") == 1


def test_call_kind_main_when_thinking_or_effort_without_tools():
    assert call_kind_from_body({"tools": [{"name": "x"}]}) == "main"
    assert call_kind_from_body({}) == "background"
    assert call_kind_from_body({"thinking": {"type": "disabled"}}) == "background"
    # no tools but reasoning active -> main (matches what CC sends)
    assert call_kind_from_body({"thinking": {"type": "adaptive"}}) == "main"
    assert call_kind_from_body({"output_config": {"effort": "high"}}) == "main"


def test_latest_real_user_text_skips_tool_results():
    loop = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "done"}]},
    ]
    assert latest_real_user_text(loop) == "fix the bug"
    assert latest_real_user_text([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]) == "hi"


def test_latest_real_user_text_ignores_system_reminders():
    def reminder(text):
        return {"role": "user", "content": [{"type": "text", "text": f"<system-reminder>\n{text}\n</system-reminder>"}]}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<system-reminder>\ntools available\n</system-reminder>"},
                {"type": "text", "text": "review the code in this directory"},
            ],
        },
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "x"}]},
        reminder("tools haven't been used recently"),  # injected mid-loop, evolving text
        {"role": "assistant", "content": [{"type": "tool_use", "id": "2", "name": "Read", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "2", "content": "y"}]},
    ]
    # anchor is the real prompt, not the injected reminder
    assert latest_real_user_text(messages) == "review the code in this directory"

    # so the whole agent loop stays one turn even as reminders are injected
    tracker = TurnTracker(store=None)
    first = tracker.turn_id("s", latest_real_user_text(messages[:3]))
    after_reminder = tracker.turn_id("s", latest_real_user_text(messages))
    assert first == after_reminder == 1
