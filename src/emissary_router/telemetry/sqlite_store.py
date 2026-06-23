from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from emissary_router.telemetry.event_record import COLUMNS, EventRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT,
    turn_id INTEGER,
    call_kind TEXT,
    requested_model TEXT,
    served_model TEXT,
    provider TEXT,
    model_id TEXT,
    route_reason TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    cost_usd REAL,
    duration_ms REAL,
    http_status INTEGER,
    raw_event TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_session_turn ON events (session_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (ts);
"""


class SqliteStore:
    """File-backed event store for telemetry and the dashboard.

    Survives restarts (single file). Uses WAL so the gateway can write while the
    dashboard reads. Each operation opens a short-lived connection, which keeps it
    thread-safe under the async server at this volume without a shared lock.
    """

    def __init__(
        self,
        path: Path,
        *,
        retention_days: int | None = None,
        max_events: int | None = None,
        prune_interval: int = 200,
    ):
        self._path = Path(path).expanduser()
        self._retention_days = retention_days
        self._max_events = max_events
        self._prune_interval = max(prune_interval, 1)
        self._writes_since_prune = 0
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            existing = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
            if "http_status" not in existing:
                conn.execute("ALTER TABLE events ADD COLUMN http_status INTEGER")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # --- write ---------------------------------------------------------------
    def write(self, record: EventRecord) -> None:
        row = asdict(record)
        placeholders = ", ".join(":" + name for name in COLUMNS)
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO events ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                row,
            )
        self._writes_since_prune += 1
        if self._writes_since_prune >= self._prune_interval:
            self._writes_since_prune = 0
            self.prune()

    # --- read ----------------------------------------------------------------
    def list_events(self, limit: int = 200, session_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM events"
        params: list[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_public_row(row) for row in rows]

    def aggregate_by_model(self) -> list[dict[str, Any]]:
        query = """
            SELECT served_model,
                   COUNT(*) AS n,
                   SUM(input_tokens) AS input_tokens,
                   SUM(output_tokens) AS output_tokens,
                   SUM(cache_read_tokens) AS cache_read_tokens,
                   SUM(cache_creation_tokens) AS cache_creation_tokens,
                   SUM(COALESCE(cost_usd, 0)) AS cost_usd
            FROM events
            GROUP BY served_model
            ORDER BY cost_usd DESC
        """
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]

    def total_events(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def turns(self, session_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Group calls into turns (session_id + turn_id), with per-model breakdown."""
        where = "WHERE session_id = ?" if session_id else ""
        params: list[Any] = [session_id] if session_id else []
        query = f"""
            SELECT session_id, turn_id, served_model, call_kind,
                   COUNT(*) AS n,
                   SUM(COALESCE(cost_usd, 0)) AS cost_usd,
                   MIN(ts) AS first_ts
            FROM events
            {where}
            GROUP BY session_id, turn_id, served_model, call_kind
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        turns: dict[tuple[Any, Any], dict[str, Any]] = {}
        for row in rows:
            key = (row["session_id"], row["turn_id"])
            turn = turns.setdefault(
                key,
                {
                    "session_id": row["session_id"],
                    "turn_id": row["turn_id"],
                    "first_ts": row["first_ts"],
                    "n_calls": 0,
                    "n_main": 0,
                    "n_background": 0,
                    "cost_usd": 0.0,
                    "models": {},
                },
            )
            turn["n_calls"] += row["n"]
            turn["cost_usd"] += row["cost_usd"] or 0.0
            turn["first_ts"] = min(turn["first_ts"], row["first_ts"])
            if row["call_kind"] == "background":
                turn["n_background"] += row["n"]
            else:
                turn["n_main"] += row["n"]
            turn["models"][row["served_model"]] = turn["models"].get(row["served_model"], 0) + row["n"]

        ordered = sorted(turns.values(), key=lambda turn: turn["first_ts"], reverse=True)
        return ordered[:limit]

    def max_turn_id(self, session_id: str) -> int:
        with self._connect() as conn:
            value = conn.execute(
                "SELECT MAX(turn_id) FROM events WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
        return int(value) if value is not None else 0

    # --- delete --------------------------------------------------------------
    def delete_event(self, event_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            return cursor.rowcount

    def delete_session(self, session_id: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            return cursor.rowcount

    # --- maintenance ---------------------------------------------------------
    def prune(self) -> int:
        removed = 0
        with self._connect() as conn:
            if self._retention_days is not None:
                cutoff = time.time() - self._retention_days * 86400
                cursor = conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
                removed += cursor.rowcount
            if self._max_events is not None:
                cursor = conn.execute(
                    """
                    DELETE FROM events WHERE id IN (
                        SELECT id FROM events ORDER BY ts DESC LIMIT -1 OFFSET ?
                    )
                    """,
                    (self._max_events,),
                )
                removed += cursor.rowcount
        return removed

    def vacuum(self) -> None:
        with self._connect() as conn:
            conn.execute("VACUUM")


def _public_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data.pop("raw_event", None)
    return data
