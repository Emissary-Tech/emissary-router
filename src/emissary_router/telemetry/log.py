from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlTelemetry:
    def __init__(
        self,
        path: Path,
        *,
        retention_days: int | None = None,
        max_events: int | None = None,
        prune_interval: int = 200,
    ):
        self._path = path
        self._retention_days = retention_days
        self._max_events = max_events
        self._prune_interval = max(prune_interval, 1)
        self._writes_since_prune = 0

    def write(self, row: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._writes_since_prune += 1
        if self._writes_since_prune >= self._prune_interval:
            self._writes_since_prune = 0
            self.prune()

    def prune(self) -> int:
        if self._retention_days is None and self._max_events is None:
            return 0
        try:
            rows = self._read_rows()
        except FileNotFoundError:
            return 0
        original_count = len(rows)
        if self._retention_days is not None:
            cutoff = time.time() - self._retention_days * 86400
            rows = [row for row in rows if _row_ts(row) >= cutoff]
        if self._max_events == 0:
            rows = []
        elif self._max_events is not None and len(rows) > self._max_events:
            rows = rows[-self._max_events:]
        if len(rows) == original_count:
            return 0
        self._rewrite(rows)
        return original_count - len(rows)

    def _read_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._path.open() as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
        return rows

    def _rewrite(self, rows: list[dict[str, Any]]) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(self._path)


def _row_ts(row: dict[str, Any]) -> float:
    value = row.get("ts")
    return float(value) if isinstance(value, (int, float)) else 0.0
