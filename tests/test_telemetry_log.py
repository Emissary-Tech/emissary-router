from __future__ import annotations

import json
import time

from emissary_router.telemetry import JsonlTelemetry


def test_jsonl_telemetry_prunes_by_retention_and_max_events(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    telemetry = JsonlTelemetry(path, retention_days=1, max_events=2)
    old = time.time() - 5 * 86400

    telemetry.write({"request_id": "old", "ts": old})
    telemetry.write({"request_id": "a", "ts": time.time()})
    telemetry.write({"request_id": "b", "ts": time.time()})
    telemetry.write({"request_id": "c", "ts": time.time()})
    telemetry.prune()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [row["request_id"] for row in rows]
    assert "old" not in ids
    assert ids == ["b", "c"]
