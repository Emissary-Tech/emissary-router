from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlTelemetry:
    def __init__(self, path: Path):
        self._path = path

    def write(self, row: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
