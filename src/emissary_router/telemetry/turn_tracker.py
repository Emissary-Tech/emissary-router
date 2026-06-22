from __future__ import annotations

import hashlib
import threading

from emissary_router.telemetry.sqlite_store import SqliteStore


def _fingerprint(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


class TurnTracker:
    """Assigns a per-session turn id that is stable across one user input.

    A user input drives several main calls (the agent loop) plus background calls.
    All of them share the same "latest user text", so the turn id only increments
    when that input changes. This is timing-independent, so long-running tools do
    not split a turn. On first sight of a session (e.g. after a restart) the counter
    is seeded from the store so turn ids do not collide with earlier turns.
    """

    def __init__(self, store: SqliteStore | None = None):
        self._store = store
        self._lock = threading.Lock()
        self._state: dict[str, tuple[str | None, int]] = {}

    def turn_id(
        self,
        session_id: str | None,
        current_input: str | None,
        call_kind: str = "main",
    ) -> int:
        if session_id is None:
            return 1
        with self._lock:
            state = self._state.get(session_id)
            if call_kind == "background":
                # Background calls (title/topic) attach to the current turn and never
                # advance it, so they do not create background-only turns.
                if state is not None:
                    return state[1]
                seed = self._store.max_turn_id(session_id) if self._store else 0
                return seed if seed > 0 else 1
            fingerprint = _fingerprint(current_input)
            if state is None:
                seed = self._store.max_turn_id(session_id) if self._store else 0
                turn_id = seed + 1
                self._state[session_id] = (fingerprint, turn_id)
                return turn_id
            last_fingerprint, turn_id = state
            if fingerprint != last_fingerprint:
                turn_id += 1
                self._state[session_id] = (fingerprint, turn_id)
            return turn_id

    def forget(self, session_id: str) -> None:
        with self._lock:
            self._state.pop(session_id, None)
