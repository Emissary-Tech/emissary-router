from __future__ import annotations

import asyncio

from emissary_router.cli import _warn_missing_env
from emissary_router.config import AppConfig
from emissary_router.pipeline import RouterPipeline
from emissary_router.telemetry import SqliteStore


class _FailingClassifier:
    def __init__(self, status: int):
        self.status = status

    async def predict(self, _input):
        resp = type("R", (), {"status_code": self.status})()
        err = Exception("unauthorized")
        err.response = resp
        raise err


def _config():
    return AppConfig.model_validate(
        {
            "models": {"claude-sonnet-4.6": True, "gemini-3.1-flash-lite": True},
            "default": "claude-sonnet-4.6",
        }
    )


def test_classifier_failure_recorded_with_status(tmp_path):
    store = SqliteStore(tmp_path / "e.sqlite3")
    pipe = RouterPipeline(_config(), store=store)
    pipe._classifier = _FailingClassifier(401)

    resp = asyncio.run(
        pipe.handle_messages(
            {"messages": [{"role": "user", "content": "hi"}]},
            {"x-claude-code-session-id": "s1"},
        )
    )
    assert resp.status_code == 502
    rows = store.list_events()
    assert len(rows) == 1
    assert rows[0]["served_model"] == "(classifier error)"
    assert rows[0]["http_status"] == 401
    assert rows[0]["session_id"] == "s1"


def test_warn_missing_env_is_provider_aware(capsys, monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY", "EMISSARY_ROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # OpenRouter-only setup: must not demand an Anthropic key
    cfg = AppConfig.model_validate(
        {
            "models": {
                "claude-haiku-4.5": {"enabled": True, "provider": "openrouter"},
                "gemini-3.1-flash-lite": True,
            },
            "default": "claude-haiku-4.5",
        }
    )
    _warn_missing_env(cfg)
    err = capsys.readouterr().err
    assert "OPENROUTER_API_KEY" in err
    assert "EMISSARY_ROUTER_API_KEY" in err
    assert "ANTHROPIC_API_KEY" not in err
