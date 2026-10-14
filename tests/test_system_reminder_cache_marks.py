"""Claude Code's per-turn conversation cache breakpoint rides on a trailing role:"system"
message (2.1.299 on newer Claude models, measured 2026-10-14). The providers convert that
message to a user reminder; the mark must survive the conversion."""
import json

from emissary_router.providers.anthropic import AnthropicProvider
from emissary_router.providers.openrouter import OpenRouterProvider

EPH = {"type": "ephemeral"}


def _body(system_content):
    return {"system": [{"type": "text", "text": "You are CC.", "cache_control": EPH}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "do it"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                {"role": "system", "content": system_content},
            ]}


def test_anthropic_trailing_system_reminder_keeps_its_cache_mark():
    body = _body([{"type": "text", "text": "Remember the budget.", "cache_control": EPH}])
    AnthropicProvider._hoist_system_messages(body)
    last = body["messages"][-1]
    assert last["role"] == "user"
    assert last["content"] == [{"type": "text", "text": "<system-reminder>\nRemember the budget.\n</system-reminder>",
                                "cache_control": EPH}]
    assert body["system"][0]["cache_control"] == EPH          # the head is untouched


def test_anthropic_unmarked_reminder_gets_no_mark():
    body = _body("plain reminder")
    AnthropicProvider._hoist_system_messages(body)
    assert "cache_control" not in body["messages"][-1]["content"][0]


def test_openrouter_trailing_system_reminder_keeps_its_cache_mark():
    body = _body([{"type": "text", "text": "Remember the budget.", "cache_control": EPH}])
    msgs = OpenRouterProvider._messages(body)
    last = msgs[-1]
    assert last["role"] == "user"
    assert last["content"][-1]["cache_control"] == EPH
    assert "<system-reminder>" in last["content"][-1]["text"]
    # unmarked reminders stay the plain string they were
    body = _body("plain reminder")
    assert isinstance(OpenRouterProvider._messages(body)[-1]["content"], str)
