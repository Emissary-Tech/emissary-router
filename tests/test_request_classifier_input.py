from __future__ import annotations

from emissary_router.routing.request_to_classifier_input import request_to_classifier_input


def _task_line(classifier_input: str) -> str:
    # "Task:\n<the task>\n\n..." -> the task text
    return classifier_input.split("\n", 2)[1]


def test_strips_session_wrapper_keeping_inner_request():
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<session>\nCan you review the codebase here?\n</session>"}
            ]}
        ],
        "tools": [{"name": "Read"}],
    }
    ci, _ = request_to_classifier_input(body)
    assert _task_line(ci) == "Can you review the codebase here?"
    assert "<session>" not in ci


def test_strips_system_reminder_block():
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>\nnoise\n</system-reminder>\nFix the auth bug"}
            ]}
        ],
        "tools": [{"name": "Read"}],
    }
    ci, _ = request_to_classifier_input(body)
    assert _task_line(ci) == "Fix the auth bug"


def test_session_stripped_across_all_turns_in_multiturn():
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "<session>add tests</session>"}]},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "1", "name": "Bash", "input": {"cmd": "pytest"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "3 passed"}
            ]},
            {"role": "user", "content": [{"type": "text", "text": "<session>now refactor</session>"}]},
        ],
        "tools": [{"name": "Bash"}],
    }
    ci, meta = request_to_classifier_input(body)
    assert _task_line(ci) == "now refactor"
    assert "Earlier user turns:" in ci and "1. add tests" in ci
    assert "<session>" not in ci
    assert meta["turn"] == 1 and meta["steps"] == 1
