from __future__ import annotations

import json
import re
from typing import Any

from router.routing.classifier_input import render_classifier_input


REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or json.dumps(block))
            else:
                parts.append(str(block))
        return "\n".join(map(str, parts))
    return str(content)


def _clean_user_text(blocks: list[dict[str, Any]]) -> str:
    parts = [
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return REMINDER_RE.sub("", "\n".join(parts)).strip()


def request_to_classifier_input(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    messages = body.get("messages", []) or []
    tools = body.get("tools", []) or []

    user_turns: list[str] = []
    steps_by_id: dict[str, dict[str, Any]] = {}
    ordered_steps: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]

        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if role == "assistant" and block_type == "tool_use":
                args = json.dumps(block.get("input", {}), ensure_ascii=False)
                step = {"call": f"{block.get('name', '?')}({args})", "output": "", "error": False}
                steps_by_id[block.get("id")] = step
                ordered_steps.append(step)
            elif role == "user" and block_type == "tool_result":
                step = steps_by_id.get(block.get("tool_use_id"))
                if step is not None:
                    step["output"] = _stringify(block.get("content"))
                    step["error"] = bool(block.get("is_error"))

        if role == "user":
            text = _clean_user_text(blocks)
            if text:
                user_turns.append(text)

    task = user_turns[-1] if user_turns else "(no task)"
    earlier = user_turns[:-1]
    tool_names = ", ".join(tool.get("name", "") for tool in tools) or "(none)"
    turn = max(len(user_turns) - 1, 0)
    classifier_input = render_classifier_input(
        task=task,
        earlier_user_turns=earlier,
        recent_steps=ordered_steps,
        turn=turn,
        tools_block=tool_names,
    )
    metadata = {
        "turn": turn,
        "steps": len(ordered_steps),
        "last_error": bool(ordered_steps[-1]["error"]) if ordered_steps else False,
        "n_tools": len(tools),
        "classifier_input_len": len(classifier_input),
    }
    return classifier_input, metadata
