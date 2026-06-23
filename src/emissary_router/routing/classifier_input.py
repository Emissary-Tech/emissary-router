"""Single source of truth for classifier input serialization.

Both the training-data builder and the live gateway/proxy import `render_classifier_input`
from here, so what the router sees at training time and at inference time is rendered
by the exact same code (no train/serve skew).

The classifier input:

    Task:                current user request
    Earlier user turns:  (multi-turn only)
    Recent steps:        executed tool calls + outputs, or "(none — turn start)"
    State:               turn index, steps so far, last-step error flag
    Tools:               tool names (+ short descriptions when available)
    Decide the next action.
"""

from __future__ import annotations

# Truncation budget. These are part of the train/serve contract — changing them
# means retraining, so keep them here, in one place.
MAX_TASK_CHARS = 4000
MAX_EARLIER_TURN_CHARS = 240
MAX_CALL_CHARS = 220
MAX_OK_OUTPUT_CHARS = 280
MAX_ERR_OUTPUT_CHARS = 600
MAX_TOOL_DESC_CHARS = 110
RECENT_STEPS_KEPT = 8


def clip(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + " …"


def clip_task(text: str, limit: int = MAX_TASK_CHARS) -> str:
    # The actual question often sits at the END of a task (math problems,
    # multi-part instructions), so oversized tasks keep head AND tail.
    text = str(text)
    if len(text) <= limit:
        return text
    head = int(limit * 0.75)
    tail = limit - head
    return text[:head] + "\n …[middle omitted]… \n" + text[-tail:]


def render_classifier_input(
    task: str,
    earlier_user_turns: list[str] | None = None,
    recent_steps: list[dict] | None = None,
    turn: int = 0,
    tools_block: str = "(none)",
) -> str:
    """Render one classifier-visible input.

    recent_steps: list of {"call": str, "output": str, "error": bool}.
    """
    lines = ["Task:", clip_task(task.strip(), MAX_TASK_CHARS), ""]

    if earlier_user_turns:
        lines.append("Earlier user turns:")
        for i, t in enumerate(earlier_user_turns):
            lines.append(f"{i + 1}. {clip(' '.join(t.split()), MAX_EARLIER_TURN_CHARS)}")
        lines.append("")

    lines.append("Recent steps:")
    if not recent_steps:
        lines.append("(none — turn start)")
        steps_so_far = 0
        last_error = False
    else:
        steps_so_far = len(recent_steps)
        kept = recent_steps[-RECENT_STEPS_KEPT:]
        omitted = steps_so_far - len(kept)
        if omitted > 0:
            lines.append(f"(... {omitted} earlier steps omitted)")
        for step in kept:
            lines.append(f"[assistant] -> {clip(step['call'], MAX_CALL_CHARS)}")
            if step.get("error"):
                lines.append(f"[result] error: {clip(step['output'], MAX_ERR_OUTPUT_CHARS)}")
            else:
                lines.append(f"[result] ok: {clip(step['output'], MAX_OK_OUTPUT_CHARS)}")
        last_error = bool(kept[-1].get("error"))
    lines.append("")

    lines.append("State:")
    lines.append(
        f"turn={turn}, steps_so_far={steps_so_far}, last_step_error={'true' if last_error else 'false'}"
    )
    lines.append("")

    lines.append("Tools:")
    lines.append(tools_block)
    lines.append("")
    lines.append("Decide the next action.")
    return "\n".join(lines)
