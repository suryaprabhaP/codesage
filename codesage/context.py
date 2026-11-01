"""Keeps what is sent to the model inside its context window.

The full history stays in the agent and the database. fit_to_context only
builds a shorter copy for one request, so nothing is lost for good.
"""

import json
from dataclasses import dataclass

# Room left for the model's reasoning and reply.
RESERVED_FOR_REPLY = 2048
# The current turn's newest tool results are the ones the model still needs.
KEEP_RECENT_TOOL_RESULTS = 2
CHARS_PER_TOKEN = 4
REMOVED_OUTPUT = "[Old output removed to save space. Run the tool again if you need it.]"
OUTPUT_CUT = "\n... (output cut to save space)"


@dataclass
class FitResult:
    messages: list[dict]
    tokens: int
    budget: int
    outputs_removed: int = 0
    messages_dropped: int = 0


def estimate_tokens(value) -> int:
    """A quick estimate: about 4 characters per token, with no tokenizer needed."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return len(text) // CHARS_PER_TOKEN


def _shrink_tool_results(messages: list[dict], protected: int, over: int) -> tuple[int, int]:
    """Replace the oldest tool outputs until `over` tokens are saved, keeping the
    newest `protected` ones of the current turn. The messages themselves stay,
    because every tool call must keep its matching result or the API rejects
    the history."""
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    tool_positions = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    kept = set([i for i in tool_positions if i > last_user][-protected:])
    removed = 0
    for i in tool_positions:
        if i in kept:
            continue
        if over <= 0:
            break
        saved = estimate_tokens(messages[i]["content"]) - estimate_tokens(REMOVED_OUTPUT)
        if saved > 0:
            messages[i] = {**messages[i], "content": REMOVED_OUTPUT}
            over -= saved
            removed += 1
    return removed, over


def fit_to_context(messages: list[dict], tools: list[dict], max_tokens: int) -> FitResult:
    """Return a copy of messages that fits in max_tokens, in order of least harm:

    1. Remove old tool outputs (they can be fetched again; earlier answers can't).
    2. Drop the oldest whole turns (a user message and everything up to the next one).
    3. Remove the current turn's other tool outputs, except the newest.
    4. Cut the newest tool output to whatever room is left.

    The system prompt and the latest user message are always kept.
    """
    budget = max_tokens - RESERVED_FOR_REPLY - estimate_tokens(tools)
    system, rest = messages[0], list(messages[1:])

    def over_budget() -> int:
        return estimate_tokens(system) + estimate_tokens(rest) - budget

    outputs_removed = dropped = 0
    over = over_budget()
    if over > 0:
        outputs_removed, over = _shrink_tool_results(rest, KEEP_RECENT_TOOL_RESULTS, over)

    while over > 0:
        user_positions = [i for i, m in enumerate(rest) if m["role"] == "user"]
        if len(user_positions) < 2:
            break
        # Whole turns only: splitting one could separate a tool call from its result.
        dropped += user_positions[1]
        del rest[: user_positions[1]]
        over = over_budget()

    if over > 0:
        removed, over = _shrink_tool_results(rest, 1, over)
        outputs_removed += removed

    if over > 0:
        tool_positions = [i for i, m in enumerate(rest) if m["role"] == "tool"]
        if tool_positions:
            newest = rest[tool_positions[-1]]
            # Leave room for the note itself and for rounding in the estimate.
            room = estimate_tokens(newest["content"]) - over - estimate_tokens(OUTPUT_CUT) - 8
            rest[tool_positions[-1]] = {
                **newest,
                "content": newest["content"][: max(0, room) * CHARS_PER_TOKEN] + OUTPUT_CUT,
            }

    if dropped:
        note = f"\n\nNote: {dropped} earlier messages were removed to save space."
        system = {**system, "content": system["content"] + note}

    fitted = [system, *rest]
    return FitResult(fitted, estimate_tokens(fitted), budget, outputs_removed, dropped)
