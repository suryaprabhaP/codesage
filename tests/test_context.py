import copy

from codesage.context import REMOVED_OUTPUT, RESERVED_FOR_REPLY, fit_to_context

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
MAX_TOKENS = RESERVED_FOR_REPLY + 1000  # leaves a budget of 1000 with no tools


def tool_turn(number: int, output_chars: int = 40) -> list[dict]:
    """One whole turn: question, tool call, tool result, answer."""
    call_id = f"call_{number}"
    return [
        {"role": "user", "content": f"question {number}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "x" * output_chars},
        {"role": "assistant", "content": f"answer {number}"},
    ]


def assert_tool_pairs_intact(messages: list[dict]) -> None:
    called = [c["id"] for m in messages for c in m.get("tool_calls", [])]
    answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert called == answered


def test_short_conversation_is_unchanged():
    messages = [SYSTEM, *tool_turn(1)]
    fit = fit_to_context(messages, [], MAX_TOKENS)
    assert fit.messages == messages
    assert (fit.outputs_removed, fit.messages_dropped) == (0, 0)


def test_original_history_is_never_changed():
    messages = [SYSTEM, *tool_turn(1, 8000), *tool_turn(2, 8000), *tool_turn(3, 8000)]
    before = copy.deepcopy(messages)
    fit_to_context(messages, [], MAX_TOKENS)
    assert messages == before


def test_old_outputs_are_removed_before_any_messages():
    messages = [SYSTEM, *tool_turn(1, 2400), *tool_turn(2, 1000), *tool_turn(3, 1000)]
    fit = fit_to_context(messages, [], MAX_TOKENS)
    assert fit.messages_dropped == 0
    assert fit.messages[3]["content"] == REMOVED_OUTPUT
    # The two newest outputs are still there in full.
    assert fit.messages[7]["content"] == "x" * 1000
    assert fit.messages[11]["content"] == "x" * 1000


def test_earlier_answers_outlive_old_outputs():
    messages = [SYSTEM, *tool_turn(1, 3000), *tool_turn(2, 3000), {"role": "user", "content": "next"}]
    fit = fit_to_context(messages, [], MAX_TOKENS)
    assert fit.messages_dropped == 0
    assert fit.messages[3]["content"] == REMOVED_OUTPUT
    assert fit.messages[4]["content"] == "answer 1"


def test_current_turn_outputs_are_protected_first():
    current = [m for n in (2, 3) for m in tool_turn(n, 1400)[1:3]]
    messages = [SYSTEM, *tool_turn(1, 1400), {"role": "user", "content": "go"}, *current]
    fit = fit_to_context(messages, [], MAX_TOKENS)
    assert fit.messages[3]["content"] == REMOVED_OUTPUT
    assert fit.messages[-1]["content"] == "x" * 1400
    assert fit.messages[-3]["content"] == "x" * 1400


def test_new_conversation_with_only_system_prompt():
    fit = fit_to_context([SYSTEM], [], MAX_TOKENS)
    assert fit.messages == [SYSTEM]


def test_oldest_turns_are_dropped_when_needed():
    messages = [SYSTEM]
    for n in range(100):
        messages += tool_turn(n)
    fit = fit_to_context(messages, [], MAX_TOKENS)
    assert fit.messages_dropped > 0
    assert fit.messages[1]["role"] == "user"
    assert fit.messages[-1] == messages[-1]
    assert "earlier messages were removed" in fit.messages[0]["content"]
    assert_tool_pairs_intact(fit.messages)
    assert fit.tokens <= fit.budget


def test_one_huge_turn_still_fits_and_keeps_pairs():
    messages = [SYSTEM, {"role": "user", "content": "go"}]
    for n in range(3):
        messages += tool_turn(n, 20000)[1:3]
    fit = fit_to_context(messages, [], MAX_TOKENS)
    assert fit.messages[1]["content"] == "go"
    assert_tool_pairs_intact(fit.messages)
    assert "output cut to save space" in fit.messages[-1]["content"]
    assert fit.tokens <= fit.budget


def test_tool_descriptions_use_up_budget():
    messages = [SYSTEM, *tool_turn(1, 2000), *tool_turn(2, 40), *tool_turn(3, 40)]
    without_tools = fit_to_context(messages, [], MAX_TOKENS)
    with_tools = fit_to_context(messages, [{"description": "y" * 3200}], MAX_TOKENS)
    assert without_tools.outputs_removed == 0
    assert with_tools.outputs_removed == 1
