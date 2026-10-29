"""The agent loop: talk to the model, run the tools it asks for, repeat.

The agent knows nothing about the terminal. It talks to the screen through a
small UI interface, so tests can drive it with a fake model and a fake UI.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from codesage.llm import LLM
from codesage.tools import TOOLS, FileChange, run_tool

# Safety limit so a confused model can't call tools forever.
MAX_TOOL_ROUNDS = 10

SYSTEM_PROMPT = """You are CodeSage, an AI coding assistant running in the user's terminal.
You are working in the user's project folder: {cwd}

Tools for looking at the project: list_files, read_file, search_code.
Use them whenever a question is about the user's code, instead of guessing.
search_code only shows single lines: read_file the match before explaining it
or naming a line number.

Tools for changing the project: edit_file and write_file.
- Always read_file before editing, then use edit_file with old_text copied exactly.
- Use write_file only for new files or when rewriting a whole file.
- The user sees a diff and must approve every change. If they reject it, ask
  what they would like instead; do not retry the same change.

All paths are relative to the project folder.
Be concise and direct. Use Markdown, and put code in fenced code blocks."""


class UI(Protocol):
    """What the agent needs from the user interface."""

    def stream_reply(self, events: Iterator[tuple[str, str | dict]]) -> tuple[str, list[dict]]:
        """Show a streaming reply; return the answer text and any tool calls."""

    def show_tool_call(self, call: dict, result: str) -> None:
        """Show that a tool ran and what it returned."""

    def confirm_change(self, change: FileChange) -> bool:
        """Show a proposed file change and ask the user to approve it."""

    def warn(self, text: str) -> None:
        """Show a warning."""


class Agent:
    def __init__(self, llm: LLM, root: Path, ui: UI):
        self.llm = llm
        self.root = root
        self.ui = ui
        self.reset()

    def reset(self) -> None:
        """Start a fresh conversation."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT.format(cwd=self.root)}]

    def ask(self, user_input: str) -> None:
        """Handle one user message. If anything fails, the history is restored."""
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            self._run_tool_loop()
        except BaseException:
            del self.messages[turn_start:]
            raise

    def _run_tool_loop(self) -> None:
        """Model asks for a tool -> we run it -> the result goes back into
        messages -> the model is called again, until it gives a final answer."""
        for _ in range(MAX_TOOL_ROUNDS):
            events = self.llm.stream_chat(self.messages, tools=TOOLS)
            answer, tool_calls = self.ui.stream_reply(events)
            if not tool_calls:
                self.messages.append({"role": "assistant", "content": answer})
                return

            # The assistant message must list its tool calls, and each tool result
            # must point back to its call by id, or the API rejects the history.
            self.messages.append({
                "role": "assistant",
                "content": answer,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in tool_calls
                ],
            })
            for call in tool_calls:
                result = run_tool(
                    self.root, call["name"], call["arguments"], confirm=self.ui.confirm_change
                )
                self.ui.show_tool_call(call, result)
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        self.ui.warn(f"Stopped after {MAX_TOOL_ROUNDS} rounds of tool calls.")
