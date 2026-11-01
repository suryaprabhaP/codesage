"""The agent loop: talk to the model, run the tools it asks for, repeat.

The agent knows nothing about the terminal. It talks to the screen through a
small UI interface, so tests can drive it with a fake model and a fake UI.
"""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from codesage.config import DEFAULT_CONTEXT_TOKENS
from codesage.context import FitResult, fit_to_context
from codesage.llm import LLM
from codesage.storage import SessionStore
from codesage.tools import SHELL_NAME, TOOLS, Command, FileChange, run_tool

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

Tool for running commands: run_command (e.g. run tests or a script).
- Commands run in {shell} inside the project folder.{shell_note}
- The user must approve each command. Never start servers or interactive
  programs; they will time out.

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

    def confirm_command(self, command: Command) -> str:
        """Show a command and ask: "yes" (once), "always" (this session) or "no"."""

    def warn(self, text: str) -> None:
        """Show a warning."""


class Agent:
    def __init__(
        self,
        llm: LLM,
        root: Path,
        ui: UI,
        store: SessionStore | None = None,
        max_context_tokens: int = DEFAULT_CONTEXT_TOKENS,
    ):
        self.llm = llm
        self.root = root
        self.ui = ui
        self.store = store
        self.max_context_tokens = max_context_tokens
        # Commands the user chose "always" for. Kept until CodeSage exits,
        # even across /clear, because it is a choice about the user's machine.
        # Never saved to disk: a resumed session must ask again.
        self.allowed_commands: set[str] = set()
        self.reset()

    def reset(self) -> None:
        """Start a fresh conversation."""
        shell_note = " Use ';' between commands, not '&&'." if SHELL_NAME == "Windows PowerShell" else ""
        prompt = SYSTEM_PROMPT.format(cwd=self.root, shell=SHELL_NAME, shell_note=shell_note)
        self.messages = [{"role": "system", "content": prompt}]
        self.session_id: int | None = None
        self._saved_count = 1  # the system prompt is rebuilt, never saved
        self._dropped_warned = 0

    def resume(self, session_id: int) -> bool:
        """Continue a saved conversation. False if this project has no such session."""
        saved = self.store.load_messages(self.root, session_id) if self.store else None
        if saved is None:
            return False
        self.reset()
        self.messages.extend(saved)
        self.session_id = session_id
        self._saved_count = len(self.messages)
        return True

    def context_usage(self) -> FitResult:
        """What the next request would send to the model."""
        return fit_to_context(self.messages, TOOLS, self.max_context_tokens)

    def _save(self) -> None:
        """Save the new messages. A database problem must not lose the user's answer."""
        if self.store is None:
            return
        try:
            if self.session_id is None:
                self.session_id = self.store.create_session(self.root, self.messages[1]["content"])
            self.store.add_messages(self.session_id, self.messages[self._saved_count:])
            self._saved_count = len(self.messages)
        except sqlite3.Error as e:
            self.ui.warn(f"Could not save this conversation: {e}")

    def confirm_command(self, command: Command) -> bool:
        """Ask the user about a command, unless they already chose "always" for it."""
        if command.command in self.allowed_commands:
            return True
        answer = self.ui.confirm_command(command)
        if answer == "always":
            self.allowed_commands.add(command.command)
        return answer in ("yes", "always")

    def ask(self, user_input: str) -> None:
        """Handle one user message. If anything fails, the history is restored."""
        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": user_input})
        try:
            self._run_tool_loop()
        except BaseException:
            del self.messages[turn_start:]
            raise
        self._save()

    def _run_tool_loop(self) -> None:
        """Model asks for a tool -> we run it -> the result goes back into
        messages -> the model is called again, until it gives a final answer."""
        for _ in range(MAX_TOOL_ROUNDS):
            fit = self.context_usage()
            if fit.messages_dropped > self._dropped_warned:
                self._dropped_warned = fit.messages_dropped
                self.ui.warn(
                    f"This conversation is long: the model no longer sees the oldest "
                    f"{fit.messages_dropped} messages. They are still saved."
                )
            events = self.llm.stream_chat(fit.messages, tools=TOOLS)
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
                    self.root,
                    call["name"],
                    call["arguments"],
                    confirm_change=self.ui.confirm_change,
                    confirm_command=self.confirm_command,
                )
                self.ui.show_tool_call(call, result)
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        self.ui.warn(f"Stopped after {MAX_TOOL_ROUNDS} rounds of tool calls.")
