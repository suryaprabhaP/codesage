"""Interactive terminal front end for the agent."""

import argparse
import json
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import openai
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.spinner import Spinner
from rich.text import Text

from codesage import __version__
from codesage.agent import Agent
from codesage.config import load_config
from codesage.llm import LLM
from codesage.storage import SessionStore
from codesage.tools import COMMAND_REJECTED_MESSAGE, REJECTED_MESSAGE, Command, FileChange, get_index

console = Console()

# Long diffs and command output are cut short on screen so the prompt stays visible.
MAX_DIFF_LINES = 80
MAX_OUTPUT_PREVIEW_LINES = 8

HELP = """[bold]Commands[/bold]
  /help         show this help
  /clear        start a fresh conversation
  /sessions     list saved conversations for this project
  /resume N     continue saved conversation number N
  /context      show how much of the model's context is in use
  /index        update the search index for this project now
  /exit         quit (or press Ctrl+D)
Press Ctrl+C while a reply is streaming to stop it."""


def render_diff(diff: str) -> Text:
    """Color a unified diff: additions green, removals red, positions cyan."""
    lines = diff.splitlines()
    text = Text()
    for line in lines[:MAX_DIFF_LINES]:
        if line.startswith(("+++", "---")):
            style = "bold"
        elif line.startswith("+"):
            style = "green"
        elif line.startswith("-"):
            style = "red"
        elif line.startswith("@@"):
            style = "cyan"
        else:
            style = "dim"
        text.append(line + "\n", style=style)
    if len(lines) > MAX_DIFF_LINES:
        text.append(f"... {len(lines) - MAX_DIFF_LINES} more lines\n", style="dim italic")
    return text


class TerminalUI:
    """Implements the agent's UI interface with rich and prompt_toolkit."""

    def stream_reply(self, events: Iterator[tuple[str, str | dict]]) -> tuple[str, list[dict]]:
        thinking, answer = "", ""
        tool_calls = []
        start = time.monotonic()
        done = False

        def render():
            seconds = int(time.monotonic() - start)
            if answer:
                return Markdown(answer)
            if done:
                # Don't leave a frozen spinner behind when the reply was only tool calls.
                return Text(f"thought for {seconds}s" if thinking else "", style="dim italic")
            if thinking:
                # Show only the tail of the reasoning so it doesn't flood the screen.
                tail = thinking[-300:].replace("\n", " ")
                return Group(
                    Spinner("dots", text=Text(f"Thinking… {seconds}s", style="cyan")),
                    Text(tail, style="dim italic"),
                )
            return Spinner("dots", text=Text(f"Waiting for the model… {seconds}s", style="cyan"))

        # get_renderable makes Live redraw on its own, so the timer keeps
        # ticking even while the model is silent.
        with Live(get_renderable=render, console=console, refresh_per_second=12) as live:
            try:
                for kind, value in events:
                    if kind == "thinking":
                        thinking += value
                    elif kind == "text":
                        answer += value
                    else:
                        tool_calls.append(value)
            except KeyboardInterrupt:
                answer += "\n\n*[stopped]*"
                live.refresh()
                return answer, []
            done = True
            live.refresh()

        if answer.strip():
            console.print(f"[dim]{time.monotonic() - start:.1f}s[/dim]")
        return answer, tool_calls

    def show_tool_call(self, call: dict, result: str) -> None:
        try:
            args = json.loads(call["arguments"] or "{}")
            # File contents are shown in the diff, so keep this line short.
            args_text = ", ".join(
                f"{k}={v!r}" for k, v in args.items() if k not in ("content", "old_text", "new_text")
            )
        except (json.JSONDecodeError, AttributeError):
            args_text = call["arguments"]
        console.print(f"[cyan]●[/cyan] [bold]{call['name']}[/bold]([dim]{args_text}[/dim])")

        if result.startswith("Error:"):
            console.print(f"  [red]└ {result}[/red]")
        elif result in (REJECTED_MESSAGE, COMMAND_REJECTED_MESSAGE):
            console.print("  [yellow]└ Rejected[/yellow]")
        elif call["name"] == "run_command":
            exit_line, _, output = result.partition("\n")
            color = "green" if exit_line == "Exit code: 0" else "red"
            console.print(f"  [{color}]└ {exit_line}[/{color}]")
            lines = output.splitlines()
            for line in lines[:MAX_OUTPUT_PREVIEW_LINES]:
                console.print(Text(f"    {line}", style="dim"))
            if len(lines) > MAX_OUTPUT_PREVIEW_LINES:
                console.print(f"    [dim]... {len(lines) - MAX_OUTPUT_PREVIEW_LINES} more lines[/dim]")
        elif call["name"] in ("write_file", "edit_file"):
            console.print(f"  [green]└ {result}[/green]")
        else:
            console.print(f"  [dim]└ {len(result.splitlines())} lines[/dim]")

    def confirm_change(self, change: FileChange) -> bool:
        action = "Create" if change.old is None else "Edit"
        console.print(
            Panel(render_diff(change.diff()), title=f"{action} {change.rel_path}", border_style="yellow")
        )
        try:
            return Confirm.ask("Apply this change?", default=False, console=console)
        except (KeyboardInterrupt, EOFError):
            return False

    def confirm_command(self, command: Command) -> str:
        console.print(
            Panel(
                Text(f"$ {command.command}", style="bold"),
                title="Run command",
                subtitle=f"in {command.cwd} · timeout {command.timeout}s",
                border_style="yellow",
            )
        )
        try:
            answer = Prompt.ask(
                "Run it? [bold]y[/bold]es / [bold]a[/bold]lways this session / [bold]n[/bold]o",
                choices=["y", "a", "n"],
                default="n",
                show_choices=False,
                console=console,
            )
        except (KeyboardInterrupt, EOFError):
            return "no"
        return {"y": "yes", "a": "always"}.get(answer, "no")

    def warn(self, text: str) -> None:
        console.print(f"[yellow]{text}[/yellow]")


def show_sessions(agent: Agent) -> None:
    sessions = agent.store.list_sessions(agent.root) if agent.store else []
    if not sessions:
        console.print("[dim]No saved conversations for this project yet.[/dim]")
        return
    for s in sessions:
        current = " [cyan](current)[/cyan]" if s.id == agent.session_id else ""
        console.print(
            f"  [bold]{s.id}[/bold]  {escape(s.title)}  "
            f"[dim]{s.updated_at} · {s.message_count} messages[/dim]{current}"
        )
    console.print("[dim]Continue one with /resume N[/dim]")


def resume_session(agent: Agent, session_id: int) -> None:
    if agent.resume(session_id):
        console.print(f"[dim]Resumed conversation {session_id} ({len(agent.messages) - 1} messages).[/dim]")
    else:
        console.print(f"[red]No saved conversation {session_id} for this project.[/red] Try /sessions.")


def show_context(agent: Agent) -> None:
    fit = agent.context_usage()
    console.print(
        f"[dim]About {fit.tokens:,} of {fit.budget:,} available tokens in use "
        f"({agent.max_context_tokens:,} total, minus tool descriptions and room for the reply). "
        f"{len(agent.messages) - 1} messages; {fit.outputs_removed} old outputs removed "
        f"and {fit.messages_dropped} messages hidden from the model.[/dim]"
    )


def update_index(agent: Agent) -> None:
    """Index now, so the agent's first meaning search doesn't wait for it."""
    try:
        with console.status("Indexing the project…"):
            stats = get_index(agent.root).update()
    except Exception as e:  # missing library, model download failure, corrupt index...
        console.print(f"[red]Could not update the index:[/red] {escape(str(e))}")
        return
    console.print(
        f"[dim]{stats.total_chunks} pieces of code indexed: {stats.files_embedded} files embedded, "
        f"{stats.files_unchanged} unchanged, {stats.files_removed} removed.[/dim]"
    )


def main():
    parser = argparse.ArgumentParser(prog="codesage", description="A terminal AI coding agent.")
    parser.add_argument(
        "-c", "--continue", dest="resume_latest", action="store_true",
        help="continue the latest saved conversation for this project",
    )
    args = parser.parse_args()

    try:
        config = load_config()
    except ValueError as e:
        console.print(f"[red]Config error:[/red] {e}")
        return

    root = Path.cwd().resolve()
    store = None
    try:
        store = SessionStore()
    except (sqlite3.Error, OSError) as e:
        console.print(f"[yellow]Conversations won't be saved: {e}[/yellow]")

    agent = Agent(LLM(config), root, TerminalUI(), store, config.context_tokens)
    console.print(
        Panel.fit(
            f"[bold cyan]CodeSage[/bold cyan] v{__version__}\n"
            f"[dim]model:[/dim] {config.model} [dim]via[/dim] {config.provider}\n"
            f"[dim]project:[/dim] {agent.root}\n"
            f"[dim]type /help for commands[/dim]",
            border_style="cyan",
        )
    )
    if args.resume_latest:
        latest = store.list_sessions(root, limit=1) if store else []
        if latest:
            resume_session(agent, latest[0].id)
        else:
            console.print("[dim]No saved conversation to continue; starting a new one.[/dim]")

    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            user_input = session.prompt("\n› ").strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/help":
            console.print(HELP)
            continue
        if user_input == "/clear":
            agent.reset()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if user_input == "/sessions":
            show_sessions(agent)
            continue
        if user_input == "/context":
            show_context(agent)
            continue
        if user_input == "/index":
            update_index(agent)
            continue
        if user_input.startswith("/resume"):
            number = user_input.removeprefix("/resume").strip()
            if number.isdigit():
                resume_session(agent, int(number))
            else:
                console.print("Usage: /resume N  (see /sessions for the numbers)")
            continue

        try:
            agent.ask(user_input)
        except openai.APIConnectionError:
            console.print(
                f"[red]Can't reach {config.provider} at {config.base_url}.[/red]"
                + (" Is Ollama running? Try: [bold]ollama serve[/bold]" if config.provider == "ollama" else "")
            )
        except openai.APIError as e:
            console.print(f"[red]API error:[/red] {e}")
        except KeyboardInterrupt:
            console.print("[dim]Stopped.[/dim]")

    if store:
        store.close()
    console.print("[dim]Bye![/dim]")


if __name__ == "__main__":
    main()
