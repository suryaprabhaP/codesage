"""Interactive terminal front end for the agent."""

import json
from collections.abc import Iterator
from pathlib import Path

import openai
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text

from codesage import __version__
from codesage.agent import Agent
from codesage.config import load_config
from codesage.llm import LLM
from codesage.tools import REJECTED_MESSAGE, FileChange

console = Console()

# Long diffs are cut short on screen so the prompt stays visible.
MAX_DIFF_LINES = 80

HELP = """[bold]Commands[/bold]
  /help    show this help
  /clear   start a fresh conversation
  /exit    quit (or press Ctrl+D)
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

        def render():
            parts = []
            if thinking and not answer:
                # Show only the tail of the reasoning so it doesn't flood the screen.
                parts.append(Text("thinking… " + thinking[-300:].replace("\n", " "), style="dim italic"))
            if answer:
                parts.append(Markdown(answer))
            return Group(*parts)

        with Live(render(), console=console, refresh_per_second=12, transient=False) as live:
            try:
                for kind, value in events:
                    if kind == "thinking":
                        thinking += value
                    elif kind == "text":
                        answer += value
                    else:
                        tool_calls.append(value)
                    live.update(render())
            except KeyboardInterrupt:
                answer += "\n\n*[stopped]*"
                live.update(render())
                return answer, []
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
        elif result == REJECTED_MESSAGE:
            console.print("  [yellow]└ Rejected[/yellow]")
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

    def warn(self, text: str) -> None:
        console.print(f"[yellow]{text}[/yellow]")


def main():
    try:
        config = load_config()
    except ValueError as e:
        console.print(f"[red]Config error:[/red] {e}")
        return

    agent = Agent(LLM(config), Path.cwd().resolve(), TerminalUI())
    console.print(
        Panel.fit(
            f"[bold cyan]CodeSage[/bold cyan] v{__version__}\n"
            f"[dim]model:[/dim] {config.model} [dim]via[/dim] {config.provider}\n"
            f"[dim]project:[/dim] {agent.root}\n"
            f"[dim]type /help for commands[/dim]",
            border_style="cyan",
        )
    )
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

    console.print("[dim]Bye![/dim]")


if __name__ == "__main__":
    main()
