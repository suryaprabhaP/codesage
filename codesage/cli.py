"""Interactive terminal chat loop."""

import json
from pathlib import Path

import openai
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from codesage import __version__
from codesage.config import load_config
from codesage.llm import LLM
from codesage.tools import TOOLS, run_tool

console = Console()

# Safety limit so a confused model can't call tools forever.
MAX_TOOL_ROUNDS = 10

SYSTEM_PROMPT = """You are CodeSage, an AI coding assistant running in the user's terminal.
You are working in the user's project folder: {cwd}

You have tools to look at the project: list_files, read_file and search_code.
Use them whenever a question is about the user's code, instead of guessing.
search_code only shows single lines: read_file the match before explaining it
or naming a line number. All paths are relative to the project folder.

Be concise and direct. Use Markdown, and put code in fenced code blocks."""

HELP = """[bold]Commands[/bold]
  /help    show this help
  /clear   start a fresh conversation
  /exit    quit (or press Ctrl+D)
Press Ctrl+C while a reply is streaming to stop it."""


def stream_reply(llm: LLM, messages: list[dict]) -> tuple[str, list[dict]]:
    """Stream the model's reply to the screen.

    Returns the answer text and any tool calls the model asked for.
    """
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
            for kind, value in llm.stream_chat(messages, tools=TOOLS):
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


def show_tool_call(call: dict, result: str) -> None:
    """Print one line for the call and a short summary of its result."""
    try:
        args = json.loads(call["arguments"] or "{}")
        args_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    except json.JSONDecodeError:
        args_text = call["arguments"]
    console.print(f"[cyan]●[/cyan] [bold]{call['name']}[/bold]([dim]{args_text}[/dim])")

    if result.startswith("Error:"):
        console.print(f"  [red]└ {result}[/red]")
    else:
        console.print(f"  [dim]└ {len(result.splitlines())} lines[/dim]")


def run_turn(llm: LLM, messages: list[dict], root: Path) -> None:
    """Get the model's reply, running any tools it asks for, until it gives a final answer.

    This is the agent loop: model asks for a tool -> we run it -> the result goes
    back into messages -> the model is called again with that new information.
    """
    for _ in range(MAX_TOOL_ROUNDS):
        answer, tool_calls = stream_reply(llm, messages)
        if not tool_calls:
            messages.append({"role": "assistant", "content": answer})
            return

        # The assistant message must list its tool calls, and each tool result
        # must point back to its call by id, or the API rejects the history.
        messages.append({
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
            result = run_tool(root, call["name"], call["arguments"])
            show_tool_call(call, result)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

    console.print(f"[yellow]Stopped after {MAX_TOOL_ROUNDS} rounds of tool calls.[/yellow]")


def main():
    try:
        config = load_config()
    except ValueError as e:
        console.print(f"[red]Config error:[/red] {e}")
        return

    llm = LLM(config)
    console.print(
        Panel.fit(
            f"[bold cyan]CodeSage[/bold cyan] v{__version__}\n"
            f"[dim]model:[/dim] {config.model} [dim]via[/dim] {config.provider}\n"
            f"[dim]type /help for commands[/dim]",
            border_style="cyan",
        )
    )

    root = Path.cwd().resolve()
    system = {"role": "system", "content": SYSTEM_PROMPT.format(cwd=root)}
    messages = [system]
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
            messages = [system]
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        # Remember where this turn started, so a failed turn can be undone cleanly.
        turn_start = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            run_turn(llm, messages, root)
        except openai.APIConnectionError:
            console.print(
                f"[red]Can't reach {config.provider} at {config.base_url}.[/red]"
                + (" Is Ollama running? Try: [bold]ollama serve[/bold]" if config.provider == "ollama" else "")
            )
            del messages[turn_start:]
        except openai.APIError as e:
            console.print(f"[red]API error:[/red] {e}")
            del messages[turn_start:]

    console.print("[dim]Bye![/dim]")


if __name__ == "__main__":
    main()
