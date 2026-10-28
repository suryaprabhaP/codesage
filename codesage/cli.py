"""Interactive terminal chat loop."""

import os

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

console = Console()

SYSTEM_PROMPT = """You are CodeSage, an AI coding assistant running in the user's terminal.
The user's current working directory is: {cwd}
Be concise and direct. Use Markdown, and put code in fenced code blocks."""

HELP = """[bold]Commands[/bold]
  /help    show this help
  /clear   start a fresh conversation
  /exit    quit (or press Ctrl+D)
Press Ctrl+C while a reply is streaming to stop it."""


def stream_reply(llm: LLM, messages: list[dict]) -> str:
    """Stream the model's reply to the screen and return the full answer text."""
    thinking, answer = "", ""

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
            for kind, chunk in llm.stream_chat(messages):
                if kind == "thinking":
                    thinking += chunk
                else:
                    answer += chunk
                live.update(render())
        except KeyboardInterrupt:
            answer += "\n\n*[stopped]*"
            live.update(render())
    return answer


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

    system = {"role": "system", "content": SYSTEM_PROMPT.format(cwd=os.getcwd())}
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

        messages.append({"role": "user", "content": user_input})
        try:
            reply = stream_reply(llm, messages)
        except openai.APIConnectionError:
            console.print(
                f"[red]Can't reach {config.provider} at {config.base_url}.[/red]"
                + (" Is Ollama running? Try: [bold]ollama serve[/bold]" if config.provider == "ollama" else "")
            )
            messages.pop()
            continue
        except openai.APIError as e:
            console.print(f"[red]API error:[/red] {e}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})

    console.print("[dim]Bye![/dim]")


if __name__ == "__main__":
    main()
