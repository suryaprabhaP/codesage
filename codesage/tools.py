"""Tools the model can call to look at and change the user's project.

Each tool is a plain Python function plus a JSON schema (TOOLS) that tells
the model what the tool does and which arguments it takes.

Read tools return text. Edit tools return a FileChange and run_command
returns a Command: nothing is written or run until the user says yes
(see run_tool).
"""

import difflib
import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build",
}
MAX_LIST_ENTRIES = 200
MAX_READ_LINES = 400
MAX_SEARCH_RESULTS = 50
MAX_FILE_BYTES = 1_000_000
MAX_OUTPUT_CHARS = 10_000
DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600

REJECTED_MESSAGE = "The user rejected this change, so nothing was written. Ask them what they would like instead."
COMMAND_REJECTED_MESSAGE = "The user did not allow this command, so it was not run. Ask them what they would like instead."

# Windows runs commands in PowerShell, everything else in sh.
SHELL_NAME = "Windows PowerShell" if os.name == "nt" else "sh"

# Refused outright, even if the user would say yes: easy to approve by
# accident and impossible to undo. The yes/no prompt is the real safety;
# this list is a second line of defence, not a complete one.
BLOCKED_COMMANDS = [
    r"\brm\s+-\w*\s+(/|~|\*)(\s|$)",           # rm -rf /  rm -rf ~  rm -rf *
    r"\bremove-item\b.*\s(/|~|\*|[a-z]:\\?)(\s|$)",
    r"\bdel\s+/[sq]\b",
    r"\b(format|mkfs|diskpart)\b",
    r"\b(shutdown|reboot|restart-computer|stop-computer)\b",
    r"\bgit\s+push\b.*(--force|-f\b)",
    r"\b(curl|wget|iwr|invoke-webrequest)\b.*\|\s*(sh|bash|iex|invoke-expression)\b",
]


class ToolError(Exception):
    """A problem the model should be told about, e.g. a missing file."""


@dataclass
class FileChange:
    """A proposed change to one file, shown to the user before anything is written."""

    path: Path
    rel_path: str
    old: str | None  # None means the file does not exist yet
    new: str

    def diff(self) -> str:
        lines = difflib.unified_diff(
            (self.old or "").splitlines(),
            self.new.splitlines(),
            fromfile="/dev/null" if self.old is None else f"a/{self.rel_path}",
            tofile=f"b/{self.rel_path}",
            lineterm="",
        )
        return "\n".join(lines)


@dataclass
class Command:
    """A shell command the model wants to run, shown to the user before it runs."""

    command: str
    cwd: Path
    timeout: int


def _resolve(root: Path, path: str) -> Path:
    """Turn a project-relative path into an absolute one, refusing anything outside root."""
    full = (root / path).resolve()
    if full != root and root not in full.parents:
        raise ToolError(f"'{path}' is outside the project folder.")
    return full


def _walk(base: Path):
    """Yield every file under base, skipping ignored folders."""
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(
            d for d in dirnames if d not in IGNORED_DIRS and not d.endswith(".egg-info")
        )
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _read_text(path: Path) -> str | None:
    """Return a file's text, or None if it is too big or not text."""
    if path.stat().st_size > MAX_FILE_BYTES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


# ---- Read tools ----------------------------------------------------------

def list_files(root: Path, path: str = ".") -> str:
    base = _resolve(root, path)
    if not base.is_dir():
        raise ToolError(f"'{path}' is not a folder.")

    files = []
    for f in _walk(base):
        if len(files) == MAX_LIST_ENTRIES:
            files.append(f"... (stopped at {MAX_LIST_ENTRIES} files)")
            break
        files.append(f.relative_to(root).as_posix())
    return "\n".join(files) or "(no files)"


def read_file(root: Path, path: str, start_line: int = 1, end_line: int | None = None) -> str:
    full = _resolve(root, path)
    if not full.is_file():
        raise ToolError(f"'{path}' is not a file.")
    text = _read_text(full)
    if text is None:
        raise ToolError(f"'{path}' is not a text file or is larger than {MAX_FILE_BYTES} bytes.")

    lines = text.splitlines()
    start = max(start_line, 1)
    end = min(end_line or len(lines), len(lines), start + MAX_READ_LINES - 1)
    if start > end:
        return f"(no lines in that range; the file has {len(lines)} lines)"

    # Line numbers let the model refer to exact places in the file.
    numbered = [f"{i:>4}  {lines[i - 1]}" for i in range(start, end + 1)]
    if end < len(lines):
        numbered.append(f"... ({len(lines)} lines total; use start_line to read more)")
    return "\n".join(numbered)


def search_code(root: Path, pattern: str, path: str = ".") -> str:
    base = _resolve(root, path)
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # Small models often send plain text with regex symbols in it.
        regex = re.compile(re.escape(pattern), re.IGNORECASE)

    files = [base] if base.is_file() else _walk(base)
    matches = []
    for f in files:
        text = _read_text(f)
        if text is None:
            continue
        for n, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{f.relative_to(root).as_posix()}:{n}: {line.strip()}")
                if len(matches) == MAX_SEARCH_RESULTS:
                    matches.append(f"... (stopped at {MAX_SEARCH_RESULTS} matches)")
                    return "\n".join(matches)
    return "\n".join(matches) or f"No matches for '{pattern}'."


# ---- Edit tools (return a FileChange; nothing is written here) -----------

def write_file(root: Path, path: str, content: str) -> FileChange:
    full = _resolve(root, path)
    if full.is_dir():
        raise ToolError(f"'{path}' is a folder.")
    old = None
    if full.exists():
        old = _read_text(full)
        if old is None:
            raise ToolError(f"'{path}' is not a text file, so it can't be overwritten.")
    return FileChange(full, full.relative_to(root).as_posix(), old, content)


def edit_file(root: Path, path: str, old_text: str, new_text: str) -> FileChange:
    full = _resolve(root, path)
    if not full.is_file():
        raise ToolError(f"'{path}' is not a file. Use write_file to create a new file.")
    text = _read_text(full)
    if text is None:
        raise ToolError(f"'{path}' is not a text file.")
    if not old_text:
        raise ToolError("old_text must not be empty.")

    # Requiring exactly one match means the edit can never land in the wrong place.
    count = text.count(old_text)
    if count == 0:
        raise ToolError(
            "old_text was not found. Read the file and copy the text exactly, including indentation."
        )
    if count > 1:
        raise ToolError(
            f"old_text appears {count} times. Include more surrounding lines so it matches once."
        )
    return FileChange(full, full.relative_to(root).as_posix(), text, text.replace(old_text, new_text, 1))


def apply_change(change: FileChange) -> None:
    # Keep Windows line endings in files that already use them.
    newline = "\n"
    if change.old is not None and b"\r\n" in change.path.read_bytes():
        newline = "\r\n"
    change.path.parent.mkdir(parents=True, exist_ok=True)
    with open(change.path, "w", encoding="utf-8", newline=newline) as f:
        f.write(change.new)


# ---- run_command (returns a Command; nothing is run here) ----------------

def run_command(root: Path, command: str, timeout: int = DEFAULT_TIMEOUT) -> Command:
    command = command.strip()
    if not command:
        raise ToolError("command must not be empty.")
    for pattern in BLOCKED_COMMANDS:
        if re.search(pattern, command, re.IGNORECASE):
            raise ToolError(f"'{command}' is blocked for safety and will not be run.")
    return Command(command, root, max(1, min(int(timeout), MAX_TIMEOUT)))


def _truncate(text: str) -> str:
    """Keep the start and end of long output; the middle is usually least useful."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return f"{text[:half]}\n... ({len(text) - MAX_OUTPUT_CHARS} characters cut) ...\n{text[-half:]}"


def execute(cmd: Command) -> str:
    if os.name == "nt":
        # Make PowerShell send UTF-8 so non-English output isn't garbled, and pass
        # on the program's real exit code (PowerShell otherwise reports 1 for any failure).
        script = (
            f"[Console]::OutputEncoding = [Text.Encoding]::UTF8; {cmd.command}\n"
            "if ($LASTEXITCODE) { exit $LASTEXITCODE } elseif (-not $?) { exit 1 }"
        )
        args = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    else:
        args = ["/bin/sh", "-c", cmd.command]

    try:
        proc = subprocess.run(
            args,
            cwd=cmd.cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=cmd.timeout,
            stdin=subprocess.DEVNULL,  # a command waiting for input would hang forever
        )
    except subprocess.TimeoutExpired:
        return f"Error: the command did not finish within {cmd.timeout} seconds and was stopped."
    except OSError as e:
        return f"Error: could not start the command: {e}"

    output = (proc.stdout + proc.stderr).strip()
    return f"Exit code: {proc.returncode}\n{_truncate(output) or '(no output)'}"


# ---- Schemas the model reads ---------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the project (recursively). Use this first to see what exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder relative to the project root. Default: '.'"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file. Returns the content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the project root."},
                    "start_line": {"type": "integer", "description": "First line to read (1-based). Default: 1"},
                    "end_line": {"type": "integer", "description": "Last line to read. Default: end of file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search all project files for a regex or text (case-insensitive). Returns file:line: text matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex or plain text to search for."},
                    "path": {"type": "string", "description": "Folder or file to search in. Default: '.'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file, or replace a whole file. The user must approve it. Prefer edit_file for small changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the project root."},
                    "content": {"type": "string", "description": "The complete new content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace one exact piece of text in an existing file. old_text must match the file "
                "exactly (including indentation) and appear only once. The user must approve it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the project root."},
                    "old_text": {"type": "string", "description": "The exact text to replace, copied from the file."},
                    "new_text": {"type": "string", "description": "The text to put in its place."},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                f"Run a shell command ({SHELL_NAME}) in the project folder and return its output "
                "and exit code, e.g. to run tests or a script. The user must approve it. "
                "Do not start servers or interactive programs: they will time out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run."},
                    "timeout": {
                        "type": "integer",
                        "description": f"Seconds before the command is stopped. Default {DEFAULT_TIMEOUT}, max {MAX_TIMEOUT}.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

_READ_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
}
_EDIT_TOOLS = {
    "write_file": write_file,
    "edit_file": edit_file,
}


def run_tool(
    root: Path,
    name: str,
    arguments: str | dict,
    confirm_change: Callable[[FileChange], bool] | None = None,
    confirm_command: Callable[[Command], bool] | None = None,
) -> str:
    """Run one tool call from the model. Always returns text and never raises,
    so a mistake by the model becomes an error message it can react to.

    Edit tools only write after confirm_change(change) returns True, and
    run_command only runs after confirm_command(command) returns True.
    """
    if name not in _READ_TOOLS and name not in _EDIT_TOOLS and name != "run_command":
        return f"Error: unknown tool '{name}'."

    if isinstance(arguments, dict):
        args = arguments
    else:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return f"Error: arguments are not valid JSON: {arguments}"

    try:
        if name in _READ_TOOLS:
            return _READ_TOOLS[name](root, **args)

        if name == "run_command":
            cmd = run_command(root, **args)
            if confirm_command is None or not confirm_command(cmd):
                return COMMAND_REJECTED_MESSAGE
            return execute(cmd)

        change = _EDIT_TOOLS[name](root, **args)
        if change.old == change.new:
            return f"No change: {change.rel_path} already has that content."
        if confirm_change is None or not confirm_change(change):
            return REJECTED_MESSAGE
        apply_change(change)
        return f"{'Created' if change.old is None else 'Updated'} {change.rel_path}."
    except ToolError as e:
        return f"Error: {e}"
    except (TypeError, ValueError) as e:
        return f"Error: wrong arguments for {name}: {e}"
    except OSError as e:
        return f"Error: could not write the file: {e}"
