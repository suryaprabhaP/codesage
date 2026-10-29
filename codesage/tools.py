"""Tools the model can call to look at and change the user's project.

Each tool is a plain Python function plus a JSON schema (TOOLS) that tells
the model what the tool does and which arguments it takes.

Read tools return text. Edit tools return a FileChange: nothing is written
until the user has seen the diff and said yes (see run_tool).
"""

import difflib
import json
import os
import re
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

REJECTED_MESSAGE = "The user rejected this change, so nothing was written. Ask them what they would like instead."


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
    confirm: Callable[[FileChange], bool] | None = None,
) -> str:
    """Run one tool call from the model. Always returns text and never raises,
    so a mistake by the model becomes an error message it can react to.

    Edit tools only write after confirm(change) returns True.
    """
    if name not in _READ_TOOLS and name not in _EDIT_TOOLS:
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

        change = _EDIT_TOOLS[name](root, **args)
        if change.old == change.new:
            return f"No change: {change.rel_path} already has that content."
        if confirm is None or not confirm(change):
            return REJECTED_MESSAGE
        apply_change(change)
        return f"{'Created' if change.old is None else 'Updated'} {change.rel_path}."
    except ToolError as e:
        return f"Error: {e}"
    except TypeError as e:
        return f"Error: wrong arguments for {name}: {e}"
    except OSError as e:
        return f"Error: could not write the file: {e}"
