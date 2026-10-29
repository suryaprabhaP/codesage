"""Tools the model can call to look at the user's project.

Each tool is a plain Python function plus a JSON schema (TOOLS) that tells
the model what the tool does and which arguments it takes.
"""

import json
import os
import re
from pathlib import Path

IGNORED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "dist", "build",
}
MAX_LIST_ENTRIES = 200
MAX_READ_LINES = 400
MAX_SEARCH_RESULTS = 50
MAX_FILE_BYTES = 1_000_000


class ToolError(Exception):
    """A problem the model should be told about, e.g. a missing file."""


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
]

_FUNCTIONS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
}


def run_tool(root: Path, name: str, arguments: str | dict) -> str:
    """Run one tool call from the model. Always returns text and never raises,
    so a mistake by the model becomes an error message it can react to."""
    func = _FUNCTIONS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'."

    if isinstance(arguments, dict):
        args = arguments
    else:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return f"Error: arguments are not valid JSON: {arguments}"

    try:
        return func(root, **args)
    except ToolError as e:
        return f"Error: {e}"
    except TypeError as e:
        return f"Error: wrong arguments for {name}: {e}"
