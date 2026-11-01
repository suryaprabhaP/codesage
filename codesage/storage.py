"""Saves conversations in a SQLite database so they can be resumed later.

The database lives in the user's home folder, not in the project, so
conversations never end up in the project's git repository.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".codesage" / "sessions.db"
MAX_TITLE_CHARS = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    project    TEXT NOT NULL,
    title      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    data       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project, updated_at);
"""


@dataclass
class SessionInfo:
    id: int
    title: str
    updated_at: str
    message_count: int


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


class SessionStore:
    def __init__(self, path: Path = DEFAULT_DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def create_session(self, project: Path, title: str) -> int:
        short_title = " ".join(title.split())[:MAX_TITLE_CHARS]
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO sessions (project, title, updated_at) VALUES (?, ?, ?)",
                (str(project), short_title, _now()),
            )
        return cursor.lastrowid

    def add_messages(self, session_id: int, messages: list[dict]) -> None:
        # "with conn" is a transaction: either all messages are saved, or none.
        # Placeholders (?) keep message text from being treated as SQL.
        with self.conn:
            self.conn.executemany(
                "INSERT INTO messages (session_id, data) VALUES (?, ?)",
                [(session_id, json.dumps(m, ensure_ascii=False)) for m in messages],
            )
            self.conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
            )

    def list_sessions(self, project: Path, limit: int = 10) -> list[SessionInfo]:
        """The project's most recent sessions, newest first."""
        rows = self.conn.execute(
            """
            SELECT s.id, s.title, s.updated_at, COUNT(m.id)
            FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
            WHERE s.project = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC, s.id DESC
            LIMIT ?
            """,
            (str(project), limit),
        ).fetchall()
        return [SessionInfo(*row) for row in rows]

    def load_messages(self, project: Path, session_id: int) -> list[dict] | None:
        """A session's messages, or None if it doesn't belong to this project."""
        found = self.conn.execute(
            "SELECT 1 FROM sessions WHERE id = ? AND project = ?", (session_id, str(project))
        ).fetchone()
        if found is None:
            return None
        rows = self.conn.execute(
            "SELECT data FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def close(self) -> None:
        self.conn.close()
