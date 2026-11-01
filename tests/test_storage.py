from pathlib import Path

import pytest

from codesage.storage import MAX_TITLE_CHARS, SessionStore

PROJECT = Path("/projects/demo")
OTHER_PROJECT = Path("/projects/other")


@pytest.fixture
def store(tmp_path: Path):
    s = SessionStore(tmp_path / "sub" / "sessions.db")
    yield s
    s.close()


def test_messages_come_back_identical_and_in_order(store):
    messages = [
        {"role": "user", "content": "héllo 世界"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "x", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "result"},
    ]
    session_id = store.create_session(PROJECT, "first")
    store.add_messages(session_id, messages[:1])
    store.add_messages(session_id, messages[1:])
    assert store.load_messages(PROJECT, session_id) == messages


def test_data_survives_reopening(tmp_path):
    path = tmp_path / "sessions.db"
    first = SessionStore(path)
    session_id = first.create_session(PROJECT, "keep me")
    first.add_messages(session_id, [{"role": "user", "content": "hi"}])
    first.close()

    second = SessionStore(path)
    assert second.load_messages(PROJECT, session_id) == [{"role": "user", "content": "hi"}]
    second.close()


def test_sessions_are_separate_per_project(store):
    session_id = store.create_session(PROJECT, "mine")
    assert store.list_sessions(OTHER_PROJECT) == []
    assert store.load_messages(OTHER_PROJECT, session_id) is None


def test_list_shows_newest_first_with_counts(store):
    older = store.create_session(PROJECT, "older")
    newer = store.create_session(PROJECT, "newer")
    store.add_messages(newer, [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    sessions = store.list_sessions(PROJECT)
    assert [s.id for s in sessions] == [newer, older]
    assert [s.message_count for s in sessions] == [2, 0]


def test_title_is_shortened_to_one_line(store):
    store.create_session(PROJECT, "line one\nline two " + "x" * 200)
    title = store.list_sessions(PROJECT)[0].title
    assert title.startswith("line one line two")
    assert len(title) == MAX_TITLE_CHARS


def test_sql_in_a_message_is_just_text(store):
    evil = "'); DROP TABLE messages; --"
    session_id = store.create_session(PROJECT, evil)
    store.add_messages(session_id, [{"role": "user", "content": evil}])
    assert store.load_messages(PROJECT, session_id)[0]["content"] == evil
