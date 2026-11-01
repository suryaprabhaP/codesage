"""RAG tests with a small fake embedding, so they need no model download."""

import hashlib
import re
from pathlib import Path

import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

from codesage.rag import MAX_CHUNK_LINES, CodeIndex, chunk_file, chunk_python, format_results


class WordEmbedding(EmbeddingFunction):
    """Each word switches on one of 64 slots: texts sharing words are similar."""

    calls: list[int] = []

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        WordEmbedding.calls.append(len(input))
        vectors = []
        for text in input:
            vector = [0.0] * 64
            for word in re.findall(r"[a-z]+", text.lower()):
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            vectors.append(vector if any(vector) else [1.0] + [0.0] * 63)
        return vectors

    @staticmethod
    def name() -> str:
        return "word-embedding-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "WordEmbedding":
        return WordEmbedding()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / "save.py").write_text("def store_session():\n    database insert session rows\n")
    (root / "net.py").write_text("def fetch_page():\n    download remote page\n")
    return root.resolve()


@pytest.fixture
def index(project: Path, tmp_path: Path):
    WordEmbedding.calls = []
    return CodeIndex(project, tmp_path / "index", WordEmbedding())


# ---- Chunking -------------------------------------------------------------

SOURCE = '''"""Module docstring."""
import os

LIMIT = 5


@decorated
def first():
    return 1


class Thing:
    """A thing."""

    def method(self):
        return 2


def second():
    return 3
'''


def test_python_gets_one_chunk_per_function():
    chunks = chunk_python("a.py", SOURCE)
    bodies = [c.text for c in chunks]
    assert any(b.startswith("@decorated\ndef first") for b in bodies)
    assert any(b.startswith("class Thing:\n    def method") for b in bodies)
    assert any(b.startswith("def second") for b in bodies)


def test_code_between_functions_is_kept():
    joined = "\n".join(c.text for c in chunk_python("a.py", SOURCE))
    assert "LIMIT = 5" in joined
    assert '"""A thing."""' in joined


def test_line_numbers_point_at_the_real_code():
    lines = SOURCE.splitlines()
    second = next(c for c in chunk_python("a.py", SOURCE) if c.text.startswith("def second"))
    assert lines[second.start_line - 1] == "def second():"
    assert lines[second.end_line - 1] == "    return 3"


def test_broken_python_still_gets_indexed():
    chunks = chunk_python("bad.py", "def broken(:\n    pass\n")
    assert len(chunks) == 1 and "broken" in chunks[0].text


def test_long_function_is_split_with_overlap_and_unique_ids():
    source = "def long():\n" + "".join(f"    x = {i}\n" for i in range(150))
    chunks = chunk_python("a.py", source)
    assert len(chunks) > 1
    assert all(c.end_line - c.start_line < MAX_CHUNK_LINES for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[1].start_line <= chunks[0].end_line
    assert chunks[-1].end_line == 151


def test_text_files_use_windows_and_blank_files_give_nothing():
    assert chunk_file("empty.md", "\n\n  \n") == []
    assert len(chunk_file("notes.md", "line\n" * 120)) > 1


# ---- Index ----------------------------------------------------------------

def test_search_finds_the_matching_file(index):
    index.update()
    assert index.search("database session", limit=1)[0].chunk.path == "save.py"
    assert index.search("download page", limit=1)[0].chunk.path == "net.py"


def test_results_have_clean_text_and_lines(index):
    index.update()
    top = index.search("database session", limit=1)[0]
    assert top.chunk.text.startswith("def store_session")
    assert (top.chunk.start_line, top.chunk.end_line) == (1, 2)
    assert 0 < top.similarity <= 1


def test_unchanged_files_are_not_embedded_again(index):
    index.update()
    WordEmbedding.calls = []
    stats = index.update()
    assert (stats.files_embedded, stats.files_unchanged) == (0, 2)
    assert WordEmbedding.calls == []


def test_changed_file_replaces_its_old_chunks(index, project):
    index.update()
    (project / "save.py").write_text("def other():\n    weather forecast rain\n")
    assert index.update().files_embedded == 1
    assert index.search("weather forecast", limit=1)[0].chunk.path == "save.py"
    texts = [r.chunk.text for r in index.search("database session", limit=5)]
    assert not any("store_session" in t for t in texts)


def test_deleted_file_leaves_the_index(index, project):
    index.update()
    (project / "net.py").unlink()
    stats = index.update()
    assert stats.files_removed == 1
    assert {r.chunk.path for r in index.search("download page")} == {"save.py"}


def test_secrets_and_ignored_folders_are_not_indexed(index, project):
    (project / ".env").write_text("API_KEY=secret\n")
    (project / ".venv").mkdir()
    (project / ".venv" / "lib.py").write_text("def hidden(): pass\n")
    index.update()
    paths = {r.chunk.path for r in index.search("secret hidden api key", limit=10)}
    assert paths == {"save.py", "net.py"}


def test_index_survives_reopening(index, project, tmp_path):
    index.update()
    WordEmbedding.calls = []
    reopened = CodeIndex(project, tmp_path / "index", WordEmbedding())
    assert reopened.update().files_embedded == 0
    assert reopened.search("database session", limit=1)[0].chunk.path == "save.py"


def test_empty_project_returns_a_message(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    index = CodeIndex(empty.resolve(), tmp_path / "i2", WordEmbedding())
    index.update()
    assert format_results(index.search("anything")) == "No indexed code found."


def test_format_shows_location_and_similarity(index):
    index.update()
    text = format_results(index.search("database session", limit=1))
    assert text.startswith("save.py lines 1-2 (similarity ")
    assert "def store_session" in text
