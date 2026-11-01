"""RAG over the project's code: find code by what it does, not by exact words.

1. Chunk:  split files into pieces (Python by function, other text by lines).
2. Embed:  turn each piece into a vector, where similar meanings sit close together.
3. Store:  keep the vectors in ChromaDB, outside the project.
4. Search: embed the question and return the closest pieces.

Embeddings run locally on the CPU with all-MiniLM-L6-v2, so code never leaves the machine.
"""

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from codesage.tools import _read_text, _walk

DEFAULT_INDEX_DIR = Path.home() / ".codesage" / "index"
# A list rather than "everything": this keeps out secrets like .env, and binary files.
INDEXED_SUFFIXES = {
    ".py", ".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini",
    ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".java", ".c", ".h", ".cpp", ".go", ".rs", ".sh",
}
MAX_CHUNK_LINES = 60
WINDOW_OVERLAP_LINES = 10
BATCH_SIZE = 64
MAX_RESULT_LINES = 30


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    def embedding_text(self) -> str:
        # The path helps: "storage.py" says something about what the code is for.
        return f"{self.path}\n{self.text}"


@dataclass
class SearchResult:
    chunk: Chunk
    similarity: float


@dataclass
class UpdateStats:
    files_embedded: int
    files_removed: int
    files_unchanged: int
    total_chunks: int


def _window_chunks(path: str, lines: list[str], first_line: int = 1) -> list[Chunk]:
    """Split lines into overlapping windows, so code at a boundary appears whole in one."""
    chunks = []
    step = MAX_CHUNK_LINES - WINDOW_OVERLAP_LINES
    for start in range(0, len(lines), step):
        window = lines[start : start + MAX_CHUNK_LINES]
        if "\n".join(window).strip():
            chunks.append(
                Chunk(path, first_line + start, first_line + start + len(window) - 1, "\n".join(window))
            )
        if start + MAX_CHUNK_LINES >= len(lines):
            break
    return chunks


def _python_spans(tree: ast.Module) -> list[tuple[int, int, str]]:
    """(first line, last line, prefix) for each function and method, in file order."""
    def span(node) -> tuple[int, int]:
        first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        return first, node.end_lineno

    spans = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((*span(node), ""))
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if methods:
                # The class line and docstring stay with the module-level code;
                # each method records which class it belongs to.
                spans.extend((*span(m), f"class {node.name}:\n") for m in methods)
            else:
                spans.append((*span(node), ""))
    return spans


def chunk_python(path: str, source: str) -> list[Chunk]:
    """One chunk per function or method, plus chunks for the code between them."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _window_chunks(path, source.splitlines())

    lines = source.splitlines()
    chunks = []
    position = 1

    def add_gap(first: int, last: int) -> None:
        gap = lines[first - 1 : last]
        if "\n".join(gap).strip():
            chunks.extend(_window_chunks(path, gap, first))

    for first, last, prefix in _python_spans(tree):
        add_gap(position, first - 1)
        body = lines[first - 1 : last]
        if len(body) > MAX_CHUNK_LINES:
            chunks.extend(_window_chunks(path, body, first))
        else:
            chunks.append(Chunk(path, first, last, prefix + "\n".join(body)))
        position = last + 1
    add_gap(position, len(lines))
    return chunks


def chunk_file(path: str, source: str) -> list[Chunk]:
    if path.endswith(".py"):
        return chunk_python(path, source)
    return _window_chunks(path, source.splitlines())


class CodeIndex:
    """A searchable index of one project. embedding_function can be replaced in tests."""

    def __init__(self, root: Path, index_dir: Path = DEFAULT_INDEX_DIR, embedding_function=None):
        self.root = root
        # One folder per project; telemetry off, because nothing should leave this machine.
        folder = index_dir / hashlib.sha256(str(root).encode()).hexdigest()[:16]
        self.client = chromadb.PersistentClient(
            path=str(folder), settings=Settings(anonymized_telemetry=False)
        )
        options = {"embedding_function": embedding_function} if embedding_function else {}
        self.collection = self.client.get_or_create_collection(
            "code_chunks", metadata={"hnsw:space": "cosine"}, **options
        )

    def _project_files(self) -> dict[str, str]:
        files = {}
        for f in _walk(self.root):
            if f.suffix.lower() in INDEXED_SUFFIXES and not f.is_symlink():
                text = _read_text(f)
                if text is not None:
                    files[f.relative_to(self.root).as_posix()] = text
        return files

    def update(self) -> UpdateStats:
        """Bring the index up to date, embedding only files whose content changed."""
        files = self._project_files()
        stored = self.collection.get(include=["metadatas"])
        stored_ids: dict[str, list[str]] = {}
        stored_hashes: dict[str, str] = {}
        for chunk_id, meta in zip(stored["ids"], stored["metadatas"]):
            stored_ids.setdefault(meta["path"], []).append(chunk_id)
            stored_hashes[meta["path"]] = meta["file_hash"]

        hashes = {p: hashlib.sha256(t.encode()).hexdigest() for p, t in files.items()}
        changed = [p for p in files if stored_hashes.get(p) != hashes[p]]
        removed = [p for p in stored_ids if p not in files]

        stale = [i for p in [*changed, *removed] for i in stored_ids.get(p, [])]
        if stale:
            self.collection.delete(ids=stale)

        chunks = [(c, hashes[p]) for p in changed for c in chunk_file(p, files[p])]
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            self.collection.add(
                ids=[c.id for c, _ in batch],
                documents=[c.embedding_text() for c, _ in batch],
                metadatas=[
                    {"path": c.path, "start_line": c.start_line, "end_line": c.end_line, "file_hash": h}
                    for c, h in batch
                ],
            )
        return UpdateStats(
            len(changed), len(removed), len(files) - len(changed), self.collection.count()
        )

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        count = self.collection.count()
        if count == 0:
            return []
        found = self.collection.query(
            query_texts=[query],
            n_results=min(limit, count),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for document, meta, distance in zip(
            found["documents"][0], found["metadatas"][0], found["distances"][0]
        ):
            text = document.split("\n", 1)[1] if "\n" in document else ""
            chunk = Chunk(meta["path"], meta["start_line"], meta["end_line"], text)
            results.append(SearchResult(chunk, 1 - distance))
        return results


def format_results(results: list[SearchResult]) -> str:
    """Turn results into text for the model, keeping each one short."""
    if not results:
        return "No indexed code found."
    parts = []
    for r in results:
        lines = r.chunk.text.splitlines()
        shown = "\n".join(lines[:MAX_RESULT_LINES])
        if len(lines) > MAX_RESULT_LINES:
            shown += f"\n... ({len(lines) - MAX_RESULT_LINES} more lines; use read_file)"
        parts.append(
            f"{r.chunk.path} lines {r.chunk.start_line}-{r.chunk.end_line} "
            f"(similarity {r.similarity:.2f})\n{shown}"
        )
    return "\n\n".join(parts)
