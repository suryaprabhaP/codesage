"""Does meaning search find the right code more often than keyword search?

Run from the repository root:  python evals/retrieval_eval.py

Each question describes some of CodeSage's own code in plain English, without
using its names. A search scores a hit when one of its top results contains the
expected code. Both searches rank the same chunks, so only the ranking differs.
"""

import re
import tempfile
from pathlib import Path

from codesage.rag import CodeIndex, chunk_file

QUESTIONS = [
    ("how does the program prevent reading files outside the project folder", "def _resolve"),
    ("where are conversations written to the database", "def add_messages"),
    ("reduce the conversation so it fits the model's memory limit", "def fit_to_context"),
    ("how are dangerous shell commands refused", "BLOCKED_COMMANDS"),
    ("show additions in green and deletions in red", "def render_diff"),
    ("ask the user before running something in the terminal", "def confirm_command"),
    ("pieces of a tool request that arrive in fragments get joined", "tool_calls.setdefault"),
    ("which cloud and local model servers are supported", "PROVIDERS = {"),
    ("stop a program that runs too long", "def execute"),
    ("keep windows style line endings when saving", "def apply_change"),
    ("make sure a replacement only happens in one place", "def edit_file"),
    ("continue a previous chat", "def resume"),
    ("command line flags", "argparse"),
    ("split a python source file by function", "def chunk_python"),
    ("only re-embed files whose contents changed", "def update"),
    ("loop until the model gives a final answer", "_run_tool_loop"),
    ("the instructions given to the model", "SYSTEM_PROMPT ="),
    ("undo the conversation when a request fails", "def ask"),
    ("animate progress while waiting for a reply", "Spinner("),
    ("keep the start and end of very long program output", "def _truncate"),
]

STOPWORDS = set(
    "a an and are as at by for from get how in into is it of on one only or so the to until "
    "when where which while with".split()
)
TOP_K = (1, 3, 5)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


def keyword_search(chunks: list, question: str, limit: int) -> list[str]:
    """Rank chunks by how many of the question's words they contain."""
    wanted = words(question)
    scored = [(len(wanted & words(c.embedding_text())), i) for i, c in enumerate(chunks)]
    best = sorted(scored, key=lambda s: (-s[0], s[1]))[:limit]
    return [chunks[i].text for score, i in best if score > 0]


def rank_of_hit(results: list[str], expected: str) -> int | None:
    for rank, text in enumerate(results, start=1):
        if expected in text:
            return rank
    return None


def report(name: str, ranks: list[int | None]) -> None:
    scores = "  ".join(
        f"top {k}: {sum(r is not None and r <= k for r in ranks)}/{len(ranks)}" for k in TOP_K
    )
    print(f"{name:10} {scores}")


def main() -> None:
    root = (Path(__file__).parent.parent / "codesage").resolve()
    chunks = []
    for f in sorted(root.glob("*.py")):
        chunks.extend(chunk_file(f.name, f.read_text(encoding="utf-8")))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        index = CodeIndex(root, Path(folder))
        index.update()
        print(f"{len(chunks)} chunks\n")
        meaning, keyword = [], []
        for question, expected in QUESTIONS:
            found = [r.chunk.text for r in index.search(question, limit=max(TOP_K))]
            meaning.append(rank_of_hit(found, expected))
            keyword.append(rank_of_hit(keyword_search(chunks, question, max(TOP_K)), expected))
            print(f"meaning {meaning[-1] or '-':>2}  keyword {keyword[-1] or '-':>2}  {question}")
        print()
        report("Meaning", meaning)
        report("Keyword", keyword)
        index.client.clear_system_cache()


if __name__ == "__main__":
    main()
