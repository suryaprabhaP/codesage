# CodeSage

A terminal AI coding agent that runs for free on a local LLM.

> Work in progress: built day by day over one week. See the roadmap below.

## Features (so far)

- Chat with an LLM in your terminal, with replies streamed live as Markdown
- Runs fully locally with [Ollama](https://ollama.com) (tested with `qwen3:4b` on an RTX 3050 6 GB)
- Also works with the Gemini or Groq free tiers by changing one setting
- Shows the model's reasoning ("thinking") dimmed, separate from the answer
- Conversation memory, `/clear`, `/help`, `/exit`, Ctrl+C to stop a reply
- **Tool calling:** the agent explores your project with `list_files`, `read_file` and `search_code`
- Tools are sandboxed to the project folder and skip `.git`, `.venv`, `node_modules` and caches
- **File editing:** `edit_file` and `write_file`, shown as a colored diff that you approve or reject first
- **Running commands:** `run_command` runs tests or scripts after you approve; answer "always" to stop being asked about the same command for the rest of the session
- Dangerous commands (`rm -rf /`, `format`, `shutdown`, `git push --force`, piping downloads into a shell, ...) are blocked outright
- A live timer while the model works, and how long each step took
- **Saved sessions:** every conversation is stored in SQLite per project; continue with `codesage --continue`, `/sessions` and `/resume N`
- **Context management:** long conversations are trimmed automatically to fit the model's context window; `/context` shows usage
- **RAG over your code:** `search_meaning` finds code by what it does, using local embeddings and ChromaDB; `/index` updates the index on demand

## How it works

All three providers accept the OpenAI chat API format, so CodeSage uses one client (`openai` SDK) and only the base URL changes. Each turn, the full `messages` history is sent to the model, because the model itself keeps no memory between calls.

**The agent loop:** CodeSage sends the model a JSON description of each tool. When the model asks for a tool, CodeSage runs it, adds the result to `messages` as a `tool` message, and calls the model again. This repeats until the model replies with a final answer (at most 10 rounds).

**Safe editing:** edit tools never write directly. They return a proposed change; the user sees a diff and answers yes or no, and only then is the file written. `edit_file` replaces an exact piece of text that must appear exactly once, so an edit can never land in the wrong place.

**Safe commands:** `run_command` follows the same pattern. The user sees the exact command and answers yes, always (for this session) or no. Commands run in the project folder (PowerShell on Windows, `sh` elsewhere) with no input, a timeout, the real exit code, and long output cut to its start and end. A short blocklist refuses commands that are easy to approve by accident and impossible to undo; the yes/no prompt is the main safety, the blocklist is a second line of defence.

**Saved sessions:** after each completed turn, new messages are written to `~/.codesage/sessions.db` in a transaction. The database is outside the project so conversations never reach git. "Always allow" answers are deliberately not saved, so a resumed conversation must ask again.

**Context management:** before each request, `context.py` builds a shorter copy of the history that fits the window, leaving 2,048 tokens for the reply. It removes in order of least harm: old tool outputs (they can be fetched again), then the oldest whole turns (so a tool call is never separated from its result), then the current turn's other outputs, then part of the newest output. The full history stays in memory and in the database. Sizes use a 4-characters-per-token estimate, which measured about 12% above the model's real count on one test, the safe direction.

**RAG:** `rag.py` splits Python files by function and other text files into overlapping line windows, embeds each piece with all-MiniLM-L6-v2 on the CPU, and stores the vectors in ChromaDB under `~/.codesage/index`. A question is embedded the same way and the closest pieces are returned. Only files whose content hash changed are embedded again, so the index stays current on every search. Only listed text file types are indexed, which keeps out `.env` secrets, and ChromaDB's telemetry is off. The only download is the embedding model, on first use (about 166 MB on disk).

The agent loop (`agent.py`) knows nothing about the terminal: it talks to the screen through a small UI interface. That keeps the loop testable with a fake model and a fake UI.

```
codesage/
├── codesage/
│   ├── config.py      # settings + provider URL table
│   ├── llm.py         # streaming client: thinking, answer and tool calls
│   ├── tools.py       # tool functions, their JSON schemas, safe file changes
│   ├── context.py     # trims what is sent to fit the context window
│   ├── storage.py     # saved sessions in SQLite
│   ├── rag.py         # chunking, embeddings and meaning search
│   ├── agent.py       # the agent loop (UI-independent)
│   └── cli.py         # terminal UI: streaming, diffs, confirmations
├── evals/
│   └── retrieval_eval.py # meaning search vs keyword search
└── tests/
    ├── test_tools.py   # unit tests for the tools
    ├── test_context.py # trimming rules
    ├── test_storage.py # saving and loading sessions
    ├── test_rag.py     # chunking and the index, with a fake embedding
    └── test_agent.py   # agent loop tests with a scripted fake model
```

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
ollama pull qwen3:4b
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e .
codesage
```

To use Gemini or Groq instead, copy `.env.example` to `.env` and set `CODESAGE_PROVIDER`, `CODESAGE_MODEL` and `CODESAGE_API_KEY`.

Run `codesage` from inside the project you want it to work on; that folder is what its tools can see.

**Context size:** Ollama's default context window is 4096 tokens, which the tool descriptions plus qwen3's reasoning can overflow. Start Ollama with a larger window (8192 fits `qwen3:4b` in 6 GB of VRAM):

```bash
# PowerShell
$env:OLLAMA_CONTEXT_LENGTH = "8192"; ollama serve
```

If you use a different size, set `CODESAGE_CONTEXT_TOKENS` to the same number so trimming matches the real window.

## Speed

`qwen3:4b` reasons before every answer, so a multi-step task takes a few minutes on a 6 GB GPU. Turning that reasoning off does not help with this model: with `reasoning_effort="none"` (or Ollama's native `think: false`) qwen3:4b moves its reasoning into the answer text instead. On the same run-fix-rerun task, normal mode took 196 s and "thinking off" took 489 s with a cluttered answer, so CodeSage keeps normal mode. For faster replies, use a larger GPU or the Groq free tier (`CODESAGE_PROVIDER=groq`).

## Does RAG help?

`evals/retrieval_eval.py` asks 20 plain-English questions about CodeSage's own code, avoiding its exact names, and checks whether each search finds the expected code among 95 chunks:

| Right code found in | Meaning search | Keyword search |
|---|---|---|
| Top 1 | 10/20 | 10/20 |
| Top 3 | 15/20 | 12/20 |
| Top 5 | 18/20 | 15/20 |

They tie at the top result, and meaning search leads further down. They also miss different questions, so the agent keeps both tools. This is a small test on a small project, with questions written by the author, so treat it as a rough signal.

Retrieval finding the code doesn't guarantee a right answer: in live tests `qwen3:4b` sometimes names a function that calls the relevant code rather than the function itself.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Day 1: terminal chat with streaming, local LLM via Ollama
- [x] Day 2: tool calling (`read_file`, `list_files`, `search_code`)
- [x] Day 3: agent loop + file editing with confirmation
- [x] Day 4: `run_command` with permission prompts, nicer UI
- [x] Day 5: saved sessions (SQLite) + context management
- [x] Day 6: RAG over the codebase (ChromaDB + local embeddings)
- [ ] Day 7: demo GIF, docs, release
