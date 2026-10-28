# CodeSage

A terminal AI coding agent that runs for free on a local LLM.

> Work in progress: built day by day over one week. See the roadmap below.

## Features (so far)

- Chat with an LLM in your terminal, with replies streamed live as Markdown
- Runs fully locally with [Ollama](https://ollama.com) (tested with `qwen3:4b` on an RTX 3050 6 GB)
- Also works with the Gemini or Groq free tiers by changing one setting
- Shows the model's reasoning ("thinking") dimmed, separate from the answer
- Conversation memory, `/clear`, `/help`, `/exit`, Ctrl+C to stop a reply

## How it works

All three providers accept the OpenAI chat API format, so CodeSage uses one client (`openai` SDK) and only the base URL changes. Each turn, the full `messages` history is sent to the model, because the model itself keeps no memory between calls.

```
codesage/
├── config.py   # settings + provider URL table
├── llm.py      # streaming client, splits thinking from answer
└── cli.py      # terminal chat loop (rich + prompt_toolkit)
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

## Roadmap

- [x] Day 1: terminal chat with streaming, local LLM via Ollama
- [ ] Day 2: tool calling (`read_file`, `list_files`, `search_code`)
- [ ] Day 3: agent loop + file editing with confirmation
- [ ] Day 4: `run_command` with permission prompts, nicer UI
- [ ] Day 5: saved sessions (SQLite) + context management
- [ ] Day 6: RAG over the codebase (ChromaDB + sentence-transformers)
- [ ] Day 7: demo GIF, docs, release
