"""Settings loaded from environment variables or a .env file."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Every provider here speaks the OpenAI-compatible chat API,
# so one client works for all of them -- only the URL changes.
PROVIDERS = {
    "ollama": "http://localhost:11434/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
}

DEFAULT_MODELS = {
    "ollama": "qwen3:4b",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
}


# Must match the context size the model really runs with (for Ollama, OLLAMA_CONTEXT_LENGTH).
DEFAULT_CONTEXT_TOKENS = 8192
MIN_CONTEXT_TOKENS = 4096


@dataclass
class Config:
    provider: str
    model: str
    base_url: str
    api_key: str
    context_tokens: int = DEFAULT_CONTEXT_TOKENS


def load_config() -> Config:
    load_dotenv()

    provider = os.getenv("CODESAGE_PROVIDER", "ollama").lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}"
        )

    model = os.getenv("CODESAGE_MODEL") or DEFAULT_MODELS[provider]
    # Ollama ignores the key, but the OpenAI client requires a non-empty one.
    api_key = os.getenv("CODESAGE_API_KEY") or "ollama"
    if provider != "ollama" and api_key == "ollama":
        raise ValueError(f"CODESAGE_API_KEY is required for provider '{provider}'.")

    setting = os.getenv("CODESAGE_CONTEXT_TOKENS") or str(DEFAULT_CONTEXT_TOKENS)
    if not setting.isdigit() or int(setting) < MIN_CONTEXT_TOKENS:
        raise ValueError(f"CODESAGE_CONTEXT_TOKENS must be a number of at least {MIN_CONTEXT_TOKENS}.")

    return Config(provider, model, PROVIDERS[provider], api_key, int(setting))
