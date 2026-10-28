"""Thin wrapper around the OpenAI-compatible chat API."""

from collections.abc import Iterator

from openai import OpenAI

from codesage.config import Config


class LLM:
    def __init__(self, config: Config):
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.model = config.model

    def stream_chat(self, messages: list[dict]) -> Iterator[tuple[str, str]]:
        """Stream a reply token by token.

        Yields (kind, text) pairs where kind is:
          "thinking" -- the model's reasoning (qwen3 and other thinking models)
          "text"     -- the actual answer
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Ollama puts reasoning in a non-standard "reasoning" field.
            reasoning = getattr(delta, "reasoning", None) or getattr(
                delta, "reasoning_content", None
            )
            if reasoning:
                yield "thinking", reasoning
            if delta.content:
                yield "text", delta.content
