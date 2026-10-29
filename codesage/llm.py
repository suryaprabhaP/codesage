"""Thin wrapper around the OpenAI-compatible chat API."""

from collections.abc import Iterator

from openai import OpenAI

from codesage.config import Config


class LLM:
    def __init__(self, config: Config):
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.model = config.model

    def stream_chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> Iterator[tuple[str, str | dict]]:
        """Stream a reply token by token.

        Yields (kind, value) pairs where kind is:
          "thinking"  -- the model's reasoning (qwen3 and other thinking models)
          "text"      -- the actual answer
          "tool_call" -- a tool the model wants to run, as {"id", "name", "arguments"}
                         (sent once the stream ends, when the call is complete)
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **({"tools": tools} if tools else {}),
        )

        # Tool calls arrive in fragments spread over many chunks (the name first,
        # then the JSON arguments a few characters at a time), so collect them by index.
        tool_calls: dict[int, dict] = {}

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
            for tc in delta.tool_calls or []:
                index = tc.index if tc.index is not None else len(tool_calls)
                call = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    call["id"] = tc.id
                if tc.function and tc.function.name:
                    call["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    call["arguments"] += tc.function.arguments

        for index, call in sorted(tool_calls.items()):
            call["id"] = call["id"] or f"call_{index}"
            yield "tool_call", call
