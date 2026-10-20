"""MiniMax Chat Completions adapter preserving reasoning across tool turns."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI


class ChatMiniMax(ChatOpenAI):
    """Retain MiniMax's reasoning_content extension in responses and history."""

    def _get_request_payload(
        self, input_: LanguageModelInput, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # Only Chat Completions uses messages; leave explicit Responses requests alone.
        if "messages" in payload:
            messages = self._convert_input(input_).to_messages()
            for message, serialized in zip(messages, payload["messages"], strict=False):
                if isinstance(message, AIMessage) and "reasoning_content" in message.additional_kwargs:
                    serialized["reasoning_content"] = message.additional_kwargs["reasoning_content"]
        return payload

    def _create_chat_result(self, response: Any, generation_info: dict[str, Any] | None = None) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        data = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, data["choices"], strict=False):
            if "reasoning_content" in choice["message"]:
                generation.message.additional_kwargs["reasoning_content"] = choice["message"]["reasoning_content"]
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict[str, Any], default_chunk_class: type, base_generation_info: dict[str, Any] | None
    ) -> ChatGenerationChunk | None:
        result = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        choices = chunk.get("choices", [])
        if result is not None and choices:
            reasoning = (choices[0].get("delta") or {}).get("reasoning_content")
            if reasoning is not None:
                result.message.additional_kwargs["reasoning_content"] = reasoning
        return result
