"""Anthropic provider implementation."""

import os
from typing import Any

from anthropic import Anthropic

from computer_use.providers.base import ContentBlock, LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    """Direct Anthropic SDK provider."""

    models_env_var = "ANTHROPIC_MODELS"

    def __init__(self) -> None:
        """Initialize the Anthropic client."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = Anthropic(api_key=api_key)

    def get_default_models(self) -> list[str]:
        """Return default Anthropic models."""
        return ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"]

    def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
        enable_caching: bool = True,
    ) -> LLMResponse:
        """Send a message to Anthropic API and return normalized response.

        Args:
            model: Model identifier.
            messages: Conversation messages.
            tools: Tool definitions.
            system: System prompt.
            max_tokens: Maximum tokens in response.
            enable_caching: Whether to enable prompt caching.

        Returns:
            Normalized LLM response.
        """
        betas = self.get_beta_headers(model)

        if enable_caching:
            betas.append("prompt-caching-2024-07-31")

            system_with_cache: str | list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

            cached_tools = [*tools]
            if cached_tools:
                cached_tools[-1] = {
                    **cached_tools[-1],
                    "cache_control": {"type": "ephemeral"},
                }
        else:
            system_with_cache = system
            cached_tools = tools

        response = self.client.beta.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_with_cache,
            messages=messages,
            tools=cached_tools,
            betas=betas,
        )

        content_blocks = []
        for block in response.content:
            if block.type == "text":
                content_blocks.append(
                    ContentBlock(
                        type="text",
                        text=block.text,
                    )
                )
            elif block.type == "tool_use":
                content_blocks.append(
                    ContentBlock(
                        type="tool_use",
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    )
                )

        return LLMResponse(
            content=content_blocks,
            stop_reason=response.stop_reason or "",
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def create_simple_message(
        self,
        model: str,
        system: str,
        user_message: str,
        max_tokens: int,
    ) -> str:
        """Make a text-only completion call with no tools.

        Args:
            model: Model identifier.
            system: System prompt.
            user_message: User message content.
            max_tokens: Maximum tokens in response.

        Returns:
            Text content of the response.
        """
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
