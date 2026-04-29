"""LLM providers package."""

import os

from computer_use.providers.base import ContentBlock, LLMProvider, LLMResponse


def get_provider() -> LLMProvider:
    """Get the configured LLM provider based on environment variables.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider name is unknown.
    """
    provider_name = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider_name == "anthropic":
        from computer_use.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()

    if provider_name == "litellm":
        from computer_use.providers.litellm_provider import LiteLLMProvider

        return LiteLLMProvider()

    if provider_name == "bedrock":
        from computer_use.providers.bedrock_provider import BedrockProvider

        return BedrockProvider()

    if provider_name == "openai":
        from computer_use.providers.openai_provider import OpenAIProvider

        return OpenAIProvider()

    raise ValueError(f"Unknown LLM provider: {provider_name}")


__all__ = [
    "ContentBlock",
    "LLMProvider",
    "LLMResponse",
    "get_provider",
]
