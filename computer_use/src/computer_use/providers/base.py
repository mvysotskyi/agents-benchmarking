"""Base classes for LLM providers."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentBlock:
    """Normalized content block (text or tool_use)."""

    type: str
    text: str | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: list[ContentBlock]
    stop_reason: str
    usage: dict[str, int]


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    models_env_var: str = ""
    requires_screenshots: bool = False
    manages_own_context: bool = False

    def reset_conversation(self) -> None:
        """Reset provider conversation state (no-op by default)."""

    @abstractmethod
    def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> LLMResponse:
        """Send a message and return normalized response.

        Args:
            model: Model identifier.
            messages: Conversation messages.
            tools: Tool definitions.
            system: System prompt.
            max_tokens: Maximum tokens in response.

        Returns:
            Normalized LLMResponse.
        """
        pass

    def get_available_models(self) -> list[str]:
        """Return list of available model identifiers from env var."""
        models_str = os.getenv(self.models_env_var, "")
        if not models_str:
            return self.get_default_models()
        return [m.strip() for m in models_str.split(",") if m.strip()]

    @abstractmethod
    def get_default_models(self) -> list[str]:
        """Return default models if env var is not set."""
        pass

    @abstractmethod
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
        pass

    @staticmethod
    def get_tool_version(model: str) -> str:
        """Get the tool version for a model based on its name."""
        if "opus" in model.lower():
            return "computer_20251124"
        return "computer_20250124"

    @staticmethod
    def get_beta_headers(model: str) -> list[str]:
        """Get the beta headers for a model based on its name."""
        if "opus" in model.lower():
            return ["computer-use-2025-11-24"]
        return ["computer-use-2025-01-24"]
