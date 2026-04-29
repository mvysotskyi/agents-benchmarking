"""Configuration module for Claude Computer Use Agent."""

import os
from typing import Any

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Model specifications including pricing and context limits."""

    context_window: int
    max_output_tokens: int
    input_price_per_mtok: float
    output_price_per_mtok: float


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "claude-opus-4-5": ModelSpec(
        context_window=200_000,
        max_output_tokens=64_000,
        input_price_per_mtok=15.0,
        output_price_per_mtok=75.0,
    ),
    "claude-sonnet-4-5": ModelSpec(
        context_window=200_000,
        max_output_tokens=64_000,
        input_price_per_mtok=3.0,
        output_price_per_mtok=15.0,
    ),
    "claude-haiku-4-5": ModelSpec(
        context_window=200_000,
        max_output_tokens=64_000,
        input_price_per_mtok=0.80,
        output_price_per_mtok=4.0,
    ),
    "gpt-5.4": ModelSpec(
        context_window=200_000,
        max_output_tokens=32_000,
        input_price_per_mtok=5.0,
        output_price_per_mtok=15.0,
    ),
}


def get_model_spec(model: str) -> ModelSpec | None:
    """Get model specification by name with fuzzy matching.

    Args:
        model: Model identifier string.

    Returns:
        ModelSpec if found, None otherwise.
    """
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model]

    for key, spec in MODEL_REGISTRY.items():
        if key in model.lower() or model.lower() in key:
            return spec

    return None


class DisplayConfig(BaseModel):
    """Display configuration for the browser viewport."""

    width: int = 1024
    height: int = 768
    display_number: int = 1


def _get_auto_summarize_default() -> bool:
    """Get default value for auto_summarize from env var."""
    env_val = os.getenv("AUTO_SUMMARIZE", "true").lower()
    return env_val not in ("false", "0", "no", "off")


class TokenOptimizationConfig(BaseModel):
    """Token optimization settings for reducing API costs.

    These settings help minimize token consumption by:
    - Pruning old screenshots from conversation history (keeping last 2)
    - Skipping screenshots for non-visual actions
    - Enabling prompt caching for Anthropic provider
    - Auto-summarizing context when nearing capacity
    """

    max_screenshots_in_history: int = Field(
        default=2,
        ge=1,
        description="Keep only last N screenshots in message history",
    )
    prune_history: bool = Field(
        default=True,
        description="Enable screenshot history pruning",
    )
    lazy_screenshots: bool = Field(
        default=True,
        description="Skip screenshots for non-visual actions (wait, mouse_move)",
    )
    lazy_screenshot_actions: frozenset[str] = Field(
        default=frozenset({"wait", "mouse_move"}),
        description="Actions that return text instead of screenshots",
    )
    enable_prompt_caching: bool = Field(
        default=True,
        description="Enable prompt caching for Anthropic provider",
    )
    auto_summarize: bool = Field(
        default_factory=_get_auto_summarize_default,
        description="Enable automatic context summarization (env: AUTO_SUMMARIZE)",
    )
    summarize_threshold_percent: float = Field(
        default=70.0,
        ge=50.0,
        le=90.0,
        description="Context fill percentage that triggers summarization",
    )
    summarize_preserve_recent: int = Field(
        default=4,
        ge=2,
        description="Number of recent message pairs to preserve during summarization",
    )


class AgentConfig(BaseModel):
    """Agent configuration settings."""

    model: str = "claude-sonnet-4-5"
    max_tokens: int = 4096
    max_iterations: int = 50
    display: DisplayConfig = DisplayConfig()
    verbose_selectors: bool = False
    additional_instructions: str = ""
    token_optimization: TokenOptimizationConfig = Field(
        default_factory=TokenOptimizationConfig
    )

    def get_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        instructions = self.additional_instructions.strip()
        if not instructions:
            return DEFAULT_SYSTEM_PROMPT
        return (
            f"{DEFAULT_SYSTEM_PROMPT.rstrip()}\n\n"
            "Additional instructions:\n"
            f"{instructions}"
        )

    def get_tool_definition(self) -> dict[str, Any]:
        """Get the computer use tool definition for the configured model."""
        tool_version = get_tool_version(self.model)
        tool_def: dict[str, Any] = {
            "type": tool_version,
            "name": "computer",
            "display_width_px": self.display.width,
            "display_height_px": self.display.height,
            "display_number": self.display.display_number,
        }

        if "opus" in self.model.lower():
            tool_def["enable_zoom"] = True

        return tool_def


def get_tool_version(model: str) -> str:
    """Get the tool version for a model based on its name.

    Args:
        model: Model identifier string.

    Returns:
        Tool version string.
    """
    if model.lower().startswith("gpt"):
        return "computer (OpenAI Responses API)"
    if "opus" in model.lower():
        return "computer_20251124"
    return "computer_20250124"


def get_beta_headers(model: str) -> list[str]:
    """Get the beta headers for a model based on its name.

    Args:
        model: Model identifier string.

    Returns:
        List of beta header strings.
    """
    if "opus" in model.lower():
        return ["computer-use-2025-11-24"]
    return ["computer-use-2025-01-24"]


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that can control a web browser to complete tasks.

When given a task:
1. First take a screenshot to see the current state of the browser
2. Analyze what you see and plan your next action
3. Execute actions step by step, taking screenshots after each action to verify results
4. If something doesn't work as expected, try alternative approaches
5. Communicate your progress and any issues encountered

Be precise with clicks - aim for the center of buttons and links.
For text input, make sure the input field is focused before typing.
"""
