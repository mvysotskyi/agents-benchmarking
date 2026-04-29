"""LiteLLM provider implementation using OpenAI SDK."""

import json
import os
from typing import Any

import openai

from computer_use.providers.base import ContentBlock, LLMProvider, LLMResponse


class LiteLLMProvider(LLMProvider):
    """LiteLLM proxy provider using OpenAI SDK for direct proxy communication."""

    models_env_var = "LITELLM_MODELS"

    def __init__(self) -> None:
        """Initialize LiteLLM configuration."""
        self.base_url = os.getenv("LITELLM_BASE_URL")
        if not self.base_url:
            raise ValueError("LITELLM_BASE_URL environment variable is required")

        self.api_key = os.getenv("LITELLM_API_KEY")
        if not self.api_key:
            raise ValueError("LITELLM_API_KEY environment variable is required")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def get_default_models(self) -> list[str]:
        """Return default LiteLLM models."""
        return ["anthropic/claude-sonnet-4-5", "anthropic/claude-haiku-4-5"]

    def _is_bedrock_model(self, model: str) -> bool:
        """Check if model is a Bedrock model."""
        return model.lower().startswith("bedrock/")

    def _is_legacy_bedrock_model(self, model: str) -> bool:
        """Check if model is a legacy Claude 3.x Bedrock model."""
        model_lower = model.lower()
        legacy_patterns = ["claude-3-5", "claude-3-", "claude3"]
        return any(pattern in model_lower for pattern in legacy_patterns)

    def _convert_tools_for_bedrock(
        self, tools: list[dict[str, Any]], is_legacy: bool
    ) -> list[dict[str, Any]]:
        """Convert tools to Bedrock-compatible format."""
        converted_tools = []
        for tool in tools:
            tool_copy = tool.copy()
            tool_type = tool_copy.get("type", "")
            if tool_type.startswith("computer_"):
                if is_legacy:
                    tool_copy["type"] = "computer_20241022"
                else:
                    tool_copy["type"] = "computer_20250124"
                tool_copy.pop("enable_zoom", None)
            converted_tools.append(tool_copy)
        return converted_tools

    def _convert_messages_to_openai(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic-style messages to OpenAI format."""
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                # Check if this is a tool_result message
                if role == "user" and any(
                    isinstance(c, dict) and c.get("type") == "tool_result"
                    for c in content
                ):
                    pending_images = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            tool_content = item.get("content", "")
                            if isinstance(tool_content, list):
                                text_parts = []
                                for tc in tool_content:
                                    if isinstance(tc, dict) and tc.get("type") == "image":
                                        source = tc.get("source", {})
                                        if source.get("type") == "base64":
                                            media_type = source.get("media_type", "image/png")
                                            data = source.get("data", "")
                                            pending_images.append({
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{media_type};base64,{data}"
                                                }
                                            })
                                    elif isinstance(tc, dict) and tc.get("type") == "text":
                                        text_parts.append(tc.get("text", ""))
                                    elif isinstance(tc, str):
                                        text_parts.append(tc)
                                tool_content = " ".join(text_parts) if text_parts else "Action completed."
                            converted.append({
                                "role": "tool",
                                "tool_call_id": item.get("tool_use_id", ""),
                                "content": str(tool_content),
                            })

                    # Add images as a separate user message with vision format
                    if pending_images:
                        image_content: list[dict[str, Any]] = [
                            {"type": "text", "text": "Here is the screenshot after the action:"}
                        ]
                        image_content.extend(pending_images)
                        converted.append({
                            "role": "user",
                            "content": image_content,
                        })
                    continue

                # Check if this is an assistant message with tool_use
                if role == "assistant":
                    text_content = ""
                    tool_calls = []

                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_content = item.get("text", "")
                            elif item.get("type") == "tool_use":
                                tool_calls.append({
                                    "id": item.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name", ""),
                                        "arguments": json.dumps(item.get("input", {})),
                                    },
                                })

                    if tool_calls:
                        converted.append({
                            "role": "assistant",
                            "content": text_content or None,
                            "tool_calls": tool_calls,
                        })
                    else:
                        converted.append({
                            "role": "assistant",
                            "content": text_content,
                        })
                    continue

                # Default: convert content blocks to text
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                converted.append({
                    "role": role,
                    "content": " ".join(text_parts) if text_parts else str(content),
                })
            else:
                converted.append({"role": role, "content": str(content)})

        return converted

    def _convert_tools_to_openai(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic computer use tools to OpenAI function format."""
        openai_tools = []
        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type.startswith("computer_"):
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", "computer"),
                        "description": "Control the computer through GUI actions",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "screenshot", "left_click", "right_click",
                                        "double_click", "triple_click", "middle_click",
                                        "type", "key", "scroll", "mouse_move",
                                        "left_click_drag", "wait"
                                    ],
                                },
                                "coordinate": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "[x, y] coordinates",
                                },
                                "text": {"type": "string"},
                                "key": {"type": "string"},
                                "scroll_direction": {
                                    "type": "string",
                                    "enum": ["up", "down", "left", "right"],
                                },
                                "scroll_amount": {"type": "integer"},
                                "start_coordinate": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                },
                                "duration": {"type": "integer"},
                            },
                            "required": ["action"],
                        },
                    },
                })
            else:
                openai_tools.append(tool)
        return openai_tools

    def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> LLMResponse:
        """Send a message via LiteLLM proxy and return normalized response."""
        # Convert messages to OpenAI format
        converted_messages = self._convert_messages_to_openai(messages)
        full_messages = [{"role": "system", "content": system}] + converted_messages

        # Convert tools to OpenAI format
        openai_tools = self._convert_tools_to_openai(tools)

        # For Bedrock models, also adjust tool version
        if self._is_bedrock_model(model):
            is_legacy = self._is_legacy_bedrock_model(model)
            tools = self._convert_tools_for_bedrock(tools, is_legacy)

        response = self.client.chat.completions.create(
            model=model,
            messages=full_messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=max_tokens,
        )

        content_blocks = []
        choice = response.choices[0]
        message = choice.message

        if message.content:
            content_blocks.append(
                ContentBlock(
                    type="text",
                    text=message.content,
                )
            )

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type == "function":
                    content_blocks.append(
                        ContentBlock(
                            type="tool_use",
                            id=tool_call.id,
                            name=tool_call.function.name,
                            input=json.loads(tool_call.function.arguments),
                        )
                    )

        stop_reason = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "stop":
            stop_reason = "end_turn"

        return LLMResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
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
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
