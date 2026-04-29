"""AWS Bedrock provider implementation."""

import os
from typing import Any

import boto3

from computer_use.providers.base import ContentBlock, LLMProvider, LLMResponse


class BedrockProvider(LLMProvider):
    """AWS Bedrock Converse API provider."""

    models_env_var = "BEDROCK_MODELS"

    def __init__(self) -> None:
        """Initialize Bedrock client."""
        region = os.getenv("AWS_REGION", "us-east-1")
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def get_default_models(self) -> list[str]:
        """Return default Bedrock models."""
        return ["anthropic.claude-3-5-sonnet-20241022-v2:0"]

    def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> LLMResponse:
        """Send a message via Bedrock Converse API and return normalized response."""
        bedrock_messages = self._convert_messages(messages)
        is_legacy = self._is_legacy_model(model)
        bedrock_tools = self._convert_tools(tools, is_legacy)

        beta_header = (
            "computer-use-2024-10-22" if is_legacy else "computer-use-2025-01-24"
        )

        response = self.client.converse(
            modelId=model,
            messages=bedrock_messages,
            system=[{"text": system}],
            additionalModelRequestFields={
                "tools": bedrock_tools,
                "anthropic_beta": [beta_header],
            },
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": "placeholder",
                            "description": "Placeholder tool for Converse API compatibility",
                            "inputSchema": {"json": {"type": "object"}},
                        }
                    }
                ]
            },
            inferenceConfig={"maxTokens": max_tokens},
        )

        content_blocks = self._parse_response(response)

        stop_reason = response.get("stopReason", "end_turn")
        if stop_reason == "tool_use":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        usage = response.get("usage", {})
        return LLMResponse(
            content=content_blocks,
            stop_reason=stop_reason,
            usage={
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
            },
        )

    def _is_legacy_model(self, model: str) -> bool:
        """Check if model is a legacy Claude 3.x model.

        Legacy models (Claude 3.5 Sonnet, Claude 3 Opus, etc.) require
        computer_20241022 tool version. Newer Claude 4.x models support
        computer_20250124.
        """
        model_lower = model.lower()
        legacy_patterns = ["claude-3-5", "claude-3-", "claude3"]
        return any(pattern in model_lower for pattern in legacy_patterns)

    def _convert_tools(
        self, tools: list[dict[str, Any]], is_legacy: bool
    ) -> list[dict[str, Any]]:
        """Convert tools to Bedrock-compatible format.

        Legacy Claude 3.x models only support computer_20241022.
        Newer Claude 4.x models support computer_20250124.
        """
        bedrock_tools = []
        for tool in tools:
            tool_copy = tool.copy()
            tool_type = tool_copy.get("type", "")
            if tool_type.startswith("computer_"):
                if is_legacy:
                    tool_copy["type"] = "computer_20241022"
                    tool_copy.pop("enable_zoom", None)
                else:
                    tool_copy["type"] = "computer_20250124"
                    tool_copy.pop("enable_zoom", None)
            bedrock_tools.append(tool_copy)
        return bedrock_tools

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert standard messages to Bedrock Converse format."""
        bedrock_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if isinstance(content, str):
                bedrock_messages.append(
                    {"role": role, "content": [{"text": content}]}
                )
            elif isinstance(content, list):
                bedrock_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            bedrock_content.append({"text": item.get("text", "")})
                        elif item.get("type") == "image":
                            source = item.get("source", {})
                            if source.get("type") == "base64":
                                bedrock_content.append(
                                    {
                                        "image": {
                                            "format": "png",
                                            "source": {
                                                "bytes": source.get("data", "").encode()
                                            },
                                        }
                                    }
                                )
                        elif item.get("type") == "tool_result":
                            tool_content = item.get("content", [])
                            result_content = []
                            for tc in tool_content:
                                if isinstance(tc, dict) and tc.get("type") == "image":
                                    src = tc.get("source", {})
                                    if src.get("type") == "base64":
                                        import base64

                                        result_content.append(
                                            {
                                                "image": {
                                                    "format": "png",
                                                    "source": {
                                                        "bytes": base64.b64decode(
                                                            src.get("data", "")
                                                        )
                                                    },
                                                }
                                            }
                                        )
                                elif isinstance(tc, str):
                                    result_content.append({"text": tc})
                            bedrock_content.append(
                                {
                                    "toolResult": {
                                        "toolUseId": item.get("tool_use_id", ""),
                                        "content": result_content,
                                    }
                                }
                            )
                        elif item.get("type") == "tool_use":
                            bedrock_content.append(
                                {
                                    "toolUse": {
                                        "toolUseId": item.get("id", ""),
                                        "name": item.get("name", ""),
                                        "input": item.get("input", {}),
                                    }
                                }
                            )
                    elif isinstance(item, str):
                        bedrock_content.append({"text": item})

                if bedrock_content:
                    bedrock_messages.append({"role": role, "content": bedrock_content})

        return bedrock_messages

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
        response = self.client.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            system=[{"text": system}],
            inferenceConfig={"maxTokens": max_tokens},
        )
        for item in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in item:
                return item["text"]
        return ""

    def _parse_response(self, response: dict[str, Any]) -> list[ContentBlock]:
        """Parse Bedrock Converse response to ContentBlocks."""
        content_blocks = []
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])

        for item in content:
            if "text" in item:
                content_blocks.append(
                    ContentBlock(
                        type="text",
                        text=item["text"],
                    )
                )
            elif "toolUse" in item:
                tool_use = item["toolUse"]
                content_blocks.append(
                    ContentBlock(
                        type="tool_use",
                        id=tool_use.get("toolUseId", ""),
                        name=tool_use.get("name", ""),
                        input=tool_use.get("input", {}),
                    )
                )

        return content_blocks
