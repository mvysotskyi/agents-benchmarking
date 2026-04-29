"""OpenAI Responses API provider for GPT computer use."""

import os
from typing import Any

from openai import OpenAI

from computer_use.providers.base import ContentBlock, LLMProvider, LLMResponse

_OPENAI_KEY_MAP: dict[str, str] = {
    "SPACE": " ",
    "ENTER": "Enter",
    "TAB": "Tab",
    "ESCAPE": "Escape",
    "BACKSPACE": "Backspace",
    "DELETE": "Delete",
    "ARROWUP": "ArrowUp",
    "ARROWDOWN": "ArrowDown",
    "ARROWLEFT": "ArrowLeft",
    "ARROWRIGHT": "ArrowRight",
}


class OpenAIProvider(LLMProvider):
    """OpenAI Responses API provider for computer use with GPT models.

    Uses the built-in ``computer`` tool from the OpenAI Responses API.
    Conversation continuity is managed via ``previous_response_id``,
    and batched ``computer_call`` actions are translated to individual
    Anthropic-style ``tool_use`` ContentBlocks so the existing agent
    loop can execute them unchanged.
    """

    models_env_var = "OPENAI_MODELS"
    requires_screenshots: bool = True
    manages_own_context: bool = True

    def __init__(self) -> None:
        """Initialize the OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=api_key)
        self._previous_response_id: str | None = None
        self._pending_call_id: str | None = None

    def get_default_models(self) -> list[str]:
        """Return default OpenAI models with computer use support."""
        return ["gpt-5.4"]

    def reset_conversation(self) -> None:
        """Clear conversation state for a new task."""
        self._previous_response_id = None
        self._pending_call_id = None

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
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    def create_message(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str,
        max_tokens: int,
    ) -> LLMResponse:
        """Send a request to the OpenAI Responses API and return a normalized response.

        Args:
            model: Model identifier (e.g. ``gpt-5.4``).
            messages: Conversation messages (Anthropic-style).
            tools: Tool definitions (ignored — the provider uses its own).
            system: System prompt passed as ``instructions``.
            max_tokens: Maximum tokens in response.

        Returns:
            Normalized LLMResponse with actions as tool_use ContentBlocks.
        """
        reasoning_cfg = {"summary": "concise"}

        if self._previous_response_id is None:
            prompt = self._extract_prompt(messages)
            response = self.client.responses.create(
                model=model,
                tools=[{"type": "computer"}],
                input=prompt,
                instructions=system,
                reasoning=reasoning_cfg,
            )
        else:
            if not self._pending_call_id:
                raise ValueError(
                    "OpenAI computer-use continuation is missing the pending call ID."
                )

            screenshot_b64 = self._extract_last_screenshot(messages)
            if not screenshot_b64:
                return LLMResponse(
                    content=[
                        ContentBlock(
                            type="text",
                            text=(
                                "Retrying screenshot capture because the previous "
                                "screenshot payload was empty."
                            ),
                        ),
                        self._action_block(
                            self._pending_call_id,
                            0,
                            action="screenshot",
                        ),
                    ],
                    stop_reason="tool_use",
                    usage={},
                )

            input_items: list[dict[str, Any]] = [
                {
                    "type": "computer_call_output",
                    "call_id": self._pending_call_id,
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": f"data:image/png;base64,{screenshot_b64}",
                        "detail": "original",
                    },
                }
            ]
            response = self.client.responses.create(
                model=model,
                tools=[{"type": "computer"}],
                previous_response_id=self._previous_response_id,
                input=input_items,
                reasoning=reasoning_cfg,
            )

        self._previous_response_id = response.id
        return self._normalize_response(response)

    # ------------------------------------------------------------------
    # Prompt / screenshot extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prompt(messages: list[dict[str, Any]]) -> str:
        """Extract the user prompt from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    @staticmethod
    def _extract_last_screenshot(messages: list[dict[str, Any]]) -> str:
        """Find the last base64 screenshot from tool_result entries in messages."""
        for msg in reversed(messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for item in reversed(content):
                if not isinstance(item, dict) or item.get("type") != "tool_result":
                    continue
                tool_content = item.get("content", [])
                if not isinstance(tool_content, list):
                    continue
                for tc in reversed(tool_content):
                    if (
                        isinstance(tc, dict)
                        and tc.get("type") == "image"
                        and tc.get("source", {}).get("type") == "base64"
                    ):
                        data = tc["source"].get("data", "")
                        if isinstance(data, str) and data:
                            return data
        return ""

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    def _normalize_response(self, response: Any) -> LLMResponse:
        """Convert an OpenAI Responses API response to LLMResponse."""
        content_blocks: list[ContentBlock] = []
        action_counter = 0

        for item in response.output:
            item_type = getattr(item, "type", None)

            if item_type == "reasoning":
                text = self._extract_reasoning_text(item)
                if text:
                    content_blocks.append(ContentBlock(type="text", text=text))

            elif item_type == "computer_call":
                self._pending_call_id = getattr(item, "call_id", None)
                actions = getattr(item, "actions", None) or []
                if not actions:
                    single = getattr(item, "action", None)
                    if single is not None:
                        actions = [single]
                for action in actions:
                    translated = self._translate_action(action, action_counter)
                    content_blocks.extend(translated)
                    action_counter += len(translated)

            elif item_type == "message":
                text = self._extract_message_text(item)
                if text:
                    content_blocks.append(ContentBlock(type="text", text=text))

        has_tool_use = any(b.type == "tool_use" for b in content_blocks)
        stop_reason = "tool_use" if has_tool_use else "end_turn"
        usage = self._extract_usage(response)

        return LLMResponse(content=content_blocks, stop_reason=stop_reason, usage=usage)

    # ------------------------------------------------------------------
    # Action translation  (OpenAI → Anthropic-style tool_use input)
    # ------------------------------------------------------------------

    def _translate_action(self, action: Any, index: int) -> list[ContentBlock]:
        """Translate one OpenAI computer action into ContentBlock(s)."""
        action_type = getattr(action, "type", "unknown")
        call_id = self._pending_call_id or "unknown"

        if action_type == "click":
            button = getattr(action, "button", "left")
            if button == "wheel":
                button = "middle"
            elif button in ("back", "forward"):
                button = "left"
            return [
                self._action_block(
                    call_id,
                    index,
                    action=f"{button}_click",
                    coordinate=[getattr(action, "x", 0), getattr(action, "y", 0)],
                )
            ]

        if action_type == "double_click":
            return [
                self._action_block(
                    call_id,
                    index,
                    action="double_click",
                    coordinate=[getattr(action, "x", 0), getattr(action, "y", 0)],
                )
            ]

        if action_type == "scroll":
            return [self._build_scroll_block(action, call_id, index)]

        if action_type == "type":
            return [
                self._action_block(
                    call_id,
                    index,
                    action="type",
                    text=getattr(action, "text", ""),
                )
            ]

        if action_type == "keypress":
            return self._build_keypress_blocks(action, call_id, index)

        if action_type == "wait":
            return [self._action_block(call_id, index, action="wait", duration=2)]

        if action_type == "drag":
            path = getattr(action, "path", [])
            if len(path) >= 2:
                start = path[0]
                end = path[-1]
                start_coord = [getattr(start, "x", 0), getattr(start, "y", 0)]
                end_coord = [getattr(end, "x", 0), getattr(end, "y", 0)]
            else:
                start_coord = [0, 0]
                end_coord = [0, 0]
            return [
                self._action_block(
                    call_id,
                    index,
                    action="left_click_drag",
                    start_coordinate=start_coord,
                    coordinate=end_coord,
                )
            ]

        if action_type == "move":
            return [
                self._action_block(
                    call_id,
                    index,
                    action="mouse_move",
                    coordinate=[getattr(action, "x", 0), getattr(action, "y", 0)],
                )
            ]

        if action_type == "screenshot":
            return [self._action_block(call_id, index, action="screenshot")]

        return [self._action_block(call_id, index, action="screenshot")]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_block(call_id: str, index: int, **input_fields: Any) -> ContentBlock:
        return ContentBlock(
            type="tool_use",
            id=f"{call_id}_{index}",
            name="computer",
            input=input_fields,
        )

    @staticmethod
    def _build_scroll_block(action: Any, call_id: str, index: int) -> ContentBlock:
        x = getattr(action, "x", 512)
        y = getattr(action, "y", 384)
        scroll_x = getattr(action, "scroll_x", 0)
        scroll_y = getattr(action, "scroll_y", 0)

        if abs(scroll_y) >= abs(scroll_x):
            direction = "down" if scroll_y >= 0 else "up"
            amount = max(1, abs(scroll_y) // 100)
        else:
            direction = "right" if scroll_x >= 0 else "left"
            amount = max(1, abs(scroll_x) // 100)

        return ContentBlock(
            type="tool_use",
            id=f"{call_id}_{index}",
            name="computer",
            input={
                "action": "scroll",
                "coordinate": [x, y],
                "scroll_direction": direction,
                "scroll_amount": amount,
            },
        )

    @staticmethod
    def _build_keypress_blocks(
        action: Any,
        call_id: str,
        start_index: int,
    ) -> list[ContentBlock]:
        keys: list[str] = getattr(action, "keys", [])
        if not keys:
            return [
                ContentBlock(
                    type="tool_use",
                    id=f"{call_id}_{start_index}",
                    name="computer",
                    input={"action": "key", "key": "Enter"},
                )
            ]

        blocks: list[ContentBlock] = []
        for i, key in enumerate(keys):
            mapped = _OPENAI_KEY_MAP.get(key.upper(), key) if isinstance(key, str) else str(key)
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    id=f"{call_id}_{start_index + i}",
                    name="computer",
                    input={"action": "key", "key": mapped},
                )
            )
        return blocks

    @staticmethod
    def _extract_reasoning_text(item: Any) -> str:
        summary = getattr(item, "summary", None)
        if isinstance(summary, list):
            texts = [
                getattr(s, "text", "")
                for s in summary
                if isinstance(getattr(s, "text", None), str) and getattr(s, "text", "")
            ]
            if texts:
                return "\n".join(texts)
        return ""

    @staticmethod
    def _extract_message_text(item: Any) -> str:
        parts = getattr(item, "content", None)
        if isinstance(parts, list):
            texts = [
                getattr(p, "text", "") for p in parts if isinstance(getattr(p, "text", None), str)
            ]
            if texts:
                return "\n".join(texts)
        return ""

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage:
            return {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            }
        return {"input_tokens": 0, "output_tokens": 0}
