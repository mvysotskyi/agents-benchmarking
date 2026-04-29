"""Agent module implementing the Claude Computer Use agent loop."""

import base64
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Page
from rich.console import Console

from computer_use.config import AgentConfig, DEFAULT_SYSTEM_PROMPT, get_model_spec
from computer_use.playwright_helpers import get_element_at_point
from computer_use.providers import LLMProvider
from computer_use.token_tracker import IterationUsage, TokenUsageTracker


class ComputerUseAgent:
    """Agent that uses Claude to control a browser via computer use API."""

    def __init__(
        self,
        page: Page,
        config: AgentConfig,
        provider: LLMProvider,
        console: Console | None = None,
    ) -> None:
        """Initialize the computer use agent.

        Args:
            page: Playwright page to control.
            config: Agent configuration.
            provider: LLM provider instance.
            console: Rich console for output (optional).
        """
        self.page = page
        self.config = config
        self.provider = provider
        self.console = console or Console()
        self.messages: list[dict[str, Any]] = []
        self.token_tracker: TokenUsageTracker | None = None
        self.task_summary: str = ""
        self.completion_steps: int | str | None = None
        self.screenshot_dir: Path | None = None
        self._screenshot_counter: int = 0

    def _take_screenshot(self) -> tuple[str, str]:
        """Capture screenshot and return as base64 string with media type.

        When screenshot_dir is set, also saves the raw PNG to disk as
        screenshot_step_{N}.png.

        Returns:
            Tuple of (base64_data, media_type).
        """
        screenshot_bytes = self.page.screenshot(type="png", full_page=False)

        if self.screenshot_dir:
            dest = self.screenshot_dir / f"screenshot_step_{self._screenshot_counter}.png"
            dest.write_bytes(screenshot_bytes)
            self._screenshot_counter += 1

        return base64.b64encode(screenshot_bytes).decode("utf-8"), "image/png"

    def _prune_screenshot_history(self) -> None:
        """Replace old screenshots with placeholder text to reduce token usage.

        Keeps only the last max_screenshots_in_history screenshots in the message
        history. Earlier screenshots are replaced with a text placeholder.
        """
        opt = self.config.token_optimization
        if not opt.prune_history:
            return

        screenshot_locations: list[tuple[int, int, int]] = []

        for msg_idx, msg in enumerate(self.messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for content_idx, item in enumerate(content):
                if not isinstance(item, dict):
                    continue

                if item.get("type") != "tool_result":
                    continue

                tool_content = item.get("content", [])
                if not isinstance(tool_content, list):
                    continue

                for item_idx, tc in enumerate(tool_content):
                    if isinstance(tc, dict) and tc.get("type") == "image":
                        screenshot_locations.append((msg_idx, content_idx, item_idx))

        to_replace = screenshot_locations[: -opt.max_screenshots_in_history]

        for msg_idx, content_idx, item_idx in to_replace:
            content = self.messages[msg_idx]["content"]
            tool_result = content[content_idx]
            tool_content = tool_result.get("content", [])
            if isinstance(tool_content, list) and item_idx < len(tool_content):
                tool_content[item_idx] = {
                    "type": "text",
                    "text": "[Screenshot omitted - see recent screenshots for current state]",
                }

    def _print_element_info(self, x: int, y: int) -> None:
        """Print element information if verbose_selectors is enabled."""
        if not self.config.verbose_selectors:
            return

        element_info = get_element_at_point(self.page, x, y)
        if element_info:
            self.console.print(f"  [magenta]Element:[/magenta] {element_info}")

    def _execute_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Execute a computer use action on the browser.

        Args:
            action: The action type (screenshot, click, type, etc.).
            params: Action parameters.

        Returns:
            Result dictionary with screenshot or error info.
        """
        try:
            if action == "screenshot":
                return self._handle_screenshot()

            if action == "left_click":
                return self._handle_click(params, "left")

            if action == "right_click":
                return self._handle_click(params, "right")

            if action == "middle_click":
                return self._handle_click(params, "middle")

            if action == "double_click":
                return self._handle_double_click(params)

            if action == "triple_click":
                return self._handle_triple_click(params)

            if action == "type":
                return self._handle_type(params)

            if action == "key":
                return self._handle_key(params)

            if action == "mouse_move":
                return self._handle_mouse_move(params)

            if action == "scroll":
                return self._handle_scroll(params)

            if action == "left_click_drag":
                return self._handle_drag(params)

            if action == "wait":
                return self._handle_wait(params)

            return {"error": f"Unknown action: {action}"}

        except Exception as e:
            return {"error": str(e)}

    def _handle_screenshot(self) -> dict[str, Any]:
        """Handle screenshot action."""
        screenshot_b64, media_type = self._take_screenshot()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": screenshot_b64,
            },
        }

    def _handle_click(self, params: dict[str, Any], button: str) -> dict[str, Any]:
        """Handle click action."""
        x, y = params.get("coordinate", [0, 0])
        self.console.print(f"  [dim]Clicking {button} at ({x}, {y})[/dim]")
        self._print_element_info(x, y)
        self.page.mouse.click(x, y, button=button)
        self.page.wait_for_timeout(500)
        return self._handle_screenshot()

    def _handle_double_click(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle double click action."""
        x, y = params.get("coordinate", [0, 0])
        self.console.print(f"  [dim]Double-clicking at ({x}, {y})[/dim]")
        self._print_element_info(x, y)
        self.page.mouse.dblclick(x, y)
        self.page.wait_for_timeout(500)
        return self._handle_screenshot()

    def _handle_triple_click(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle triple click action."""
        x, y = params.get("coordinate", [0, 0])
        self.console.print(f"  [dim]Triple-clicking at ({x}, {y})[/dim]")
        self._print_element_info(x, y)
        self.page.mouse.click(x, y, click_count=3)
        self.page.wait_for_timeout(500)
        return self._handle_screenshot()

    def _handle_type(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle type action."""
        text = params.get("text", "")
        display_text = text[:50] + "..." if len(text) > 50 else text
        self.console.print(f'  [dim]Typing: "{display_text}"[/dim]')
        self.page.keyboard.type(text)
        self.page.wait_for_timeout(300)
        return self._handle_screenshot()

    def _handle_key(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle key press action."""
        key = params.get("key", "")
        self.console.print(f"  [dim]Pressing key: {key}[/dim]")
        self.page.keyboard.press(key)
        self.page.wait_for_timeout(300)
        return self._handle_screenshot()

    def _handle_mouse_move(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle mouse move action."""
        x, y = params.get("coordinate", [0, 0])
        self.console.print(f"  [dim]Moving mouse to ({x}, {y})[/dim]")
        self.page.mouse.move(x, y)

        if (
            self.config.token_optimization.lazy_screenshots
            and not self.provider.requires_screenshots
        ):
            return {
                "type": "text",
                "text": f"Mouse moved to ({x}, {y}). Screen unchanged.",
            }
        return self._handle_screenshot()

    def _handle_scroll(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle scroll action."""
        x, y = params.get("coordinate", [512, 384])
        direction = params.get("scroll_direction", "down")
        amount = params.get("scroll_amount", 3)

        delta_y = -100 * amount if direction == "up" else 100 * amount
        delta_x = 0
        if direction == "left":
            delta_x = -100 * amount
            delta_y = 0
        elif direction == "right":
            delta_x = 100 * amount
            delta_y = 0

        self.console.print(f"  [dim]Scrolling {direction} by {amount} at ({x}, {y})[/dim]")
        self.page.mouse.move(x, y)
        self.page.mouse.wheel(delta_x, delta_y)
        self.page.wait_for_timeout(500)
        return self._handle_screenshot()

    def _handle_drag(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle click and drag action."""
        start_x, start_y = params.get("start_coordinate", [0, 0])
        end_x, end_y = params.get("coordinate", [0, 0])
        self.console.print(
            f"  [dim]Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})[/dim]"
        )
        self.page.mouse.move(start_x, start_y)
        self.page.mouse.down()
        self.page.mouse.move(end_x, end_y)
        self.page.mouse.up()
        self.page.wait_for_timeout(500)
        return self._handle_screenshot()

    def _handle_wait(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle wait action."""
        duration = params.get("duration", 1)
        self.console.print(f"  [dim]Waiting {duration} seconds[/dim]")
        self.page.wait_for_timeout(int(duration * 1000))

        if (
            self.config.token_optimization.lazy_screenshots
            and not self.provider.requires_screenshots
        ):
            return {
                "type": "text",
                "text": f"Waited {duration} seconds. Request screenshot to see current state.",
            }
        return self._handle_screenshot()

    def _print_token_usage(self, iter_usage: IterationUsage) -> None:
        """Print iteration token usage with context % and cost.

        Args:
            iter_usage: Token usage for this iteration.
        """
        if not self.token_tracker:
            return

        context_pct = self.token_tracker.get_context_fill_percentage()
        context_window = self.token_tracker.format_context_window()
        cost = self.token_tracker.calculate_iteration_cost(iter_usage)

        self.console.print(
            f"[dim]Tokens: {iter_usage.input_tokens:,} in / {iter_usage.output_tokens:,} out | "
            f"Context: {context_pct:.1f}% of {context_window} | "
            f"Cost: ${cost:.4f}[/dim]"
        )

    def _print_session_summary(self) -> None:
        """Print final session statistics."""
        if not self.token_tracker:
            return

        total_in = self.token_tracker.total_input_tokens
        total_out = self.token_tracker.total_output_tokens
        total = self.token_tracker.total_tokens
        peak_pct = self.token_tracker.get_peak_context_percentage()
        cost = self.token_tracker.calculate_cost()
        num_iterations = len(self.token_tracker.iterations)

        self.console.print(
            f"\n[bold]Session Stats:[/bold] {num_iterations} iterations | "
            f"{total:,} tokens ({total_in:,} in / {total_out:,} out)"
        )
        self.console.print(
            f"[bold]Context Peak:[/bold] {peak_pct:.1f}% | "
            f"[bold]Estimated Cost:[/bold] ${cost:.4f}\n"
        )

    def _should_summarize(self) -> bool:
        """Check if context summarization is needed.

        Returns:
            True if context fill percentage exceeds threshold.
        """
        if self.provider.manages_own_context:
            return False
        opt = self.config.token_optimization
        if not opt.auto_summarize:
            return False
        if not self.token_tracker:
            return False

        fill_percent = self.token_tracker.get_context_fill_percentage()
        return fill_percent >= opt.summarize_threshold_percent

    def _format_messages_for_summary(self, messages: list[dict[str, Any]]) -> str:
        """Format messages for summarization prompt.

        Args:
            messages: List of messages to format.

        Returns:
            Formatted string representation of messages.
        """
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str):
                lines.append(f"{role}: {content[:500]}")
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", "")[:200])
                        elif item.get("type") == "tool_use":
                            action = item.get("input", {}).get("action", "unknown")
                            text_parts.append(f"[Action: {action}]")
                        elif item.get("type") == "tool_result":
                            text_parts.append("[Tool result]")
                if text_parts:
                    lines.append(f"{role}: {' | '.join(text_parts)}")

        return "\n".join(lines)

    def _summarize_context(self) -> None:
        """Summarize older messages to reduce context size."""
        opt = self.config.token_optimization
        preserve_count = opt.summarize_preserve_recent * 2

        if len(self.messages) <= preserve_count + 2:
            return

        messages_to_summarize = self.messages[1:-preserve_count]
        preserved_messages = self.messages[-preserve_count:]

        summary_prompt = (
            "Summarize this conversation history concisely. Include:\n"
            "1. The user's original goal\n"
            "2. Key actions taken (clicks, navigation, text input)\n"
            "3. Current progress status\n"
            "4. Any errors or issues\n\n"
            f"Conversation:\n{self._format_messages_for_summary(messages_to_summarize)}"
        )

        try:
            summary_response = self.provider.create_message(
                model=self.config.model,
                messages=[{"role": "user", "content": summary_prompt}],
                tools=[],
                system="You are a conversation summarizer. Be concise and factual.",
                max_tokens=1024,
            )

            summary_text = ""
            for block in summary_response.content:
                if block.type == "text" and block.text:
                    summary_text = block.text
                    break

            self.messages = [
                self.messages[0],
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"[Previous conversation summary]\n{summary_text}",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": "[Continuing from summary. Recent actions follow.]",
                },
                *preserved_messages,
            ]

            self.console.print(
                "\n[yellow]Context summarized to reduce token usage:[/yellow]"
            )
            self.console.print(f"[dim]{summary_text}[/dim]\n")

        except Exception as e:
            self.console.print(
                f"[dim]Warning: Could not summarize context: {e}[/dim]"
            )

    def generate_task_summary(self, original_prompt: str, result: str) -> str:
        """Generate compact summary of completed task for session context.

        Args:
            original_prompt: The original user task prompt.
            result: The result/response from the task.

        Returns:
            Compact summary string for session context.
        """
        iterations = len(self.token_tracker.iterations) if self.token_tracker else 0
        status = "completed" if "incomplete" not in result.lower() else "incomplete"

        summary_parts = [
            f"Task: {original_prompt[:100]}",
            f"Status: {status}",
            f"Iterations: {iterations}",
        ]

        if result and len(result) < 200:
            summary_parts.append(f"Result: {result}")
        elif result:
            summary_parts.append(f"Result: {result[:200]}...")

        self.task_summary = " | ".join(summary_parts)
        return self.task_summary

    def run(self, prompt: str) -> str:
        """Run the agent loop for a given prompt.

        Args:
            prompt: User's task description.

        Returns:
            Final response from Claude.
        """
        model_spec = get_model_spec(self.config.model)
        self.completion_steps = None
        self._screenshot_counter = 0
        if model_spec:
            self.token_tracker = TokenUsageTracker(model_spec)
        else:
            self.token_tracker = None

        self.provider.reset_conversation()

        self.messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        tools = [self.config.get_tool_definition()]
        iteration = 0

        self.console.print(f"\n[bold blue]Starting task:[/bold blue] {prompt}\n")

        while iteration < self.config.max_iterations:
            iteration += 1
            self.console.print(f"[yellow]Iteration {iteration}[/yellow]")

            iter_start = time.perf_counter()

            if self._should_summarize():
                self._summarize_context()

            step_start = time.perf_counter()
            response = self.provider.create_message(
                model=self.config.model,
                messages=self.messages,
                tools=tools,
                system=self.config.get_system_prompt(),
                max_tokens=self.config.max_tokens,
            )
            step_elapsed = time.perf_counter() - step_start

            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            self.messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            final_text = ""

            model_label = self.config.model.split("/")[-1].split("-")[0].upper()

            for block in response.content:
                if block.type == "text":
                    final_text = block.text or ""
                    self.console.print(f"[green]{model_label}:[/green] {block.text}")

                elif block.type == "tool_use":
                    action = block.input.get("action", "unknown") if block.input else "unknown"
                    self.console.print(f"[cyan]Action:[/cyan] {action}")

                    result = self._execute_action(action, block.input or {})

                    if "error" in result:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Error: {result['error']}",
                                "is_error": True,
                            }
                        )
                    else:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": [result],
                            }
                        )

            agent_elapsed = time.perf_counter() - iter_start

            if self.token_tracker:
                iter_usage = self.token_tracker.record_iteration(
                    iteration,
                    response.usage,
                    step_elapsed=step_elapsed,
                    agent_elapsed=agent_elapsed,
                )
                self._print_token_usage(iter_usage)

            if not tool_results:
                self.console.print("\n[bold green]Task completed![/bold green]")
                self._print_session_summary()
                self.completion_steps = iteration
                return final_text

            self.messages.append({"role": "user", "content": tool_results})
            self._prune_screenshot_history()

        self.console.print("\n[bold red]Max iterations reached![/bold red]")
        self._print_session_summary()
        self.completion_steps = "MAX_ITER_ERROR"
        return "Task incomplete: maximum iterations reached."
