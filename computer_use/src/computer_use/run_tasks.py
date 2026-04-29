"""CLI script to run the agent for multiple test cases from a YAML file."""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from computer_use.agent import ComputerUseAgent
from computer_use.browser import BrowserSession, create_browser
from computer_use.config import AgentConfig, DisplayConfig, get_tool_version
from computer_use.providers import LLMProvider, get_provider

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "square": (1000, 1000),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse resolution string into (width, height).

    Args:
        value: Resolution as 'WIDTHxHEIGHT' or preset name.

    Returns:
        Tuple of (width, height).

    Raises:
        argparse.ArgumentTypeError: If format is invalid.
    """
    value_lower = value.lower()

    if value_lower in RESOLUTION_PRESETS:
        return RESOLUTION_PRESETS[value_lower]

    match = re.match(r"^(\d+)x(\d+)$", value_lower)
    if match:
        return int(match.group(1)), int(match.group(2))

    preset_names = ", ".join(RESOLUTION_PRESETS.keys())
    raise argparse.ArgumentTypeError(
        f"Invalid resolution: '{value}'. Use WIDTHxHEIGHT (e.g., 1920x1080) "
        f"or a preset: {preset_names}"
    )


def parse_tasks_range(value: str) -> tuple[int, int]:
    """Parse a tasks range string like '10:20' into (start, end) 1-based inclusive.

    Supports formats:
        '5'      -> run only task 5
        '10:20'  -> run tasks 10 through 20 (inclusive)
        ':15'    -> run tasks 1 through 15
        '10:'    -> run tasks 10 through the last

    Args:
        value: Range string.

    Returns:
        Tuple of (start, end) where 0 means unbounded.

    Raises:
        argparse.ArgumentTypeError: If format is invalid.
    """
    if ":" in value:
        parts = value.split(":", 1)
        start = int(parts[0]) if parts[0].strip() else 0
        end = int(parts[1]) if parts[1].strip() else 0
        if start < 0 or end < 0:
            raise argparse.ArgumentTypeError(f"Range values must be non-negative: '{value}'")
        return start, end

    try:
        idx = int(value)
        return idx, idx
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid tasks range: '{value}'. Use N, START:END, :END, or START:"
        ) from None


def load_test_cases(path: str) -> list[dict[str, Any]]:
    """Load test cases from a YAML file.

    Args:
        path: Path to YAML file with 'test_cases' key.

    Returns:
        List of test case dicts.
    """
    with open(path) as f:
        return yaml.safe_load(f)["test_cases"]


def select_test_cases(
    test_cases: list[dict[str, Any]],
    tasks_range: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    """Filter test cases by 1-based inclusive range.

    Args:
        test_cases: Full list of test cases.
        tasks_range: Optional (start, end) tuple; 0 means unbounded.

    Returns:
        Filtered list of test cases.
    """
    if tasks_range is None:
        return test_cases

    start, end = tasks_range
    start_idx = max(start - 1, 0) if start > 0 else 0
    end_idx = end if end > 0 else len(test_cases)
    return test_cases[start_idx:end_idx]


# ── Post-run helpers ───────────────────────────────────────────────────────────


def _post_run_fetch_url(
    page: Any,
    base_url: str,
    post_run_url: str,
    console: Console,
) -> dict[str, Any]:
    """Navigate to a post-run URL and capture page content.

    Args:
        page: Playwright page (same session the agent used).
        base_url: Start URL used for the test case.
        post_run_url: Relative path to navigate to (e.g. "/finish").
        console: Rich console for status output.

    Returns:
        Dict with post_run_page_url, post_run_page_content, post_run_page_html,
        and post_run_page_error (if any).
    """
    full_url = urljoin(base_url, post_run_url)
    console.print(f"  [dim]Post-run URL: {full_url}[/dim]")
    result: dict[str, Any] = {"post_run_page_url": full_url}
    try:
        page.goto(full_url, wait_until="networkidle", timeout=30_000)
        result["post_run_page_content"] = page.inner_text("body")
        result["post_run_page_html"] = page.content()
        result["post_run_page_error"] = None
    except Exception as e:
        console.print(f"  [bold red]Failed to fetch {full_url}: {e}[/bold red]")
        result["post_run_page_content"] = None
        result["post_run_page_html"] = None
        result["post_run_page_error"] = str(e)
    return result


def _post_run_execute_js(
    page: Any,
    js_snippet: str,
    console: Console,
) -> dict[str, Any]:
    """Execute a JavaScript snippet in the page after the agent run.

    Args:
        page: Playwright page.
        js_snippet: JS code to evaluate.
        console: Rich console for status output.

    Returns:
        Dict with post_run_js_result and post_run_js_error.
    """
    console.print("  [dim]Executing post-run JS snippet...[/dim]")
    try:
        js_result = page.evaluate(js_snippet)
        return {"post_run_js_result": js_result, "post_run_js_error": None}
    except Exception as e:
        console.print(f"  [bold red]JS snippet failed: {e}[/bold red]")
        return {"post_run_js_result": None, "post_run_js_error": str(e)}


# ── Single task runner ─────────────────────────────────────────────────────────


def run_single_task(
    test_case: dict[str, Any],
    start_url: str,
    display_config: DisplayConfig,
    model: str,
    provider: LLMProvider,
    console: Console,
    post_run_url: str | None = None,
    post_run_js_snippet: str | None = None,
    headless: bool = False,
    screen: int | None = None,
    max_iterations: int = 100,
    additional_instructions: str = "",
    screenshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Launch a browser, run the agent for one test case, and return results.

    Args:
        test_case: Dict with at least 'id' and 'prompt' keys.
        start_url: URL to navigate to before starting.
        display_config: Viewport configuration.
        model: Model identifier string.
        provider: LLM provider instance.
        console: Rich console for output.
        post_run_url: Optional relative URL to navigate to after the agent finishes.
        post_run_js_snippet: Optional JS code to execute after the agent finishes.
        headless: Run browser headless.
        screen: Screen number for multi-monitor setups.
        additional_instructions: Extra instructions appended to the system prompt.
        screenshot_dir: Directory to save per-step screenshots (screenshot_step_N.png).

    Returns:
        Dict with raw_response, completion_steps, and post-run data.
    """
    tc_url = test_case.get("start_url", start_url)
    session: BrowserSession | None = None
    try:
        session = create_browser(
            display_config=display_config,
            start_url=tc_url,
            headless=headless,
            screen=screen,
        )
        agent_config = AgentConfig(
            model=model,
            max_iterations=max_iterations,
            display=display_config,
            additional_instructions=additional_instructions,
        )
        agent = ComputerUseAgent(
            page=session.page,
            config=agent_config,
            provider=provider,
            console=console,
        )

        if screenshot_dir:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            agent.screenshot_dir = screenshot_dir

        raw_response = agent.run(test_case["prompt"])

        result: dict[str, Any] = {
            "raw_response": raw_response,
            "completion_steps": agent.completion_steps,
            "messages": agent.messages,
            "token_tracker": agent.token_tracker,
        }

        if post_run_js_snippet:
            result.update(_post_run_execute_js(
                page=session.page,
                js_snippet=post_run_js_snippet,
                console=console,
            ))

        if post_run_url:
            result.update(_post_run_fetch_url(
                page=session.page,
                base_url=tc_url,
                post_run_url=post_run_url,
                console=console,
            ))

        return result
    finally:
        if session:
            session.close()


# ── Trajectory saving (gpt54-compatible format) ───────────────────────────────


def _build_trajectory_dir_name(
    model: str,
    test_id: str,
    start_time: datetime,
) -> str:
    """Build the per-task directory name matching the gpt54 convention.

    Format: {timestamp}_{model}_on_eval.{test_id}_{short_uuid}

    Args:
        model: Model identifier.
        test_id: Test case ID.
        start_time: When the task started.

    Returns:
        Directory name string.
    """
    ts = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    short_model = model.replace("/", "_")
    short_uuid = uuid.uuid4().hex[:32]
    return f"{ts}_{short_model}_on_eval.{test_id}_{short_uuid}"


def _extract_steps_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract a per-step trajectory list from the agent's message history.

    Each assistant turn that contains tool_use blocks is recorded as a step.

    Args:
        messages: Agent message history (user/assistant alternation).

    Returns:
        List of step dicts with action, model_response, err_msg fields.
    """
    steps: list[dict[str, Any]] = []
    step_idx = 0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        text_parts: list[str] = []
        actions: list[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_input = block.get("input", {})
                action = tool_input.get("action", "unknown")
                action_params = {k: v for k, v in tool_input.items() if k != "action"}
                actions.append(json.dumps({"action": action, **action_params}))

        model_response = "\n".join(text_parts + actions) if (text_parts or actions) else ""

        err_msg = None
        step_entry: dict[str, Any] = {
            "step": step_idx,
            "action": "\n".join(actions) if actions else model_response,
            "model_response": model_response,
            "raw_model_response": model_response,
            "chat_messages": None,
            "err_msg": err_msg,
            "stack_trace": None,
            "terminated": False,
            "truncated": False,
        }
        steps.append(step_entry)
        step_idx += 1

    return steps


def save_trajectory(
    task_dir: Path,
    test_case: dict[str, Any],
    result: dict[str, Any],
    model: str,
    start_url: str,
    post_run_url: str | None,
    post_run_js_snippet_path: str | None,
    console: Console,
) -> Path:
    """Save a single task's trajectory in gpt54-compatible format.

    Writes summary_info.json, agent_output.txt, and agent_outputs.json into
    the pre-created task_dir (which already contains screenshot_step_N.png
    files saved during the run).

    Args:
        task_dir: Pre-created directory for this task's output.
        test_case: The test case dict.
        result: Result dict from run_single_task.
        model: Model identifier.
        start_url: Starting URL for this test case.
        post_run_url: Relative post-run URL (if used).
        post_run_js_snippet_path: Original path to JS snippet file (if used).
        console: Rich console for output.

    Returns:
        Path to the trajectory directory.
    """
    test_id: str = test_case["id"]

    raw_response: str = result.get("raw_response", "")
    completion_steps = result.get("completion_steps")
    messages = result.get("messages", [])
    steps = _extract_steps_from_messages(messages)

    is_completed = isinstance(completion_steps, int)
    is_error = completion_steps == "ERROR"
    n_steps = completion_steps if isinstance(completion_steps, int) else len(steps)

    summary_info: dict[str, Any] = {
        "task_name": f"eval.{test_id}",
        "model_name": model,
        "n_steps": n_steps,
        "experiment_status": "completed" if is_completed else "error" if is_error else "max_steps",
        "completed": is_completed,
        "success": False,
        "error": is_error,
        "err_msg": raw_response if is_error else None,
        "stack_trace": None,
        "full_prompt": test_case.get("prompt", ""),
        "agent_response": raw_response,
        "raw_agent_response": raw_response,
        "post_run_url": post_run_url,
        "post_run_js_snippet_path": post_run_js_snippet_path,
        "post_run_js_result": result.get("post_run_js_result"),
        "post_run_js_error": result.get("post_run_js_error"),
        "post_run_page_url": result.get("post_run_page_url"),
        "post_run_page_content": result.get("post_run_page_content"),
        "post_run_page_html": result.get("post_run_page_html"),
        "post_run_page_error": result.get("post_run_page_error"),
    }

    token_tracker = result.get("token_tracker")
    if token_tracker is not None:
        for key, value in token_tracker.get_summary_stats().items():
            summary_info[f"stats.{key}"] = value

    (task_dir / "summary_info.json").write_text(
        json.dumps(summary_info, indent=4), encoding="utf-8"
    )

    (task_dir / "agent_output.txt").write_text(raw_response, encoding="utf-8")

    agent_outputs: dict[str, Any] = {
        "primary_output": raw_response,
        "agent_response": raw_response,
        "raw_agent_response": raw_response,
        "post_run_url": post_run_url,
        "post_run_js_result": result.get("post_run_js_result"),
        "post_run_js_error": result.get("post_run_js_error"),
        "post_run_js_snippet_path": post_run_js_snippet_path,
        "post_run_page_url": result.get("post_run_page_url"),
        "post_run_page_content": result.get("post_run_page_content"),
        "post_run_page_html": result.get("post_run_page_html"),
        "post_run_page_error": result.get("post_run_page_error"),
        "steps": steps,
    }

    (task_dir / "agent_outputs.json").write_text(
        json.dumps(agent_outputs, indent=4), encoding="utf-8"
    )

    console.print(f"  [dim]Trajectory saved: {task_dir}[/dim]")
    return task_dir


# ── Batch runner ───────────────────────────────────────────────────────────────


def run_all_tasks(
    test_cases: list[dict[str, Any]],
    start_url: str,
    display_config: DisplayConfig,
    model: str,
    provider: LLMProvider,
    console: Console,
    output_path: Path,
    *,
    post_run_url: str | None = None,
    post_run_js_snippet: str | None = None,
    post_run_js_snippet_path: str | None = None,
    results_dir: Path | None = None,
    headless: bool = False,
    screen: int | None = None,
    additional_instructions: str = "",
) -> list[dict[str, Any]]:
    """Run the agent for every test case and save results incrementally.

    Args:
        test_cases: Test cases to execute.
        start_url: Default starting URL.
        display_config: Viewport configuration.
        model: Model identifier string.
        provider: LLM provider instance.
        console: Rich console for output.
        output_path: Path to write aggregated results JSON.
        post_run_url: Relative URL to navigate to after each run.
        post_run_js_snippet: JS code to execute after each run.
        post_run_js_snippet_path: Original file path of the JS snippet (for metadata).
        results_dir: Directory for per-task trajectory output (gpt54 format).
        headless: Run browser headless.
        screen: Screen number for multi-monitor.
        additional_instructions: Extra system prompt instructions.

    Returns:
        List of response dicts.
    """
    responses: list[dict[str, Any]] = []

    metadata: dict[str, Any] = {
        "model": model,
        "start_url": start_url,
        "resolution": f"{display_config.width}x{display_config.height}",
        "timestamp": datetime.now(UTC).isoformat(),
        "total_tasks": len(test_cases),
        "headless": headless,
    }

    for i, tc in enumerate(test_cases, 1):
        test_id: str = tc["id"]
        description: str = tc.get("description", "")

        console.rule(f"[bold cyan]Task {i}/{len(test_cases)}: {test_id}[/bold cyan]")
        if description:
            console.print(f"[dim]{description}[/dim]")

        task_start = datetime.now(UTC)

        task_dir: Path | None = None
        if results_dir:
            dir_name = _build_trajectory_dir_name(model, test_id, task_start)
            task_dir = results_dir / dir_name
            task_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = run_single_task(
                test_case=tc,
                start_url=start_url,
                display_config=display_config,
                model=model,
                provider=provider,
                console=console,
                post_run_url=post_run_url,
                post_run_js_snippet=post_run_js_snippet,
                headless=headless,
                screen=screen,
                additional_instructions=additional_instructions,
                screenshot_dir=task_dir,
            )
        except Exception as e:
            console.print(f"[bold red]Error on {test_id}: {e}[/bold red]")
            result = {
                "raw_response": f"ERROR: {e}",
                "completion_steps": "ERROR",
                "messages": [],
            }

        entry: dict[str, Any] = {
            "test_id": test_id,
            "raw_response": result["raw_response"],
            "completion_steps": result.get("completion_steps"),
        }
        for key in (
            "post_run_js_result",
            "post_run_js_error",
            "post_run_page_url",
            "post_run_page_content",
        ):
            if key in result:
                entry[key] = result[key]

        responses.append(entry)
        console.print(f"[green]Saved response for {test_id}[/green]\n")

        _save_incremental(output_path, metadata, responses)

        if task_dir:
            save_trajectory(
                task_dir=task_dir,
                test_case=tc,
                result=result,
                model=model,
                start_url=tc.get("start_url", start_url),
                post_run_url=post_run_url,
                post_run_js_snippet_path=post_run_js_snippet_path,
                console=console,
            )

    return responses


def _save_incremental(
    output_path: Path,
    metadata: dict[str, Any],
    responses: list[dict[str, Any]],
) -> None:
    """Persist current progress so far to the output file.

    Args:
        output_path: Destination JSON path.
        metadata: Run metadata dict.
        responses: Responses collected so far.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"metadata": metadata, "responses": responses}
    output_path.write_text(json.dumps(output_data, indent=2))


def print_summary_table(
    console: Console,
    test_cases: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> None:
    """Print a summary table of all task results.

    Args:
        console: Rich console for output.
        test_cases: Original test case dicts.
        responses: Collected response dicts.
    """
    table = Table(title="Task Results Summary")
    table.add_column("#", style="dim", width=4)
    table.add_column("Test ID", style="cyan")
    table.add_column("Steps", justify="right")
    table.add_column("Status", justify="center")

    for i, resp in enumerate(responses, 1):
        steps = resp.get("completion_steps")
        if steps is None or steps == "ERROR":
            status = "[red]FAILED[/red]"
            steps_str = str(steps or "?")
        elif steps == "MAX_ITER_ERROR":
            status = "[yellow]MAX_ITER[/yellow]"
            steps_str = "max"
        else:
            status = "[green]OK[/green]"
            steps_str = str(steps)

        table.add_row(str(i), resp["test_id"], steps_str, status)

    console.print()
    console.print(table)


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run computer-use agent on multiple test cases from a YAML file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --test-cases test_cases/circuit.yaml --start-url https://example.com
  %(prog)s --test-cases test_cases/circuit.yaml --start-url https://example.com --tasks-range 10:20
  %(prog)s --test-cases test_cases/circuit.yaml --start-url https://example.com \\
      --results-dir results/sonnet --post-run-url /finish
  %(prog)s --test-cases test_cases/video.yaml --start-url http://localhost:5173 \\
      --results-dir results/run1 --post-run-js-snippet scripts/export.js --post-run-url /finish
""",
    )

    parser.add_argument(
        "--test-cases",
        required=True,
        type=str,
        help="Path to YAML file containing test cases",
    )
    parser.add_argument(
        "--start-url",
        required=True,
        type=str,
        help="Default starting URL for each browser session",
    )
    parser.add_argument(
        "--tasks-range",
        type=parse_tasks_range,
        default=None,
        help=(
            "Run a subset of tasks: N (single), START:END (inclusive), "
            ":END (from first), START: (to last)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        help="Model to use (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=None,
        help="Browser resolution: WIDTHxHEIGHT or preset (square, hd, fhd)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save aggregated responses JSON (default: responses/<yaml_stem>_responses.json)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save per-task trajectory results (gpt54-compatible format)",
    )
    parser.add_argument(
        "--screen",
        type=int,
        default=None,
        help="Screen number to open browser on (1 = primary, 2 = secondary, etc.)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )
    parser.add_argument(
        "--post-run-url",
        type=str,
        default=None,
        help="Relative URL to navigate to after each agent run (e.g. /finish)",
    )
    parser.add_argument(
        "--post-run-js-snippet",
        type=str,
        default=None,
        help="Path to a JS file to execute in the page after each agent run",
    )
    parser.add_argument(
        "--additional-instructions",
        type=str,
        default="",
        help="Extra instructions appended to the system prompt",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the batch task runner CLI."""
    load_dotenv()
    args = parse_args()
    console = Console()

    title = Text("Computer Use Agent — Batch Runner", style="bold magenta")
    console.print(Panel(title, expand=False))

    provider_name = os.getenv("LLM_PROVIDER", "anthropic")
    console.print(f"[dim]Provider: {provider_name}[/dim]")

    try:
        provider = get_provider()
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    tool_version = get_tool_version(args.model)
    console.print(f"[dim]Model:      {args.model}[/dim]")
    console.print(f"[dim]Tool:       {tool_version}[/dim]")
    console.print(f"[dim]Start URL:  {args.start_url}[/dim]")
    console.print(f"[dim]Test cases: {args.test_cases}[/dim]")
    console.print(f"[dim]Headless:   {args.headless}[/dim]")

    if args.resolution:
        width, height = args.resolution
        display_config = DisplayConfig(width=width, height=height)
    else:
        display_config = DisplayConfig()
    console.print(f"[dim]Resolution: {display_config.width}x{display_config.height}[/dim]")

    all_test_cases = load_test_cases(args.test_cases)
    selected = select_test_cases(all_test_cases, args.tasks_range)

    if not selected:
        console.print("[bold red]No test cases matched the given range.[/bold red]")
        sys.exit(1)

    range_label = f"{args.tasks_range[0]}:{args.tasks_range[1]}" if args.tasks_range else "all"
    console.print(
        f"[dim]Tasks:      {len(selected)}/{len(all_test_cases)} (range: {range_label})[/dim]"
    )

    yaml_stem = Path(args.test_cases).stem
    output_path = Path(args.output) if args.output else Path(f"responses/{yaml_stem}_responses.json")
    console.print(f"[dim]Output:     {output_path}[/dim]")

    results_dir: Path | None = None
    if args.results_dir:
        results_dir = Path(args.results_dir)
        console.print(f"[dim]Results dir: {results_dir}[/dim]")

    post_run_js_snippet: str | None = None
    post_run_js_snippet_path: str | None = None
    if args.post_run_js_snippet:
        snippet_file = Path(args.post_run_js_snippet)
        if not snippet_file.is_file():
            console.print(
                f"[bold red]Error:[/bold red] JS snippet not found: {args.post_run_js_snippet}"
            )
            sys.exit(1)
        post_run_js_snippet = snippet_file.read_text()
        post_run_js_snippet_path = args.post_run_js_snippet
        console.print(f"[dim]JS snippet:  {args.post_run_js_snippet}[/dim]")

    if args.post_run_url:
        console.print(f"[dim]Post-run URL: {args.post_run_url}[/dim]")

    console.print()

    responses = run_all_tasks(
        test_cases=selected,
        start_url=args.start_url,
        display_config=display_config,
        model=args.model,
        provider=provider,
        console=console,
        output_path=output_path,
        post_run_url=args.post_run_url,
        post_run_js_snippet=post_run_js_snippet,
        post_run_js_snippet_path=post_run_js_snippet_path,
        results_dir=results_dir,
        headless=args.headless,
        screen=args.screen,
        additional_instructions=args.additional_instructions,
    )

    print_summary_table(console, selected, responses)
    console.print(
        f"\n[bold green]All {len(responses)} responses saved to: {output_path}[/bold green]"
    )
    if results_dir:
        console.print(f"[bold green]Trajectories saved in: {results_dir}[/bold green]")


if __name__ == "__main__":
    main()
