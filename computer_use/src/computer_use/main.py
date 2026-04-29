"""Main entry point for the Claude Computer Use Agent CLI."""

import argparse
import os
import re
import sys
import time

from dotenv import load_dotenv
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from computer_use.agent import ComputerUseAgent
from computer_use.browser import BrowserSession, create_browser
from computer_use.config import AgentConfig, DisplayConfig, get_tool_version
from computer_use.providers import LLMProvider, get_provider

RESOLUTION_PRESETS = {
    "square": (1000, 1000),
    "hd": (1280, 720),
    "fhd": (1920, 1080),
}


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse resolution string into width and height.

    Args:
        value: Resolution as 'WIDTHxHEIGHT' or preset name (square, hd, fhd).

    Returns:
        Tuple of (width, height).

    Raises:
        argparse.ArgumentTypeError: If resolution format is invalid.
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude Computer Use Agent - Control a browser with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Resolution presets:
  square    1000x1000
  hd        1280x720
  fhd       1920x1080

Examples:
  %(prog)s --start-url https://google.com --resolution fhd
  %(prog)s --start-url https://example.com --prompt "Click the login button"
  %(prog)s --resolution 1920x1080 --prompt "Search for Claude AI"
  %(prog)s --screen 2 --start-url https://example.com  # Open on second monitor
""",
    )

    parser.add_argument(
        "--start-url",
        type=str,
        help="Starting URL for the browser",
    )

    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        help="Browser resolution: WIDTHxHEIGHT or preset (square, hd, fhd)",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Initial task to execute (skips interactive prompt for first task)",
    )

    parser.add_argument(
        "--screen",
        type=int,
        help="Screen number to open browser on (1 = primary, 2 = secondary, etc.)",
    )

    parser.add_argument(
        "--verbose-selectors",
        action="store_true",
        help="Display HTML element info for click coordinates (useful for debugging)",
    )

    return parser.parse_args()


def select_model(console: Console, provider: LLMProvider) -> str:
    """Prompt user to select a model from available options.

    Args:
        console: Rich console for output.
        provider: LLM provider instance.

    Returns:
        Selected model identifier.
    """
    console.print("\n[bold]Select Model[/bold]\n")

    models = provider.get_available_models()
    if not models:
        console.print("[bold red]Error:[/bold red] No models available.")
        sys.exit(1)

    choices = [Choice(value=model, name=model) for model in models]

    model = inquirer.select(
        message="Choose a model:",
        choices=choices,
        default=models[0],
    ).execute()

    return model


def select_start_url(console: Console) -> str:
    """Prompt user for starting URL."""
    url = inquirer.text(
        message="Starting URL:",
    ).execute()

    return url


def _build_prompt_with_context(prompt: str, session_summaries: list[str]) -> str:
    """Build prompt with session context from previous tasks.

    Args:
        prompt: Current task prompt.
        session_summaries: List of summaries from previous tasks.

    Returns:
        Full prompt with context if available.
    """
    if not session_summaries:
        return prompt

    context = "## Previous Tasks in This Session\n" + "\n".join(
        f"- {summary}" for summary in session_summaries
    )
    return f"{context}\n\n## Current Task\n{prompt}"


def _print_session_context(console: Console, session_summaries: list[str]) -> None:
    """Print previous task summaries to the console.

    Args:
        console: Rich console for output.
        session_summaries: List of summaries from previous tasks.
    """
    if not session_summaries:
        return

    console.print("\n[bold magenta]Previous Tasks in This Session:[/bold magenta]")
    for i, summary in enumerate(session_summaries, 1):
        console.print(f"  [dim]{i}.[/dim] {summary}")
    console.print()


def run_interactive_loop(
    agent: ComputerUseAgent,
    console: Console,
    initial_prompt: str | None = None,
) -> None:
    """Run the interactive prompt loop.

    Args:
        agent: The computer use agent.
        console: Rich console for output.
        initial_prompt: Optional initial task to execute before interactive loop.
    """
    console.print("\n[bold green]Agent ready![/bold green]")
    console.print(
        "Enter your tasks. [dim](Esc+Enter to submit, 'quit' to exit)[/dim]\n"
    )

    session_summaries: list[str] = []

    if initial_prompt:
        console.print(f"[bold cyan]Task:[/bold cyan] {initial_prompt}\n")
        result = agent.run(initial_prompt)
        summary = agent.generate_task_summary(initial_prompt, result)
        session_summaries.append(summary)

    last_interrupt_time: float = 0

    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _submit(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.validate_and_handle()

    session: PromptSession[str] = PromptSession(
        multiline=True,
        key_bindings=bindings,
    )

    while True:
        try:
            prompt = session.prompt("Task: ", in_thread=True)

            if not prompt.strip():
                console.print("[dim]Please enter a task[/dim]")
                continue

            if prompt.strip().lower() in ("quit", "exit", "q"):
                console.print("\n[yellow]Goodbye![/yellow]\n")
                break

            _print_session_context(console, session_summaries)
            full_prompt = _build_prompt_with_context(prompt, session_summaries)
            result = agent.run(full_prompt)
            summary = agent.generate_task_summary(prompt, result)
            session_summaries.append(summary)

        except KeyboardInterrupt:
            current_time = time.time()
            if current_time - last_interrupt_time < 2.0:
                console.print("\n[yellow]Goodbye![/yellow]\n")
                break
            last_interrupt_time = current_time
            console.print(
                "\n[yellow]Interrupted. Press Ctrl+C again to exit.[/yellow]"
            )
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]\n")
            break


def main() -> None:
    """Main entry point for the CLI."""
    load_dotenv()
    args = parse_args()

    console = Console()

    title = Text("Computer Use Agent", style="bold magenta")
    console.print(Panel(title, expand=False))

    provider_name = os.getenv("LLM_PROVIDER", "anthropic")
    console.print(f"[dim]Provider: {provider_name}[/dim]")

    try:
        provider = get_provider()
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    model = select_model(console, provider)
    tool_version = get_tool_version(model)

    console.print(f"\n[dim]Selected model: {model}[/dim]")
    console.print(f"[dim]Tool version: {tool_version}[/dim]")

    if args.start_url:
        start_url = args.start_url
        console.print(f"[dim]Start URL: {start_url}[/dim]")
    else:
        start_url = select_start_url(console)

    if args.resolution:
        width, height = args.resolution
        display_config = DisplayConfig(width=width, height=height)
        console.print(f"[dim]Resolution: {width}x{height}[/dim]")
    else:
        display_config = DisplayConfig()

    agent_config = AgentConfig(
        model=model,
        display=display_config,
        verbose_selectors=args.verbose_selectors,
    )

    if args.verbose_selectors:
        console.print("[dim]Verbose selectors: enabled[/dim]")

    console.print("\n[bold]Launching browser...[/bold]")

    if args.screen:
        console.print(f"[dim]Screen: {args.screen}[/dim]")

    session: BrowserSession | None = None
    try:
        session = create_browser(
            display_config=display_config,
            start_url=start_url,
            screen=args.screen,
        )

        agent = ComputerUseAgent(
            page=session.page,
            config=agent_config,
            provider=provider,
            console=console,
        )

        run_interactive_loop(agent, console, initial_prompt=args.prompt)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    finally:
        if session:
            console.print("[dim]Closing browser...[/dim]")
            session.close()


if __name__ == "__main__":
    main()
