import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from eval.register import register_evaluation_tasks
from src.agisdk import REAL
from src.agisdk.REAL.browsergym.webclones.task_config import load_tasks_from_file

load_dotenv()

console = Console()

THIS_FILE_PATH = Path(__file__).parent
TASKS_PATH = THIS_FILE_PATH / "tasks" / "eval"
TASK_SCOPE_NAME = "eval"


def create_header():
    title = Text("Agent Task Runner", style="bold magenta")
    subtitle = Text("Run tasks and save task artifacts", style="italic cyan")

    header_table = Table.grid(padding=1)
    header_table.add_column(justify="center")
    header_table.add_row(title)
    header_table.add_row(subtitle)

    return Panel(
        Align.center(header_table),
        box=box.DOUBLE,
        border_style="magenta",
        padding=(1, 2),
    )


def create_table_options(args: argparse.Namespace, num_workers: int, task_names: list[str]) -> Table:
    table = Table(
        title="[bold cyan]Configuration[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        title_style="bold cyan",
        show_header=True,
        header_style="bold magenta",
        row_styles=["", "dim"],
    )
    table.add_column("Option", style="bold", no_wrap=True, width=16)
    table.add_column("Value", style="green", min_width=20)
    table.add_row("Model", f"[yellow]{args.model}[/yellow]")
    table.add_row("Concurrent", "[green]Yes[/green]" if args.concurrent else "[red]No[/red]")
    table.add_row("Headless", "[green]Yes[/green]" if args.headless else "[red]No[/red]")
    table.add_row("Workers", f"[cyan]{num_workers}[/cyan]")
    table.add_row("Results Dir", f"[magenta]{Path(args.results_dir).resolve()}[/magenta]")
    table.add_row("Max Steps", f"[yellow]{args.max_steps}[/yellow]")

    if len(task_names) == 1:
        table.add_row("Task", f"[cyan]{task_names[0]}[/cyan]")
    else:
        table.add_row("Tasks", f"[yellow]{len(task_names)} total[/yellow]")

    if args.application:
        table.add_row("Application", f"[magenta]{args.application}[/magenta]")

    if args.iterations > 1:
        table.add_row("Iterations", f"[yellow]{args.iterations}[/yellow]")

    if args.task_file:
        table.add_row("Task File", f"[magenta]{Path(args.task_file).resolve()}[/magenta]")
    if args.url:
        table.add_row("URL", f"[magenta]{args.url}[/magenta]")
    if args.js_snippet_file:
        table.add_row("JS Snippet", f"[magenta]{Path(args.js_snippet_file).resolve()}[/magenta]")
    if args.post_run_url:
        table.add_row("Post-run URL", f"[magenta]{args.post_run_url}[/magenta]")

    return table


def create_run_summary_table(manifest_entries: list[dict], manifest_path: Path) -> Table:
    completed = sum(1 for entry in manifest_entries if entry["status"] == "completed")
    truncated = sum(1 for entry in manifest_entries if entry["status"] == "truncated")
    errored = sum(1 for entry in manifest_entries if entry["status"] == "error")
    total_step_files = sum(entry["step_state_count"] for entry in manifest_entries)
    total_screenshots = sum(entry["screenshot_count"] for entry in manifest_entries)

    table = Table(
        title="[bold magenta]Saved Run Artifacts[/bold magenta]",
        box=box.ROUNDED,
        border_style="magenta",
        header_style="bold cyan",
        row_styles=["", "dim"],
    )
    table.add_column("Metric", style="bold", width=18)
    table.add_column("Value", min_width=20)
    table.add_row("Task Runs", f"[cyan]{len(manifest_entries)}[/cyan]")
    table.add_row("Completed", f"[green]{completed}[/green]")
    table.add_row("Truncated", f"[yellow]{truncated}[/yellow]")
    table.add_row("Errors", f"[red]{errored}[/red]")
    table.add_row("Step States", f"[cyan]{total_step_files}[/cyan]")
    table.add_row("Screenshots", f"[cyan]{total_screenshots}[/cyan]")
    table.add_row("Manifest", f"[magenta]{manifest_path}[/magenta]")
    return table


def create_task_artifacts_table(record: dict) -> Table:
    status_colors = {
        "completed": "green",
        "truncated": "yellow",
        "error": "red",
        "finished": "cyan",
    }
    status_color = status_colors.get(record["status"], "cyan")

    table = Table(
        title=f"[bold magenta]Artifacts: {record['task_name']}[/bold magenta]",
        box=box.ROUNDED,
        border_style="magenta",
        header_style="bold cyan",
        row_styles=["", "dim"],
    )
    table.add_column("Field", style="bold", width=18)
    table.add_column("Value", min_width=20)
    table.add_row("Status", f"[{status_color}]{record['status']}[/{status_color}]")
    table.add_row("Iteration", f"[cyan]{record['iteration']}[/cyan]")
    table.add_row("Task Steps", f"[cyan]{record['task_steps']}[/cyan]")
    table.add_row("Step States", f"[cyan]{record['step_state_count']}[/cyan]")
    table.add_row("Screenshots", f"[cyan]{record['screenshot_count']}[/cyan]")
    table.add_row("Summary", f"[magenta]{record['summary_info_path']}[/magenta]")
    table.add_row("Log", f"[magenta]{record['experiment_log_path']}[/magenta]")
    table.add_row("Agent Output", f"[magenta]{record['agent_output_text_path']}[/magenta]")
    table.add_row("Agent Outputs", f"[magenta]{record['agent_outputs_path']}[/magenta]")
    table.add_row("Run Dir", f"[magenta]{record['exp_dir']}[/magenta]")
    return table


def log_error(message):
    console.print(f"[red]{message}[/red]")


def log_info(message):
    console.print(f"{message}")


def log_success(message):
    console.print(f"[green]{message}[/green]")


def get_task_name(task_file: dict) -> str:
    return f"{TASK_SCOPE_NAME}.{task_file['id']}"


def list_tasks(task_source: Path) -> list[dict]:
    if task_source.is_file():
        return load_tasks_from_file(task_source)

    tasks = []
    for pattern in ("**/*.json", "**/*.yaml", "**/*.yml"):
        for task_file in sorted(task_source.glob(pattern)):
            tasks.extend(load_tasks_from_file(task_file))
    return tasks


def get_random_tasks(task_list: list[dict], num_tasks: int) -> list[dict]:
    if num_tasks >= len(task_list):
        shuffled_tasks = task_list.copy()
        random.shuffle(shuffled_tasks)
        return shuffled_tasks
    return random.sample(task_list, num_tasks)


def filter_tasks_by_application(tasks: list[dict], application: str) -> list[dict]:
    return [task for task in tasks if task.get("website", {}).get("id") == application]


def filter_tasks_by_id(tasks: list[dict], task_id: str) -> list[dict]:
    normalized_task_id = task_id.split(".", 1)[1] if task_id.startswith(f"{TASK_SCOPE_NAME}.") else task_id
    return [task for task in tasks if task.get("id") == normalized_task_id]


def get_task_status(result: dict) -> str:
    if result.get("err_msg") or result.get("stack_trace") or result.get("error"):
        return "error"
    if result.get("truncated"):
        return "truncated"
    if result.get("terminated") or result.get("completed"):
        return "completed"
    return "finished"


def build_artifact_record(
    task_name: str,
    result: dict,
    iteration: int,
    include_finish_page: bool = True,
) -> dict:
    exp_dir_value = result.get("exp_dir")
    exp_dir = Path(exp_dir_value).resolve() if exp_dir_value else None

    step_state_files = []
    screenshot_files = []
    summary_info_path = None
    experiment_log_path = None
    agent_outputs_path = None
    agent_output_text_path = None

    if exp_dir and exp_dir.exists():
        step_state_files = sorted(str(path.resolve()) for path in exp_dir.glob("step_*.pkl.gz"))
        screenshot_files = sorted(str(path.resolve()) for path in exp_dir.glob("screenshot_step_*.png"))

        potential_summary_path = exp_dir / "summary_info.json"
        if potential_summary_path.exists():
            summary_info_path = str(potential_summary_path.resolve())

        potential_log_path = exp_dir / "experiment.log"
        if potential_log_path.exists():
            experiment_log_path = str(potential_log_path.resolve())

        potential_agent_outputs_path = exp_dir / "agent_outputs.json"
        if potential_agent_outputs_path.exists():
            agent_outputs_path = str(potential_agent_outputs_path.resolve())

        potential_agent_output_text_path = exp_dir / "agent_output.txt"
        if potential_agent_output_text_path.exists():
            agent_output_text_path = str(potential_agent_output_text_path.resolve())

    return {
        "iteration": iteration + 1,
        "task_name": task_name,
        "task_id": task_name.split(".", 1)[1] if "." in task_name else task_name,
        "status": get_task_status(result),
        "task_steps": result.get("n_steps", 0),
        "exp_dir": str(exp_dir) if exp_dir else None,
        "summary_info_path": summary_info_path,
        "experiment_log_path": experiment_log_path,
        "agent_outputs_path": agent_outputs_path,
        "agent_output_text_path": agent_output_text_path,
        "step_state_files": step_state_files,
        "step_state_count": len(step_state_files),
        "screenshot_files": screenshot_files,
        "screenshot_count": len(screenshot_files),
        "agent_response": result.get("agent_response", ""),
        "finish_page_content": result.get("finish_page_content") if include_finish_page else None,
        "finish_page_html": result.get("finish_page_html") if include_finish_page else None,
        "finish_page_axtree": result.get("finish_page_axtree") if include_finish_page else None,
        "post_run_js_snippet_path": result.get("post_run_js_snippet_path"),
        "post_run_js_result": result.get("post_run_js_result"),
        "post_run_js_error": result.get("post_run_js_error"),
        "post_run_page_url": result.get("post_run_page_url"),
        "post_run_page_content": result.get("post_run_page_content"),
        "post_run_page_html": result.get("post_run_page_html"),
        "post_run_page_axtree": result.get("post_run_page_axtree"),
        "post_run_page_error": result.get("post_run_page_error"),
        "terminated": result.get("terminated", False),
        "truncated": result.get("truncated", False),
        "error": bool(result.get("err_msg") or result.get("stack_trace") or result.get("error")),
    }


def save_run_manifest(
    args: argparse.Namespace,
    task_names: list[str],
    manifest_entries: list[dict],
    num_workers: int,
) -> Path:
    results_dir = Path(args.results_dir).resolve()
    manifests_dir = results_dir / "run_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest_path = manifests_dir / f"run_{timestamp}.json"

    payload = {
        "created_at": datetime.now().isoformat(),
        "model": args.model,
        "headless": args.headless,
        "concurrent": args.concurrent,
        "num_workers": num_workers,
        "max_steps": args.max_steps,
        "results_dir": str(results_dir),
        "task_file": str(Path(args.task_file).resolve()) if args.task_file else None,
        "url": args.url or None,
        "js_snippet_file": str(Path(args.js_snippet_file).resolve()) if args.js_snippet_file else None,
        "post_run_url": args.post_run_url or None,
        "tasks": task_names,
        "iterations": args.iterations,
        "runs": manifest_entries,
    }

    with open(manifest_path, "w") as file:
        json.dump(payload, file, indent=2)

    return manifest_path


def main():
    overall_start_time = time.time()

    console.print(create_header())
    console.print()

    parser = argparse.ArgumentParser(description="Run agent tasks and save artifacts")
    parser.add_argument("--model", "-m", type=str, help="Model to use", default="o3")
    parser.add_argument("--concurrent", "-c", action="store_true", help="Run tasks in parallel", default=False)
    parser.add_argument("--workers", type=int, default=0, help="Number of workers to use with --concurrent (default: auto)")
    parser.add_argument("--task", "--id", type=str, help="Task to run", default="")
    parser.add_argument("--run-all", "-a", action="store_true", help="Run all selected tasks", default=False)
    parser.add_argument("--headless", action="store_true", help="Run in headless mode", default=False)
    parser.add_argument("--verbose", action="store_true", help="Verbose output", default=False)
    parser.add_argument("--application", "--app", type=str, help="Filter tasks by application name", default="")
    parser.add_argument("--run-random", "-r", action="store_true", help="Run random tasks from the selected pool", default=False)
    parser.add_argument("--iterations", "-i", type=int, help="Number of iterations to run", default=1)
    parser.add_argument("--max-tasks", "-n", type=int, help="Maximum number of tasks to run (0 means all)")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum number of agent steps per task")
    parser.add_argument("--task-file", type=str, default="", help="Path to a YAML or JSON testcase file")
    parser.add_argument("--url", type=str, default="", help="Start URL override for testcase files that do not define one")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory where task artifacts will be saved")
    parser.add_argument("--js-snippet-file", type=str, default="", help="Path to a JavaScript file to execute on the final page for each testcase")
    parser.add_argument("--post-run-url", type=str, default="", help="Optional URL to open after each testcase before capturing extra page data and running the JS snippet")
    parser.add_argument("--system-prompt", type=str, default="", help="Append extra instructions to the built-in agent system prompt")
    parser.add_argument("--system-prompt-file", type=str, default="", help="Append extra system instructions from a text file")
    args = parser.parse_args()

    os.environ["VERBOSE"] = "1" if args.verbose else "0"
    os.environ["LLM_MODEL_NAME"] = args.model
    if args.url:
        os.environ["WEBCLONE_URL"] = args.url

    js_snippet_source = None
    js_snippet_path = None
    system_prompt_append = None
    if args.js_snippet_file:
        js_snippet_path = Path(args.js_snippet_file).expanduser().resolve()
        if not js_snippet_path.exists():
            log_error(f"JavaScript snippet file not found: {js_snippet_path}")
            return {}
        js_snippet_source = js_snippet_path.read_text()

    if args.system_prompt and args.system_prompt_file:
        log_error("Provide either --system-prompt or --system-prompt-file, not both.")
        return {}

    if args.system_prompt_file:
        system_prompt_path = Path(args.system_prompt_file).expanduser().resolve()
        if not system_prompt_path.exists():
            log_error(f"System prompt file not found: {system_prompt_path}")
            return {}
        system_prompt_append = system_prompt_path.read_text()
    elif args.system_prompt:
        system_prompt_append = args.system_prompt

    task_source = Path(args.task_file).expanduser().resolve() if args.task_file else TASKS_PATH
    if not task_source.exists():
        log_error(f"Task source not found: {task_source}")
        return {}

    register_evaluation_tasks([task_source])
    all_tasks = list_tasks(task_source)

    if not args.url and not os.environ.get("WEBCLONE_URL"):
        tasks_missing_urls = [task["id"] for task in all_tasks if not task.get("website", {}).get("url")]
        if tasks_missing_urls:
            log_error("Some tasks do not define a URL. Provide --url or set WEBCLONE_URL.")
            return {}

    if args.application:
        all_tasks = filter_tasks_by_application(all_tasks, args.application)
        if not all_tasks:
            log_error(f"No tasks found for application '{args.application}'")
            return {}

    if args.task:
        all_tasks = filter_tasks_by_id(all_tasks, args.task)
        if not all_tasks:
            log_error(f"Task {args.task} not found")
            return {}

    if args.run_random:
        num_tasks = len(all_tasks) if not args.max_tasks else min(args.max_tasks, len(all_tasks))
        all_tasks = get_random_tasks(all_tasks, num_tasks)
    elif args.max_tasks:
        all_tasks = all_tasks[:args.max_tasks]

    task_names = [get_task_name(task) for task in all_tasks]
    if not task_names:
        log_error("No tasks found to run")
        return {}

    if args.workers < 0:
        log_error("--workers must be 0 or greater")
        return {}

    if args.concurrent:
        if args.workers > 0:
            num_workers = min(len(task_names), args.workers)
        else:
            num_workers = min(len(task_names), 6)
    else:
        num_workers = 1

    console.print(create_table_options(args, num_workers, task_names))
    console.print()

    all_iteration_results = defaultdict(list)
    manifest_entries = []
    include_finish_page_in_manifest = not (args.js_snippet_file or args.post_run_url)

    harness = REAL.harness(
        model=args.model,
        headless=args.headless,
        max_steps=args.max_steps,
        use_html=False,
        use_axtree=True,
        use_screenshot=True,
        system_prompt_append=system_prompt_append,
        use_cache=False,
        force_refresh=True,
        results_dir=args.results_dir,
        num_workers=num_workers,
        save_step_screenshots=True,
        save_step_info=True,
        show_task_completion_summary=False,
        post_run_js_snippet=js_snippet_source,
        post_run_js_snippet_path=str(js_snippet_path) if js_snippet_path else None,
        post_run_url=args.post_run_url or None,
        registration_paths=[str(task_source.resolve())],
    )

    total_task_runs = len(task_names) * args.iterations

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        progress_task_id = progress.add_task("Running tasks", total=total_task_runs)

        for iteration in range(args.iterations):
            if args.iterations > 1:
                log_info(f"\n🔄 Running iteration {iteration + 1}/{args.iterations}")

            try:
                iteration_results = harness.run(tasks=task_names, format_results=False)

                for task_name, result in iteration_results.items():
                    all_iteration_results[task_name].append(result)
                    artifact_record = build_artifact_record(
                        task_name,
                        result,
                        iteration,
                        include_finish_page=include_finish_page_in_manifest,
                    )
                    manifest_entries.append(artifact_record)
                    progress.advance(progress_task_id)

                    status = artifact_record["status"]
                    status_icon = "✓" if status == "completed" else "⏱️" if status == "truncated" else "✗"
                    status_color = "green" if status == "completed" else "yellow" if status == "truncated" else "red"
                    progress.console.print(
                        f"  [{status_color}]{status_icon}[/] {task_name} -> {artifact_record['exp_dir']}",
                        highlight=False,
                    )

                if args.iterations > 1:
                    log_success(f"✓ Iteration {iteration + 1} completed")
                else:
                    log_success("\n✓ All tasks completed")
            except Exception as exc:
                log_error(f"\n✗ Error running iteration {iteration + 1}: {exc}")

    manifest_path = save_run_manifest(args, task_names, manifest_entries, num_workers)

    overall_end_time = time.time()
    log_info(f"\nAll task runs finished in {overall_end_time - overall_start_time:.2f}s")
    console.print(create_run_summary_table(manifest_entries, manifest_path))
    console.print()

    if len(manifest_entries) == 1:
        console.print(create_task_artifacts_table(manifest_entries[0]))
        console.print()

    return all_iteration_results


if __name__ == "__main__":
    main()
