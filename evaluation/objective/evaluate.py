"""CLI script to evaluate saved agent responses against test case ground truth."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from computer_use.evals.evaluation_framework import (
    LLMJudgeTestResult,
    ObjectiveTestResult,
    evaluate_llm_judge,
    evaluate_objective,
    load_test_cases,
)
from computer_use.providers import LLMProvider, get_provider


# ── Response loading ───────────────────────────────────────────────────────────


def load_responses(path: str) -> dict[str, Any]:
    """Load agent responses from a JSON file or benchmark results directory.

    Args:
        path: Path to a responses JSON file or benchmark results directory.

    Returns:
        Dict with 'metadata' and 'responses' keys.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Responses path does not exist: {input_path}")

    if input_path.is_dir():
        return load_responses_from_results_dir(input_path)

    return json.loads(input_path.read_text())


def extract_test_id(task_identifier: str) -> str | None:
    """Extract normalized test id from a task identifier string.

    Examples:
        - eval.tc_frad_001 -> tc_frad_001
        - eval.tc_frad_001_301 -> tc_frad_001
        - tc_circ_004 -> tc_circ_004
    """
    match = re.search(r"(tc_[a-zA-Z0-9]+_\d+)", task_identifier)
    return match.group(1) if match else None


def load_responses_from_results_dir(results_dir: Path) -> dict[str, Any]:
    """Load responses from benchmark run directory structure.

    Expects per-run subdirectories that contain `agent_outputs.json` and
    optionally `summary_info.json`.
    """
    response_map: dict[str, str] = {}
    model_names: set[str] = set()

    run_dirs = [
        p for p in sorted(results_dir.iterdir())
        if p.is_dir() and p.name != "run_manifests"
    ]

    for run_dir in run_dirs:
        outputs_path = run_dir / "agent_outputs.json"
        if not outputs_path.exists():
            continue

        outputs_data = json.loads(outputs_path.read_text())
        raw_response: str | None = (
            outputs_data.get("raw_agent_response")
            or outputs_data.get("agent_response")
            or outputs_data.get("primary_output")
        )
        if not isinstance(raw_response, str) or not raw_response.strip():
            continue

        test_id: str | None = None
        summary_path = run_dir / "summary_info.json"
        if summary_path.exists():
            summary_data = json.loads(summary_path.read_text())
            model_name = summary_data.get("model_name")
            if isinstance(model_name, str) and model_name:
                model_names.add(model_name)

            task_name = summary_data.get("task_name")
            if isinstance(task_name, str):
                test_id = extract_test_id(task_name)

            if not test_id:
                task_id = summary_data.get("task_id")
                if isinstance(task_id, str):
                    test_id = extract_test_id(task_id)

        if not test_id:
            test_id = extract_test_id(run_dir.name)

        if test_id:
            response_map[test_id] = raw_response

    metadata: dict[str, Any] = {
        "source": "results_dir",
        "results_dir": str(results_dir),
        "runs_discovered": len(run_dirs),
        "responses_loaded": len(response_map),
    }
    if len(model_names) == 1:
        metadata["model"] = next(iter(model_names))
    elif model_names:
        metadata["models"] = sorted(model_names)

    return {
        "metadata": metadata,
        "responses": [
            {"test_id": test_id, "raw_response": response_map[test_id]}
            for test_id in sorted(response_map)
        ],
    }


def objective_evaluation_output_path(responses_path: str) -> Path:
    """Return the lightweight objective-evaluation path near the responses input."""
    input_path = Path(responses_path)
    base_dir = input_path if input_path.is_dir() else input_path.parent
    return base_dir / "objective_evaluation.json"


# ── Evaluation orchestrator ────────────────────────────────────────────────────


def run_evaluation_from_file(
    test_cases_path: str,
    responses_path: str,
    output_dir: Path,
    console: Console,
    provider: LLMProvider | None = None,
    judge_model: str | None = None,
) -> None:
    """Evaluate saved responses against test case ground truth.

    Runs objective (JSON field matching) evaluation for every test case.
    Optionally runs LLM-as-judge evaluation when provider and judge_model
    are supplied.

    Args:
        test_cases_path: Path to YAML test cases file.
        responses_path: Path to JSON responses file from agent run.
        output_dir: Directory to write result JSON files.
        console: Rich console for progress output.
        provider: LLM provider for judge evaluation (optional).
        judge_model: Model to use as LLM judge (optional, requires provider).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    test_cases = load_test_cases(test_cases_path)

    responses_data = load_responses(responses_path)
    response_map: dict[str, str] = {
        r["test_id"]: r["raw_response"] for r in responses_data["responses"]
    }
    metadata = responses_data.get("metadata", {})

    if metadata:
        console.print(f"[dim]Responses from model: {metadata.get('model', 'unknown')}[/dim]")
        console.print(f"[dim]Collected at: {metadata.get('timestamp', 'unknown')}[/dim]")

    run_judge = provider is not None and judge_model is not None

    objective_results: list[ObjectiveTestResult] = []
    llm_judge_results: list[LLMJudgeTestResult] = []

    for i, tc in enumerate(test_cases, 1):
        test_id: str = tc["id"]
        description: str = tc.get("description", "")
        gt: dict = tc.get("gt", {})
        llm_judge_gt: str = tc.get("llm_judge_gt", "")
        eval_config: dict | None = tc.get("eval_config")

        console.print(f"\n[bold cyan]Test {i}/{len(test_cases)}: {test_id}[/bold cyan]")
        console.print(f"[dim]{description}[/dim]")

        raw_response = response_map.get(test_id)
        if raw_response is None:
            console.print(f"[bold red]No response found for {test_id}, skipping.[/bold red]")
            continue

        obj_result = evaluate_objective(raw_response, gt, test_id, description, eval_config)
        objective_results.append(obj_result)

        console.print(f"  [dim]Raw response (first 200 chars):[/dim] {raw_response[:200]!r}")
        console.print(f"  [dim]Extracted prediction:[/dim] {obj_result.prediction}")
        console.print(f"  [dim]Ground truth:[/dim] {gt}")
        for field_name, fr in obj_result.field_results.items():
            field_color = "green" if fr.score == 1 else "red"
            console.print(
                f"  [{field_color}]  Field '{field_name}': "
                f"expected={fr.expected!r}  actual={fr.actual!r}  "
                f"{'MATCH' if fr.score == 1 else 'MISMATCH'}[/{field_color}]"
            )

        obj_color = "green" if obj_result.score == 1 else "red"
        console.print(
            f"[{obj_color}]Objective: {'PASS' if obj_result.score == 1 else 'FAIL'}[/{obj_color}]"
        )

        if run_judge:
            judge_result = evaluate_llm_judge(
                raw_response=raw_response,
                llm_judge_gt=llm_judge_gt,
                test_id=test_id,
                description=description,
                provider=provider,
                judge_model=judge_model,
            )
            llm_judge_results.append(judge_result)

            judge_color = "green" if judge_result.score == 1 else "red"
            console.print(
                f"[{judge_color}]LLM Judge: {'PASS' if judge_result.score == 1 else 'FAIL'}"
                f" — {judge_result.justification}[/{judge_color}]"
            )

    obj_correct = sum(r.score for r in objective_results)
    obj_total = len(objective_results)
    obj_accuracy = obj_correct / obj_total if obj_total else 0.0

    objective_output = {
        "metadata": metadata,
        "test_cases": [
            {
                "test_id": r.test_id,
                "description": r.description,
                "ground_truth": r.ground_truth,
                "prediction": r.prediction,
                "field_results": {
                    k: {"expected": v.expected, "actual": v.actual, "score": v.score}
                    for k, v in r.field_results.items()
                },
                "score": r.score,
                "raw_response": r.raw_response,
            }
            for r in objective_results
        ],
        "correct": obj_correct,
        "total": obj_total,
        "accuracy": round(obj_accuracy, 4),
    }

    obj_path = output_dir / "objective_results.json"
    obj_path.write_text(json.dumps(objective_output, indent=2, default=str))
    objective_eval_path = objective_evaluation_output_path(responses_path)
    objective_eval_path.write_text(
        json.dumps({r.test_id: int(r.score) for r in objective_results}, indent=2, default=str),
        encoding="utf-8",
    )

    console.print("\n[bold green]Evaluation complete![/bold green]")
    console.print(f"Objective accuracy: {obj_correct}/{obj_total} ({obj_accuracy:.1%})")

    if run_judge and llm_judge_results:
        judge_correct = sum(r.score for r in llm_judge_results)
        judge_total = len(llm_judge_results)
        judge_accuracy = judge_correct / judge_total if judge_total else 0.0

        llm_judge_output = {
            "metadata": metadata,
            "test_cases": [
                {
                    "test_id": r.test_id,
                    "description": r.description,
                    "ground_truth": r.ground_truth,
                    "prediction": r.prediction,
                    "score": r.score,
                    "justification": r.justification,
                }
                for r in llm_judge_results
            ],
            "correct": judge_correct,
            "total": judge_total,
            "accuracy": round(judge_accuracy, 4),
        }

        judge_path = output_dir / "llm_judge_results.json"
        judge_path.write_text(json.dumps(llm_judge_output, indent=2, default=str))
        console.print(f"LLM judge accuracy: {judge_correct}/{judge_total} ({judge_accuracy:.1%})")

    console.print(f"Results saved to: {output_dir}/")
    console.print(f"Objective evaluation saved to: {objective_eval_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved agent responses against test case ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m computer_use.evals.evaluate \\
      --test-cases test_cases/softradar.yaml \\
      --responses responses/run1.json

    # Using benchmark result directories:
    python -m computer_use.evals.evaluate \
            --test-cases test_cases/flightradar.yaml \
            --responses results_opus46_frad

  # With LLM judge:
  python -m computer_use.evals.evaluate \\
      --test-cases test_cases/softradar.yaml \\
      --responses responses/run1.json \\
      --judge-model claude-haiku-4-5
""",
    )
    parser.add_argument(
        "--test-cases", required=True, type=str,
        help="Path to YAML test cases file",
    )
    parser.add_argument(
        "--responses", required=True, type=str,
        help="Path to responses JSON file or benchmark results directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save result JSON files (default: results)",
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
        help="Model to use as LLM judge (requires LLM_PROVIDER env var)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the evaluation CLI."""
    load_dotenv()
    args = parse_args()
    console = Console()

    provider: LLMProvider | None = None
    judge_model: str | None = args.judge_model

    if judge_model:
        try:
            provider = get_provider()
        except ValueError as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)

    console.print("[bold]Evaluation Framework[/bold]")
    console.print(f"Test cases:  {args.test_cases}")
    console.print(f"Responses:   {args.responses}")
    console.print(f"Output dir:  {args.output_dir}")
    if judge_model:
        console.print(f"Judge model: {judge_model}")
    else:
        console.print("[dim]LLM judge: disabled (use --judge-model to enable)[/dim]")

    run_evaluation_from_file(
        test_cases_path=args.test_cases,
        responses_path=args.responses,
        output_dir=Path(args.output_dir),
        console=console,
        provider=provider,
        judge_model=judge_model,
    )


if __name__ == "__main__":
    main()
