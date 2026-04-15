"""Evaluate circuit-export similarity from benchmark result directories."""

import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from evaluation.objective.circuit.build_graph import build_component_graph
from evaluation.objective.circuit.calc_similarity import compare_circuit_exports
from evaluation.objective.evaluation_framework import load_test_cases


def extract_test_id(task_identifier: str) -> str | None:
    """Extract a normalized circuit test id from a task identifier."""
    match = re.search(r"(tc_[a-zA-Z0-9]+_\d+)", task_identifier)
    return match.group(1) if match else None


def _unwrap_circuit_answer(answer: str) -> str:
    """Unwrap a gt.answer that may be a JSON envelope around the circuit export.

    The new YAML format stores the circuit data as::

        gt: {answer: '{"answer": "<cir ...>...</cir>"}'}

    This helper detects the JSON envelope and returns the inner ``answer``
    value.  If the string is not a JSON envelope it is returned as-is
    (legacy plain-text format).
    """
    stripped = answer.strip()
    if not stripped.startswith("{"):
        return answer
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return answer
    if isinstance(parsed, dict) and "answer" in parsed:
        inner = parsed["answer"]
        if isinstance(inner, str) and inner.strip():
            return inner
    return answer


def load_ground_truth_exports(test_cases_path: str) -> dict[str, str]:
    """Load `gt.answer` exports keyed by test id."""
    ground_truth_exports: dict[str, str] = {}
    for test_case in load_test_cases(test_cases_path):
        test_id = test_case["id"]
        gt = test_case.get("gt", {})
        answer = gt.get("answer")
        if isinstance(answer, str) and answer.strip():
            ground_truth_exports[test_id] = _unwrap_circuit_answer(answer)
    return ground_truth_exports


def _find_summary_file(run_dir: Path) -> Path | None:
    """Support the current filename and the common typo variant."""
    for filename in ("summary_info.json", "summmary_info.json"):
        summary_path = run_dir / filename
        if summary_path.exists():
            return summary_path
    return None


def load_predicted_exports(results_dir: str) -> dict[str, str]:
    """Load `post_run_js_result` exports from a benchmark results directory."""
    base_dir = Path(results_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Results path does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {base_dir}")

    predicted_exports: dict[str, str] = {}

    run_dirs = [
        path for path in sorted(base_dir.iterdir())
        if path.is_dir() and path.name != "run_manifests"
    ]

    for run_dir in run_dirs:
        summary_path = _find_summary_file(run_dir)
        if summary_path is None:
            continue

        summary_data = json.loads(summary_path.read_text())

        test_id: str | None = None
        task_name = summary_data.get("task_name")
        if isinstance(task_name, str):
            test_id = extract_test_id(task_name)

        if not test_id:
            task_id = summary_data.get("task_id")
            if isinstance(task_id, str):
                test_id = extract_test_id(task_id)

        if not test_id:
            test_id = extract_test_id(run_dir.name)

        export_text = summary_data.get("post_run_js_result")
        if test_id and isinstance(export_text, str) and export_text.strip():
            predicted_exports[test_id] = export_text

    return predicted_exports


def scheme_similarity_output_path(results_dir: str) -> Path:
    """Return the output path for per-test scheme similarity scores."""
    return Path(results_dir) / "scheme_similarity.json"


def objective_evaluation_output_path(results_dir: str) -> Path:
    """Return the output path for binary exact-match circuit scores."""
    return Path(results_dir) / "objective_evaluation.json"


def evaluate_circuit_schemes(
    test_cases_path: str,
    results_dir: str,
    console: Console,
) -> dict[str, float]:
    """Compute per-test scheme similarity for a circuit benchmark run directory."""
    ground_truth_exports = load_ground_truth_exports(test_cases_path)
    predicted_exports = load_predicted_exports(results_dir)

    similarities: dict[str, float] = {}
    missing_predictions = 0
    failed_predictions = 0

    skipped_gt = 0
    oversized_predictions = 0

    for test_id in sorted(ground_truth_exports):
        gt_export = ground_truth_exports[test_id]

        gt_graph = build_component_graph(gt_export)
        if gt_graph.number_of_nodes() == 0:
            similarities[test_id] = 0.0
            skipped_gt += 1
            console.print(
                f"[yellow]{test_id}: ground truth has no parseable circuit components, "
                f"score 0.0[/yellow]"
            )
            continue

        pred_export = predicted_exports.get(test_id)

        if not pred_export:
            similarities[test_id] = 0.0
            missing_predictions += 1
            console.print(f"[yellow]{test_id}: missing `post_run_js_result`, score 0.0[/yellow]")
            continue

        pred_graph = build_component_graph(pred_export)
        gt_count = gt_graph.number_of_nodes()
        pred_count = pred_graph.number_of_nodes()
        if gt_count > 0 and pred_count > gt_count * 1.5:
            similarities[test_id] = 0.0
            oversized_predictions += 1
            console.print(
                f"[yellow]{test_id}: predicted scheme has {pred_count} elements "
                f"vs {gt_count} in GT (>{50}% larger), score 0.0[/yellow]"
            )
            continue

        try:
            comparison = compare_circuit_exports(gt_export, pred_export)
        except Exception as exc:
            similarities[test_id] = 0.0
            failed_predictions += 1
            console.print(f"[red]{test_id}: failed to score export ({exc}), score 0.0[/red]")
            continue

        similarity = comparison["similarity"]
        similarities[test_id] = round(float(similarity), 6)
        if (
            comparison.get("truth_table_applicable")
            and comparison.get("truth_table_equivalent")
            and comparison.get("structural_similarity") != 1.0
        ):
            console.print(
                f"[green]{test_id}: {similarities[test_id]:.6f}[/green] "
                f"[dim](truth-table equivalent; structural "
                f"{comparison['structural_similarity']:.6f})[/dim]"
            )
        elif comparison.get("truth_table_applicable"):
            console.print(
                f"[green]{test_id}: {similarities[test_id]:.6f}[/green] "
                f"[dim](truth-table {comparison['truth_table_similarity']:.6f})[/dim]"
            )
        else:
            console.print(f"[green]{test_id}: {similarities[test_id]:.6f}[/green]")

    output_path = scheme_similarity_output_path(results_dir)
    output_path.write_text(json.dumps(similarities, indent=2), encoding="utf-8")

    objective_scores = {
        test_id: int(similarity == 1.0)
        for test_id, similarity in similarities.items()
    }
    objective_path = objective_evaluation_output_path(results_dir)
    objective_path.write_text(json.dumps(objective_scores, indent=2), encoding="utf-8")

    scored = len(similarities) - missing_predictions - failed_predictions - skipped_gt - oversized_predictions
    mean_similarity = sum(similarities.values()) / len(similarities) if similarities else 0.0

    console.print("\n[bold green]Circuit scheme evaluation complete[/bold green]")
    console.print(f"Scored: {scored}/{len(similarities)}")
    console.print(f"Skipped (unparseable GT): {skipped_gt}")
    console.print(f"Missing predictions: {missing_predictions}")
    console.print(f"Oversized predictions: {oversized_predictions}")
    console.print(f"Failed predictions: {failed_predictions}")
    console.print(f"Mean similarity: {mean_similarity:.4f}")
    console.print(f"Saved to: {output_path}")
    console.print(f"Objective evaluation saved to: {objective_path}")

    return similarities


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate circuit-export similarity from benchmark result directories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m computer_use.evals.evaluate_circuit_scheme \\
      --test-cases test_cases/circuit.yaml \\
      --responses results_gemini31pro_circuit
""",
    )
    parser.add_argument(
        "--test-cases",
        type=str,
        default="test_cases/circuit.yaml",
        help="Path to the circuit YAML test cases file",
    )
    parser.add_argument(
        "--responses",
        required=True,
        type=str,
        help="Path to the benchmark results directory",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the circuit scheme evaluation CLI."""
    load_dotenv()
    args = parse_args()
    console = Console()

    console.print("[bold]Circuit Scheme Evaluation[/bold]")
    console.print(f"Test cases: {args.test_cases}")
    console.print(f"Responses:  {args.responses}")

    try:
        evaluate_circuit_schemes(
            test_cases_path=args.test_cases,
            results_dir=args.responses,
            console=console,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
