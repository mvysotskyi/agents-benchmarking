"""Evaluate circuit benchmark results.

For each test case the ground truth ``gt.answer`` is classified as:

- **circuit XML** (starts with ``<cir``) — agent is scored by comparing the
  truth tables of the GT and predicted circuit exports. Structural / scheme
  similarity is no longer used.
- **JSON answer** (any other JSON object, e.g. analog measurements) — agent
  is scored only by its JSON answer; the predicted circuit export is ignored.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console

from evaluation.objective.circuit.truth_table import compare_circuit_truth_tables
from evaluation.objective.evaluation_framework import (
    compare_values,
    extract_json_best_effort,
    load_test_cases,
)


def extract_test_id(task_identifier: str) -> str | None:
    """Extract a normalized circuit test id from a task identifier."""
    match = re.search(r"(tc_[a-zA-Z0-9]+_\d+)", task_identifier)
    return match.group(1) if match else None


def _classify_ground_truth(answer: str) -> tuple[str, Any]:
    """Classify a raw ``gt.answer`` string.

    Returns ``("xml", export_text)`` for circuit XML ground truths and
    ``("json", payload_dict)`` for JSON answer ground truths.
    Returns ``("unknown", None)`` if the value cannot be classified.
    """
    if not isinstance(answer, str) or not answer.strip():
        return ("unknown", None)

    stripped = answer.strip()
    inner_text = stripped

    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if isinstance(parsed, dict):
            if (
                "answer" in parsed
                and isinstance(parsed["answer"], str)
                and parsed["answer"].strip()
            ):
                inner_text = parsed["answer"].strip()
            else:
                return ("json", parsed)

    if inner_text.lstrip().startswith("<cir"):
        return ("xml", inner_text)

    try:
        parsed_inner = json.loads(inner_text)
    except (json.JSONDecodeError, TypeError):
        return ("unknown", None)

    if isinstance(parsed_inner, dict):
        return ("json", parsed_inner)

    return ("unknown", None)


def _find_summary_file(run_dir: Path) -> Path | None:
    """Support the current filename and the common typo variant."""
    for filename in ("summary_info.json", "summmary_info.json"):
        summary_path = run_dir / filename
        if summary_path.exists():
            return summary_path
    return None


def _load_run_data(results_dir: str) -> dict[str, dict[str, Any]]:
    """Index per-test run data from a benchmark results directory.

    Returns a dict keyed by test id with values containing ``post_run_js_result``
    (str | None) and ``primary_output`` (str | None) for the latest run that
    maps to that test id.
    """
    base_dir = Path(results_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Results path does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {base_dir}")

    run_dirs = [
        path for path in sorted(base_dir.iterdir())
        if path.is_dir() and path.name != "run_manifests"
    ]

    indexed: dict[str, dict[str, Any]] = {}

    for run_dir in run_dirs:
        summary_path = _find_summary_file(run_dir)
        if summary_path is None:
            continue

        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        test_id: str | None = None
        for key in ("task_name", "task_id"):
            value = summary_data.get(key)
            if isinstance(value, str):
                test_id = extract_test_id(value)
                if test_id:
                    break
        if not test_id:
            test_id = extract_test_id(run_dir.name)
        if not test_id:
            continue

        post_run_js_result = summary_data.get("post_run_js_result")
        primary_output = summary_data.get("agent_response")

        agent_outputs_path = run_dir / "agent_outputs.json"
        if agent_outputs_path.exists():
            try:
                agent_outputs = json.loads(agent_outputs_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                agent_outputs = {}
            primary_output = (
                agent_outputs.get("primary_output")
                or agent_outputs.get("agent_response")
                or primary_output
            )
            if not post_run_js_result:
                post_run_js_result = agent_outputs.get("post_run_js_result")

        indexed[test_id] = {
            "post_run_js_result": post_run_js_result,
            "primary_output": primary_output,
        }

    return indexed


def scheme_similarity_output_path(results_dir: str) -> Path:
    return Path(results_dir) / "scheme_similarity.json"


def objective_evaluation_output_path(results_dir: str) -> Path:
    return Path(results_dir) / "objective_evaluation.json"


def _score_xml_test(
    test_id: str,
    gt_export: str,
    pred_export: str | None,
    console: Console,
) -> float:
    if not isinstance(pred_export, str) or not pred_export.strip():
        console.print(f"[yellow]{test_id}: missing predicted circuit export, score 0.0[/yellow]")
        return 0.0

    try:
        result = compare_circuit_truth_tables(gt_export, pred_export)
    except Exception as exc:
        console.print(f"[red]{test_id}: truth-table comparison failed ({exc}), score 0.0[/red]")
        return 0.0

    if not result.get("applicable"):
        console.print(
            f"[yellow]{test_id}: truth table not applicable "
            f"({result.get('reason')}), score 0.0[/yellow]"
        )
        return 0.0

    if result.get("equivalent"):
        console.print(f"[green]{test_id}: 1.000000 (truth-table equivalent)[/green]")
        return 1.0

    similarity = round(float(result.get("similarity", 0.0)), 6)
    console.print(
        f"[red]{test_id}: {similarity:.6f} "
        f"(truth tables differ: {result.get('reason')})[/red]"
    )
    return similarity


def _score_json_test(
    test_id: str,
    gt_payload: dict,
    primary_output: str | None,
    console: Console,
) -> float:
    if not isinstance(primary_output, str) or not primary_output.strip():
        console.print(f"[yellow]{test_id}: missing agent answer, score 0.0[/yellow]")
        return 0.0

    prediction = extract_json_best_effort(primary_output)
    if prediction is None:
        console.print(f"[red]{test_id}: could not parse JSON from agent answer, score 0.0[/red]")
        return 0.0

    expected_fields = list(gt_payload.items())
    if not expected_fields:
        console.print(f"[yellow]{test_id}: empty ground-truth JSON, score 0.0[/yellow]")
        return 0.0

    matched = sum(
        1 for key, expected in expected_fields
        if compare_values(prediction.get(key), expected)
    )
    similarity = round(matched / len(expected_fields), 6)

    if matched == len(expected_fields):
        console.print(f"[green]{test_id}: 1.000000 (JSON answer matches)[/green]")
    else:
        console.print(
            f"[red]{test_id}: {similarity:.6f} "
            f"(JSON answer mismatch: {matched}/{len(expected_fields)} fields)[/red]"
        )
    return similarity


def evaluate_circuit_schemes(
    test_cases_path: str,
    results_dir: str,
    console: Console,
) -> dict[str, float]:
    """Compute per-test scores for a circuit benchmark run directory."""
    test_cases = load_test_cases(test_cases_path)
    run_data = _load_run_data(results_dir)

    similarities: dict[str, float] = {}
    xml_count = 0
    json_count = 0
    skipped = 0

    for tc in test_cases:
        test_id = tc["id"]
        gt = tc.get("gt", {})
        answer = gt.get("answer")
        kind, payload = _classify_ground_truth(answer)

        if kind == "unknown":
            skipped += 1
            console.print(f"[yellow]{test_id}: unrecognized ground truth, skipped[/yellow]")
            continue

        run_entry = run_data.get(test_id, {})

        if kind == "xml":
            xml_count += 1
            similarities[test_id] = _score_xml_test(
                test_id, payload, run_entry.get("post_run_js_result"), console
            )
        else:
            json_count += 1
            similarities[test_id] = _score_json_test(
                test_id, payload, run_entry.get("primary_output"), console
            )

    output_path = scheme_similarity_output_path(results_dir)
    output_path.write_text(json.dumps(similarities, indent=2), encoding="utf-8")

    objective_scores = {
        test_id: int(similarity == 1.0)
        for test_id, similarity in similarities.items()
    }
    objective_path = objective_evaluation_output_path(results_dir)
    objective_path.write_text(json.dumps(objective_scores, indent=2), encoding="utf-8")

    passed = sum(objective_scores.values())
    mean_similarity = sum(similarities.values()) / len(similarities) if similarities else 0.0

    console.print("\n[bold green]Circuit evaluation complete[/bold green]")
    console.print(f"Test cases scored: {len(similarities)} (xml={xml_count}, json={json_count})")
    console.print(f"Skipped: {skipped}")
    console.print(f"Passed: {passed}/{len(similarities)}")
    console.print(f"Mean similarity: {mean_similarity:.4f}")
    console.print(f"Saved to: {output_path}")
    console.print(f"Objective evaluation saved to: {objective_path}")

    return similarities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate circuit benchmark results (truth tables for XML GT, JSON answers for JSON GT).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.objective.evaluate_circuit_scheme \\
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
    load_dotenv()
    args = parse_args()
    console = Console()

    console.print("[bold]Circuit Evaluation[/bold]")
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
