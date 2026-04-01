"""Evaluation framework for Claude Computer Use Agent.

Shared evaluation logic: dataclasses, comparison utilities, and scoring
functions used by both the agent runner and the evaluator scripts.
"""

import itertools
import json
from pathlib import Path
import re
from dataclasses import dataclass
import sys
from typing import Any

from rich.console import Console
from rich.console import Console
import yaml

import argparse
from computer_use.providers import LLMProvider, get_provider


# ── Dataclasses ────────────────────────────────────────────────────────────────


@dataclass
class FieldResult:
    """Result for a single ground truth field."""

    expected: Any
    actual: Any
    score: int  # 0 or 1


@dataclass
class ObjectiveTestResult:
    """Result for objective (structured JSON) evaluation of a single test case."""

    test_id: str
    description: str
    ground_truth: dict
    prediction: dict | None
    raw_response: str
    field_results: dict[str, FieldResult]
    score: int  # 1 if ALL fields match, else 0


@dataclass
class LLMJudgeTestResult:
    """Result for LLM-as-judge evaluation of a single test case."""

    test_id: str
    description: str
    ground_truth: str
    prediction: str
    score: int  # 0 or 1
    justification: str


@dataclass
class AgentOutput:
    """Persisted output from running the agent on a single test case."""

    test_id: str
    description: str
    prompt: str
    raw_response: str
    ground_truth: dict
    llm_judge_gt: str
    eval_config: dict | None


# ── Utility functions ──────────────────────────────────────────────────────────


def load_test_cases(path: str) -> list[dict]:
    """Load test cases from YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        List of test case dicts.
    """
    with open(path) as f:
        return yaml.safe_load(f)["test_cases"]


def write_agent_output(output: AgentOutput, path: Path) -> None:
    """Serialize an AgentOutput to a JSON file.

    Args:
        output: AgentOutput to serialize.
        path: Directory in which to write ``{test_id}_output.json``.
    """
    dest = path / f"{output.test_id}_output.json"
    data = {
        "test_id": output.test_id,
        "description": output.description,
        "prompt": output.prompt,
        "raw_response": output.raw_response,
        "ground_truth": output.ground_truth,
        "llm_judge_gt": output.llm_judge_gt,
        "eval_config": output.eval_config,
    }
    dest.write_text(json.dumps(data, indent=2, default=str))


def load_agent_output(path: Path) -> AgentOutput:
    """Deserialize an AgentOutput from a JSON file.

    Args:
        path: Path to a ``{test_id}_output.json`` file.

    Returns:
        AgentOutput populated from the file.
    """
    data = json.loads(path.read_text())
    return AgentOutput(
        test_id=data["test_id"],
        description=data["description"],
        prompt=data["prompt"],
        raw_response=data["raw_response"],
        ground_truth=data["ground_truth"],
        llm_judge_gt=data["llm_judge_gt"],
        eval_config=data.get("eval_config"),
    )


def extract_json_from_response(response: str) -> dict | None:
    """Extract and validate JSON from an LLM response.

    Tries three strategies in order:
    1. Parse the entire response as JSON.
    2. Extract from a ```json fenced code block.
    3. Find the first {...} block via bracket matching.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed dict if successful, None otherwise.
    """
    text = response.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def extract_json_from_fenced_block(response: str) -> dict | None:
    """Extract JSON exclusively from a ```json fenced code block.

    Returns None if no ```json block is found or if the block content is empty/invalid.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed dict if a valid ```json block is found, None otherwise.
    """
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", response.strip(), re.DOTALL)
    if not fence_match:
        return None
    content = fence_match.group(1).strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def extract_json_from_send_msg(response: str) -> dict | None:
    """Extract JSON from a ``send_msg_to_user(...)`` call.

    Handles both single- and double-quoted string arguments with common
    Python escape sequences (``\\n``, ``\\t``, ``\\"``, ``\\'``, ``\\\\``).

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed dict if a valid JSON payload is found inside send_msg_to_user,
        None otherwise.
    """
    match = re.search(r"send_msg_to_user\(", response)
    if not match:
        return None

    pos = match.end()
    while pos < len(response) and response[pos] in " \t":
        pos += 1
    if pos >= len(response) or response[pos] not in ("'", '"'):
        return None

    quote = response[pos]
    pos += 1
    chars: list[str] = []
    while pos < len(response):
        ch = response[pos]
        if ch == "\\" and pos + 1 < len(response):
            nxt = response[pos + 1]
            escape_map = {"n": "\n", "t": "\t", "\\": "\\", "'": "'", '"': '"'}
            chars.append(escape_map.get(nxt, ch + nxt))
            pos += 2
        elif ch == quote:
            break
        else:
            chars.append(ch)
            pos += 1

    content = "".join(chars)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def extract_json_best_effort(response: str) -> dict | None:
    """Try multiple strategies to extract JSON from an agent response.

    Strategies tried in order:
    1. ``extract_json_from_fenced_block`` — ` ```json ` code blocks.
    2. ``extract_json_from_send_msg``  — ``send_msg_to_user(...)`` calls.
    3. ``extract_json_from_response``  — whole-string parse / bracket matching.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed dict from the first successful strategy, or None.
    """
    return (
        extract_json_from_fenced_block(response)
        or extract_json_from_send_msg(response)
        or extract_json_from_response(response)
    )


def normalize(value: Any) -> str:
    """Normalize a value for comparison (lowercase, stripped).

    Args:
        value: Any value to normalize.

    Returns:
        Normalized string.
    """
    return str(value).strip().lower()


def normalize_gt_value(value: Any) -> Any:
    """Normalize a ground truth value from YAML.

    Converts YAML list-of-single-key-dicts (e.g., tc_003 format produced by YAML
    block sequences under a mapping key) into a flat dict for comparison.
    Passes all other types through unchanged.

    Args:
        value: Raw YAML value.

    Returns:
        Flat dict if input was a list-of-single-key-dicts, otherwise the original value.
    """
    if not isinstance(value, list):
        return value

    if all(isinstance(item, dict) and len(item) == 1 for item in value):
        merged: dict[str, Any] = {}
        for item in value:
            merged.update(item)
        return merged

    return value


def compare_values(actual: Any, expected: Any) -> bool:
    """Recursively compare actual and expected values.

    Lists are compared order-independently by sorted string representation.
    Dicts are compared by recursing on each key in expected.
    Scalars are normalized (lowercase, stripped) before comparison.

    Args:
        actual: Value from agent response.
        expected: Ground truth value.

    Returns:
        True if values match according to the rules above.
    """
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if len(actual) != len(expected):
            return False
        return sorted(normalize(x) for x in actual) == sorted(normalize(x) for x in expected)

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(compare_values(actual.get(k), v) for k, v in expected.items())

    return normalize(actual) == normalize(expected)


def compare_with_tolerance(actual: Any, expected: Any, tolerance: float) -> bool:
    """Compare two numeric values within an absolute tolerance.

    Falls back to normalized string comparison if either value cannot be
    parsed as a float.

    Args:
        actual: Value from agent response.
        expected: Ground truth value.
        tolerance: Maximum allowed absolute difference.

    Returns:
        True if abs(actual - expected) <= tolerance, or string-equal as fallback.
    """
    try:
        return abs(float(str(actual)) - float(str(expected))) <= tolerance
    except (TypeError, ValueError):
        return normalize(actual) == normalize(expected)


def compare_unordered_aircraft(
    gt_objects: list[Any],
    pred_objects: list[Any],
) -> bool:
    """Check if two lists of aircraft dicts match in any order.

    Tries all permutations of pred_objects and returns True if any permutation
    makes every (gt, pred) pair match via compare_values.

    Args:
        gt_objects: Ground truth aircraft dicts (one per aircraft slot).
        pred_objects: Predicted aircraft dicts (one per aircraft slot).

    Returns:
        True if a permutation of pred_objects matches gt_objects element-wise.
    """
    if len(gt_objects) != len(pred_objects):
        return False

    for pred_permutation in itertools.permutations(pred_objects):
        if all(
            compare_values(pred_aircraft, gt_aircraft)
            for pred_aircraft, gt_aircraft in zip(pred_permutation, gt_objects)
        ):
            return True

    return False


# ── Evaluation functions ───────────────────────────────────────────────────────


def _unwrap_answer_gt(gt: dict, prediction: dict | None) -> dict:
    """Unwrap a GT that stores the real fields as a JSON string under ``answer``.

    Many YAML test cases use the pattern::

        gt: {answer: '{"airport_id": "ADL"}'}

    When the prediction dict does not contain an ``answer`` key but the
    parsed inner JSON matches the prediction's key set, the inner dict is
    returned so that field-by-field comparison works correctly.

    Args:
        gt: Original ground-truth dict from YAML.
        prediction: Parsed prediction dict (may be None).

    Returns:
        Unwrapped GT dict if the pattern is detected, otherwise the original gt.
    """
    if list(gt.keys()) != ["answer"] or not isinstance(gt["answer"], str):
        return gt

    try:
        inner = json.loads(gt["answer"])
    except (json.JSONDecodeError, TypeError):
        return gt

    if not isinstance(inner, dict):
        return gt

    if prediction is None:
        return inner

    inner_overlap = len(set(inner.keys()) & set(prediction.keys()))
    outer_overlap = 1 if "answer" in prediction else 0
    if inner_overlap >= outer_overlap:
        return inner

    return gt


def evaluate_objective(
    raw_response: str,
    gt: dict,
    test_id: str,
    description: str,
    eval_config: dict | None = None,
) -> ObjectiveTestResult:
    """Evaluate agent response against objective (structured JSON) ground truth.

    Args:
        raw_response: Raw string response from the agent.
        gt: Ground truth dict from YAML (keyed by field name).
        test_id: Test case identifier.
        description: Human-readable test description.
        eval_config: Optional evaluation configuration dict. Supports
            ``unordered_object_keys`` (list[str]) to compare those fields
            in any order via permutation matching, and ``tolerance_fields``
            (dict[str, float]) to compare numeric fields within an absolute
            tolerance.

    Returns:
        ObjectiveTestResult with per-field scores and overall pass/fail.
    """
    prediction = extract_json_best_effort(raw_response)
    field_results: dict[str, FieldResult] = {}

    gt = _unwrap_answer_gt(gt, prediction)

    unordered_keys: list[str] = []
    tolerance_fields: dict[str, float] = {}
    if eval_config is not None:
        unordered_keys = eval_config.get("unordered_object_keys", [])
        tolerance_fields = eval_config.get("tolerance_fields", {})

    if unordered_keys:
        gt_objects = [normalize_gt_value(gt[k]) for k in unordered_keys if k in gt]
        pred_objects = [
            prediction.get(k) if prediction is not None else None
            for k in unordered_keys
            if k in gt
        ]
        unordered_match = compare_unordered_aircraft(gt_objects, pred_objects)
        for k in unordered_keys:
            if k not in gt:
                continue
            field_results[k] = FieldResult(
                expected=normalize_gt_value(gt[k]),
                actual=prediction.get(k) if prediction is not None else None,
                score=1 if unordered_match else 0,
            )

    for key, raw_expected in gt.items():
        if key in unordered_keys:
            continue
        expected = normalize_gt_value(raw_expected)
        actual = prediction.get(key) if prediction is not None else None
        if key in tolerance_fields:
            match = compare_with_tolerance(actual, expected, tolerance_fields[key])
        else:
            match = compare_values(actual, expected)
        field_results[key] = FieldResult(
            expected=expected,
            actual=actual,
            score=1 if match else 0,
        )

    all_match = bool(field_results) and all(fr.score == 1 for fr in field_results.values())

    return ObjectiveTestResult(
        test_id=test_id,
        description=description,
        ground_truth=gt,
        prediction=prediction,
        raw_response=raw_response,
        field_results=field_results,
        score=1 if all_match else 0,
    )


def evaluate_llm_judge(
    raw_response: str,
    llm_judge_gt: str,
    test_id: str,
    description: str,
    provider: LLMProvider,
    judge_model: str,
) -> LLMJudgeTestResult:
    """Evaluate agent response using LLM-as-judge.

    Args:
        raw_response: Raw string response from the agent.
        llm_judge_gt: Ground truth string from YAML for the LLM judge.
        test_id: Test case identifier.
        description: Human-readable test description.
        provider: LLM provider instance.
        judge_model: Model to use as judge.

    Returns:
        LLMJudgeTestResult with binary score and one-sentence justification.
    """
    system_prompt = (
        "You are a strict answer evaluator. Compare an AI agent's response to ground truth.\n"
        'Respond ONLY with JSON: {"score": 0 or 1, "justification": "one sentence"}.\n'
        "Score 1 if the response contains the correct information matching the ground truth.\n"
        "List items may be in any order. Numeric values must match exactly unless the ground truth "
        "explicitly states a tolerance (e.g., ±2 min) — in that case use only that stated "
        "tolerance. Do not apply any implicit rounding or approximate matching. "
        "Typos in callsigns or airport codes = score 0."
    )
    user_message = f"Ground truth: {llm_judge_gt}\nAgent response: {raw_response}"

    try:
        judge_text = provider.create_simple_message(
            model=judge_model,
            system=system_prompt,
            user_message=user_message,
            max_tokens=256,
        )

        judge_result = extract_json_from_response(judge_text)
        if judge_result is None:
            return LLMJudgeTestResult(
                test_id=test_id,
                description=description,
                ground_truth=llm_judge_gt,
                prediction=raw_response,
                score=0,
                justification=f"Could not parse judge response: {judge_text}",
            )

        return LLMJudgeTestResult(
            test_id=test_id,
            description=description,
            ground_truth=llm_judge_gt,
            prediction=raw_response,
            score=int(judge_result.get("score", 0)),
            justification=str(judge_result.get("justification", "")),
        )

    except Exception as e:
        return LLMJudgeTestResult(
            test_id=test_id,
            description=description,
            ground_truth=llm_judge_gt,
            prediction=raw_response,
            score=0,
            justification=f"Judge error: {e}",
        )

def evaluate_from_outputs(
    output_dir: Path,
    provider: LLMProvider,
    judge_model: str,
    console: Console,
) -> None:
    """Evaluate persisted agent outputs and write result JSON files.

    Args:
        output_dir: Directory containing ``*_output.json`` files.
        provider: LLM provider instance for LLM-as-judge calls.
        judge_model: Model identifier to use for LLM-as-judge evaluation.
        console: Rich console for progress output.
    """
    if not output_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] output directory '{output_dir}' does not exist"
        )
        return

    output_files = sorted(output_dir.glob("*_output.json"))
    if not output_files:
        console.print(
            f"[bold red]Error:[/bold red] no '*_output.json' files found in '{output_dir}'"
        )
        return

    objective_results: list[ObjectiveTestResult] = []
    llm_judge_results: list[LLMJudgeTestResult] = []

    for i, path in enumerate(output_files, 1):
        agent_output = load_agent_output(path)
        console.print(
            f"\n[bold cyan]Evaluating {i}/{len(output_files)}: {agent_output.test_id}[/bold cyan]"
        )

        obj_result = evaluate_objective(
            agent_output.raw_response,
            agent_output.ground_truth,
            agent_output.test_id,
            agent_output.description,
            agent_output.eval_config,
        )
        objective_results.append(obj_result)

        judge_result = evaluate_llm_judge(
            raw_response=agent_output.raw_response,
            llm_judge_gt=agent_output.llm_judge_gt,
            test_id=agent_output.test_id,
            description=agent_output.description,
            provider=provider,
            judge_model=judge_model,
        )
        llm_judge_results.append(judge_result)

        obj_color = "green" if obj_result.score == 1 else "red"
        judge_color = "green" if judge_result.score == 1 else "red"
        console.print(
            f"[{obj_color}]Objective: {'PASS' if obj_result.score == 1 else 'FAIL'}[/{obj_color}]"
        )
        console.print(
            f"[{judge_color}]LLM Judge: {'PASS' if judge_result.score == 1 else 'FAIL'}"
            f" — {judge_result.justification}[/{judge_color}]"
        )

    obj_correct = sum(r.score for r in objective_results)
    obj_total = len(objective_results)
    obj_accuracy = obj_correct / obj_total if obj_total else 0.0

    judge_correct = sum(r.score for r in llm_judge_results)
    judge_total = len(llm_judge_results)
    judge_accuracy = judge_correct / judge_total if judge_total else 0.0

    objective_output = {
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

    llm_judge_output = {
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

    obj_path = output_dir / "objective_results.json"
    judge_path = output_dir / "llm_judge_results.json"

    obj_path.write_text(json.dumps(objective_output, indent=2, default=str))
    judge_path.write_text(json.dumps(llm_judge_output, indent=2, default=str))

    console.print("\n[bold green]Evaluation complete![/bold green]")
    console.print(f"Objective accuracy: {obj_correct}/{obj_total} ({obj_accuracy:.1%})")
    console.print(f"LLM judge accuracy: {judge_correct}/{judge_total} ({judge_accuracy:.1%})")
    console.print(f"Results saved to: {output_dir}/")



def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    eval_parser = argparse.ArgumentParser(
        description="Evaluation framework for Claude Computer Use Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate persisted outputs
  python -m computer_use.evals.evaluation_framework \\
      --output-dir results/haiku
""",
    )
    eval_parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory containing *_output.json files",
    )
    eval_parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model to use as LLM judge (default: first model from provider)",
    )

    return eval_parser.parse_args()


def main() -> None:
    """Main entry point for the evaluation CLI."""
    args = parse_args()

    console = Console()

    try:
        provider = get_provider()
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    judge_model = args.judge_model or provider.get_available_models()[0]

    console.print("[bold]Evaluation Framework — evaluate[/bold]")
    console.print(f"Judge model: {judge_model}")
    console.print(f"Output dir:  {args.output_dir}")

    evaluate_from_outputs(
        output_dir=Path(args.output_dir),
        provider=provider,
        judge_model=judge_model,
        console=console,
    )


if __name__ == "__main__":
    main()
