#!/usr/bin/env python3
"""Objective evaluator for Flightradar benchmark results.

Compares agent responses from run directories against ground-truth answers
from the flightradar test cases YAML file.  The agent's final answer is read
from the ``primary_output`` field in ``agent_outputs.json``.  Ground-truth
answers live in ``test_cases/flightradar.yaml`` under each test case's ``gt``
field (JSON-encoded string under the ``answer`` key).

Usage:
    python -m evaluation.objective.eval_flightradar <results_dir> [<test_cases_yaml>] [--verbose]

Examples:
    python -m evaluation.objective.eval_flightradar final_results/flightradar/gemini31pro
    python -m evaluation.objective.eval_flightradar final_results/flightradar/opus46 test_cases/flightradar.yaml --verbose
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import fire

from evaluation.objective.evaluation_framework import (
    evaluate_objective,
    extract_json_from_fenced_block,
    extract_json_from_response,
    load_test_cases,
)


def _extract_send_msg_content(response: str) -> str | None:
    """Extract the decoded string argument from send_msg_to_user(...)."""
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
    return "".join(chars)


def _normalize_response(response: str) -> str:
    """Normalize agent responses so standard JSON extractors can parse them.

    Opus46 frequently wraps JSON in markdown fences inside send_msg_to_user,
    e.g.  send_msg_to_user('```json\\n{...}\\n```')  or embeds JSON at the
    end of a prose message inside the call.  Standard extraction fails on the
    raw response because the send_msg content is not bare JSON.

    This function decodes the send_msg argument and, if it contains a JSON
    object (directly or inside a code fence), returns just that JSON so the
    normal extractors in evaluation_framework succeed.
    """
    content = _extract_send_msg_content(response)
    if content is None:
        return response

    # Try to find a JSON object inside the decoded content
    extracted = (
        extract_json_from_fenced_block(content)
        or extract_json_from_response(content)
    )
    if extracted is not None:
        return json.dumps(extracted)

    return response

TIME_TOLERANCE_MINUTES = 1.0


def _parse_minutes(value: Any) -> float | None:
    """Parse a time string like 'HH:MM UTC' or 'HH:MM:SS UTC' to minutes since midnight."""
    if value is None:
        return None
    text = str(value).strip().upper()
    text = text.removesuffix("UTC").strip()
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60.0
    except (ValueError, IndexError):
        pass
    return None


def _within_time_tolerance(actual: Any, expected: Any) -> bool:
    """Return True if both values parse as times and differ by at most TIME_TOLERANCE_MINUTES."""
    exp = _parse_minutes(expected)
    act = _parse_minutes(actual)
    if exp is None or act is None:
        return False
    return abs(exp - act) <= TIME_TOLERANCE_MINUTES


def _apply_time_tolerance(
    field_results: dict, prediction: dict | None
) -> tuple[dict[str, int], int]:
    """Re-evaluate failed fields with time tolerance.

    For fields where the actual value is None (key name mismatch) but the
    expected value is a time, also searches all prediction values for a
    match within tolerance.  This handles cases where the agent uses a
    different key name (e.g. ``time_utc``) than the ground truth (``answer``).

    Returns updated per-field scores (0/1) and the new overall score.
    """
    adjusted: dict[str, int] = {k: fr.score for k, fr in field_results.items()}
    for field, fr in field_results.items():
        if fr.score == 1:
            continue
        actual = fr.actual
        if actual is None and prediction is not None and _parse_minutes(fr.expected) is not None:
            for v in prediction.values():
                if _within_time_tolerance(v, fr.expected):
                    actual = v
                    break
        if _within_time_tolerance(actual, fr.expected):
            adjusted[field] = 1
    new_score = 1 if (adjusted and all(v == 1 for v in adjusted.values())) else 0
    return adjusted, new_score

TESTCASE_ID_PATTERN = re.compile(r"(tc_frad_\d+)")


def _extract_test_id(value: str) -> str | None:
    match = TESTCASE_ID_PATTERN.search(value)
    return match.group(1) if match else None


def _load_run_responses(results_dir: Path) -> list[dict[str, Any]]:
    """Discover run directories and extract test IDs and primary_output.

    When multiple runs map to the same test ID, the latest run with a
    non-empty primary_output is kept.
    """
    candidates: dict[str, dict[str, Any]] = {}

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        test_id = _extract_test_id(run_dir.name)

        if not test_id:
            summary_path = run_dir / "summary_info.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                test_id = _extract_test_id(
                    summary.get("task_name", "")
                ) or _extract_test_id(summary.get("task_id", ""))
        if not test_id:
            continue

        agent_outputs_path = run_dir / "agent_outputs.json"
        if not agent_outputs_path.exists():
            continue

        agent_outputs = json.loads(agent_outputs_path.read_text(encoding="utf-8"))
        primary_output: str = (
            agent_outputs.get("primary_output")
            or agent_outputs.get("agent_response")
            or ""
        )

        entry: dict[str, Any] = {
            "test_id": test_id,
            "run_dir": str(run_dir),
            "primary_output": primary_output,
        }

        prev = candidates.get(test_id)
        if prev is None or (primary_output and not prev.get("primary_output")) or primary_output:
            candidates[test_id] = entry

    return list(candidates.values())


def evaluate_all(
    results_dir: str,
    test_cases_yaml: str = "test_cases/flightradar.yaml",
    verbose: bool = False,
) -> None:
    """Evaluate all Flightradar test cases from a results directory.

    Args:
        results_dir:      Path to the model's benchmark results directory
                          (e.g. final_results/flightradar/gemini31pro).
        test_cases_yaml:  Path to the flightradar test cases YAML file.
        verbose:          Print per-field comparison details for every case.
    """
    results_path = Path(results_dir)
    yaml_path = Path(test_cases_yaml)

    if not results_path.exists():
        print("ERROR: results directory not found: %s" % results_path, file=sys.stderr)
        sys.exit(1)
    if not yaml_path.exists():
        print("ERROR: test cases YAML not found: %s" % yaml_path, file=sys.stderr)
        sys.exit(1)

    test_cases = load_test_cases(str(yaml_path))
    gt_by_id = {tc["id"]: tc for tc in test_cases}

    responses = _load_run_responses(results_path)
    if not responses:
        print(
            "ERROR: no run directories with agent_outputs.json found in %s" % results_path,
            file=sys.stderr,
        )
        sys.exit(1)

    print("Evaluating %d test case(s)" % len(responses))
    print("-" * 60)

    total = 0
    passed = 0
    failed_ids: list[str] = []
    objective_evaluation: dict[str, int] = {}

    for response in sorted(responses, key=lambda r: r["test_id"]):
        test_id = response["test_id"]
        primary_output = response["primary_output"]

        if test_id not in gt_by_id:
            print("\n[SKIP] %s — no ground truth in YAML" % test_id)
            continue

        tc = gt_by_id[test_id]
        gt = tc.get("gt", {})
        description = tc.get("prompt", "")[:80].replace("\n", " ")
        eval_config = tc.get("eval_config")

        result = evaluate_objective(
            raw_response=_normalize_response(primary_output),
            gt=gt,
            test_id=test_id,
            description=description,
            eval_config=eval_config,
        )

        adjusted_fields, final_score = _apply_time_tolerance(result.field_results, result.prediction)

        total += 1
        status = "PASS" if final_score == 1 else "FAIL"
        if final_score == 1:
            passed += 1
            objective_evaluation[test_id] = 1
        else:
            failed_ids.append(test_id)
            objective_evaluation[test_id] = 0

        print("\n[%s] %s" % (status, test_id))

        if result.prediction is None:
            print("  (could not extract JSON from agent response)")

        if verbose or final_score == 0:
            for field, fr in result.field_results.items():
                adj = adjusted_fields[field]
                if adj == 1 and fr.score == 0:
                    match_str = "OK(±1min)"
                elif adj == 1:
                    match_str = "OK"
                else:
                    match_str = "FAIL"
                print("  [%s] %s: expected=%r  got=%r" % (match_str, field, fr.expected, fr.actual))

    objective_path = results_path / "objective_evaluation.json"
    objective_path.write_text(json.dumps(objective_evaluation, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("TOTAL: %d/%d passed" % (passed, total))
    if failed_ids:
        print("FAILED: %s" % ", ".join(sorted(failed_ids)))
    print("Saved objective evaluation to: %s" % objective_path)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    fire.Fire(evaluate_all)
