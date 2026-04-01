#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.llm_judge.read_step_states import as_dict, load_pickle
from evaluation.llm_judge.screenshot_diff import ScreenshotDiffThresholds, compare_screenshots, sorted_screenshot_paths

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_ID_RE = re.compile(r"(tc_[a-z]+_\d+)")
CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]+\n)?(.*?)```", re.DOTALL)
STAGE2_SYSTEM_PROMPT_PATH = REPO_ROOT / "stage2_sys_prompt.txt"
STAGE2_TASK_PROMPT_PATH = REPO_ROOT / "stage2_task_prompt.txt"
STAGE3_SYSTEM_PROMPT_PATH = REPO_ROOT / "stage3_sys_prompt.txt"
STAGE3_TASK_PROMPT_PATH = REPO_ROOT / "stage3_task_prompt.txt"
DEFAULT_STAGE2_MODEL = "gpt-4o-mini"
DEFAULT_STAGE3_MODEL = "gpt-5.1"
DEFAULT_OUTPUT_FILENAME = "llm_judgments.json"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestCase:
    id: str
    description: str
    prompt: str
    raw_fields: dict[str, str]


@dataclass(frozen=True)
class Stage2Judgment:
    visible_action: str
    visible_change_summary: str
    visible_change_type: str
    task_relevance: str
    progress: str
    confidence: str
    uncertainty_note: str
    updated_visible_state_summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage3Judgment:
    score: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeConfig:
    stage2_model: str = DEFAULT_STAGE2_MODEL
    stage3_model: str = DEFAULT_STAGE3_MODEL
    stage3_reasoning_effort: str = "low"
    stage2_image_detail: str = "low"
    stage3_image_detail: str = "high"
    compare_size: tuple[int, int] = (320, 200)
    stage1_thresholds: ScreenshotDiffThresholds = ScreenshotDiffThresholds()
    max_judgments: int | None = None
    dry_run: bool = False
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 120
    enable_progress: bool = True
    objective_evaluation_path: Path | None = None


def run_pipeline(
    results_dir: Path,
    testcases_yaml_path: Path,
    output_path: Path | None = None,
    config: JudgeConfig | None = None,
) -> dict[str, Any]:
    config = config or JudgeConfig()
    stage2_system_prompt = STAGE2_SYSTEM_PROMPT_PATH.read_text()
    stage2_task_prompt_template = STAGE2_TASK_PROMPT_PATH.read_text()
    stage3_system_prompt = STAGE3_SYSTEM_PROMPT_PATH.read_text()
    stage3_task_prompt_template = STAGE3_TASK_PROMPT_PATH.read_text()
    testcases = load_testcases(testcases_yaml_path)
    objective_evaluation_results, objective_evaluation_path = load_objective_evaluation_results(
        results_dir=results_dir,
        configured_path=config.objective_evaluation_path,
    )
    run_dirs = sorted_run_dirs(results_dir)

    LOGGER.info(
        "Starting LLM judge on %d run(s) from %s using testcases %s",
        len(run_dirs),
        results_dir,
        testcases_yaml_path,
    )

    aggregate: dict[str, Any] = {
        "results_dir": str(results_dir),
        "testcases_yaml": str(testcases_yaml_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "stage2_model": config.stage2_model,
        "stage3_model": config.stage3_model,
        "stage3_reasoning_effort": config.stage3_reasoning_effort,
        "stage2_image_detail": config.stage2_image_detail,
        "stage3_image_detail": config.stage3_image_detail,
        "objective_evaluation_path": str(objective_evaluation_path) if objective_evaluation_path else "",
        "stage1_thresholds": {
            "rmse": config.stage1_thresholds.rmse,
            "phash": config.stage1_thresholds.phash,
            "changed_fraction": config.stage1_thresholds.changed_fraction,
        },
        "compare_size": {"width": config.compare_size[0], "height": config.compare_size[1]},
        "dry_run": config.dry_run,
        "runs": [],
    }

    judged_pairs = 0
    run_iterable = maybe_progress(
        run_dirs,
        total=len(run_dirs),
        desc="Testcases",
        enabled=config.enable_progress,
        position=0,
        leave=True,
    )
    for run_dir in run_iterable:
        update_progress_label(run_iterable, task_label=run_dir.name)
        run_result = process_run(
            run_dir=run_dir,
            testcases=testcases,
            stage2_system_prompt=stage2_system_prompt,
            stage2_task_prompt_template=stage2_task_prompt_template,
            stage3_system_prompt=stage3_system_prompt,
            stage3_task_prompt_template=stage3_task_prompt_template,
            config=config,
            objective_evaluation_results=objective_evaluation_results,
            judged_pairs_so_far=judged_pairs,
        )
        judged_pairs += sum(1 for pair in run_result["pairs"] if pair["stage2_judgment"] is not None)
        aggregate["runs"].append(run_result)
        if output_path is not None:
            output_path.write_text(json.dumps(aggregate, indent=2))

    aggregate["summary"] = summarize_aggregate(aggregate)
    LOGGER.info(
        "Finished LLM judge: stage1 flagged %d/%d pairs, stage2 judged %d pairs, stage3 judged %d runs, errors(stage2=%d, stage3=%d)",
        aggregate["summary"]["pairs_flagged_stage1"],
        aggregate["summary"]["pairs_total"],
        aggregate["summary"]["pairs_judged_stage2"],
        aggregate["summary"]["runs_judged_stage3"],
        aggregate["summary"]["stage2_error_count"],
        aggregate["summary"]["stage3_error_count"],
    )
    if output_path is not None:
        output_path.write_text(json.dumps(aggregate, indent=2))
    return aggregate


def process_run(
    run_dir: Path,
    testcases: dict[str, TestCase],
    stage2_system_prompt: str,
    stage2_task_prompt_template: str,
    stage3_system_prompt: str,
    stage3_task_prompt_template: str,
    config: JudgeConfig,
    objective_evaluation_results: dict[str, int],
    judged_pairs_so_far: int,
) -> dict[str, Any]:
    screenshots = sorted_screenshot_paths(run_dir)
    summary_info = read_json_if_exists(run_dir / "summary_info.json")
    task_id = detect_task_id(run_dir, summary_info)
    testcase = testcases.get(task_id) if task_id else None
    if testcase is None:
        LOGGER.warning("No testcase mapping found for run %s (detected task_id=%s)", run_dir.name, task_id or "unknown")
    else:
        LOGGER.info("Processing %s for testcase %s", run_dir.name, testcase.id)

    pairs: list[dict[str, Any]] = []
    prior_visible_state_summary = ""
    stage1_flagged_so_far = 0
    stage2_completed_so_far = 0
    pair_iterable = zip(screenshots, screenshots[1:])
    total_pairs = max(0, len(screenshots) - 1)
    pair_iterable = maybe_progress(
        pair_iterable,
        total=total_pairs,
        desc=f"{(taskcase_label(testcase, task_id) or run_dir.name)[:32]}",
        enabled=config.enable_progress,
        position=1,
        leave=False,
    )

    for first, second in pair_iterable:
        first_step = extract_step_from_name(first.name)
        second_step = extract_step_from_name(second.name)
        action_text = load_action_text_for_step(run_dir, first_step)

        diff_result = compare_screenshots(
            first,
            second,
            thresholds=config.stage1_thresholds,
            compare_size=config.compare_size,
        )

        pair_record: dict[str, Any] = {
            "first_step": first_step,
            "second_step": second_step,
            "first_screenshot": str(first),
            "second_screenshot": str(second),
            "action_text": action_text,
            "stage1": diff_result.to_dict(),
            "prior_visible_state_summary": prior_visible_state_summary,
            "stage2_judgment": None,
            "stage2_error": None,
        }

        if diff_result.is_significant:
            stage1_flagged_so_far += 1

        should_judge = diff_result.is_significant and testcase is not None
        if should_judge and not limit_reached(config.max_judgments, judged_pairs_so_far):
            try:
                judgment = judge_screenshot_pair(
                    testcase=testcase,
                    action_text=action_text,
                    prior_visible_state_summary=prior_visible_state_summary,
                    previous_screenshot_path=first,
                    current_screenshot_path=second,
                    system_prompt=stage2_system_prompt,
                    task_prompt_template=stage2_task_prompt_template,
                    config=config,
                )
                pair_record["stage2_judgment"] = judgment.to_dict()
                prior_visible_state_summary = judgment.updated_visible_state_summary.strip()
                judged_pairs_so_far += 1
                stage2_completed_so_far += 1
            except Exception as exc:
                pair_record["stage2_error"] = str(exc)
                LOGGER.warning(
                    "Stage 2 judgment failed for %s steps %d->%d: %s",
                    run_dir.name,
                    first_step,
                    second_step,
                    exc,
                )

        pairs.append(pair_record)
        update_progress_counts(pair_iterable, stage1_flagged_so_far, stage2_completed_so_far)

    final_screenshot_path = screenshots[-1] if screenshots else None
    trajectory_events = build_stage3_trajectory_events(pairs)
    stage3_judgment = None
    stage3_error = None
    objective_evaluation_result = objective_evaluation_results.get(task_id) if task_id else None
    if testcase is not None and final_screenshot_path is not None:
        try:
            stage3_judgment = judge_final_outcome(
                testcase=testcase,
                trajectory_events=trajectory_events,
                final_screenshot_path=final_screenshot_path,
                final_agent_answer=extract_final_agent_answer(summary_info),
                objective_evaluation_result=objective_evaluation_result,
                system_prompt=stage3_system_prompt,
                task_prompt_template=stage3_task_prompt_template,
                config=config,
            ).to_dict()
        except Exception as exc:
            stage3_error = str(exc)
            LOGGER.warning("Stage 3 judgment failed for %s: %s", run_dir.name, exc)

    result = {
        "run_dir": str(run_dir),
        "task_id": task_id,
        "task_found_in_yaml": testcase is not None,
        "task_description": testcase.description if testcase else "",
        "task_prompt": testcase.prompt if testcase else "",
        "success_condition": extract_success_condition(testcase.prompt) if testcase else "",
        "summary_info": summary_info,
        "pairs_total": max(0, len(screenshots) - 1),
        "pairs_flagged_stage1": sum(1 for pair in pairs if pair["stage1"]["is_significant"]),
        "pairs_judged_stage2": sum(1 for pair in pairs if pair["stage2_judgment"] is not None),
        "pairs": pairs,
        "stage3": stage3_judgment,
        "stage3_error": stage3_error,
        "stage3_inputs": {
            "llm_judge_gt": testcase.raw_fields.get("llm_judge_gt", "") if testcase else "",
            "objective_evaluation_result": objective_evaluation_result,
            "trajectory_event_count": len(trajectory_events),
            "final_screenshot": str(final_screenshot_path) if final_screenshot_path else "",
            "final_agent_answer": extract_final_agent_answer(summary_info),
        },
    }
    LOGGER.info(
        "Finished %s: stage1 flagged %d/%d pairs, stage2 judged %d, stage3=%s, errors=%d/%d",
        run_dir.name,
        result["pairs_flagged_stage1"],
        result["pairs_total"],
        result["pairs_judged_stage2"],
        "yes" if result["stage3"] is not None else "no",
        sum(1 for pair in pairs if pair["stage2_error"] is not None),
        1 if stage3_error else 0,
    )
    return result


def judge_screenshot_pair(
    testcase: TestCase,
    action_text: str,
    prior_visible_state_summary: str,
    previous_screenshot_path: Path,
    current_screenshot_path: Path,
    system_prompt: str,
    task_prompt_template: str,
    config: JudgeConfig,
) -> Stage2Judgment:
    if config.dry_run:
        return Stage2Judgment(
            visible_action=action_text or "unknown",
            visible_change_summary="Dry run placeholder. No model call was made.",
            visible_change_type="other",
            task_relevance="none",
            progress="neutral",
            confidence="low",
            uncertainty_note="dry_run",
            updated_visible_state_summary=prior_visible_state_summary,
        )

    api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")

    success_condition = extract_success_condition(testcase.prompt)
    prior_summary_json = json.dumps(
        {
            "visible_state_summary": prior_visible_state_summary,
        },
        ensure_ascii=False,
    )
    user_prompt = render_stage2_task_prompt(
        task_prompt_template=task_prompt_template,
        task_prompt=testcase.prompt,
        success_condition=success_condition,
        action_text=action_text or "unknown",
        prior_state_summary_json=prior_summary_json,
    )

    payload = {
        "model": config.stage2_model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_path_to_data_url(previous_screenshot_path),
                        "detail": config.stage2_image_detail,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_path_to_data_url(current_screenshot_path),
                        "detail": config.stage2_image_detail,
                    },
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "stage2_trajectory_judgment",
                "strict": True,
                "schema": stage2_json_schema(),
            }
        },
    }

    response_json = post_responses_api(
        payload=payload,
        api_key=api_key,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
    )
    output_text = extract_response_output_text(response_json)
    parsed = json.loads(output_text)
    validate_stage2_judgment(parsed)
    return Stage2Judgment(**parsed)


def judge_final_outcome(
    testcase: TestCase,
    trajectory_events: list[dict[str, Any]],
    final_screenshot_path: Path,
    final_agent_answer: str,
    objective_evaluation_result: int | None,
    system_prompt: str,
    task_prompt_template: str,
    config: JudgeConfig,
) -> Stage3Judgment:
    if config.dry_run:
        return Stage3Judgment(
            score=1,
            reason="Dry run placeholder. No model call was made.",
        )

    api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used.")

    success_condition = extract_success_condition(testcase.prompt)
    user_prompt = render_stage3_task_prompt(
        task_prompt_template=task_prompt_template,
        task_prompt=testcase.prompt,
        success_condition=success_condition,
        llm_judge_gt=testcase.raw_fields.get("llm_judge_gt", ""),
        objective_evaluation_result=objective_evaluation_result,
        trajectory_events_json=json.dumps(trajectory_events, ensure_ascii=False, indent=2),
        final_agent_answer=final_agent_answer or "No final agent answer provided",
    )

    payload = {
        "model": config.stage3_model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_path_to_data_url(final_screenshot_path),
                        "detail": config.stage3_image_detail,
                    },
                ],
            },
        ],
        "reasoning": {
            "effort": config.stage3_reasoning_effort,
        },
        "text": {
            "format": {
                "type": "json_schema",
                "name": "stage3_final_outcome_judgment",
                "strict": True,
                "schema": stage3_json_schema(),
            }
        },
    }

    response_json = post_responses_api(
        payload=payload,
        api_key=api_key,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
    )
    output_text = extract_response_output_text(response_json)
    parsed = json.loads(output_text)
    validate_stage3_judgment(parsed, objective_evaluation_result=objective_evaluation_result)
    return Stage3Judgment(**parsed)


def summarize_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    runs = aggregate["runs"]
    pair_total = sum(run["pairs_total"] for run in runs)
    flagged_total = sum(run["pairs_flagged_stage1"] for run in runs)
    judged_total = sum(run["pairs_judged_stage2"] for run in runs)
    error_total = sum(
        1
        for run in runs
        for pair in run["pairs"]
        if pair["stage2_error"] is not None
    )
    stage3_judged_total = sum(1 for run in runs if run["stage3"] is not None)
    stage3_error_total = sum(1 for run in runs if run["stage3_error"] is not None)
    return {
        "run_count": len(runs),
        "pairs_total": pair_total,
        "pairs_flagged_stage1": flagged_total,
        "pairs_judged_stage2": judged_total,
        "stage2_error_count": error_total,
        "runs_judged_stage3": stage3_judged_total,
        "stage3_error_count": stage3_error_total,
    }


def sorted_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name != "run_manifests"
    )


def load_testcases(path: Path) -> dict[str, TestCase]:
    raw_cases = parse_simple_testcases_yaml(path)
    testcases = {}
    for raw in raw_cases:
        testcase = TestCase(
            id=raw.get("id", ""),
            description=raw.get("description", ""),
            prompt=raw.get("prompt", ""),
            raw_fields=raw,
        )
        if testcase.id:
            testcases[testcase.id] = testcase
    return testcases


def parse_simple_testcases_yaml(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "test_cases:":
            index += 1
            continue

        if stripped.startswith("- "):
            if current is not None:
                items.append(current)
            current = {}
            key, raw_value = split_yaml_key_value(stripped[2:])
            value, index = consume_yaml_value(lines, index, 0, raw_value)
            current[key] = value
            continue

        indent = len(line) - len(line.lstrip(" "))
        if current is not None and indent >= 2 and ":" in stripped:
            key, raw_value = split_yaml_key_value(stripped)
            value, index = consume_yaml_value(lines, index, indent, raw_value)
            current[key] = value
            continue

        index += 1

    if current is not None:
        items.append(current)
    return items


def split_yaml_key_value(text: str) -> tuple[str, str]:
    key, value = text.split(":", 1)
    return key.strip(), value.strip()


def consume_yaml_value(
    lines: list[str],
    index: int,
    parent_indent: int,
    raw_value: str,
) -> tuple[str, int]:
    if raw_value in {"|", "|-", ">", ">-"}:
        collected: list[str] = []
        block_indent: int | None = None
        cursor = index + 1
        while cursor < len(lines):
            line = lines[cursor]
            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))
            if stripped and indent <= parent_indent:
                break
            if block_indent is None and stripped:
                block_indent = indent
            if block_indent is None:
                collected.append("")
            else:
                collected.append(line[block_indent:])
            cursor += 1
        return normalize_block_scalar(collected, raw_value), cursor
    return parse_yaml_scalar(raw_value), index + 1


def normalize_block_scalar(lines: list[str], style: str) -> str:
    if style.startswith("|"):
        return "\n".join(lines).strip()

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
            continue
        if current:
            paragraphs.append(" ".join(current).strip())
            current = []
        if paragraphs and paragraphs[-1] != "":
            paragraphs.append("")
    if current:
        paragraphs.append(" ".join(current).strip())
    return "\n".join(paragraphs).strip()


def parse_yaml_scalar(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def detect_task_id(run_dir: Path, summary_info: dict[str, Any]) -> str:
    candidates = [
        run_dir.name,
        str(summary_info.get("task_name", "")),
        str(summary_info.get("task_id", "")),
    ]
    for candidate in candidates:
        match = TASK_ID_RE.search(candidate)
        if match:
            return match.group(1)
    return ""


def extract_success_condition(prompt: str) -> str:
    marker = "SUCCESS CONDITION:"
    if marker not in prompt:
        return ""
    return prompt.split(marker, 1)[1].strip()


def load_action_text_for_step(run_dir: Path, step_index: int) -> str:
    step_path = run_dir / f"step_{step_index}.pkl.gz"
    if not step_path.exists():
        return ""
    step_data = as_dict(load_pickle(step_path))
    return extract_executed_action_text(step_data.get("action"))


def extract_executed_action_text(raw_action: Any) -> str:
    if raw_action is None:
        return ""
    text = str(raw_action).strip()
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text


def render_stage2_task_prompt(
    task_prompt_template: str,
    task_prompt: str,
    success_condition: str,
    action_text: str,
    prior_state_summary_json: str,
) -> str:
    rendered = task_prompt_template
    rendered = rendered.replace("{TASK_PROMPT}", task_prompt)
    rendered = rendered.replace("{SUCCESS_CONDITION}", success_condition or "Not provided")
    rendered = rendered.replace("{ACTION_TEXT}", action_text or "unknown")
    rendered = rendered.replace("{PRIOR_STATE_SUMMARY_JSON}", prior_state_summary_json)
    return rendered


def render_stage3_task_prompt(
    task_prompt_template: str,
    task_prompt: str,
    success_condition: str,
    llm_judge_gt: str,
    objective_evaluation_result: int | None,
    trajectory_events_json: str,
    final_agent_answer: str,
) -> str:
    rendered = task_prompt_template
    rendered = rendered.replace("{TASK_PROMPT}", task_prompt)
    rendered = rendered.replace("{SUCCESS_CONDITION}", success_condition or "Not provided")
    rendered = rendered.replace("{LLM_JUDGE_GT}", llm_judge_gt or "Not provided")
    rendered = rendered.replace(
        "{OBJECTIVE_EVALUATION_RESULT}",
        "Not provided" if objective_evaluation_result is None else str(objective_evaluation_result),
    )
    rendered = rendered.replace("{TRAJECTORY_EVENTS_JSON}", trajectory_events_json)
    rendered = rendered.replace("{FINAL_AGENT_ANSWER}", final_agent_answer or "No final agent answer provided")
    return rendered


def load_objective_evaluation_results(
    results_dir: Path,
    configured_path: Path | None,
) -> tuple[dict[str, int], Path | None]:
    path = resolve_objective_evaluation_path(results_dir=results_dir, configured_path=configured_path)
    if path is None:
        return {}, None

    raw = json.loads(path.read_text())
    parsed = parse_objective_evaluation_payload(raw)
    LOGGER.info("Loaded %d objective evaluation result(s) from %s", len(parsed), path)
    return parsed, path


def resolve_objective_evaluation_path(results_dir: Path, configured_path: Path | None) -> Path | None:
    candidate_paths: list[Path] = []
    if configured_path is not None:
        candidate_paths.append(configured_path)
    candidate_paths.extend(
        [
            results_dir / "objective_evaluation.json",
            results_dir.parent / "objective_evaluation.json",
            REPO_ROOT / "objective_evaluation.json",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def parse_objective_evaluation_payload(raw: Any) -> dict[str, int]:
    results: dict[str, int] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            task_id, parsed_value = parse_objective_evaluation_entry(key, value)
            if task_id is not None:
                results[task_id] = parsed_value
        return results

    if isinstance(raw, list):
        for item in raw:
            task_id, parsed_value = parse_objective_evaluation_entry(item, None)
            if task_id is not None:
                results[task_id] = parsed_value
        return results

    raise RuntimeError("objective_evaluation.json must contain a JSON object or array.")


def parse_objective_evaluation_entry(raw_key: Any, raw_value: Any) -> tuple[str | None, int]:
    if isinstance(raw_key, str):
        match = re.search(r"(tc_[a-z]+_\d+)\s*:?\s*([01])?\s*$", raw_key.strip())
        if match and match.group(2) is not None:
            return match.group(1), int(match.group(2))
        if match and raw_value is not None:
            return match.group(1), parse_binary_flag(raw_value)

    if isinstance(raw_key, dict):
        task_id = raw_key.get("id") or raw_key.get("task_id")
        value = raw_key.get("result") if "result" in raw_key else raw_key.get("value")
        if isinstance(task_id, str) and value is not None:
            return task_id.strip(), parse_binary_flag(value)

    return None, 0


def parse_binary_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise RuntimeError(f"Objective evaluation value must be binary 0/1, got {value!r}")


def image_path_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def stage2_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "visible_action": {"type": "string"},
            "visible_change_summary": {"type": "string"},
            "visible_change_type": {
                "type": "string",
                "enum": [
                    "navigation",
                    "ui_interaction",
                    "content_update",
                    "object_added",
                    "object_removed",
                    "selection_changed",
                    "text_changed",
                    "dialog_opened",
                    "dialog_closed",
                    "error",
                    "no_clear_change",
                    "other",
                ],
            },
            "task_relevance": {
                "type": "string",
                "enum": ["high", "medium", "low", "none"],
            },
            "progress": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"],
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "uncertainty_note": {"type": "string"},
            "updated_visible_state_summary": {"type": "string"},
        },
        "required": [
            "visible_action",
            "visible_change_summary",
            "visible_change_type",
            "task_relevance",
            "progress",
            "confidence",
            "uncertainty_note",
            "updated_visible_state_summary",
        ],
    }


def stage3_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"],
    }


def post_responses_api(
    payload: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/responses"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API request failed ({exc.code}): {body}") from exc


def extract_response_output_text(response_json: dict[str, Any]) -> str:
    top_level_output_text = response_json.get("output_text")
    if isinstance(top_level_output_text, str) and top_level_output_text.strip():
        return top_level_output_text

    for output_item in response_json.get("output", []):
        for content_item in output_item.get("content", []):
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                return text
    raise RuntimeError(f"Could not extract output text from OpenAI response: {json.dumps(response_json)[:1000]}")


def validate_stage2_judgment(payload: dict[str, Any]) -> None:
    required = {
        "visible_action",
        "visible_change_summary",
        "visible_change_type",
        "task_relevance",
        "progress",
        "confidence",
        "uncertainty_note",
        "updated_visible_state_summary",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Stage 2 judgment is missing required keys: {sorted(missing)}")


def validate_stage3_judgment(
    payload: dict[str, Any],
    objective_evaluation_result: int | None = None,
) -> None:
    required = {"score", "reason"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Stage 3 judgment is missing required keys: {sorted(missing)}")
    if not isinstance(payload["score"], int) or not 1 <= payload["score"] <= 5:
        raise RuntimeError(f"Stage 3 score must be an integer between 1 and 5, got {payload['score']!r}")
    if objective_evaluation_result == 0 and payload["score"] == 5:
        raise RuntimeError("Stage 3 score cannot be 5 when OBJECTIVE EVALUATION RESULT is 0.")


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def extract_step_from_name(filename: str) -> int:
    match = re.fullmatch(r"screenshot_step_(\d+)\.png", filename)
    if not match:
        raise ValueError(f"Unexpected screenshot filename: {filename}")
    return int(match.group(1))


def limit_reached(limit: int | None, current: int) -> bool:
    return limit is not None and current >= limit


def extract_final_agent_answer(summary_info: dict[str, Any]) -> str:
    for key in ("raw_agent_response", "agent_response"):
        value = summary_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_stage3_trajectory_events(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for pair in pairs:
        judgment = pair.get("stage2_judgment")
        if judgment is None:
            continue
        events.append(
            {
                "first_step": pair["first_step"],
                "second_step": pair["second_step"],
                "visible_action": judgment["visible_action"],
                "visible_change_summary": judgment["visible_change_summary"],
                "visible_change_type": judgment["visible_change_type"],
                "task_relevance": judgment["task_relevance"],
                "progress": judgment["progress"],
                "confidence": judgment["confidence"],
            }
        )
    return events


def maybe_progress(
    iterable: Any,
    total: int,
    desc: str,
    enabled: bool,
    position: int = 0,
    leave: bool = False,
) -> Any:
    if not enabled:
        return iterable
    if tqdm is None:
        return iterable
    if not sys.stderr.isatty():
        return iterable
    return tqdm(iterable, total=total, desc=desc, position=position, leave=leave)


def taskcase_label(testcase: TestCase | None, task_id: str) -> str:
    if testcase is not None:
        return testcase.id
    return task_id


def update_progress_counts(progress_bar: Any, stage1_flagged: int, stage2_completed: int) -> None:
    if not hasattr(progress_bar, "set_postfix_str"):
        return
    progress_bar.set_postfix_str(f"flagged={stage1_flagged} stage2={stage2_completed}")


def update_progress_label(progress_bar: Any, task_label: str) -> None:
    if not hasattr(progress_bar, "set_postfix_str"):
        return
    progress_bar.set_postfix_str(task_label[:40])
