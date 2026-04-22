#!/usr/bin/env python3
"""Automatic evaluator for voidcut /finish endpoint outputs.

Compares agent responses from either a run_agent JSON file or a results
directory full of testcase subfolders against per-test-case ground-truth
JSON files using strict binary block-by-block matching.

Usage:
    python eval_voidcut.py <responses_path> [<gt_dir>] [--tolerance_ms=1000] [--verbose]
    python eval_voidcut.py responses/run1.json assets/video_ground_truth --verbose
    python eval_voidcut.py results_gpt54_video assets/video_ground_truth --verbose

Ground-truth directory format::

    assets/video_ground_truth/
        tc_vid_001.json   ->  {"scenario": 1, "endpoint_content": {"operations": [...]}}
        tc_vid_002.json
        ...

Each file is named ``<test_id>.json``. ``scenario`` maps to effect-validation
rules (6-9 have special checks).  ``endpoint_content`` is the expected
/finish JSON payload.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire


def get_export_complete(data: dict) -> dict | None:
    """Return the last EXPORT_COMPLETE operation's data, or None."""
    ops = data.get("operations", [])
    for op in reversed(ops):
        if op.get("type") == "EXPORT_COMPLETE":
            return op.get("data", {})
    return None


@dataclass
class EvalResult:
    scenario: int
    passed: bool
    matched_blocks: int = 0
    missing_gt_blocks: int = 0
    extra_pred_blocks: int = 0
    failure_reasons: list[str] | None = None

    def summary(self) -> str:
        lines = [f"{'PASS' if self.passed else 'FAIL'}  scenario={self.scenario}"]
        lines.append(
            f"matched={self.matched_blocks} missing_gt={self.missing_gt_blocks} extra_pred={self.extra_pred_blocks}"
        )
        if self.failure_reasons:
            lines.append("")
            lines.append("Reasons:")
            for reason in self.failure_reasons:
                lines.append(f"- {reason}")
        return "\n".join(lines)


TEMPORAL_FIELDS = ("startTimeMs", "durationMs", "trimFromStartMs", "trimFromEndMs")
REQUIRED_BLOCK_FIELDS = ("mediaName", "mediaType", "startTimeMs", "durationMs")
DEFAULT_LIGHT_PARAMS = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}
DEFAULT_FADE_PARAMS = {"fromOpacity": 1.0, "toOpacity": 0.0, "curve": "linear"}
TESTCASE_ID_PATTERN = re.compile(r"(tc_vid_\d+)")


@dataclass(frozen=True)
class FlatBlock:
    block_id: str
    track_index: int
    raw: dict[str, Any]


def _flatten_timeline_blocks(export_data: dict) -> list[FlatBlock]:
    blocks: list[FlatBlock] = []
    timeline = export_data.get("timeline")
    if not isinstance(timeline, list):
        return blocks

    for track_pos, track in enumerate(timeline):
        if not isinstance(track, dict):
            continue
        track_index = track.get("trackIndex")
        if not isinstance(track_index, int):
            track_index = track_pos

        elements = track.get("elements", [])
        if not isinstance(elements, list):
            continue

        for element_pos, element in enumerate(elements):
            if not isinstance(element, dict):
                continue
            block_id = str(element.get("elementId") or f"t{track_index}-e{element_pos}")
            blocks.append(FlatBlock(block_id=block_id, track_index=track_index, raw=element))

    return blocks


def _order_key(block: FlatBlock) -> tuple:
    def _safe_float(value: Any) -> float:
        if _is_number(value):
            return float(value)
        return 0.0

    media_name = str(block.raw.get("mediaName", ""))
    media_type = str(block.raw.get("mediaType", "")).upper()
    start = _safe_float(block.raw.get("startTimeMs", 0))
    duration = _safe_float(block.raw.get("durationMs", 0))
    trim_start = _safe_float(block.raw.get("trimFromStartMs", 0))
    trim_end = _safe_float(block.raw.get("trimFromEndMs", 0))
    return (media_type, media_name, start, duration, trim_start, trim_end, block.track_index, block.block_id)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _to_float(value: Any, fallback: float = 0.0) -> float:
    return float(value) if _is_number(value) else fallback


def _nearly_equal(a: Any, b: Any, epsilon: float = 1e-6) -> bool:
    if not _is_number(a) or not _is_number(b):
        return False
    return abs(float(a) - float(b)) <= epsilon


def _normalize_media_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return text


def _flatten_effect_items(export_data: dict) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    timeline = export_data.get("timeline")
    if not isinstance(timeline, list):
        return effects
    for track in timeline:
        if not isinstance(track, dict):
            continue
        subtrack = track.get("effectSubTrack")
        if not isinstance(subtrack, dict):
            continue
        items = subtrack.get("items")
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                effects.append(item)
    return effects


def _light_params_ok(
    params: dict[str, Any],
    *,
    boosted: str,
    errors: list[str],
    context: str,
) -> None:
    brightness = _to_float(params.get("brightness"), DEFAULT_LIGHT_PARAMS["brightness"])
    contrast = _to_float(params.get("contrast"), DEFAULT_LIGHT_PARAMS["contrast"])
    saturation = _to_float(params.get("saturation"), DEFAULT_LIGHT_PARAMS["saturation"])

    if boosted == "contrast":
        if contrast <= DEFAULT_LIGHT_PARAMS["contrast"]:
            errors.append(f"{context}: contrast must be increased (>1)")
        if not _nearly_equal(brightness, DEFAULT_LIGHT_PARAMS["brightness"]):
            errors.append(f"{context}: brightness must stay unchanged at 0")
        if not _nearly_equal(saturation, DEFAULT_LIGHT_PARAMS["saturation"]):
            errors.append(f"{context}: saturation must stay unchanged at 1")
    elif boosted == "brightness":
        if brightness <= DEFAULT_LIGHT_PARAMS["brightness"]:
            errors.append(f"{context}: brightness must be increased (>0)")
        if not _nearly_equal(contrast, DEFAULT_LIGHT_PARAMS["contrast"]):
            errors.append(f"{context}: contrast must stay unchanged at 1")
        if not _nearly_equal(saturation, DEFAULT_LIGHT_PARAMS["saturation"]):
            errors.append(f"{context}: saturation must stay unchanged at 1")
    elif boosted == "saturation":
        if saturation <= DEFAULT_LIGHT_PARAMS["saturation"]:
            errors.append(f"{context}: saturation must be increased (>1)")
        if not _nearly_equal(brightness, DEFAULT_LIGHT_PARAMS["brightness"]):
            errors.append(f"{context}: brightness must stay unchanged at 0")
        if not _nearly_equal(contrast, DEFAULT_LIGHT_PARAMS["contrast"]):
            errors.append(f"{context}: contrast must stay unchanged at 1")


def _find_clip(blocks: list[FlatBlock], media_name: str) -> FlatBlock | None:
    target = media_name.lower()
    for block in blocks:
        normalized = _normalize_media_name(block.raw.get("mediaName"))
        if normalized == target:
            return block
    return None


def _find_full_clip_light_effect(
    effects: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    epsilon_ms: float,
) -> dict[str, Any] | None:
    for effect in effects:
        if str(effect.get("type", "")).lower() != "light-adjustment":
            continue
        start = _to_float(effect.get("startTimeMs"))
        end = _to_float(effect.get("endTimeMs"))
        if abs(start - clip_start) <= epsilon_ms and abs(end - clip_end) <= epsilon_ms:
            return effect
    return None


def _validate_scenario_effects(
    scenario: int,
    pred_blocks: list[FlatBlock],
    pred_export: dict,
    epsilon_ms: float,
) -> list[str]:
    errors: list[str] = []
    if scenario not in {7, 8, 9, 14}:
        return errors

    effects = _flatten_effect_items(pred_export)
    fade_effects = [e for e in effects if str(e.get("type", "")).lower() == "fade-out"]
    light_effects = [e for e in effects if str(e.get("type", "")).lower() == "light-adjustment"]

    if scenario == 7:
        flower = _find_clip(pred_blocks, "flower video")
        if flower is None:
            errors.append("missing clip: Flower Video")
            return errors
        if light_effects:
            clip_start = _to_float(flower.raw.get("startTimeMs"))
            clip_end = clip_start + _to_float(flower.raw.get("durationMs"))
            effect = _find_full_clip_light_effect(light_effects, clip_start, clip_end, epsilon_ms)
            if effect is None:
                errors.append("light-adjustment must cover the whole Flower Video clip")
                return errors
            params = effect.get("params", {}) if isinstance(effect.get("params"), dict) else {}
            _light_params_ok(params, boosted="contrast", errors=errors, context="Flower Video light-adjustment")
        return errors

    if scenario == 8:
        return errors

    if scenario == 9:
        return errors

    # scenario == 14
    first = _find_clip(pred_blocks, "tuning a radio")
    second = _find_clip(pred_blocks, "flower video")
    if first is None:
        errors.append("missing clip: Tuning a Radio")
        return errors
    if second is None:
        errors.append("missing clip: Flower Video")
        return errors

    first_start = min(
        _to_float(b.raw.get("startTimeMs"))
        for b in pred_blocks if _normalize_media_name(b.raw.get("mediaName")) == "tuning a radio"
    )
    first_end = max(
        _to_float(b.raw.get("startTimeMs")) + _to_float(b.raw.get("durationMs"))
        for b in pred_blocks if _normalize_media_name(b.raw.get("mediaName")) == "tuning a radio"
    )
    second_start = min(
        _to_float(b.raw.get("startTimeMs"))
        for b in pred_blocks if _normalize_media_name(b.raw.get("mediaName")) == "flower video"
    )
    second_end = max(
        _to_float(b.raw.get("startTimeMs")) + _to_float(b.raw.get("durationMs"))
        for b in pred_blocks if _normalize_media_name(b.raw.get("mediaName")) == "flower video"
    )

    light_segment: dict[str, Any] | None = None
    for effect in light_effects:
        start = _to_float(effect.get("startTimeMs"))
        end = _to_float(effect.get("endTimeMs"))
        if abs(start - first_start) <= epsilon_ms and abs(end - (first_start + 4000.0)) <= epsilon_ms:
            light_segment = effect
            break
    if light_segment is None:
        errors.append("Tuning a Radio: missing 4s segment light-adjustment from clip start")
    else:
        params = light_segment.get("params", {}) if isinstance(light_segment.get("params"), dict) else {}
        _light_params_ok(
            params,
            boosted="contrast",
            errors=errors,
            context="Tuning a Radio light-adjustment",
        )
        seg_end = _to_float(light_segment.get("endTimeMs"))
        if seg_end > first_end + epsilon_ms:
            errors.append("Tuning a Radio light-adjustment must stay within first clip")

    if len(fade_effects) != 1:
        errors.append(f"expected exactly 1 fade-out effect, found {len(fade_effects)}")
        return errors

    fade = fade_effects[0]
    fade_start = _to_float(fade.get("startTimeMs"))
    fade_end = _to_float(fade.get("endTimeMs"))
    expected_fade_start = second_end - 3000.0
    if abs(fade_end - second_end) > epsilon_ms:
        errors.append("Flower Video fade-out must end at clip end (last 3s)")
    if abs(fade_start - expected_fade_start) > epsilon_ms:
        errors.append("Flower Video fade-out must start around clip_end-3000ms")
    if fade_start < second_start - epsilon_ms:
        errors.append("Flower Video fade-out must be applied on the second clip")

    params = fade.get("params", {}) if isinstance(fade.get("params"), dict) else {}
    if str(params.get("curve", "")).lower() != "ease-out":
        errors.append("Flower Video fade-out must use ease-out curve")
    if not _nearly_equal(_to_float(params.get("fromOpacity")), DEFAULT_FADE_PARAMS["fromOpacity"]):
        errors.append("Flower Video fade-out fromOpacity must stay unchanged at 1")
    if not _nearly_equal(_to_float(params.get("toOpacity")), DEFAULT_FADE_PARAMS["toOpacity"]):
        errors.append("Flower Video fade-out toOpacity must stay unchanged at 0")
    return errors


def _validate_required_fields(block: FlatBlock, side: str) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_BLOCK_FIELDS:
        value = block.raw.get(field)
        if value is None:
            errors.append(f"{side} block {block.block_id} missing required field '{field}'")
            continue
        if field in ("startTimeMs", "durationMs") and not _is_number(value):
            errors.append(f"{side} block {block.block_id} has non-numeric '{field}'")
    media_type = block.raw.get("mediaType")
    if media_type is not None and not isinstance(media_type, str):
        errors.append(f"{side} block {block.block_id} has non-string 'mediaType'")
    media_name = block.raw.get("mediaName")
    if media_name is not None and not isinstance(media_name, str):
        errors.append(f"{side} block {block.block_id} has non-string 'mediaName'")
    return errors


def _compare_temporal_fields(gt_block: FlatBlock, pred_block: FlatBlock, epsilon_ms: float) -> str | None:
    for field in TEMPORAL_FIELDS:
        gt_has = field in gt_block.raw and gt_block.raw.get(field) is not None
        pred_has = field in pred_block.raw and pred_block.raw.get(field) is not None

        if not gt_has and not pred_has:
            continue
        if gt_has != pred_has:
            return (
                f"field '{field}' presence mismatch "
                f"(gt={gt_block.raw.get(field)!r}, pred={pred_block.raw.get(field)!r})"
            )

        gt_value = gt_block.raw.get(field)
        pred_value = pred_block.raw.get(field)
        if not _is_number(gt_value) or not _is_number(pred_value):
            return f"field '{field}' is non-numeric (gt={gt_value!r}, pred={pred_value!r})"

        diff = abs(float(gt_value) - float(pred_value))
        if diff > epsilon_ms:
            return f"field '{field}' differs by {diff:.3f}ms (> {epsilon_ms:.3f}ms)"
    return None


def _is_overlapped_by_video(
    block: FlatBlock, all_blocks: list[FlatBlock], tolerance_ms: float = 500.0,
) -> bool:
    """Check if a block is temporally overlapped by any VIDEO block.

    Returns True only when the overlap exceeds *tolerance_ms* so that
    adjacent or barely-touching clips are not treated as overlapping.
    """
    block_start = _to_float(block.raw.get("startTimeMs"))
    block_end = block_start + _to_float(block.raw.get("durationMs"))

    for other in all_blocks:
        if str(other.raw.get("mediaType", "")).upper() != "VIDEO":
            continue
        video_start = _to_float(other.raw.get("startTimeMs"))
        video_end = video_start + _to_float(other.raw.get("durationMs"))
        overlap = min(block_end, video_end) - max(block_start, video_start)
        if overlap > tolerance_ms:
            return True
    return False


def _candidate_mismatch_reason(
    gt_block: FlatBlock,
    pred_block: FlatBlock,
    epsilon_ms: float,
    gt_blocks: list[FlatBlock] | None = None,
) -> str:
    gt_media_type = str(gt_block.raw.get("mediaType", "")).upper()
    pred_media_type = str(pred_block.raw.get("mediaType", "")).upper()
    gt_media_name = str(gt_block.raw.get("mediaName", ""))
    pred_media_name = str(pred_block.raw.get("mediaName", ""))

    if gt_media_type != pred_media_type:
        return f"mediaType mismatch (gt={gt_media_type!r}, pred={pred_media_type!r})"
    if gt_media_name != pred_media_name:
        return f"mediaName mismatch (gt={gt_media_name!r}, pred={pred_media_name!r})"

    # Video and audio blocks are track-agnostic.
    # Text blocks are track-agnostic unless overlapped by a video block.
    track_agnostic = gt_media_type in ("VIDEO", "AUDIO")
    if not track_agnostic and gt_media_type == "TEXT" and gt_blocks is not None:
        track_agnostic = not _is_overlapped_by_video(gt_block, gt_blocks)

    if not track_agnostic and gt_block.track_index != pred_block.track_index:
        return (
            f"trackIndex mismatch for {gt_media_type} "
            f"(gt={gt_block.track_index}, pred={pred_block.track_index})"
        )

    temporal_error = _compare_temporal_fields(gt_block, pred_block, epsilon_ms)
    if temporal_error:
        return temporal_error
    return ""


def _validate_exports(gt_export: dict | None, pred_export: dict | None) -> list[str]:
    errors: list[str] = []
    if gt_export is None:
        errors.append("ground truth export missing EXPORT_COMPLETE")
    if pred_export is None:
        errors.append("result export missing EXPORT_COMPLETE")
    return errors


def _extract_testcase_id(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        match = TESTCASE_ID_PATTERN.search(str(value))
        if match:
            return match.group(1)
    return None


def _extract_localstorage_operations(raw: str) -> str:
    """Convert eval-harness-localstorage format to {"operations": [...]} JSON string."""
    if not raw:
        return ""
    marker = "--- eval-harness-localstorage ---"
    idx = raw.find(marker)
    if idx == -1:
        return raw
    after = raw[idx + len(marker):].lstrip("\n")
    # Content ends at next section separator or end of string
    end = after.find("\n--- ")
    section = after[:end].strip() if end != -1 else after.strip()
    try:
        ops = json.loads(section)
    except json.JSONDecodeError:
        return ""
    return json.dumps({"operations": ops})


def _load_agent_responses(responses_path: Path) -> tuple[list[dict[str, Any]], str]:
    if responses_path.is_file():
        responses_data = json.loads(responses_path.read_text(encoding="utf-8"))
        agent_responses: list[dict[str, Any]] = responses_data.get("responses", [])
        model = responses_data.get("metadata", {}).get("model", "unknown")
        return agent_responses, str(model)

    if not responses_path.is_dir():
        raise ValueError(f"responses path must be a file or directory: {responses_path}")

    agent_responses: list[dict[str, Any]] = []
    model_names: set[str] = set()
    summary_paths = sorted(
        path
        for path in responses_path.iterdir()
        if path.is_dir() and (path / "summary_info.json").is_file()
    )

    for run_dir in summary_paths:
        summary_path = run_dir / "summary_info.json"
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        test_id = _extract_testcase_id(
            summary_data.get("task_name"),
            summary_data.get("task_id"),
            run_dir.name,
        )
        if not test_id:
            continue

        endpoint_content = summary_data.get("post_run_page_content")
        if not endpoint_content:
            endpoint_content = _extract_localstorage_operations(
                summary_data.get("post_run_js_result") or ""
            )

        model_name = summary_data.get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            model_names.add(model_name.strip())

        agent_responses.append(
            {
                "test_id": test_id,
                "endpoint_content": endpoint_content,
                "summary_info_path": str(summary_path),
            }
        )

    if not model_names:
        model = responses_path.name
    elif len(model_names) == 1:
        model = next(iter(model_names))
    else:
        model = ", ".join(sorted(model_names))

    return agent_responses, model


def _evaluated_output_dir(responses_path: Path) -> Path:
    """Return the directory where evaluation artifacts should be stored."""
    return responses_path if responses_path.is_dir() else responses_path.parent


def evaluate_binary(scenario: int, gt_data: dict, result_data: dict, epsilon_ms: float) -> EvalResult:
    gt_export = get_export_complete(gt_data)
    pred_export = get_export_complete(result_data)

    errors = _validate_exports(gt_export, pred_export)
    if errors:
        return EvalResult(
            scenario=scenario,
            passed=False,
            failure_reasons=errors,
        )

    assert gt_export is not None and pred_export is not None

    gt_blocks = sorted(_flatten_timeline_blocks(gt_export), key=_order_key)
    pred_blocks = sorted(_flatten_timeline_blocks(pred_export), key=_order_key)

    for gt_block in gt_blocks:
        errors.extend(_validate_required_fields(gt_block, "gt"))
    for pred_block in pred_blocks:
        errors.extend(_validate_required_fields(pred_block, "pred"))

    if errors:
        return EvalResult(
            scenario=scenario,
            passed=False,
            failure_reasons=errors,
        )

    if len(gt_blocks) != len(pred_blocks):
        errors.append(
            f"block count mismatch (gt={len(gt_blocks)}, pred={len(pred_blocks)})"
        )

    unmatched_pred_indices = set(range(len(pred_blocks)))
    matched_count = 0

    for gt_block in gt_blocks:
        matched_idx: int | None = None
        sample_mismatch: str | None = None

        for pred_idx in sorted(unmatched_pred_indices):
            pred_block = pred_blocks[pred_idx]
            mismatch_reason = _candidate_mismatch_reason(gt_block, pred_block, epsilon_ms, gt_blocks)
            if mismatch_reason == "":
                matched_idx = pred_idx
                break
            if sample_mismatch is None:
                sample_mismatch = mismatch_reason

        if matched_idx is None:
            errors.append(
                "missing match for gt block "
                f"{gt_block.block_id} ({gt_block.raw.get('mediaType')}/{gt_block.raw.get('mediaName')})"
                + (f": {sample_mismatch}" if sample_mismatch else "")
            )
            continue

        unmatched_pred_indices.remove(matched_idx)
        matched_count += 1

    if unmatched_pred_indices:
        preview = []
        for idx in sorted(unmatched_pred_indices)[:3]:
            block = pred_blocks[idx]
            preview.append(
                f"{block.block_id}({block.raw.get('mediaType')}/{block.raw.get('mediaName')})"
            )
        errors.append(
            f"extra unmatched predicted blocks: {len(unmatched_pred_indices)} ({', '.join(preview)})"
        )

    errors.extend(_validate_scenario_effects(scenario, pred_blocks, pred_export, epsilon_ms))

    return EvalResult(
        scenario=scenario,
        passed=len(errors) == 0,
        matched_blocks=matched_count,
        missing_gt_blocks=max(0, len(gt_blocks) - matched_count),
        extra_pred_blocks=len(unmatched_pred_indices),
        failure_reasons=errors if errors else None,
    )


def _load_gt_dir(gt_dir: Path) -> dict[str, Any]:
    """Load all per-test-case GT files from a directory into a single dict."""
    gt_data: dict[str, Any] = {}
    for gt_file in sorted(gt_dir.glob("tc_vid_*.json")):
        test_id = gt_file.stem
        gt_data[test_id] = json.loads(gt_file.read_text(encoding="utf-8"))
    return gt_data


def evaluate_all(
    responses_json: str,
    gt_dir: str = "assets/video_ground_truth",
    tolerance_ms: float = 1000,
    verbose: bool = False,
) -> None:
    """Evaluate all test cases from a run_agent response file against ground truths.

    Args:
        responses_json: Path to the agent responses JSON or results directory.
        gt_dir:         Path to the ground-truth directory (one JSON per test case).
        tolerance_ms:   Timing tolerance in milliseconds (default 1000).
        verbose:        Print additional debug info.
    """
    if tolerance_ms < 0:
        print("ERROR: tolerance_ms must be non-negative.", file=sys.stderr)
        sys.exit(1)

    responses_path = Path(responses_json)
    gt_path = Path(gt_dir)

    if not responses_path.exists():
        print(f"ERROR: responses file not found: {responses_path}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.is_dir():
        print(f"ERROR: ground-truth directory not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    gt_data = _load_gt_dir(gt_path)

    try:
        agent_responses, model = _load_agent_responses(responses_path)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: unable to load responses from {responses_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not agent_responses:
        print("ERROR: no responses found in responses path.", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(agent_responses)} test case(s)  [model={model}]")
    print(f"Tolerance: {tolerance_ms:.0f}ms")
    print("-" * 60)

    total = 0
    passed = 0
    failed_ids: list[str] = []
    objective_evaluation: dict[str, int] = {}

    for response_entry in agent_responses:
        # print(response_entry)
        test_id: str = response_entry.get("test_id", "")
        endpoint_raw: str = response_entry.get("endpoint_content", "")

        if test_id not in gt_data:
            print(f"\n[SKIP] {test_id}  — no ground truth provided")
            continue

        gt_entry = gt_data[test_id]
        scenario: int = gt_entry.get("scenario", 0)
        gt_endpoint = gt_entry.get("endpoint_content", {})

        try:
            result_data = json.loads(endpoint_raw) if isinstance(endpoint_raw, str) else endpoint_raw
        except json.JSONDecodeError as exc:
            source_path = response_entry.get("summary_info_path")
            if source_path:
                print(f"\n[FAIL] {test_id}  — invalid endpoint JSON in {source_path}: {exc}")
            else:
                print(f"\n[FAIL] {test_id}  — invalid endpoint JSON: {exc}")
            total += 1
            failed_ids.append(test_id)
            objective_evaluation[test_id] = 0
            continue

        if verbose:
            gt_export = get_export_complete(gt_endpoint)
            result_export = get_export_complete(result_data)
            print(f"\n--- {test_id} Ground Truth EXPORT_COMPLETE ---")
            print(json.dumps(gt_export, indent=2) if gt_export else "(none)")
            print(f"\n--- {test_id} Result EXPORT_COMPLETE ---")
            print(json.dumps(result_export, indent=2) if result_export else "(none)")
            print()

        result = evaluate_binary(
            scenario=scenario,
            gt_data=gt_endpoint,
            result_data=result_data,
            epsilon_ms=tolerance_ms,
        )

        total += 1
        status = "PASS" if result.passed else "FAIL"
        if result.passed:
            passed += 1
            objective_evaluation[test_id] = 1
        else:
            failed_ids.append(test_id)
            objective_evaluation[test_id] = 0

        print(f"\n[{status}] {test_id}  (scenario={scenario})")
        print(f"       matched={result.matched_blocks} missing_gt={result.missing_gt_blocks} extra_pred={result.extra_pred_blocks}")
        if result.failure_reasons:
            for reason in result.failure_reasons:
                print(f"       - {reason}")

    output_dir = _evaluated_output_dir(responses_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_path = output_dir / "objective_evaluation.json"
    objective_path.write_text(json.dumps(objective_evaluation, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"TOTAL: {passed}/{total} passed")
    if failed_ids:
        print(f"FAILED: {', '.join(failed_ids)}")
    print(f"Saved objective evaluation to: {objective_path}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    fire.Fire(evaluate_all)
