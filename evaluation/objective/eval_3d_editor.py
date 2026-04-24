#!/usr/bin/env python3
"""Evaluate 3D editor scene exports against ground-truth scene files.

Compares agent results from benchmark run directories against per-task
ground-truth JSON files.  The predicted state is read from the
``post_run_page_content`` field in ``agent_outputs.json`` (falling back
to ``summary_info.json``).  Ground-truth files live under
``assets/3d_ground_truth/taskN.json``.

Usage:
    python -m evaluation.objective.eval_3d_editor <results_dir> <gt_dir> [--tolerance=0.15] [--verbose]

Examples:
    python -m evaluation.objective.eval_3d_editor results_3d_correct/results_haiku45_3d assets/3d_ground_truth
    python -m evaluation.objective.eval_3d_editor results_3d_correct/results_opus46_3d assets/3d_ground_truth --verbose
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fire


# ── Dataclasses ──────────────────────────────────────────────────────────────

TESTCASE_ID_PATTERN = re.compile(r"tc_(?:graph|clone3d|3d|vid)_(\d+)")

OBJECT_PROPS_EXACT = ("type", "wireframe")
OBJECT_PROPS_NUMERIC_ARRAY = ("position", "rotation", "scale")
OBJECT_PROPS_COLOR = ("color",)
OBJECT_PROPS_NUMERIC = ("opacity",)

Y_ROTATION_SYMMETRIC_TYPES = ("cone", "sphere", "cylinder")


def _is_y_rotation_symmetric(obj: dict) -> bool:
    """Object types whose Y-axis rotation is visually indistinguishable."""
    type_key = str(obj.get("type", "")).strip().lower()
    if type_key in Y_ROTATION_SYMMETRIC_TYPES:
        return True
    # Fallback for objects missing a type field: check the name prefix.
    name_key = str(obj.get("name", "")).strip().lower()
    return any(name_key.startswith(p) for p in Y_ROTATION_SYMMETRIC_TYPES)

SETTINGS_EXACT = ("showGrid", "showAxes")
SETTINGS_NUMERIC = ("fov", "orthoZoom")
SETTINGS_COLOR = ("backgroundColor",)


@dataclass
class ObjectResult:
    """Comparison result for a single 3D object."""

    name: str
    matched: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class SceneResult:
    """Comparison result for an entire scene."""

    scene_name: str
    found: bool
    object_results: list[ObjectResult] = field(default_factory=list)
    settings_errors: list[str] = field(default_factory=list)
    structural_errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not self.found:
            return False
        if self.structural_errors:
            return False
        if self.settings_errors:
            return False
        return all(obj.matched for obj in self.object_results)


@dataclass
class EvalResult:
    """Evaluation result for a single test case."""

    test_id: str
    task_number: int
    scene_result: SceneResult | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.scene_result is None:
            return False
        return self.scene_result.passed


# ── Helpers ──────────────────────────────────────────────────────────────────


def _normalize_color(color: Any) -> str | None:
    """Normalize a color value to lowercase hex string."""
    if color is None:
        return None
    s = str(color).strip().lower()
    if not s.startswith("#"):
        s = "#" + s
    return s


def _nearly_equal(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


def _compare_numeric_arrays(
    gt_arr: list, pred_arr: list, tolerance: float
) -> str | None:
    """Compare two numeric arrays element-wise within tolerance."""
    if not isinstance(pred_arr, list):
        return "expected list, got %s" % type(pred_arr).__name__
    if len(gt_arr) != len(pred_arr):
        return "length mismatch (gt=%d, pred=%d)" % (len(gt_arr), len(pred_arr))
    diffs = []
    for i, (g, p) in enumerate(zip(gt_arr, pred_arr)):
        try:
            gf, pf = float(g), float(p)
        except (TypeError, ValueError):
            return "non-numeric value at index %d (gt=%r, pred=%r)" % (i, g, p)
        if not _nearly_equal(gf, pf, tolerance):
            diffs.append("index %d: gt=%.4f pred=%.4f diff=%.4f" % (i, gf, pf, abs(gf - pf)))
    if diffs:
        return "; ".join(diffs)
    return None


def _extract_task_number(test_id: str) -> int | None:
    """Extract the numeric task number from a test case ID.

    E.g. tc_graph_004 -> 4, tc_clone3d_015 -> 15.
    """
    match = TESTCASE_ID_PATTERN.search(test_id)
    if match:
        return int(match.group(1))
    return None


def _extract_test_id(value: str) -> str | None:
    """Extract a test ID like tc_graph_004 or tc_clone3d_008 from a string."""
    match = re.search(r"(tc_(?:graph|clone3d|3d|vid)_\d+)", value)
    return match.group(1) if match else None


# ── Comparison logic ─────────────────────────────────────────────────────────


def _compare_object_pair(
    gt_obj: dict,
    pred_obj: dict,
    tolerance: float,
) -> list[str]:
    """Compare a single GT object against a single predicted object.

    Returns a list of human-readable property mismatches.  An empty list
    means a perfect match.  Y-axis rotation is ignored for shapes whose
    rotation around Y is visually indistinguishable (sphere/cylinder/cone).
    """
    errors: list[str] = []

    for prop in OBJECT_PROPS_EXACT:
        gt_val = gt_obj.get(prop)
        pred_val = pred_obj.get(prop)
        if gt_val is not None and pred_val is not None:
            if str(gt_val).lower() != str(pred_val).lower():
                errors.append("%s: gt=%r pred=%r" % (prop, gt_val, pred_val))

    for prop in OBJECT_PROPS_NUMERIC_ARRAY:
        gt_val = gt_obj.get(prop)
        pred_val = pred_obj.get(prop)
        if gt_val is not None and pred_val is not None:
            if (
                prop == "rotation"
                and _is_y_rotation_symmetric(gt_obj)
                and isinstance(gt_val, list)
                and isinstance(pred_val, list)
                and len(gt_val) > 1
                and len(pred_val) > 1
            ):
                pred_val = list(pred_val)
                pred_val[1] = gt_val[1]
            err = _compare_numeric_arrays(gt_val, pred_val, tolerance)
            if err:
                errors.append("%s: %s" % (prop, err))

    for prop in OBJECT_PROPS_COLOR:
        gt_val = _normalize_color(gt_obj.get(prop))
        pred_val = _normalize_color(pred_obj.get(prop))
        if gt_val is not None and pred_val is not None:
            if gt_val != pred_val:
                errors.append("%s: gt=%s pred=%s" % (prop, gt_val, pred_val))

    for prop in OBJECT_PROPS_NUMERIC:
        gt_val = gt_obj.get(prop)
        pred_val = pred_obj.get(prop)
        if gt_val is not None and pred_val is not None:
            try:
                if not _nearly_equal(float(gt_val), float(pred_val), tolerance):
                    errors.append("%s: gt=%s pred=%s" % (prop, gt_val, pred_val))
            except (TypeError, ValueError):
                errors.append("%s: non-numeric (gt=%r, pred=%r)" % (prop, gt_val, pred_val))

    return errors


def compare_objects(
    gt_objects: list[dict],
    pred_objects: list[dict],
    tolerance: float,
) -> tuple[list[ObjectResult], list[str]]:
    """Compare GT objects against predicted objects by content (name-independent).

    For every (predicted, GT) pair we compute the list of property
    mismatches, then greedily assign pairs starting with the lowest error
    count.  Unmatched GT objects fail with "no matching predicted object";
    leftover predicted objects become structural errors.  Y-rotation
    symmetry constraints for cone/sphere/cylinder are preserved.
    """
    structural_errors: list[str] = []

    pair_errors: dict[tuple[int, int], list[str]] = {}
    for pi, pred_obj in enumerate(pred_objects):
        for gi, gt_obj in enumerate(gt_objects):
            pair_errors[(pi, gi)] = _compare_object_pair(gt_obj, pred_obj, tolerance)

    sorted_pairs = sorted(
        pair_errors.items(),
        key=lambda kv: (len(kv[1]), kv[0][1], kv[0][0]),
    )

    pred_remaining: set[int] = set(range(len(pred_objects)))
    gt_remaining: set[int] = set(range(len(gt_objects)))
    matched_by_gt: dict[int, tuple[int, list[str]]] = {}

    for (pi, gi), errors in sorted_pairs:
        if pi in pred_remaining and gi in gt_remaining:
            matched_by_gt[gi] = (pi, errors)
            pred_remaining.discard(pi)
            gt_remaining.discard(gi)

    object_results: list[ObjectResult] = []
    for gi, gt_obj in enumerate(gt_objects):
        gt_name = gt_obj.get("name", "")
        if gi in matched_by_gt:
            _, errors = matched_by_gt[gi]
            object_results.append(
                ObjectResult(name=gt_name, matched=len(errors) == 0, errors=errors)
            )
        else:
            object_results.append(
                ObjectResult(
                    name=gt_name,
                    matched=False,
                    errors=["no matching predicted object found"],
                )
            )

    for pi in sorted(pred_remaining):
        pred_obj = pred_objects[pi]
        label = pred_obj.get("name") or pred_obj.get("type") or "<unnamed>"
        structural_errors.append(
            "extra predicted object '%s' (no matching GT object)" % label
        )

    if len(gt_objects) != len(pred_objects):
        structural_errors.append(
            "object count mismatch (gt=%d, pred=%d)" % (len(gt_objects), len(pred_objects))
        )

    return object_results, structural_errors


def compare_settings(
    gt_settings: dict,
    pred_settings: dict,
) -> list[str]:
    """Compare scene settings between GT and predicted."""
    errors: list[str] = []

    for prop in SETTINGS_EXACT:
        gt_val = gt_settings.get(prop)
        pred_val = pred_settings.get(prop)
        if gt_val is not None and pred_val is not None:
            if str(gt_val).lower() != str(pred_val).lower():
                errors.append("settings.%s: gt=%r pred=%r" % (prop, gt_val, pred_val))

    for prop in SETTINGS_COLOR:
        gt_val = _normalize_color(gt_settings.get(prop))
        pred_val = _normalize_color(pred_settings.get(prop))
        if gt_val is not None and pred_val is not None:
            if gt_val != pred_val:
                errors.append("settings.%s: gt=%s pred=%s" % (prop, gt_val, pred_val))

    for prop in SETTINGS_NUMERIC:
        gt_val = gt_settings.get(prop)
        pred_val = pred_settings.get(prop)
        if gt_val is not None and pred_val is not None:
            try:
                if not _nearly_equal(float(gt_val), float(pred_val), 0.01):
                    errors.append("settings.%s: gt=%s pred=%s" % (prop, gt_val, pred_val))
            except (TypeError, ValueError):
                errors.append("settings.%s: non-numeric (gt=%r, pred=%r)" % (prop, gt_val, pred_val))

    return errors


def compare_scene(
    gt_scene: dict,
    pred_scene: dict,
    tolerance: float,
) -> SceneResult:
    """Compare a GT scene against a predicted scene."""
    scene_name = gt_scene.get("name", "unknown")

    gt_objects = gt_scene.get("scene", {}).get("objects", [])
    pred_objects = pred_scene.get("scene", {}).get("objects", [])

    gt_settings = gt_scene.get("scene", {}).get("settings", {})
    pred_settings = pred_scene.get("scene", {}).get("settings", {})

    object_results, structural_errors = compare_objects(gt_objects, pred_objects, tolerance)
    settings_errors = compare_settings(gt_settings, pred_settings)

    return SceneResult(
        scene_name=scene_name,
        found=True,
        object_results=object_results,
        settings_errors=settings_errors,
        structural_errors=structural_errors,
    )


def find_scene_by_name(scenes: list[dict], name: str) -> dict | None:
    """Find a scene in the scenes list by name (case-insensitive)."""
    target = name.strip().lower()
    for scene in scenes:
        if scene.get("name", "").strip().lower() == target:
            return scene
    return None


def _count_object_name_hits(gt_objects: list[dict], pred_objects: list[dict]) -> int:
    """Count how many GT object names appear in the predicted objects list."""
    pred_names = {obj.get("name", "").strip().lower() for obj in pred_objects}
    return sum(
        1 for obj in gt_objects
        if obj.get("name", "").strip().lower() in pred_names
    )


def find_best_scene(
    scenes: list[dict],
    gt_scene: dict,
) -> dict | None:
    """Find the scene whose objects best match the GT, as a fallback.

    Used when exact name matching fails or finds an empty scene while the
    GT expects objects (e.g. "Create a new scene" tasks where the agent
    names the scene differently).
    """
    gt_objects = gt_scene.get("scene", {}).get("objects", [])
    if not gt_objects:
        return None

    best: dict | None = None
    best_hits = -1

    for scene in scenes:
        pred_objects = scene.get("scene", {}).get("objects", [])
        hits = _count_object_name_hits(gt_objects, pred_objects)
        if hits > best_hits:
            best_hits = hits
            best = scene

    return best if best_hits > 0 else None


# ── Data loading ─────────────────────────────────────────────────────────────


def load_predicted_content(run_dir: Path) -> str | None:
    """Load post_run_page_content from a run directory.

    Tries agent_outputs.json first, then summary_info.json.
    """
    for filename in ("agent_outputs.json", "summary_info.json"):
        filepath = run_dir / filename
        if not filepath.exists():
            continue
        data = json.loads(filepath.read_text(encoding="utf-8"))
        content = data.get("post_run_page_content")
        if content and isinstance(content, str) and content.strip():
            return content
    return None


def load_gt_scene(gt_dir: Path, task_number: int) -> dict | None:
    """Load a ground-truth scene file by task number."""
    gt_path = gt_dir / ("task%d.json" % task_number)
    if not gt_path.exists():
        return None
    return json.loads(gt_path.read_text(encoding="utf-8"))


def load_agent_responses(
    results_dir: Path,
) -> list[dict[str, Any]]:
    """Discover all run directories and extract test IDs + content.

    When multiple runs map to the same test ID, only the latest (last
    directory in sorted order) is kept.  Runs without
    ``post_run_page_content`` are skipped in favour of those that have it.
    """
    candidates: dict[str, dict[str, Any]] = {}

    run_dirs = [
        p for p in sorted(results_dir.iterdir())
        if p.is_dir() and p.name != "run_manifests"
    ]

    for run_dir in run_dirs:
        test_id = _extract_test_id(run_dir.name)
        if not test_id:
            # Try summary_info.json
            summary_path = run_dir / "summary_info.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                test_id = _extract_test_id(
                    summary.get("task_name", "")
                ) or _extract_test_id(
                    summary.get("task_id", "")
                )
            if not test_id:
                continue

        content = load_predicted_content(run_dir)
        entry = {
            "test_id": test_id,
            "run_dir": str(run_dir),
            "content": content,
        }

        prev = candidates.get(test_id)
        # Prefer runs that have content; among those, keep the latest
        if prev is None or (content and not prev.get("content")) or content:
            candidates[test_id] = entry

    return list(candidates.values())


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_single(
    test_id: str,
    content_str: str | None,
    gt_dir: Path,
    tolerance: float,
) -> EvalResult:
    """Evaluate a single test case."""
    task_number = _extract_task_number(test_id)
    if task_number is None:
        return EvalResult(
            test_id=test_id,
            task_number=0,
            scene_result=None,
            error="could not extract task number from test_id",
        )

    gt_scene = load_gt_scene(gt_dir, task_number)
    if gt_scene is None:
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            scene_result=None,
            error="ground truth file task%d.json not found" % task_number,
        )

    if not content_str:
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            scene_result=None,
            error="no post_run_page_content available",
        )

    try:
        content = json.loads(content_str)
    except json.JSONDecodeError as exc:
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            scene_result=None,
            error="invalid JSON in post_run_page_content: %s" % exc,
        )

    scenes = content.get("scenes", [])
    if not isinstance(scenes, list):
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            scene_result=None,
            error="post_run_page_content has no 'scenes' list",
        )

    gt_name = gt_scene.get("name", "")
    gt_objects = gt_scene.get("scene", {}).get("objects", [])
    pred_scene = find_scene_by_name(scenes, gt_name)

    # Fallback: if the name-matched scene is empty but GT expects objects,
    # or if no scene matched by name at all, search all scenes for the
    # best object-level match.  Handles "Create a new scene" tasks where
    # the agent names the scene differently from the GT.
    if pred_scene is None or (
        gt_objects
        and not pred_scene.get("scene", {}).get("objects", [])
    ):
        fallback = find_best_scene(scenes, gt_scene)
        if fallback is not None:
            pred_scene = fallback

    if pred_scene is None:
        return EvalResult(
            test_id=test_id,
            task_number=task_number,
            scene_result=SceneResult(scene_name=gt_name, found=False),
        )

    scene_result = compare_scene(gt_scene, pred_scene, tolerance)

    return EvalResult(
        test_id=test_id,
        task_number=task_number,
        scene_result=scene_result,
    )


def evaluate_all(
    results_dir: str,
    gt_dir: str,
    tolerance: float = 0.15,
    verbose: bool = False,
) -> None:
    """Evaluate all 3D editor test cases from a results directory.

    Args:
        results_dir: Path to the benchmark results directory.
        gt_dir:      Path to the ground-truth directory (assets/3d_ground_truth).
        tolerance:   Numeric tolerance for position/rotation/scale comparisons.
        verbose:     Print detailed per-object comparison info.
    """
    results_path = Path(results_dir)
    gt_path = Path(gt_dir)

    if not results_path.exists():
        print("ERROR: results directory not found: %s" % results_path, file=sys.stderr)
        sys.exit(1)
    if not gt_path.exists():
        print("ERROR: ground truth directory not found: %s" % gt_path, file=sys.stderr)
        sys.exit(1)

    responses = load_agent_responses(results_path)
    if not responses:
        print("ERROR: no test case responses found in %s" % results_path, file=sys.stderr)
        sys.exit(1)

    print("Evaluating %d test case(s)" % len(responses))
    print("Tolerance: %.3f" % tolerance)
    print("-" * 60)

    total = 0
    passed = 0
    failed_ids: list[str] = []
    objective_evaluation: dict[str, int] = {}

    for response in responses:
        test_id = response["test_id"]
        content = response["content"]

        result = evaluate_single(test_id, content, gt_path, tolerance)
        total += 1

        if result.passed:
            passed += 1
            objective_evaluation[test_id] = 1
            print("\n[PASS] %s  (task%d)" % (test_id, result.task_number))
        else:
            failed_ids.append(test_id)
            objective_evaluation[test_id] = 0
            print("\n[FAIL] %s  (task%d)" % (test_id, result.task_number))

        if result.error:
            print("       ERROR: %s" % result.error)
            continue

        sr = result.scene_result
        if sr is None:
            continue

        if not sr.found:
            print("       scene '%s' not found in predicted output" % sr.scene_name)
            continue

        # Object summary
        obj_pass = sum(1 for o in sr.object_results if o.matched)
        obj_total = len(sr.object_results)
        print("       objects: %d/%d matched" % (obj_pass, obj_total))

        if sr.structural_errors:
            for err in sr.structural_errors:
                print("       [structural] %s" % err)

        if sr.settings_errors:
            for err in sr.settings_errors:
                print("       [settings]   %s" % err)

        if verbose:
            for obj_r in sr.object_results:
                status = "OK" if obj_r.matched else "FAIL"
                print("         [%s] %s" % (status, obj_r.name))
                for err in obj_r.errors:
                    print("               %s" % err)

    # Save results
    output_dir = results_path
    output_dir.mkdir(parents=True, exist_ok=True)
    objective_path = output_dir / "objective_evaluation.json"
    objective_path.write_text(json.dumps(objective_evaluation, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("TOTAL: %d/%d passed" % (passed, total))
    if failed_ids:
        print("FAILED: %s" % ", ".join(failed_ids))
    print("Saved objective evaluation to: %s" % objective_path)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    fire.Fire(evaluate_all)
