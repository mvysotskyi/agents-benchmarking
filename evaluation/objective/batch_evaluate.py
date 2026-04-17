#!/usr/bin/env python3
"""Batch parallel objective evaluation across benchmark applications.

Discovers model result directories under a given root (e.g. clean_results/open_source)
and runs the appropriate evaluator for each benchmark app (circuit, flightradar, voidcut)
in parallel using a process pool.

Usage:
    python -m evaluation.objective.batch_evaluate clean_results/open_source
    python -m evaluation.objective.batch_evaluate clean_results/closed_source --workers 4
    python -m evaluation.objective.batch_evaluate clean_results/open_source clean_results/closed_source
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


TASKS_DIR = Path("test_cases")
GT_VOIDCUT = Path("assets/gt_all.json")
GT_3D = Path("assets/3d_ground_truth")

APP_CIRCUIT = "circuit"
APP_FRAD = "frad"
APP_VIDEO = "video"
APP_3D = "3d"

CIRCUIT_PATTERNS = ("_circuit",)
FRAD_PATTERNS = ("_frad",)
VIDEO_PATTERNS = ("_video", "_voidcut")
THREE_D_PATTERNS = ("_3d", "_clone3d", "_graph")


@dataclass
class EvalJob:
    """A single evaluation job to execute."""

    app: str
    model: str
    results_dir: Path

    @property
    def label(self) -> str:
        return f"{self.model}/{self.app}"


@dataclass
class EvalOutcome:
    """Result of a single evaluation job."""

    job: EvalJob
    success: bool
    objective_scores: dict[str, int | float] | None = None
    error: str | None = None


def classify_app(dirname: str) -> str | None:
    """Classify a results directory name into an app type."""
    lower = dirname.lower()
    for pattern in CIRCUIT_PATTERNS:
        if pattern in lower:
            return APP_CIRCUIT
    for pattern in FRAD_PATTERNS:
        if pattern in lower:
            return APP_FRAD
    for pattern in VIDEO_PATTERNS:
        if pattern in lower:
            return APP_VIDEO
    for pattern in THREE_D_PATTERNS:
        if pattern in lower:
            return APP_3D
    return None


def discover_jobs(root_dirs: list[Path]) -> list[EvalJob]:
    """Walk root directories and discover all evaluation jobs."""
    jobs: list[EvalJob] = []

    for root in root_dirs:
        if not root.is_dir():
            print(f"WARNING: skipping non-directory {root}", file=sys.stderr)
            continue

        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            for results_dir in sorted(model_dir.iterdir()):
                if not results_dir.is_dir():
                    continue
                app = classify_app(results_dir.name)
                if app is None:
                    continue
                jobs.append(EvalJob(app=app, model=model_name, results_dir=results_dir))

    return jobs


def _build_command(job: EvalJob) -> list[str]:
    """Build the subprocess command for an evaluation job."""
    if job.app == APP_CIRCUIT:
        test_cases = TASKS_DIR / "circuit.yaml"
        return [
            sys.executable, "-m", "evaluation.objective.evaluate_circuit_scheme",
            "--test-cases", str(test_cases),
            "--responses", str(job.results_dir),
        ]

    if job.app == APP_FRAD:
        test_cases = TASKS_DIR / "flightradar.yaml"
        return [
            sys.executable, "-m", "evaluation.objective.evaluate",
            "--test-cases", str(test_cases),
            "--responses", str(job.results_dir),
        ]

    if job.app == APP_VIDEO:
        return [
            sys.executable, "-m", "evaluation.objective.eval_voidcut",
            str(job.results_dir),
            str(GT_VOIDCUT),
        ]

    if job.app == APP_3D:
        return [
            sys.executable, "-m", "evaluation.objective.eval_3d_editor",
            str(job.results_dir),
            str(GT_3D),
        ]

    raise ValueError(f"Unknown app type: {job.app}")


def _find_objective_file(job: EvalJob) -> Path | None:
    """Locate the objective_evaluation.json produced by the evaluator."""
    candidate = job.results_dir / "objective_evaluation.json"
    if candidate.exists():
        return candidate
    return None


def run_job(job: EvalJob) -> EvalOutcome:
    """Execute a single evaluation job in a subprocess."""
    cmd = _build_command(job)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return EvalOutcome(job=job, success=False, error="timeout after 300s")
    except Exception as exc:
        return EvalOutcome(job=job, success=False, error=str(exc))

    obj_path = _find_objective_file(job)
    if obj_path is not None:
        try:
            scores = json.loads(obj_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return EvalOutcome(
                job=job, success=False,
                error=f"failed to read {obj_path}: {exc}\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}",
            )
        return EvalOutcome(job=job, success=True, objective_scores=scores)

    stderr_tail = result.stderr[-1000:] if result.stderr else ""
    stdout_tail = result.stdout[-1000:] if result.stdout else ""
    return EvalOutcome(
        job=job, success=False,
        error=f"no objective_evaluation.json produced (exit={result.returncode})\nstdout: {stdout_tail}\nstderr: {stderr_tail}",
    )


def _aggregate_results(outcomes: list[EvalOutcome]) -> dict:
    """Build a summary dict from all outcomes."""
    per_model: dict[str, dict[str, dict]] = {}

    for outcome in outcomes:
        model = outcome.job.model
        app = outcome.job.app
        if model not in per_model:
            per_model[model] = {}

        if outcome.success and outcome.objective_scores is not None:
            scores = outcome.objective_scores
            total = len(scores)
            passed = sum(1 for v in scores.values() if v == 1)
            per_model[model][app] = {
                "passed": passed,
                "total": total,
                "accuracy": round(passed / total, 4) if total else 0.0,
                "scores": scores,
            }
        else:
            per_model[model][app] = {
                "passed": 0,
                "total": 0,
                "accuracy": 0.0,
                "error": outcome.error,
            }

    return per_model


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run batch parallel objective evaluation across benchmark applications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root_dirs", nargs="+", type=Path,
        help="Root directories containing model subdirectories (e.g. clean_results/open_source)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Max parallel workers (default: number of jobs)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save aggregated JSON results (default: print to stdout)",
    )
    args = parser.parse_args()

    jobs = discover_jobs(args.root_dirs)
    if not jobs:
        print("No evaluation jobs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(jobs)} evaluation job(s):")
    for job in jobs:
        print(f"  {job.label:30s} {job.results_dir}")

    max_workers = args.workers or len(jobs)
    outcomes: list[EvalOutcome] = []

    print(f"\nRunning evaluations with {max_workers} workers...\n")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            outcome = future.result()
            outcomes.append(outcome)

            if outcome.success and outcome.objective_scores is not None:
                scores = outcome.objective_scores
                passed = sum(1 for v in scores.values() if v == 1)
                total = len(scores)
                print(f"  [DONE] {outcome.job.label:30s} {passed}/{total}")
            else:
                print(f"  [FAIL] {outcome.job.label:30s} {outcome.error}")

    aggregated = _aggregate_results(outcomes)

    print("\n" + "=" * 70)
    print(f"{'Model':<20s} {'Circuit':>12s} {'FlightRadar':>12s} {'VoidCut':>12s} {'3D Editor':>12s}")
    print("-" * 82)
    for model in sorted(aggregated):
        cells: list[str] = []
        for app in (APP_CIRCUIT, APP_FRAD, APP_VIDEO, APP_3D):
            info = aggregated[model].get(app)
            if info is None:
                cells.append("—")
            elif "error" in info:
                cells.append("ERROR")
            else:
                cells.append(f"{info['passed']}/{info['total']}")
        print(f"{model:<20s} {cells[0]:>12s} {cells[1]:>12s} {cells[2]:>12s} {cells[3]:>12s}")
    print("=" * 82)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
        print(f"\nAggregated results saved to: {output_path}")
    else:
        print("\nAggregated JSON:")
        print(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
