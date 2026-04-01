import argparse
import csv
import json
import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.llm_judge.llm_as_judge import DEFAULT_OUTPUT_FILENAME, JudgeConfig, run_pipeline
from evaluation.llm_judge.screenshot_diff import ScreenshotDiffThresholds


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR_RE = re.compile(r"^results_([^_]+)_(.+)$")
DEFAULT_SUMMARY_CSV_NAME = "llm_judge_batch_summary.csv"
DEFAULT_RUNS_CSV_NAME = "llm_judge_batch_runs.csv"
DEFAULT_TASK_YAMLS = {
    "circuit": REPO_ROOT / "tasks" / "circuit.yaml",
    "frad": REPO_ROOT / "tasks" / "flightradar.yaml",
    "voidcut": REPO_ROOT / "tasks" / "voidcut.yaml",
}


@dataclass(frozen=True)
class BatchJob:
    index: int
    total: int
    results_dir: Path
    task_yaml: Path
    output_path: Path


@dataclass(frozen=True)
class JobResult:
    job: BatchJob
    status: str
    aggregate: dict[str, Any] | None
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run llm-as-a-judge over every results_{model}_{app} directory under an input root, "
            "optionally in parallel, and write a summary CSV."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Root directory containing one or more results folders.")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum number of results directories to process concurrently.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=f"Summary CSV path. Defaults to <input_dir>/{DEFAULT_SUMMARY_CSV_NAME}.",
    )
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=None,
        help="Optional detailed per-run CSV path.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Filename to write inside each results directory. Defaults to {DEFAULT_OUTPUT_FILENAME}.",
    )
    parser.add_argument(
        "--task-yaml",
        action="append",
        default=[],
        metavar="APP=PATH",
        help=(
            "Override task YAML mapping for an app. Known defaults are "
            "circuit=tasks/circuit.yaml, frad=tasks/flightradar.yaml, voidcut=tasks/voidcut.yaml."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip directories where the output JSON already exists.",
    )
    parser.add_argument(
        "--limit-results-dirs",
        type=int,
        default=None,
        help="Optional limit on how many results directories to process after discovery.",
    )
    parser.add_argument(
        "--match",
        default=None,
        help="Optional substring filter applied to discovered results directory paths.",
    )
    parser.add_argument("--stage2-model", default="gpt-4o-mini", help="Responses API model to use for Stage 2.")
    parser.add_argument("--stage3-model", default="gpt-5.1", help="Responses API model to use for Stage 3.")
    parser.add_argument(
        "--stage3-reasoning-effort",
        default="medium",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort for the Stage 3 model call.",
    )
    parser.add_argument(
        "--image-detail",
        default=None,
        choices=["low", "auto", "high"],
        help="Legacy flag that sets both Stage 2 and Stage 3 image detail when stage-specific flags are omitted.",
    )
    parser.add_argument(
        "--stage2-image-detail",
        default=None,
        choices=["low", "auto", "high"],
        help="Image detail for Stage 2 screenshot-pair judgments. Defaults to low.",
    )
    parser.add_argument(
        "--stage3-image-detail",
        default=None,
        choices=["low", "auto", "high"],
        help="Image detail for Stage 3 final-outcome judgments. Defaults to high.",
    )
    parser.add_argument("--rmse-threshold", type=float, default=0.03)
    parser.add_argument("--phash-threshold", type=float, default=5.0)
    parser.add_argument("--changed-fraction-threshold", type=float, default=0.01)
    parser.add_argument("--compare-width", type=int, default=320)
    parser.add_argument("--compare-height", type=int, default=200)
    parser.add_argument("--max-judgments", type=int, default=None, help="Optional cap for Stage 2 model calls.")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--objective-evaluation-json",
        type=Path,
        default=None,
        help="Optional path to objective_evaluation.json used for every results directory.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip the OpenAI API and emit placeholder judgments.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--progress", action="store_true", help="Enable tqdm progress bars inside each worker.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="Responses API base URL.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")

    if args.parallel < 1:
        parser.error("--parallel must be at least 1.")
    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if not args.input_dir.is_dir():
        parser.error(f"Expected a directory: {args.input_dir}")

    task_yaml_overrides = parse_task_yaml_overrides(args.task_yaml, parser)
    discovered = discover_results_dirs(args.input_dir)
    if args.match:
        discovered = [path for path in discovered if args.match in str(path)]
    if args.limit_results_dirs is not None:
        discovered = discovered[: args.limit_results_dirs]
    if not discovered:
        parser.error(f"No results_*_* directories found under {args.input_dir}")

    jobs = build_jobs(
        results_dirs=discovered,
        input_dir=args.input_dir,
        output_name=args.output_name,
        task_yaml_overrides=task_yaml_overrides,
        parser=parser,
    )
    summary_csv = args.summary_csv or (args.input_dir / DEFAULT_SUMMARY_CSV_NAME)
    runs_csv = args.runs_csv

    stage2_image_detail = args.stage2_image_detail or args.image_detail or "low"
    stage3_image_detail = args.stage3_image_detail or args.image_detail or "high"
    config = JudgeConfig(
        stage2_model=args.stage2_model,
        stage3_model=args.stage3_model,
        stage3_reasoning_effort=args.stage3_reasoning_effort,
        stage2_image_detail=stage2_image_detail,
        stage3_image_detail=stage3_image_detail,
        compare_size=(args.compare_width, args.compare_height),
        stage1_thresholds=ScreenshotDiffThresholds(
            rmse=args.rmse_threshold,
            phash=args.phash_threshold,
            changed_fraction=args.changed_fraction_threshold,
        ),
        max_judgments=args.max_judgments,
        dry_run=args.dry_run,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        enable_progress=args.progress,
        objective_evaluation_path=args.objective_evaluation_json,
    )

    LOGGER.info("Discovered %d results directories under %s", len(jobs), args.input_dir)
    results = execute_jobs(jobs=jobs, config=config, parallel=args.parallel, skip_existing=args.skip_existing)
    write_summary_csv(results=results, input_dir=args.input_dir, output_path=summary_csv)
    if runs_csv is not None:
        write_runs_csv(results=results, input_dir=args.input_dir, output_path=runs_csv)

    failed = [result for result in results if result.status == "failed"]
    LOGGER.info("Wrote summary CSV to %s", summary_csv)
    if runs_csv is not None:
        LOGGER.info("Wrote detailed runs CSV to %s", runs_csv)
    if failed:
        LOGGER.error("%d job(s) failed.", len(failed))
        return 1
    return 0


def parse_task_yaml_overrides(raw_overrides: list[str], parser: argparse.ArgumentParser) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for raw_override in raw_overrides:
        app_name, separator, raw_path = raw_override.partition("=")
        if separator != "=" or not app_name.strip() or not raw_path.strip():
            parser.error(f"Invalid --task-yaml value {raw_override!r}. Expected APP=PATH.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if not path.exists():
            parser.error(f"Task YAML override does not exist: {path}")
        overrides[normalize_app_key(app_name)] = path
    return overrides


def discover_results_dirs(input_dir: Path) -> list[Path]:
    if is_results_dir(input_dir):
        return [input_dir.resolve()]
    return sorted(path.resolve() for path in input_dir.rglob("results_*_*") if path.is_dir() and is_results_dir(path))


def is_results_dir(path: Path) -> bool:
    return RESULTS_DIR_RE.fullmatch(path.name) is not None


def build_jobs(
    results_dirs: list[Path],
    input_dir: Path,
    output_name: str,
    task_yaml_overrides: dict[str, Path],
    parser: argparse.ArgumentParser,
) -> list[BatchJob]:
    jobs = []
    total = len(results_dirs)
    for index, results_dir in enumerate(results_dirs, start=1):
        task_yaml = resolve_task_yaml(results_dir=results_dir, task_yaml_overrides=task_yaml_overrides)
        if task_yaml is None:
            parser.error(
                "Could not infer task YAML for "
                f"{results_dir}. Use --task-yaml APP=PATH to provide an override."
            )
        jobs.append(
            BatchJob(
                index=index,
                total=total,
                results_dir=results_dir,
                task_yaml=task_yaml,
                output_path=results_dir / output_name,
            )
        )
    return jobs


def resolve_task_yaml(results_dir: Path, task_yaml_overrides: dict[str, Path]) -> Path | None:
    _, app = parse_results_dir_name(results_dir.name)
    normalized_app = normalize_app_key(app)
    if normalized_app in task_yaml_overrides:
        return task_yaml_overrides[normalized_app]
    return DEFAULT_TASK_YAMLS.get(normalized_app)


def parse_results_dir_name(name: str) -> tuple[str, str]:
    match = RESULTS_DIR_RE.fullmatch(name)
    if match is None:
        return "", ""
    return match.groups()


def normalize_app_key(app_name: str) -> str:
    normalized = app_name.strip().lower().replace("-", "").replace("_", "")
    if normalized in {"circuit"}:
        return "circuit"
    if normalized in {"frad", "flightradar", "flightradar24", "flightradarapp"}:
        return "frad"
    if normalized in {"video", "vid", "voidcut", "videoeditor"}:
        return "voidcut"
    return normalized


def execute_jobs(jobs: list[BatchJob], config: JudgeConfig, parallel: int, skip_existing: bool) -> list[JobResult]:
    results: list[JobResult] = []
    max_workers = min(len(jobs), parallel)
    with ThreadPoolExecutor(max_workers=max_workers if max_workers else 1) as executor:
        futures = {
            executor.submit(run_one_job, job=job, config=config, skip_existing=skip_existing): job
            for job in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            label = f"[{result.job.index}/{result.job.total}] {result.job.results_dir.name}"
            if result.status == "failed":
                LOGGER.error("%s failed: %s", label, result.error.splitlines()[-1] if result.error else "unknown error")
            else:
                LOGGER.info("%s %s", label, result.status)
    return sorted(results, key=lambda item: item.job.index)


def run_one_job(job: BatchJob, config: JudgeConfig, skip_existing: bool) -> JobResult:
    LOGGER.info(
        "[%d/%d] Starting %s with %s",
        job.index,
        job.total,
        job.results_dir,
        job.task_yaml,
    )
    if skip_existing and job.output_path.exists():
        aggregate = load_json_if_exists(job.output_path)
        return JobResult(job=job, status="skipped_existing", aggregate=aggregate)

    try:
        aggregate = run_pipeline(
            results_dir=job.results_dir,
            testcases_yaml_path=job.task_yaml,
            output_path=job.output_path,
            config=config,
        )
        return JobResult(job=job, status="completed", aggregate=aggregate)
    except Exception:
        return JobResult(job=job, status="failed", aggregate=load_json_if_exists(job.output_path), error=traceback.format_exc())


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_summary_csv(results: list[JobResult], input_dir: Path, output_path: Path) -> None:
    rows = [build_summary_row(result=result, input_dir=input_dir) for result in results]
    fieldnames = [
        "source_group",
        "collection",
        "model",
        "app",
        "results_dir",
        "task_yaml",
        "output_json",
        "status",
        "error",
        "run_count",
        "runs_judged_stage3",
        "stage3_mean_score",
        "stage3_normalized_score",
        "avg_steps",
        "avg_awl",
        "pairs_total",
        "pairs_flagged_stage1",
        "pairs_judged_stage2",
        "stage2_error_count",
        "stage3_error_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runs_csv(results: list[JobResult], input_dir: Path, output_path: Path) -> None:
    rows = []
    for result in results:
        rows.extend(build_run_rows(result=result, input_dir=input_dir))
    fieldnames = [
        "source_group",
        "collection",
        "model",
        "app",
        "results_dir",
        "run_dir",
        "task_id",
        "task_found_in_yaml",
        "stage3_score",
        "stage3_reason",
        "stage3_error",
        "objective_evaluation_result",
        "n_steps",
        "success",
        "completed",
        "terminated",
        "truncated",
        "pairs_total",
        "pairs_flagged_stage1",
        "pairs_judged_stage2",
        "output_json",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_row(result: JobResult, input_dir: Path) -> dict[str, str]:
    source_group, collection, model, app = extract_path_metadata(result.job.results_dir, input_dir)
    summary = (result.aggregate or {}).get("summary", {})
    runs = (result.aggregate or {}).get("runs", [])
    scores = [
        float(stage3["score"])
        for run in runs
        for stage3 in [run.get("stage3") or {}]
        if isinstance(stage3.get("score"), (int, float))
    ]
    step_counts = [
        float(summary_info["n_steps"])
        for run in runs
        for summary_info in [run.get("summary_info") or {}]
        if isinstance(summary_info.get("n_steps"), (int, float))
    ]
    awl_values = [
        awl_value
        for run in runs
        for awl_value in [extract_awl(run.get("summary_info") or {})]
        if awl_value is not None
    ]
    mean_score = mean(scores)
    return {
        "source_group": source_group,
        "collection": collection,
        "model": model,
        "app": app,
        "results_dir": display_path(result.job.results_dir, input_dir),
        "task_yaml": display_path(result.job.task_yaml, REPO_ROOT),
        "output_json": display_path(result.job.output_path, input_dir),
        "status": result.status,
        "error": summarize_error(result.error),
        "run_count": str(summary.get("run_count", len(runs))) if result.aggregate else "",
        "runs_judged_stage3": str(summary.get("runs_judged_stage3", "")) if result.aggregate else "",
        "stage3_mean_score": format_float(mean_score),
        "stage3_normalized_score": format_float((mean_score - 1.0) / 4.0) if mean_score is not None else "",
        "avg_steps": format_float(mean(step_counts)),
        "avg_awl": format_float(mean(awl_values)),
        "pairs_total": str(summary.get("pairs_total", "")) if result.aggregate else "",
        "pairs_flagged_stage1": str(summary.get("pairs_flagged_stage1", "")) if result.aggregate else "",
        "pairs_judged_stage2": str(summary.get("pairs_judged_stage2", "")) if result.aggregate else "",
        "stage2_error_count": str(summary.get("stage2_error_count", "")) if result.aggregate else "",
        "stage3_error_count": str(summary.get("stage3_error_count", "")) if result.aggregate else "",
    }


def build_run_rows(result: JobResult, input_dir: Path) -> list[dict[str, str]]:
    source_group, collection, model, app = extract_path_metadata(result.job.results_dir, input_dir)
    rows = []
    for run in (result.aggregate or {}).get("runs", []):
        stage3 = run.get("stage3") or {}
        summary_info = run.get("summary_info") or {}
        rows.append(
            {
                "source_group": source_group,
                "collection": collection,
                "model": model,
                "app": app,
                "results_dir": display_path(result.job.results_dir, input_dir),
                "run_dir": display_path(Path(run.get("run_dir", "")), input_dir) if run.get("run_dir") else "",
                "task_id": str(run.get("task_id", "")),
                "task_found_in_yaml": str(bool(run.get("task_found_in_yaml", False))),
                "stage3_score": str(stage3.get("score", "")),
                "stage3_reason": str(stage3.get("reason", "")),
                "stage3_error": str(run.get("stage3_error", "")),
                "objective_evaluation_result": str((run.get("stage3_inputs") or {}).get("objective_evaluation_result", "")),
                "n_steps": str(summary_info.get("n_steps", "")),
                "success": str(summary_info.get("success", "")),
                "completed": str(summary_info.get("completed", "")),
                "terminated": str(summary_info.get("terminated", "")),
                "truncated": str(summary_info.get("truncated", "")),
                "pairs_total": str(run.get("pairs_total", "")),
                "pairs_flagged_stage1": str(run.get("pairs_flagged_stage1", "")),
                "pairs_judged_stage2": str(run.get("pairs_judged_stage2", "")),
                "output_json": display_path(result.job.output_path, input_dir),
            }
        )
    return rows


def extract_path_metadata(results_dir: Path, input_dir: Path) -> tuple[str, str, str, str]:
    relative = safe_relative_path(results_dir, input_dir)
    parts = relative.parts
    source_group = parts[-3] if len(parts) >= 3 else results_dir.parent.parent.name
    collection = parts[-2] if len(parts) >= 2 else results_dir.parent.name
    model, app = parse_results_dir_name(results_dir.name)
    return source_group, collection, model, app


def safe_relative_path(path: Path, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return path.resolve()


def display_path(path: Path, base: Path) -> str:
    return str(safe_relative_path(path, base))


def extract_awl(summary_info: dict[str, Any]) -> float | None:
    for key in ("stats.cum_step_elapsed", "stats.cum_agent_elapsed"):
        value = summary_info.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def summarize_error(raw_error: str) -> str:
    if not raw_error:
        return ""
    lines = [line.strip() for line in raw_error.splitlines() if line.strip()]
    return lines[-1] if lines else raw_error.strip()


if __name__ == "__main__":
    raise SystemExit(main())
