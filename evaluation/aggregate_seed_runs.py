#!/usr/bin/env python3
"""Run objective evaluation on every model under seed_runs/ and emit aggregated CSVs.

Walks ``seed_runs/run{1,2,3}/{app}/{model}/`` directories, runs the matching
objective evaluator for each model dir (writing ``objective_evaluation.json``)
unless one already exists, then aggregates per-(model, app) results across the
three seed runs.

Outputs two CSV files:

  * success_rate.csv   — one row per model, columns Graph/Flight/Video/3D/Avg.
                         each cell is "mean ± std" of the per-run FSR % (image 1).
                         Only FSR is reported here; PSR would need partial-credit
                         scoring which the existing evaluators don't expose
                         (they emit binary 0/1 in objective_evaluation.json).
  * efficiency.csv     — one row per model, columns ACT / AWL / AS / SPF (image 2).
                         ACT = mean total tokens per task,
                         AWL = mean wall-clock seconds per task,
                         AS  = mean number of agent steps per task,
                         SPF = 1 if the model is on the (cost ↓, FSR ↑) Pareto
                               frontier across all evaluated models, else 0.

Usage:
    python -m evaluation.aggregate_seed_runs
    python -m evaluation.aggregate_seed_runs --root seed_runs --out aggregated_results
    python -m evaluation.aggregate_seed_runs --skip-eval   # only re-aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from evaluation.objective.batch_evaluate import EvalJob, run_job


# Known apps under seed_runs/run*/. Maps directory name → (app code used by
# batch_evaluate, display column heading for image-1 success table).
APPS: list[tuple[str, str, str]] = [
    ("graph",       "graph", "Graph"),
    ("flightradar", "frad", "Flight"),
    ("video",       "video", "Video"),
    ("3d",          "3d",   "3D"),
    ("circuit",     "circuit", "Circuit"),
]

# Pretty model names — fall back to the raw dir name if not listed.
MODEL_DISPLAY: dict[str, str] = {
    "opus46":      "Claude Opus 4.6",
    "opus46_cu":   "Claude Opus 4.6 Computer Use",
    "gpt54":       "GPT-5.4",
    "gpt54_cu":    "GPT-5.4 Computer Use",
    "gemini31pro": "Gemini 3.1 Pro",
    "o3pro":       "OpenAI o3-pro",
    "grok4":       "Grok 4",
    "qwen3max":    "Qwen Max",
    "qwen332b":    "Qwen3 32B",
    "llama4mav":   "Llama 4 Maverick",
    "dsv32":       "DeepSeek-V3.2",
    "mistrallarge": "Mistral Large 3",
    "gemma431b":   "Gemma 4 31B",
    "gemma327b":   "Gemma 3 27B",
    "gemini3flash": "Gemini 3 Flash",
    # size_runs/
    "sonnet46":    "Claude Sonnet 4.6",
    "haiku45":     "Claude Haiku 4.5",
    "gpt54mini":   "GPT-5.4 Mini",
    "gpt54nano":   "GPT-5.4 Nano",
    # reasoning_runs/
    "opus46-thinking":     "Claude Opus 4.6 (thinking)",
    "gpt54-noreasoning":   "GPT-5.4 (no reasoning)",
    "gemini31pro-low":     "Gemini 3.1 Pro (low reasoning)",
}

TASK_ID_RE = re.compile(r"tc_(?:graph|3d|frad|vid|clone3d|circuit)_\d+")


# ── Discovery + evaluation ───────────────────────────────────────────────────


def discover_model_dirs(root: Path) -> list[tuple[str, str, str, Path]]:
    """Return [(run_name, app_dirname, model_name, path), ...] for every model dir."""
    found: list[tuple[str, str, str, Path]] = []
    if not root.is_dir():
        raise SystemExit(f"seed runs root not found: {root}")

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("run"):
            continue
        for app_dirname, _app_code, _label in APPS:
            app_dir = run_dir / app_dirname
            if not app_dir.is_dir():
                continue
            for model_dir in sorted(app_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                found.append((run_dir.name, app_dirname, model_dir.name, model_dir))
    return found


def ensure_objective_files(
    triples: list[tuple[str, str, str, Path]],
    workers: int,
    force: bool,
) -> None:
    """Run the objective evaluator for any (run, app, model) without a fresh result file."""
    jobs: list[EvalJob] = []
    for _run, app_dirname, model, path in triples:
        if not force and (path / "objective_evaluation.json").exists():
            continue
        app_code = next(code for d, code, _ in APPS if d == app_dirname)
        jobs.append(EvalJob(app=app_code, model=model, results_dir=path))

    if not jobs:
        print("[eval] all objective_evaluation.json files already present — skipping.")
        return

    print(f"[eval] running objective evaluation for {len(jobs)} model dir(s) "
          f"with {workers} workers...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job): job for job in jobs}
        for fut in as_completed(futures):
            outcome = fut.result()
            job = outcome.job
            if outcome.success and outcome.objective_scores is not None:
                passed = sum(1 for v in outcome.objective_scores.values() if v == 1)
                total = len(outcome.objective_scores)
                print(f"  [DONE] {job.results_dir}  {passed}/{total}")
            else:
                print(f"  [FAIL] {job.results_dir}  {outcome.error}", file=sys.stderr)


# ── Stats extraction ─────────────────────────────────────────────────────────


def _load_objective(model_dir: Path) -> dict[str, int] | None:
    p = model_dir / "objective_evaluation.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _extract_test_id(name: str) -> str | None:
    m = TASK_ID_RE.search(name)
    return m.group(0) if m else None


def _per_test_stats(model_dir: Path) -> dict[str, dict]:
    """Map test_id → {tokens, elapsed_s, n_steps} from each run's summary_info.json."""
    out: dict[str, dict] = {}
    for run_dir in model_dir.iterdir():
        if not run_dir.is_dir():
            continue
        test_id = _extract_test_id(run_dir.name)
        if test_id is None:
            continue
        summary = run_dir / "summary_info.json"
        if not summary.exists():
            continue
        try:
            d = json.loads(summary.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        out[test_id] = {
            "tokens":    d.get("stats.cum_total_tokens"),
            "elapsed_s": d.get("stats.cum_step_elapsed"),
            "n_steps":   d.get("n_steps"),
            "success":   bool(d.get("success")),
        }
    return out


# ── Aggregation ──────────────────────────────────────────────────────────────


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], 0.0)
    return (statistics.mean(values), statistics.pstdev(values))


def _fmt_pct(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "—"
    return f"{mean:.1f} ± {std:.1f}"


def _fmt_num(mean: float, std: float, digits: int = 0) -> str:
    if math.isnan(mean):
        return "—"
    if digits == 0:
        return f"{mean:.0f} ± {std:.0f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def aggregate(
    triples: list[tuple[str, str, str, Path]],
) -> tuple[dict, dict, dict]:
    """Return (success_table, eff_by_model, eff_by_model_app).

    * by_model_app[model][app] = {"runs": [pass_rate_per_run, ...]}
    * eff_by_model[model]      = {"tokens": [..per-test..], "elapsed": [...],
                                  "steps": [...], "successes": [...]}     (pooled overall)
    * eff_by_model_app[model][app] = same shape as eff_by_model, but per-app.
    """
    by_model_app: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"runs": []})
    )
    eff_by_model: dict[str, dict[str, list]] = defaultdict(
        lambda: {"tokens": [], "elapsed": [], "steps": [], "successes": []}
    )
    eff_by_model_app: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: {"tokens": [], "elapsed": [], "steps": [], "successes": []})
    )

    for _run, app_dirname, model, path in triples:
        obj = _load_objective(path)
        stats = _per_test_stats(path)

        # FSR for this run.
        if obj:
            total = len(obj)
            passed = sum(1 for v in obj.values() if v == 1)
            fsr = 100.0 * passed / total if total else 0.0
            by_model_app[model][app_dirname]["runs"].append(fsr)

        # Efficiency stats — pool both overall (eff_by_model) and per-app (eff_by_model_app).
        for _test_id, st in stats.items():
            buckets = [eff_by_model[model], eff_by_model_app[model][app_dirname]]
            for b in buckets:
                if st["tokens"] is not None:
                    b["tokens"].append(float(st["tokens"]))
                if st["elapsed_s"] is not None:
                    b["elapsed"].append(float(st["elapsed_s"]))
                if st["n_steps"] is not None:
                    b["steps"].append(float(st["n_steps"]))
                b["successes"].append(1 if st["success"] else 0)

    return by_model_app, eff_by_model, eff_by_model_app


# ── CSV emit ─────────────────────────────────────────────────────────────────


def write_success_csv(out_path: Path, by_model_app: dict) -> None:
    """Image-1 layout: one cell per (model, app) holds 'mean ± std' of per-run FSR%."""
    headers = ["Model"] + [label for _d, _c, label in APPS] + ["Avg."]
    rows: list[list[str]] = []

    for model in sorted(by_model_app, key=lambda m: MODEL_DISPLAY.get(m, m)):
        per_app = by_model_app[model]
        cells: list[str] = [MODEL_DISPLAY.get(model, model)]
        means_for_avg: list[float] = []
        for app_dirname, _code, _label in APPS:
            runs = per_app.get(app_dirname, {}).get("runs", [])
            if not runs:
                cells.append("—")
                continue
            mean, std = _mean_std(runs)
            cells.append(_fmt_pct(mean, std))
            means_for_avg.append(mean)
        if means_for_avg:
            avg_mean = statistics.mean(means_for_avg)
            cells.append(f"{avg_mean:.1f}")
        else:
            cells.append("—")
        rows.append(cells)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)


def _pareto_frontier(points: dict[str, tuple[float, float]]) -> set[str]:
    """Models on the (cost ↓, success ↑) Pareto frontier.

    A model M is on the frontier iff no other model has BOTH lower-or-equal cost
    AND strictly higher success, OR strictly lower cost and equal success.
    Models with NaN cost/success are excluded.
    """
    valid = {m: (c, s) for m, (c, s) in points.items()
             if not (math.isnan(c) or math.isnan(s))}
    frontier: set[str] = set()
    for m, (cm, sm) in valid.items():
        dominated = False
        for n, (cn, sn) in valid.items():
            if n == m:
                continue
            if cn <= cm and sn >= sm and (cn < cm or sn > sm):
                dominated = True
                break
        if not dominated:
            frontier.add(m)
    return frontier


def write_efficiency_csv(
    out_path: Path,
    eff_by_model: dict,
    by_model_app: dict,
) -> None:
    """Image-2 layout."""
    # Per-model means.
    summaries: dict[str, dict[str, float]] = {}
    for model, vals in eff_by_model.items():
        tokens = vals["tokens"]
        elapsed = vals["elapsed"]
        steps = vals["steps"]
        # Overall FSR (pooled across all tasks/runs) for SPF.
        per_app_runs: list[float] = []
        for app_dirname, _c, _l in APPS:
            per_app_runs.extend(by_model_app.get(model, {}).get(app_dirname, {}).get("runs", []))
        overall_fsr = statistics.mean(per_app_runs) if per_app_runs else math.nan
        summaries[model] = {
            "act_mean":  statistics.mean(tokens) if tokens else math.nan,
            "act_std":   statistics.pstdev(tokens) if len(tokens) > 1 else (0.0 if tokens else math.nan),
            "awl_mean":  statistics.mean(elapsed) if elapsed else math.nan,
            "awl_std":   statistics.pstdev(elapsed) if len(elapsed) > 1 else (0.0 if elapsed else math.nan),
            "as_mean":   statistics.mean(steps) if steps else math.nan,
            "as_std":    statistics.pstdev(steps) if len(steps) > 1 else (0.0 if steps else math.nan),
            "fsr":       overall_fsr,
        }

    pareto_input = {
        m: (s["act_mean"], s["fsr"])
        for m, s in summaries.items()
        if not (math.isnan(s["act_mean"]) or math.isnan(s["fsr"]))
    }
    frontier = _pareto_frontier(pareto_input)

    headers = ["Model", "ACT (↓)", "AWL (↓)", "AS (↓)", "SPF (↑)"]
    rows: list[list[str]] = []
    for model in sorted(summaries, key=lambda m: MODEL_DISPLAY.get(m, m)):
        s = summaries[model]
        spf = "—" if model not in pareto_input else ("1" if model in frontier else "0")
        rows.append([
            MODEL_DISPLAY.get(model, model),
            _fmt_num(s["act_mean"], s["act_std"], digits=0),
            _fmt_num(s["awl_mean"], s["awl_std"], digits=1),
            _fmt_num(s["as_mean"],  s["as_std"],  digits=1),
            spf,
        ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)


def write_efficiency_per_app_csvs(
    out_dir: Path,
    eff_by_model_app: dict,
    by_model_app: dict,
) -> list[Path]:
    """Emit four per-app CSVs (image-1 layout, one per metric).

    Each file has rows = models, columns = Graph / Flight / Video / 3D / Avg.
      * efficiency_act_per_app.csv  — mean total tokens per task
      * efficiency_awl_per_app.csv  — mean wall-clock seconds per task
      * efficiency_as_per_app.csv   — mean number of agent steps per task
      * efficiency_spf_per_app.csv  — 1 if model is on the (cost ↓, FSR ↑) Pareto
                                      frontier *for that app*, else 0
    """
    # Per-(model, app) summaries.
    metric_to_key = {"act": "tokens", "awl": "elapsed", "as": "steps"}
    metric_to_digits = {"act": 0, "awl": 1, "as": 1}
    metric_to_label = {"act": "ACT (↓)", "awl": "AWL (↓)", "as": "AS (↓)", "spf": "SPF (↑)"}

    summaries: dict[str, dict[str, dict[str, float]]] = {}  # model → app → {metric_mean, metric_std, fsr}
    for model, per_app in eff_by_model_app.items():
        summaries[model] = {}
        for app_dirname, vals in per_app.items():
            entry: dict[str, float] = {}
            for metric, key in metric_to_key.items():
                xs = vals[key]
                entry[f"{metric}_mean"] = statistics.mean(xs) if xs else math.nan
                entry[f"{metric}_std"] = (
                    statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else math.nan)
                )
            runs = by_model_app.get(model, {}).get(app_dirname, {}).get("runs", [])
            entry["fsr"] = statistics.mean(runs) if runs else math.nan
            summaries[model][app_dirname] = entry

    # Pareto frontier per app: (mean ACT, FSR) across models that have both.
    frontier_per_app: dict[str, set[str]] = {}
    pareto_input_per_app: dict[str, dict[str, tuple[float, float]]] = {}
    for app_dirname, _c, _l in APPS:
        points = {}
        for model, per_app in summaries.items():
            entry = per_app.get(app_dirname)
            if not entry:
                continue
            if math.isnan(entry["act_mean"]) or math.isnan(entry["fsr"]):
                continue
            points[model] = (entry["act_mean"], entry["fsr"])
        pareto_input_per_app[app_dirname] = points
        frontier_per_app[app_dirname] = _pareto_frontier(points)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write(metric: str, file_name: str) -> None:
        digits = metric_to_digits.get(metric, 1)
        headers = ["Model"] + [label for _d, _c, label in APPS] + ["Avg."]
        rows: list[list[str]] = []

        for model in sorted(summaries, key=lambda m: MODEL_DISPLAY.get(m, m)):
            cells: list[str] = [MODEL_DISPLAY.get(model, model)]
            means_for_avg: list[float] = []
            for app_dirname, _c, _l in APPS:
                entry = summaries[model].get(app_dirname)
                if metric == "spf":
                    if entry is None or model not in pareto_input_per_app.get(app_dirname, {}):
                        cells.append("—")
                    else:
                        cells.append("1" if model in frontier_per_app[app_dirname] else "0")
                    continue
                if entry is None or math.isnan(entry[f"{metric}_mean"]):
                    cells.append("—")
                    continue
                cells.append(_fmt_num(entry[f"{metric}_mean"], entry[f"{metric}_std"], digits=digits))
                means_for_avg.append(entry[f"{metric}_mean"])

            if metric == "spf":
                # Avg. column for SPF: count of apps on which model is on the frontier.
                count = sum(
                    1 for app_dirname, _c, _l in APPS
                    if model in frontier_per_app.get(app_dirname, set())
                )
                cells.append(str(count))
            elif means_for_avg:
                avg_mean = statistics.mean(means_for_avg)
                if digits == 0:
                    cells.append(f"{avg_mean:.0f}")
                else:
                    cells.append(f"{avg_mean:.{digits}f}")
            else:
                cells.append("—")
            rows.append(cells)

        path = out_dir / file_name
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(headers)
            w.writerows(rows)
        written.append(path)

    _write("act", "efficiency_act_per_app.csv")
    _write("awl", "efficiency_awl_per_app.csv")
    _write("as",  "efficiency_as_per_app.csv")
    _write("spf", "efficiency_spf_per_app.csv")
    return written


# ── Main ─────────────────────────────────────────────────────────────────────


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("seed_runs"),
                   help="Root containing run1/run2/run3 subdirectories (default: seed_runs)")
    p.add_argument("--out",  type=Path, default=Path("aggregated_results"),
                   help="Output directory for CSV files (default: aggregated_results)")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel workers when running objective evaluators (default: 8)")
    p.add_argument("--skip-eval", action="store_true",
                   help="Don't run evaluators; only re-aggregate existing objective_evaluation.json files")
    p.add_argument("--force-eval", action="store_true",
                   help="Re-run evaluators even if objective_evaluation.json already exists")
    args = p.parse_args(list(argv) if argv is not None else None)

    triples = discover_model_dirs(args.root)
    if not triples:
        print(f"No model directories found under {args.root}", file=sys.stderr)
        return 1

    print(f"Found {len(triples)} (run, app, model) directories under {args.root}.")

    if not args.skip_eval:
        ensure_objective_files(triples, workers=args.workers, force=args.force_eval)

    by_model_app, eff_by_model, eff_by_model_app = aggregate(triples)

    fsr_path = args.out / "success_rate.csv"
    eff_path = args.out / "efficiency.csv"

    write_success_csv(fsr_path, by_model_app)
    write_efficiency_csv(eff_path, eff_by_model, by_model_app)
    per_app_paths = write_efficiency_per_app_csvs(args.out, eff_by_model_app, by_model_app)

    written = "\n  ".join(str(p) for p in [fsr_path, eff_path, *per_app_paths])
    print(f"\nWrote:\n  {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
