#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any


STEP_FILE_RE = re.compile(r"step_(\d+)\.pkl\.gz$")


class DummyBase:
    """Fallback type used when the original pickle class is unavailable."""


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type[Any]:
        try:
            return super().find_class(module, name)
        except Exception:
            return type(name, (DummyBase,), {"__module__": module})


def load_pickle(path: Path) -> Any:
    with gzip.open(path, "rb") as file_obj:
        return SafeUnpickler(file_obj).load()


def as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {"value": obj}


def normalize_action(action: Any) -> Any:
    if isinstance(action, str):
        stripped = action.strip()
        return stripped or None
    return action


def truncate(text: Any, limit: int | None = 140) -> str:
    if text is None:
        return "None"
    value = str(text).replace("\n", "\\n")
    if limit is None:
        return value
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def sorted_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in results_dir.iterdir()
            if path.is_dir() and path.name != "run_manifests"
        ]
    )


def sorted_step_files(run_dir: Path) -> list[Path]:
    step_files = []
    for path in run_dir.glob("step_*.pkl.gz"):
        match = STEP_FILE_RE.fullmatch(path.name)
        if match:
            step_files.append((int(match.group(1)), path))
    return [path for _, path in sorted(step_files)]


def read_summary_info(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary_info.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


def collect_report(
    results_dir: Path,
    sample_count: int,
    include_chat: bool,
) -> dict[str, Any]:
    run_dirs = sorted_run_dirs(results_dir)

    reward_counter: Counter[str] = Counter()
    raw_reward_counter: Counter[str] = Counter()
    obs_keys: Counter[str] = Counter()
    stats_keys: Counter[str] = Counter()
    task_info_types: Counter[str] = Counter()
    action_heads: Counter[str] = Counter()
    last_action_errors: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    sample_steps: list[dict[str, Any]] = []

    total_step_files = 0
    step_terminated_true = 0
    step_truncated_true = 0
    task_info_nonempty = 0
    model_response_mismatch = 0

    for run_dir in run_dirs:
        summary_info = read_summary_info(run_dir)
        step_files = sorted_step_files(run_dir)
        total_step_files += len(step_files)

        first_step_summary = None
        last_step_summary = None

        for step_file in step_files:
            step_data = as_dict(load_pickle(step_file))
            obs = as_dict(step_data.get("obs") or {})
            stats = as_dict(step_data.get("stats") or {})
            task_info = step_data.get("task_info")
            action = normalize_action(step_data.get("action"))
            model_response = (step_data.get("agent_info") or {}).get("model_response")

            reward_counter[repr(step_data.get("reward"))] += 1
            raw_reward_counter[repr(step_data.get("raw_reward"))] += 1
            step_terminated_true += bool(step_data.get("terminated"))
            step_truncated_true += bool(step_data.get("truncated"))

            for key in obs:
                obs_keys[key] += 1
            for key in stats:
                stats_keys[key] += 1

            task_info_types[type(task_info).__name__] += 1
            if isinstance(task_info, dict) and task_info:
                task_info_nonempty += 1

            if model_response is not None and model_response != step_data.get("action"):
                model_response_mismatch += 1

            if isinstance(action, str):
                action_head = action.split("(", 1)[0]
                action_heads[action_head] += 1

            last_action_error = obs.get("last_action_error")
            if last_action_error not in (None, "") and len(last_action_errors) < sample_count:
                last_action_errors.append(
                    {
                        "run": run_dir.name,
                        "step": step_data.get("step"),
                        "error": str(last_action_error),
                    }
                )

            step_summary = {
                "file": step_file.name,
                "step": step_data.get("step"),
                "action": action,
                "reward": step_data.get("reward"),
                "raw_reward": step_data.get("raw_reward"),
                "terminated": step_data.get("terminated"),
                "truncated": step_data.get("truncated"),
                "last_action": obs.get("last_action"),
                "last_action_error": obs.get("last_action_error"),
                "obs_keys": sorted(obs.keys()),
                "stats": stats,
            }
            if include_chat:
                step_summary["chat_messages"] = obs.get("chat_messages")

            if first_step_summary is None:
                first_step_summary = step_summary
            last_step_summary = step_summary

            if len(sample_steps) < sample_count:
                sample_steps.append(
                    {
                        "run": run_dir.name,
                        **step_summary,
                    }
                )

        per_run.append(
            {
                "run": run_dir.name,
                "task_name": summary_info.get("task_name"),
                "n_steps": summary_info.get("n_steps"),
                "success": summary_info.get("success"),
                "completed": summary_info.get("completed"),
                "terminated": summary_info.get("terminated"),
                "truncated": summary_info.get("truncated"),
                "score": summary_info.get("score"),
                "first_step": first_step_summary,
                "last_step": last_step_summary,
            }
        )

    summary = {
        "results_dir": str(results_dir),
        "run_count": len(run_dirs),
        "total_step_files": total_step_files,
        "step_terminated_true": step_terminated_true,
        "step_truncated_true": step_truncated_true,
        "reward_counter": reward_counter.most_common(),
        "raw_reward_counter": raw_reward_counter.most_common(),
        "obs_keys": obs_keys.most_common(),
        "stats_keys": stats_keys.most_common(),
        "task_info_types": task_info_types.most_common(),
        "task_info_nonempty": task_info_nonempty,
        "model_response_mismatch": model_response_mismatch,
        "action_heads": action_heads.most_common(),
        "sample_last_action_errors": last_action_errors,
        "sample_steps": sample_steps,
        "runs": per_run,
    }
    return summary


def print_chat_messages(chat_messages: Any, indent: str, full_chat: bool) -> None:
    if not isinstance(chat_messages, list):
        print(f"{indent}chat_messages=None")
        return

    print(f"{indent}chat_messages:")
    for index, message in enumerate(chat_messages, start=1):
        if isinstance(message, dict):
            role = message.get("role")
            text = truncate(message.get("message"), None if full_chat else 400)
            print(f"{indent}  {index}. role={role} message={text}")
        else:
            print(
                f"{indent}  {index}. "
                f"{truncate(message, None if full_chat else 400)}"
            )


def print_human_report(
    report: dict[str, Any],
    show_runs: bool,
    show_samples: bool,
    show_chat: bool,
    full_chat: bool,
) -> None:
    runs = report["runs"]
    terminated_runs = sum(1 for run in runs if run.get("terminated") is True)
    truncated_runs = sum(1 for run in runs if run.get("truncated") is True)
    successful_runs = sum(1 for run in runs if run.get("success") is True)

    print(f"Results directory: {report['results_dir']}")
    print(f"Runs: {report['run_count']}")
    print(f"Step files: {report['total_step_files']}")
    print(f"Successful runs: {successful_runs}")
    print(f"Terminated runs: {terminated_runs}")
    print(f"Truncated runs: {truncated_runs}")
    print(f"Step-level terminated=true: {report['step_terminated_true']}")
    print(f"Step-level truncated=true: {report['step_truncated_true']}")
    print()

    print("Top action heads:")
    for action, count in report["action_heads"][:10]:
        print(f"  {action}: {count}")
    print()

    print("Observed obs keys:")
    for key, count in report["obs_keys"]:
        print(f"  {key}: {count}")
    print()

    if report["sample_last_action_errors"]:
        print("Sample last_action_error values:")
        for item in report["sample_last_action_errors"]:
            print(
                f"  {item['run']} step={item['step']} error={truncate(item['error'], 180)}"
            )
        print()

    if show_runs:
        print("Per-run summary:")
        for run in runs:
            first_step = run.get("first_step") or {}
            last_step = run.get("last_step") or {}
            print(
                f"  {run['run']}: task={run.get('task_name')} n_steps={run.get('n_steps')} "
                f"success={run.get('success')} terminated={run.get('terminated')} "
                f"truncated={run.get('truncated')}"
            )
            print(
                f"    first_step={first_step.get('step')} first_action={truncate(first_step.get('action'))}"
            )
            print(
                f"    last_step={last_step.get('step')} last_action={truncate(last_step.get('action'))} "
                f"last_obs.last_action={truncate(last_step.get('last_action'))}"
            )
        print()

    if show_samples:
        print("Sample step states:")
        for sample in report["sample_steps"]:
            print(
                f"  {sample['run']} {sample['file']} step={sample['step']} "
                f"action={truncate(sample['action'])} reward={sample['reward']} "
                f"terminated={sample['terminated']} truncated={sample['truncated']}"
            )
            print(f"    last_action={truncate(sample['last_action'])}")
            print(f"    last_action_error={truncate(sample['last_action_error'])}")
            print(f"    obs_keys={', '.join(sample['obs_keys'])}")
            if show_chat:
                print_chat_messages(sample.get("chat_messages"), "    ", full_chat)
            if sample["stats"]:
                stats_text = ", ".join(
                    f"{key}={sample['stats'][key]}" for key in sorted(sample["stats"])
                )
                print(f"    stats={stats_text}")


def inspect_single_run(
    results_dir: Path,
    run_name: str,
    step_limit: int | None,
    include_chat: bool,
) -> dict[str, Any]:
    run_dir = results_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    steps: list[dict[str, Any]] = []
    for step_file in sorted_step_files(run_dir):
        step_data = as_dict(load_pickle(step_file))
        obs = as_dict(step_data.get("obs") or {})
        step_payload = {
            "file": step_file.name,
            "step": step_data.get("step"),
            "action": normalize_action(step_data.get("action")),
            "reward": step_data.get("reward"),
            "raw_reward": step_data.get("raw_reward"),
            "terminated": step_data.get("terminated"),
            "truncated": step_data.get("truncated"),
            "last_action": obs.get("last_action"),
            "last_action_error": obs.get("last_action_error"),
            "obs_keys": sorted(obs.keys()),
            "stats": as_dict(step_data.get("stats") or {}),
        }
        if include_chat:
            step_payload["chat_messages"] = obs.get("chat_messages")
        steps.append(step_payload)

    if step_limit is not None:
        steps = steps[:step_limit]

    return {
        "results_dir": str(results_dir),
        "run": run_name,
        "summary_info": read_summary_info(run_dir),
        "steps": steps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read and summarize pickled BrowserGym step state files."
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results_gpt54_circuit",
        help="Directory containing run folders with step_*.pkl.gz files.",
    )
    parser.add_argument(
        "--run",
        help="Inspect a single run directory by name instead of printing the aggregate report.",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        help="Limit the number of printed steps when using --run.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="How many sample steps or sample errors to include in the aggregate report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of the human-readable report.",
    )
    parser.add_argument(
        "--no-runs",
        action="store_true",
        help="Skip the per-run summary in human-readable mode.",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Skip sample step output in human-readable mode.",
    )
    parser.add_argument(
        "--show-chat",
        action="store_true",
        help="Print chat_messages for inspected or sampled steps.",
    )
    parser.add_argument(
        "--full-chat",
        action="store_true",
        help="Do not truncate chat_messages in human-readable mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        parser.error(f"Results directory does not exist: {results_dir}")

    if args.run:
        payload = inspect_single_run(
            results_dir,
            args.run,
            args.step_limit,
            include_chat=args.show_chat,
        )
        if args.json:
            print(json.dumps(payload, indent=2))
            return

        print(f"Run: {payload['run']}")
        summary_info = payload["summary_info"]
        if summary_info:
            print(
                "Summary: "
                f"task={summary_info.get('task_name')} "
                f"n_steps={summary_info.get('n_steps')} "
                f"success={summary_info.get('success')} "
                f"terminated={summary_info.get('terminated')} "
                f"truncated={summary_info.get('truncated')}"
            )
        print()
        for step in payload["steps"]:
            print(
                f"{step['file']} step={step['step']} action={truncate(step['action'])} "
                f"reward={step['reward']} terminated={step['terminated']} "
                f"truncated={step['truncated']}"
            )
            print(f"  last_action={truncate(step['last_action'])}")
            print(f"  last_action_error={truncate(step['last_action_error'])}")
            print(f"  obs_keys={', '.join(step['obs_keys'])}")
            if args.show_chat:
                print_chat_messages(step.get("chat_messages"), "  ", args.full_chat)
            if step["stats"]:
                stats_text = ", ".join(
                    f"{key}={step['stats'][key]}" for key in sorted(step["stats"])
                )
                print(f"  stats={stats_text}")
        return

    report = collect_report(
        results_dir,
        args.sample_count,
        include_chat=args.show_chat,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print_human_report(
        report,
        show_runs=not args.no_runs,
        show_samples=not args.no_samples,
        show_chat=args.show_chat,
        full_chat=args.full_chat,
    )


if __name__ == "__main__":
    main()
