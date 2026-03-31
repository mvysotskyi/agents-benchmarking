import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ENTRYPOINT = ROOT_DIR / "main.py"
MODEL_BLOCK_KEYS = {"defaults", "runs", "experiments", "testcases"}
EXPERIMENT_KEYS = {
    "name",
    "enabled",
    "task",
    "task_file",
    "application",
    "run_all",
    "run_random",
    "iterations",
    "max_tasks",
    "max_steps",
    "url",
    "results_dir",
    "headless",
    "use_screenshot",
    "verbose",
    "concurrent",
    "workers",
    "js_snippet",
    "post_run_js_snippet",
    "js_snippet_file",
    "post_run_js_snippet_path",
    "post_run_url",
    "system_prompt",
    "system_prompt_file",
    "task_range",
    "initial_delay",
    "extra_args",
}
PATH_FIELDS = {"task_file", "results_dir", "post_run_js_snippet_path", "system_prompt_file"}


@dataclass(slots=True)
class ExperimentSpec:
    model: str
    results_dir: str
    name: str = ""
    enabled: bool = True
    task: str = ""
    task_file: str = ""
    application: str = ""
    run_all: bool = False
    run_random: bool = False
    iterations: int = 1
    max_tasks: int | None = None
    max_steps: int = 50
    url: str = ""
    headless: bool = False
    use_screenshot: bool | None = None
    verbose: bool = False
    concurrent: bool = False
    workers: int = 0
    js_snippet: str = ""
    post_run_js_snippet_path: str = ""
    post_run_url: str = ""
    system_prompt: str = ""
    system_prompt_file: str = ""
    task_range: str = ""
    initial_delay: float = 0
    extra_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BatchConfig:
    experiments: list[ExperimentSpec]
    entrypoint: Path
    max_parallel: int
    fail_fast: bool
    poll_interval: float
    termination_grace_period: float


@dataclass(slots=True)
class RunningExperiment:
    spec: ExperimentSpec
    command: list[str]
    process: subprocess.Popen[str]
    log_handle: TextIO
    log_path: Path
    started_at: float


@dataclass(slots=True)
class FinishedExperiment:
    spec: ExperimentSpec
    return_code: int
    log_path: Path
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run many main.py experiments from a JSON config")
    parser.add_argument("--config", required=True, help="Path to a batch experiment JSON config")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Override the config parallelism limit (0 means run all experiments at once)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop all remaining experiments after the first non-zero exit code",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without starting them",
    )
    parser.add_argument(
        "--use-screenshot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override screenshot usage for all experiments in this batch",
    )
    return parser.parse_args()


def _ensure_mapping(value: object, description: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return value


def _ensure_list(value: object, description: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a JSON array.")
    return value


def _ensure_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"'{key}' must be true or false.")
    return value


def _ensure_int(value: object, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"'{key}' must be an integer.")
    return value


def _ensure_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a number.")
    return float(value)


def _ensure_string(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string.")
    return value


def _ensure_string_list(value: object, key: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"'{key}' must be an array of strings.")
    return list(value)


def _validate_experiment_keys(block: dict, description: str) -> None:
    unknown_keys = sorted(set(block) - EXPERIMENT_KEYS)
    if unknown_keys:
        raise ValueError(f"{description} has unsupported keys: {', '.join(unknown_keys)}")


def _resolve_config_path(value: str, base_dir: Path) -> str:
    return str((base_dir / value).expanduser().resolve()) if value else ""


def _coerce_experiment(raw: dict, base_dir: Path, model_name: str, index: int) -> ExperimentSpec:
    _validate_experiment_keys(raw, f"Experiment #{index + 1} for model '{model_name}'")

    for field_name in ("results_dir",):
        if not raw.get(field_name):
            raise ValueError(f"Experiment #{index + 1} for model '{model_name}' is missing '{field_name}'.")

    if raw.get("js_snippet") and raw.get("post_run_js_snippet"):
        raise ValueError(
            f"Experiment #{index + 1} for model '{model_name}' cannot set both 'js_snippet' and 'post_run_js_snippet'."
        )
    if raw.get("js_snippet") and raw.get("js_snippet_file"):
        raise ValueError(
            f"Experiment #{index + 1} for model '{model_name}' cannot set both 'js_snippet' and 'js_snippet_file'."
        )
    if raw.get("js_snippet") and raw.get("post_run_js_snippet_path"):
        raise ValueError(
            f"Experiment #{index + 1} for model '{model_name}' cannot set both 'js_snippet' and 'post_run_js_snippet_path'."
        )
    if raw.get("post_run_js_snippet") and raw.get("js_snippet_file"):
        raise ValueError(
            f"Experiment #{index + 1} for model '{model_name}' cannot set both 'post_run_js_snippet' and 'js_snippet_file'."
        )
    if raw.get("post_run_js_snippet") and raw.get("post_run_js_snippet_path"):
        raise ValueError(
            f"Experiment #{index + 1} for model '{model_name}' cannot set both 'post_run_js_snippet' and 'post_run_js_snippet_path'."
        )
    if raw.get("js_snippet_file") and raw.get("post_run_js_snippet_path"):
        raise ValueError(
            f"Experiment #{index + 1} for model '{model_name}' cannot set both 'js_snippet_file' and 'post_run_js_snippet_path'."
        )
    if raw.get("system_prompt") and raw.get("system_prompt_file"):
        raise ValueError(
            f"Experiment #{index + 1} for model '{model_name}' cannot set both 'system_prompt' and 'system_prompt_file'."
        )

    values: dict[str, object] = {
        "model": model_name,
        "name": raw.get("name", ""),
        "enabled": raw.get("enabled", True),
        "task": raw.get("task", ""),
        "task_file": raw.get("task_file", ""),
        "application": raw.get("application", ""),
        "run_all": raw.get("run_all", False),
        "run_random": raw.get("run_random", False),
        "iterations": raw.get("iterations", 1),
        "max_tasks": raw.get("max_tasks"),
        "max_steps": raw.get("max_steps", 50),
        "url": raw.get("url", ""),
        "results_dir": raw.get("results_dir", ""),
        "headless": raw.get("headless", False),
        "use_screenshot": raw.get("use_screenshot"),
        "verbose": raw.get("verbose", False),
        "concurrent": raw.get("concurrent", False),
        "workers": raw.get("workers", 0),
        "js_snippet": raw.get("js_snippet", raw.get("post_run_js_snippet", "")),
        "post_run_js_snippet_path": raw.get("post_run_js_snippet_path", raw.get("js_snippet_file", "")),
        "post_run_url": raw.get("post_run_url", ""),
        "system_prompt": raw.get("system_prompt", ""),
        "system_prompt_file": raw.get("system_prompt_file", ""),
        "task_range": raw.get("task_range", ""),
        "initial_delay": raw.get("initial_delay", 0),
        "extra_args": raw.get("extra_args", []),
    }

    values["name"] = _ensure_string(values["name"], "name")
    values["enabled"] = _ensure_bool(values["enabled"], "enabled")
    values["task"] = _ensure_string(values["task"], "task")
    values["task_file"] = _ensure_string(values["task_file"], "task_file")
    values["application"] = _ensure_string(values["application"], "application")
    values["run_all"] = _ensure_bool(values["run_all"], "run_all")
    values["run_random"] = _ensure_bool(values["run_random"], "run_random")
    values["iterations"] = _ensure_int(values["iterations"], "iterations")
    if values["max_tasks"] is not None:
        values["max_tasks"] = _ensure_int(values["max_tasks"], "max_tasks")
    values["max_steps"] = _ensure_int(values["max_steps"], "max_steps")
    values["url"] = _ensure_string(values["url"], "url")
    values["results_dir"] = _ensure_string(values["results_dir"], "results_dir")
    values["headless"] = _ensure_bool(values["headless"], "headless")
    if values["use_screenshot"] is not None:
        values["use_screenshot"] = _ensure_bool(values["use_screenshot"], "use_screenshot")
    values["verbose"] = _ensure_bool(values["verbose"], "verbose")
    values["concurrent"] = _ensure_bool(values["concurrent"], "concurrent")
    values["workers"] = _ensure_int(values["workers"], "workers")
    values["js_snippet"] = _ensure_string(values["js_snippet"], "js_snippet")
    values["post_run_js_snippet_path"] = _ensure_string(
        values["post_run_js_snippet_path"], "post_run_js_snippet_path"
    )
    values["post_run_url"] = _ensure_string(values["post_run_url"], "post_run_url")
    values["system_prompt"] = _ensure_string(values["system_prompt"], "system_prompt")
    values["system_prompt_file"] = _ensure_string(values["system_prompt_file"], "system_prompt_file")
    values["task_range"] = _ensure_string(values["task_range"], "task_range")
    values["initial_delay"] = _ensure_float(values["initial_delay"], "initial_delay")
    values["extra_args"] = _ensure_string_list(values["extra_args"], "extra_args")

    if values["iterations"] < 1:
        raise ValueError("'iterations' must be at least 1.")
    if values["max_steps"] < 1:
        raise ValueError("'max_steps' must be at least 1.")
    if values["workers"] < 0:
        raise ValueError("'workers' must be 0 or greater.")
    if values["max_tasks"] is not None and values["max_tasks"] < 0:
        raise ValueError("'max_tasks' must be 0 or greater.")
    if values["initial_delay"] < 0:
        raise ValueError("'initial_delay' must be 0 or greater.")

    for field_name in PATH_FIELDS:
        values[field_name] = _resolve_config_path(values[field_name], base_dir)

    if not values["name"]:
        task_label = Path(values["task_file"]).stem if values["task_file"] else values["task"] or f"run-{index + 1}"
        model_label = model_name.replace("/", "_").replace(":", "_")
        values["name"] = f"{model_label}-{task_label}"

    return ExperimentSpec(**values)


def load_batch_config(config_path: Path) -> BatchConfig:
    raw_config = json.loads(config_path.read_text())
    config = _ensure_mapping(raw_config, "Batch config")

    unknown_root_keys = sorted(set(config) - {"entrypoint", "max_parallel", "fail_fast", "poll_interval", "termination_grace_period", "defaults", "models"})
    if unknown_root_keys:
        raise ValueError(f"Unsupported root config keys: {', '.join(unknown_root_keys)}")

    defaults = _ensure_mapping(config.get("defaults", {}), "'defaults'")
    _validate_experiment_keys(defaults, "'defaults'")

    models = _ensure_mapping(config.get("models"), "'models'")
    experiments: list[ExperimentSpec] = []
    base_dir = config_path.parent

    for model_name, model_block in models.items():
        if not isinstance(model_name, str):
            raise ValueError("Each model key in 'models' must be a string.")

        if isinstance(model_block, list):
            model_defaults = {}
            raw_experiments = model_block
        else:
            model_mapping = _ensure_mapping(model_block, f"Model block '{model_name}'")
            unknown_model_keys = sorted(set(model_mapping) - MODEL_BLOCK_KEYS)
            if unknown_model_keys:
                raise ValueError(f"Model block '{model_name}' has unsupported keys: {', '.join(unknown_model_keys)}")

            model_defaults = _ensure_mapping(model_mapping.get("defaults", {}), f"Model defaults for '{model_name}'")
            _validate_experiment_keys(model_defaults, f"Model defaults for '{model_name}'")

            raw_experiments = None
            for field_name in ("runs", "experiments", "testcases"):
                if field_name in model_mapping:
                    raw_experiments = model_mapping[field_name]
                    break

        experiment_list = _ensure_list(raw_experiments, f"Experiment list for model '{model_name}'")
        for index, raw_experiment in enumerate(experiment_list):
            experiment_mapping = _ensure_mapping(raw_experiment, f"Experiment #{index + 1} for model '{model_name}'")
            merged = dict(defaults)
            merged.update(model_defaults)
            merged.update(experiment_mapping)
            experiments.append(_coerce_experiment(merged, base_dir, model_name, index))

    if not experiments:
        raise ValueError("No experiments were defined in the batch config.")

    entrypoint_value = config.get("entrypoint")
    entrypoint = _resolve_config_path(entrypoint_value, base_dir) if entrypoint_value else str(DEFAULT_ENTRYPOINT)
    max_parallel = config.get("max_parallel", 0)
    fail_fast = config.get("fail_fast", False)
    poll_interval = config.get("poll_interval", 0.5)
    termination_grace_period = config.get("termination_grace_period", 10.0)

    if isinstance(max_parallel, bool):
        raise ValueError("'max_parallel' must be an integer.")
    if max_parallel != 0:
        max_parallel = _ensure_int(max_parallel, "max_parallel")
    if isinstance(max_parallel, int) and max_parallel < 0:
        raise ValueError("'max_parallel' must be 0 or greater.")

    fail_fast = _ensure_bool(fail_fast, "fail_fast")
    poll_interval = _ensure_float(poll_interval, "poll_interval")
    termination_grace_period = _ensure_float(termination_grace_period, "termination_grace_period")
    if poll_interval <= 0:
        raise ValueError("'poll_interval' must be greater than 0.")
    if termination_grace_period < 0:
        raise ValueError("'termination_grace_period' must be 0 or greater.")

    entrypoint_path = Path(entrypoint).resolve()
    if not entrypoint_path.exists():
        raise ValueError(f"Configured entrypoint does not exist: {entrypoint_path}")

    return BatchConfig(
        experiments=experiments,
        entrypoint=entrypoint_path,
        max_parallel=max_parallel or len(experiments),
        fail_fast=fail_fast,
        poll_interval=poll_interval,
        termination_grace_period=termination_grace_period,
    )


def build_main_command(spec: ExperimentSpec, entrypoint: Path) -> list[str]:
    command = [sys.executable, str(entrypoint), "--model", spec.model, "--results-dir", spec.results_dir]

    if spec.concurrent:
        command.append("--concurrent")
    if spec.workers > 0:
        command.extend(["--workers", str(spec.workers)])
    if spec.task:
        command.extend(["--task", spec.task])
    if spec.run_all:
        command.append("--run-all")
    if spec.headless:
        command.append("--headless")
    if spec.use_screenshot is not None:
        command.append("--use-screenshot" if spec.use_screenshot else "--no-use-screenshot")
    if spec.verbose:
        command.append("--verbose")
    if spec.application:
        command.extend(["--application", spec.application])
    if spec.run_random:
        command.append("--run-random")
    if spec.iterations != 1:
        command.extend(["--iterations", str(spec.iterations)])
    if spec.max_tasks is not None:
        command.extend(["--max-tasks", str(spec.max_tasks)])
    if spec.max_steps != 50:
        command.extend(["--max-steps", str(spec.max_steps)])
    if spec.task_file:
        command.extend(["--task-file", spec.task_file])
    if spec.url:
        command.extend(["--url", spec.url])
    if spec.js_snippet:
        command.extend(["--js-snippet", spec.js_snippet])
    if spec.post_run_js_snippet_path:
        command.extend(["--js-snippet-file", spec.post_run_js_snippet_path])
    if spec.post_run_url:
        command.extend(["--post-run-url", spec.post_run_url])
    if spec.system_prompt:
        command.extend(["--system-prompt", spec.system_prompt])
    if spec.system_prompt_file:
        command.extend(["--system-prompt-file", spec.system_prompt_file])
    if spec.task_range:
        command.extend(["--task-range", spec.task_range])
    if spec.initial_delay > 0:
        command.extend(["--initial-delay", str(spec.initial_delay)])
    command.extend(spec.extra_args)

    return command


class BatchRunner:
    def __init__(self, config: BatchConfig):
        self.config = config
        self.pending = deque(spec for spec in config.experiments if spec.enabled)
        self.running: dict[str, RunningExperiment] = {}
        self.finished: list[FinishedExperiment] = []
        self.stop_requested = False
        self.stop_reason = ""
        self.shutdown_started = False

    def request_stop(self, reason: str) -> None:
        if not self.stop_requested:
            self.stop_requested = True
            self.stop_reason = reason

    def launch_experiment(self, spec: ExperimentSpec) -> None:
        command = build_main_command(spec, self.config.entrypoint)
        log_path = Path(spec.results_dir).resolve() / "batch_runner.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {spec.name}\n")
        log_handle.write(f"Command: {shlex.join(command)}\n\n")
        log_handle.flush()

        popen_kwargs = {
            "cwd": str(ROOT_DIR),
            "stdout": log_handle,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(command, **popen_kwargs)
        self.running[spec.name] = RunningExperiment(
            spec=spec,
            command=command,
            process=process,
            log_handle=log_handle,
            log_path=log_path,
            started_at=time.time(),
        )
        print(f"[start] {spec.name} -> {log_path}")

    def poll_finished(self) -> list[FinishedExperiment]:
        completed: list[FinishedExperiment] = []
        for name, running in list(self.running.items()):
            return_code = running.process.poll()
            if return_code is None:
                continue

            running.log_handle.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished with exit code {return_code}\n"
            )
            running.log_handle.flush()
            running.log_handle.close()

            finished = FinishedExperiment(
                spec=running.spec,
                return_code=return_code,
                log_path=running.log_path,
                duration_seconds=time.time() - running.started_at,
            )
            completed.append(finished)
            del self.running[name]
        return completed

    def _terminate_process_tree(self, running: RunningExperiment, kill: bool) -> None:
        try:
            if os.name == "nt":
                if kill:
                    running.process.kill()
                else:
                    running.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                pgid = os.getpgid(running.process.pid)
                os.killpg(pgid, signal.SIGKILL if kill else signal.SIGTERM)
        except ProcessLookupError:
            return

    def stop_all_running(self) -> None:
        if self.shutdown_started:
            return
        self.shutdown_started = True

        if not self.running:
            return

        reason = self.stop_reason or "shutdown requested"
        print(f"[stop] {reason}. Stopping {len(self.running)} running experiment(s)...")

        for running in list(self.running.values()):
            self._terminate_process_tree(running, kill=False)

        deadline = time.time() + self.config.termination_grace_period
        while self.running and time.time() < deadline:
            for finished in self.poll_finished():
                self.finished.append(finished)
            if self.running:
                time.sleep(min(self.config.poll_interval, 0.25))

        if self.running:
            print(f"[kill] Force killing {len(self.running)} remaining experiment(s)...")
            for running in list(self.running.values()):
                self._terminate_process_tree(running, kill=True)

        while self.running:
            for finished in self.poll_finished():
                self.finished.append(finished)
            if self.running:
                time.sleep(min(self.config.poll_interval, 0.25))

    def run(self) -> int:
        print(f"Loaded {len(self.pending)} enabled experiment(s). Max parallel: {self.config.max_parallel}")

        while self.pending or self.running:
            while not self.stop_requested and self.pending and len(self.running) < self.config.max_parallel:
                self.launch_experiment(self.pending.popleft())

            for finished in self.poll_finished():
                self.finished.append(finished)
                status = "ok" if finished.return_code == 0 else "failed"
                print(
                    f"[{status}] {finished.spec.name} exit={finished.return_code} "
                    f"duration={finished.duration_seconds:.1f}s log={finished.log_path}"
                )
                if finished.return_code != 0 and self.config.fail_fast:
                    self.request_stop(f"{finished.spec.name} failed with exit code {finished.return_code}")

            if self.stop_requested and not self.shutdown_started:
                self.stop_all_running()

            if not self.pending and not self.running:
                break

            if self.running or self.pending:
                time.sleep(self.config.poll_interval)

        failed = [finished for finished in self.finished if finished.return_code != 0]
        if self.stop_requested:
            return 130
        if failed:
            return 1
        return 0


def install_signal_handlers(runner: BatchRunner) -> dict[signal.Signals, object]:
    previous_handlers: dict[signal.Signals, object] = {}

    def handler(signum, _frame):
        signal_name = signal.Signals(signum).name
        runner.request_stop(f"Received {signal_name}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, handler)

    return previous_handlers


def restore_signal_handlers(previous_handlers: dict[signal.Signals, object]) -> None:
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)


def main() -> int:
    args = parse_args()
    config = load_batch_config(Path(args.config).expanduser().resolve())

    if args.max_parallel is not None:
        if args.max_parallel < 0:
            raise ValueError("--max-parallel must be 0 or greater.")
        config.max_parallel = args.max_parallel or len(config.experiments)

    if args.fail_fast:
        config.fail_fast = True
    if args.use_screenshot is not None:
        for spec in config.experiments:
            spec.use_screenshot = args.use_screenshot

    enabled_experiments = [spec for spec in config.experiments if spec.enabled]
    if not enabled_experiments:
        raise ValueError("All experiments are disabled. Nothing to run.")

    if args.dry_run:
        print(f"Loaded {len(enabled_experiments)} enabled experiment(s).")
        for spec in enabled_experiments:
            print(f"\n[{spec.name}]")
            print(shlex.join(build_main_command(spec, config.entrypoint)))
        return 0

    runner = BatchRunner(config)
    previous_handlers = install_signal_handlers(runner)
    try:
        return runner.run()
    finally:
        runner.request_stop(runner.stop_reason or "batch runner exiting")
        runner.stop_all_running()
        restore_signal_handlers(previous_handlers)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
