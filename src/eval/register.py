import os
from pathlib import Path

from agisdk.REAL.browsergym.core.registration import register_task
from agisdk.REAL.browsergym.webclones import base
from agisdk.REAL.browsergym.webclones.task_config import load_tasks_from_file


def register_evaluation_tasks(paths: list[Path]):
    if not any(os.path.exists(path) for path in paths):
        print(f"Warning: Evaluation tasks path not found: {paths}")
        return

    def register_task_file(entry: Path, prefix: str = "eval"):
        for task in load_tasks_from_file(entry):
            task_id = task["id"]
            print(f"Registering evaluation task: {entry.name}::{task_id}")
            gym_id = f"{prefix}.{task_id}"
            register_task(
                gym_id,
                base.AbstractWebCloneTask,
                task_kwargs={"task_id": task_id, "task_source": str(entry.resolve())},
            )

    def register_tasks_in_path(path: Path, prefix: str = "eval"):
        if path.is_file():
            register_task_file(path, prefix=prefix)
            return

        for pattern in ("**/*.json", "**/*.yaml", "**/*.yml"):
            for entry in path.glob(pattern):
                if entry.is_file():
                    register_task_file(entry, prefix=prefix)

    for path in paths:
        register_tasks_in_path(Path(path), prefix="eval")
