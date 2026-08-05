from pathlib import Path

import yaml

from harness.contextbench_tasks import TASK_PREFIX, load_contextbench_task


class TaskLoader:
    def __init__(self, tasks_root: str = "benchmarks/tasks", contextbench_cache=None):
        self.tasks_root = Path(tasks_root)
        self.contextbench_cache = contextbench_cache

    def load(self, task_id: str) -> dict:
        if task_id.startswith(TASK_PREFIX):
            return load_contextbench_task(task_id, path=self.contextbench_cache)

        task_dir = self.tasks_root / task_id
        config_path = task_dir / "task.yml"

        if not config_path.exists():
            raise FileNotFoundError(f"Task config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            task = yaml.safe_load(file)

        issue_path = task_dir / task["issue_file"]

        if not issue_path.exists():
            raise FileNotFoundError(f"Issue file not found: {issue_path}")

        task["id"] = task_id
        task["issue"] = issue_path.read_text(encoding="utf-8")
        task["task_dir"] = str(task_dir)
        task.setdefault("test_command", "pytest")
        task.setdefault("test_timeout", 60)
        task.setdefault("run_final_tests", True)

        return task
