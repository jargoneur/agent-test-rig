from pathlib import Path
import yaml


class TaskLoader:
    def __init__(self, tasks_root: str = "benchmarks/tasks"):
        self.tasks_root = Path(tasks_root)

    def load(self, task_id: str) -> dict:
        task_dir = self.tasks_root / task_id
        config_path = task_dir / "task.yml"

        if not config_path.exists():
            raise FileNotFoundError(f"Task config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            task = yaml.safe_load(file)

        issue_path = task_dir / task["issue_file"]

        if not issue_path.exists():
            raise FileNotFoundError(f"Issue file not found: {issue_path}")

        task["issue"] = issue_path.read_text(encoding="utf-8")
        task["task_dir"] = str(task_dir)

        return task