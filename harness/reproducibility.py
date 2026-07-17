import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_OLLAMA_SEED = 2_147_483_647


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def derive_run_seed(
    base_seed: int,
    task_id: str,
    model_name: str,
    repeat: int,
) -> int:
    """Create a stable seed paired across scaffolds.

    The scaffold name is deliberately excluded so corresponding scaffold
    conditions receive the same run seed.
    """
    payload = "\0".join(
        [str(base_seed), task_id, model_name, str(repeat)]
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    seed = value % MAX_OLLAMA_SEED
    return seed or 1


def derive_step_seed(run_seed: int, step: int) -> int:
    seed = (run_seed + step - 1) % MAX_OLLAMA_SEED
    return seed or 1


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def git_metadata() -> dict:
    status = _git(["status", "--porcelain"])
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def environment_metadata() -> dict:
    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(
            part in {".git", "__pycache__", ".pytest_cache"}
            for part in path.parts
        ):
            continue

        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def task_metadata(task: dict) -> dict:
    task_dir = Path(task["task_dir"]).resolve()
    fixture_path = task.get("fixture_path")
    repo_path = task.get("repo_path")

    metadata = {
        "id": task["id"],
        "issue_sha256": sha256_text(task["issue"]),
        "task_config": {
            key: value
            for key, value in task.items()
            if key not in {"issue", "task_dir"}
        },
    }

    if fixture_path:
        fixture = (PROJECT_ROOT / fixture_path).resolve()
        metadata["fixture_path"] = fixture_path
        metadata["fixture_sha256"] = directory_sha256(fixture)

    if repo_path:
        repo = (PROJECT_ROOT / repo_path).resolve()
        metadata["repo_path"] = repo_path
        metadata["repo_sha256"] = directory_sha256(repo)

    config_path = task_dir / "task.yml"
    issue_path = task_dir / task["issue_file"]

    if config_path.exists():
        metadata["task_yml_sha256"] = file_sha256(config_path)
    if issue_path.exists():
        metadata["issue_file_sha256"] = file_sha256(issue_path)

    return metadata


def write_json(path: str | Path, data: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
