from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = (PROJECT_ROOT / "benchmarks" / "fixtures").resolve()
WORKSPACES_ROOT = PROJECT_ROOT / ".workspaces"


def _run_git(arguments: list[str], cwd: Path) -> None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Git command failed in {cwd}: git {' '.join(arguments)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def prepare_workspace(task: dict, workspace_name: str) -> str:
    """Create a fresh local git repository from a tracked task fixture.

    Legacy tasks with repo_path keep using their existing repository.
    """
    fixture_path = task.get("fixture_path")

    if not fixture_path:
        repo_path = task.get("repo_path")
        if not repo_path:
            raise ValueError("Task must define fixture_path or repo_path")
        return repo_path

    source = (PROJECT_ROOT / fixture_path).resolve()

    if source != FIXTURES_ROOT and FIXTURES_ROOT not in source.parents:
        raise ValueError(f"Fixture path is outside benchmarks/fixtures: {fixture_path}")

    if not source.is_dir():
        raise FileNotFoundError(f"Fixture directory not found: {source}")

    WORKSPACES_ROOT.mkdir(parents=True, exist_ok=True)
    destination = WORKSPACES_ROOT / workspace_name

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)

    _run_git(["init", "-q"], destination)
    _run_git(["add", "-A"], destination)
    _run_git(
        [
            "-c",
            "user.name=Benchmark Harness",
            "-c",
            "user.email=benchmark@local",
            "commit",
            "-q",
            "-m",
            "Initial benchmark state",
        ],
        destination,
    )

    return str(destination)
