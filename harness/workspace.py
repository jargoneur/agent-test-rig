from __future__ import annotations

import fcntl
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_ROOT = (PROJECT_ROOT / "benchmarks" / "fixtures").resolve()
WORKSPACES_ROOT = PROJECT_ROOT / ".workspaces"
REPO_CACHE_ROOT = Path(
    os.environ.get("BENCHMARK_REPO_CACHE", str(PROJECT_ROOT / ".cache" / "repos"))
).expanduser()


def _run_git(arguments: List[str], cwd: Path, timeout: int = 300) -> str:
    result = subprocess.run(
        ["git"] + arguments,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Git command failed in %s: git %s\nstdout: %s\nstderr: %s"
            % (cwd, " ".join(arguments), result.stdout, result.stderr)
        )
    return result.stdout


def _safe_repo_name(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo)


def _ensure_repo_mirror(repo: str, commit: str) -> Path:
    REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    mirror = REPO_CACHE_ROOT / (_safe_repo_name(repo) + ".git")
    lock_path = REPO_CACHE_ROOT / (_safe_repo_name(repo) + ".lock")
    url = "https://github.com/%s.git" % repo

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not mirror.exists():
            result = subprocess.run(
                ["git", "clone", "--mirror", url, str(mirror)],
                text=True,
                capture_output=True,
                timeout=1800,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Failed to clone %s: %s" % (url, result.stderr.strip())
                )
        else:
            origin = _run_git(["remote", "get-url", "origin"], mirror).strip()
            if origin != url:
                raise RuntimeError(
                    "Repository cache origin mismatch for %s: %s" % (repo, origin)
                )

        check = subprocess.run(
            ["git", "cat-file", "-e", "%s^{commit}" % commit],
            cwd=mirror,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if check.returncode != 0:
            _run_git(["fetch", "--prune", "origin", commit], mirror, timeout=1800)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return mirror


def _prepare_git_checkout(task: dict, workspace_name: str) -> str:
    repo = str(task["repo"])
    commit = str(task["base_commit"])
    mirror = _ensure_repo_mirror(repo, commit)

    WORKSPACES_ROOT.mkdir(parents=True, exist_ok=True)
    destination = WORKSPACES_ROOT / workspace_name
    if destination.exists():
        shutil.rmtree(destination)

    result = subprocess.run(
        ["git", "clone", "--no-checkout", str(mirror), str(destination)],
        text=True,
        capture_output=True,
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to create cached checkout for %s: %s"
            % (repo, result.stderr.strip())
        )
    _run_git(["checkout", "--detach", commit], destination, timeout=300)
    _run_git(["config", "core.autocrlf", "false"], destination)
    _run_git(["config", "core.eol", "lf"], destination)
    return str(destination)


def prepare_workspace(task: dict, workspace_name: str) -> str:
    """Create an isolated repository workspace for a benchmark task."""
    if task.get("workspace_kind") == "git_checkout":
        return _prepare_git_checkout(task, workspace_name)

    fixture_path = task.get("fixture_path")

    if not fixture_path:
        repo_path = task.get("repo_path")
        if not repo_path:
            raise ValueError("Task must define fixture_path, repo_path, or git checkout")
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
