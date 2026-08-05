from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class FileTools:
    def __init__(self, repo_path: str, test_timeout: int = 300):
        self.repo_path = Path(repo_path).resolve()
        self.test_timeout = max(1, int(test_timeout))

    def _safe_path(self, path: str) -> Path:
        full_path = (self.repo_path / path).resolve()

        try:
            full_path.relative_to(self.repo_path)
        except ValueError as error:
            raise ValueError(f"Unsafe path outside repo: {path}") from error

        return full_path

    @staticmethod
    def _ignored(relative: Path) -> bool:
        ignored_parts = {
            "__pycache__",
            ".pytest_cache",
            ".git",
            ".venv",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules",
        }
        return any(part in ignored_parts for part in relative.parts)

    def list_files(
        self,
        glob: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        files: List[str] = []

        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue

            relative = path.relative_to(self.repo_path)
            if self._ignored(relative):
                continue
            value = relative.as_posix()
            if glob and not fnmatch.fnmatch(value, glob):
                continue
            files.append(value)

        files.sort()
        if limit is not None:
            return files[: max(0, int(limit))]
        return files

    def line_count(self, path: str) -> int:
        full_path = self._safe_path(path)
        with full_path.open("r", encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        full_path = self._safe_path(path)
        text = full_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(1, int(start_line or 1))
        end = int(end_line or len(lines))
        if end < start:
            raise ValueError("end_line must be greater than or equal to start_line")
        selected = lines[start - 1 : end]
        return "\n".join(selected)

    def search_text(
        self,
        query: str,
        glob: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, object]]:
        if not query:
            raise ValueError("search_text requires a non-empty query")
        results: List[Dict[str, object]] = []
        for relative in self.list_files(glob=glob):
            path = self._safe_path(relative)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    results.append(
                        {
                            "path": relative,
                            "line": line_number,
                            "text": line[:500],
                        }
                    )
                    if len(results) >= max(1, int(limit)):
                        return results
        return results

    def write_file(self, path: str, content: str) -> None:
        full_path = self._safe_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def _normalize_test_output(self, output: str) -> str:
        normalized = output.replace(str(self.repo_path), "<WORKSPACE>")
        normalized = re.sub(r"\b\d+(?:\.\d+)?s\b", "<TIME>", normalized)
        return normalized

    def run_tests(self, command: str = "pytest") -> dict:
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"

        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                shell=True,
                text=True,
                capture_output=True,
                timeout=self.test_timeout,
                env=environment,
            )
            return {
                "returncode": result.returncode,
                "stdout": self._normalize_test_output(result.stdout),
                "stderr": self._normalize_test_output(result.stderr),
                "passed": result.returncode == 0,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {
                "returncode": 124,
                "stdout": self._normalize_test_output(stdout),
                "stderr": self._normalize_test_output(stderr),
                "passed": False,
                "timed_out": True,
            }

    def git_diff(self) -> str:
        result = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=self.repo_path,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError("git diff failed: %s" % result.stderr)
        return result.stdout

    def reset_repo(self) -> dict:
        reset = subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            cwd=self.repo_path,
            text=True,
            capture_output=True,
            timeout=60,
        )

        clean = subprocess.run(
            ["git", "clean", "-fdx"],
            cwd=self.repo_path,
            text=True,
            capture_output=True,
            timeout=60,
        )

        return {
            "reset_returncode": reset.returncode,
            "reset_stdout": reset.stdout,
            "reset_stderr": reset.stderr,
            "clean_returncode": clean.returncode,
            "clean_stdout": clean.stdout,
            "clean_stderr": clean.stderr,
            "success": reset.returncode == 0 and clean.returncode == 0,
        }
