from pathlib import Path
import subprocess


class FileTools:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def _safe_path(self, path: str) -> Path:
        full_path = (self.repo_path / path).resolve()

        if not str(full_path).startswith(str(self.repo_path)):
            raise ValueError(f"Unsafe path outside repo: {path}")

        return full_path

    def list_files(self) -> list[str]:
        ignored_parts = {
            "__pycache__",
            ".pytest_cache",
            ".git",
            ".venv",
        }

        files = []

        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue

            relative = path.relative_to(self.repo_path)

            if any(part in ignored_parts for part in relative.parts):
                continue

            files.append(str(relative))

        return sorted(files)

    def read_file(self, path: str) -> str:
        full_path = self._safe_path(path)
        return full_path.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        full_path = self._safe_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def run_tests(self, command: str = "pytest") -> dict:
        result = subprocess.run(
            command,
            cwd=self.repo_path,
            shell=True,
            text=True,
            capture_output=True,
            timeout=60,
        )

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0,
        }

    def reset_repo(self) -> dict:
        reset = subprocess.run(
            "git reset --hard HEAD",
            cwd=self.repo_path,
            shell=True,
            text=True,
            capture_output=True,
            timeout=60,
        )

        clean = subprocess.run(
            "git clean -fdx",
            cwd=self.repo_path,
            shell=True,
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
