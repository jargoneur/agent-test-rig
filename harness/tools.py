import os
import re
import subprocess
from pathlib import Path


class FileTools:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def _safe_path(self, path: str) -> Path:
        full_path = (self.repo_path / path).resolve()

        try:
            full_path.relative_to(self.repo_path)
        except ValueError as error:
            raise ValueError