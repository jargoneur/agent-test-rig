from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "benchmarks" / "contextbench" / "tasks.jsonl"
TASK_PREFIX = "contextbench::"


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    "ContextBench cache line %d is not an object" % line_number
                )
            yield value


def load_contextbench_cache(path: Optional[str] = None) -> List[Dict[str, Any]]:
    cache = Path(path) if path else DEFAULT_CACHE
    if not cache.is_absolute():
        cache = PROJECT_ROOT / cache
    if not cache.is_file():
        raise FileNotFoundError(
            "ContextBench task cache is missing: %s. "
            "Run scripts/prepare_contextbench.py" % cache
        )
    rows = list(_read_jsonl(cache))
    if not rows:
        raise ValueError("ContextBench task cache is empty: %s" % cache)
    return rows


def contextbench_task_ids(
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[str]:
    rows = load_contextbench_cache(path)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return [TASK_PREFIX + str(row["instance_id"]) for row in rows]


def _infer_repo(original_inst_id: str) -> str:
    match = re.match(r"^(?P<org>[^_]+)__(?P<repo>[^-]+)-\d+$", original_inst_id)
    if not match:
        return ""
    return "%s/%s" % (match.group("org"), match.group("repo"))


def _first(row: Dict[str, Any], *names: str, default=None):
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def load_contextbench_task(
    task_id: str,
    path: Optional[str] = None,
) -> Dict[str, Any]:
    if not task_id.startswith(TASK_PREFIX):
        raise ValueError("Not a ContextBench task id: %s" % task_id)
    instance_id = task_id[len(TASK_PREFIX) :]
    row = next(
        (
            item
            for item in load_contextbench_cache(path)
            if str(item.get("instance_id")) == instance_id
        ),
        None,
    )
    if row is None:
        raise KeyError("ContextBench instance is not in the frozen cache: %s" % instance_id)

    original_inst_id = str(
        _first(row, "original_inst_id", "swebench_instance_id", default="")
    )
    repo = str(_first(row, "repo", default="") or _infer_repo(original_inst_id))
    base_commit = str(_first(row, "base_commit", "commit", default=""))
    issue = str(
        _first(
            row,
            "problem_statement",
            "issue",
            "issue_text",
            "description",
            default="",
        )
    )
    if not repo:
        raise ValueError("ContextBench task has no repository: %s" % instance_id)
    if not base_commit:
        raise ValueError("ContextBench task has no base commit: %s" % instance_id)
    if not issue:
        raise ValueError("ContextBench task has no problem statement: %s" % instance_id)

    return {
        "id": task_id,
        "source": "contextbench",
        "instance_id": instance_id,
        "original_inst_id": original_inst_id,
        "bench": str(row.get("bench") or "Verified"),
        "language": str(row.get("language") or "python"),
        "issue": issue,
        "repo": repo,
        "base_commit": base_commit,
        "workspace_kind": "git_checkout",
        "expected_files": [],
        "test_command": str(row.get("test_command") or "true"),
        "test_timeout": int(row.get("test_timeout") or 300),
        "run_final_tests": bool(row.get("run_final_tests", False)),
        "contextbench_metadata": row,
    }
