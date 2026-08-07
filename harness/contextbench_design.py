from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Sequence


COMPLEXITY_METRICS = (
    "patch_files",
    "patch_hunks",
    "patch_changed_lines",
    "test_changed_lines",
    "gold_context_files",
    "gold_context_characters",
    "problem_characters",
)
COMPLEXITY_BINS = ("easy", "intermediate", "hard")
_DIFF_FILE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
_HUNK = re.compile(r"^@@ ", re.MULTILINE)
_DOC = re.compile(r"(^|/)(docs?|examples?)/|\.(md|rst|adoc)$", re.IGNORECASE)
_CONFIG = re.compile(
    r"(^|/)(config|configs|scripts|workflows)/|"
    r"(^|/)(pyproject\.toml|setup\.(py|cfg)|package\.json|"
    r".*\.(ya?ml|toml|ini|cfg))$",
    re.IGNORECASE,
)
_TEST = re.compile(
    r"(^|/)(tests?|specs?)/|(^|/)test[_-]|[_-](test|spec)\.",
    re.IGNORECASE,
)


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(("%d:%s" % (seed, value)).encode("utf-8")).hexdigest()


def patch_statistics(patch: Any) -> Dict[str, Any]:
    text = str(patch or "")
    files = []
    for left, right in _DIFF_FILE.findall(text):
        path = right if right != "/dev/null" else left
        if path not in files:
            files.append(path)
    added = 0
    deleted = 0
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return {
        "files": files,
        "file_count": len(files),
        "hunks": len(_HUNK.findall(text)),
        "added_lines": added,
        "deleted_lines": deleted,
        "changed_lines": added + deleted,
    }


def gold_context_statistics(value: Any) -> Dict[str, int]:
    text = str(value or "")
    files = set()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and item.get("file"):
                files.add(str(item["file"]))
    if not files:
        files.update(
            match.strip()
            for match in re.findall(r"^\s*file:\s*(.+?)\s*$", text, re.MULTILINE)
        )
    return {
        "file_count": len(files),
        "characters": len(text),
    }


def declared_task_type(instance_id: Any) -> str:
    parts = str(instance_id or "").split("__")
    if len(parts) >= 4:
        return "%s_%s" % (parts[2], parts[3])
    return "unknown"


def solution_scope(files: Sequence[str]) -> str:
    paths = [str(value) for value in files]
    if not paths:
        return "no_gold_patch"
    docs_or_config = sum(bool(_DOC.search(path) or _CONFIG.search(path)) for path in paths)
    tests = sum(bool(_TEST.search(path)) for path in paths)
    if docs_or_config >= max(1, math.ceil(len(paths) / 2)):
        return "documentation_configuration"
    if tests >= max(1, math.ceil(len(paths) / 2)):
        return "test_maintenance"
    directories = {
        str(PurePosixPath(path).parent)
        for path in paths
    }
    if len(paths) >= 3 and len(directories) >= 2:
        return "cross_module"
    return "localized_code"


def _percentile_ranks(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, float]:
    ordered = sorted(
        rows,
        key=lambda row: (
            math.log1p(float(row["_complexity_raw"][metric])),
            str(row["instance_id"]),
        ),
    )
    denominator = max(1, len(ordered) - 1)
    return {
        str(row["instance_id"]): index / denominator
        for index, row in enumerate(ordered)
    }


def enrich_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in records:
        row = dict(source)
        patch = patch_statistics(row.get("patch"))
        tests = patch_statistics(row.get("test_patch"))
        gold = gold_context_statistics(row.get("gold_context"))
        row["_complexity_raw"] = {
            "patch_files": patch["file_count"],
            "patch_hunks": patch["hunks"],
            "patch_changed_lines": patch["changed_lines"],
            "test_changed_lines": tests["changed_lines"],
            "gold_context_files": gold["file_count"],
            "gold_context_characters": gold["characters"],
            "problem_characters": len(str(row.get("problem_statement") or "")),
        }
        row["_solution_scope"] = solution_scope(patch["files"])
        rows.append(row)

    if not rows:
        raise ValueError("ContextBench source contains no rows")

    percentiles = {
        metric: _percentile_ranks(rows, metric)
        for metric in COMPLEXITY_METRICS
    }
    for row in rows:
        instance_id = str(row["instance_id"])
        component = {
            metric: round(percentiles[metric][instance_id], 8)
            for metric in COMPLEXITY_METRICS
        }
        score = sum(component.values()) / len(component)
        row["_complexity_components"] = component
        row["_complexity_score"] = score

    ordered = sorted(
        rows,
        key=lambda row: (row["_complexity_score"], str(row["instance_id"])),
    )
    for index, row in enumerate(ordered):
        bin_index = min(2, (index * 3) // len(ordered))
        row["_complexity_bin"] = COMPLEXITY_BINS[bin_index]

    enriched = []
    for row in rows:
        raw = row.pop("_complexity_raw")
        scope = row.pop("_solution_scope")
        components = row.pop("_complexity_components")
        score = row.pop("_complexity_score")
        complexity_bin = row.pop("_complexity_bin")
        row["task_design"] = {
            "schema_version": 1,
            "source_declared_task_type": declared_task_type(row.get("instance_id")),
            "source": str(row.get("source") or ""),
            "language": str(row.get("language") or ""),
            "solution_scope_rubric_v1": scope,
            "complexity": {
                "rubric": "contextbench_complexity_v1",
                "continuous_score": round(score, 8),
                "bin": complexity_bin,
                "raw": raw,
                "percentile_components": components,
            },
        }
        enriched.append(row)
    return enriched


def _selection_cost(
    candidate: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
    pool: Sequence[Dict[str, Any]],
    target_count: int,
) -> float:
    dimensions = (
        lambda row: str(row.get("source") or ""),
        lambda row: str(row.get("language") or ""),
        lambda row: str((row.get("task_design") or {}).get("solution_scope_rubric_v1") or ""),
        lambda row: str((row.get("task_design") or {}).get("source_declared_task_type") or ""),
    )
    cost = 0.0
    for key_fn in dimensions:
        pool_counts = Counter(key_fn(row) for row in pool)
        selected_counts = Counter(key_fn(row) for row in selected)
        key = key_fn(candidate)
        target = target_count * pool_counts[key] / len(pool)
        before = selected_counts[key]
        cost += (before + 1 - target) ** 2 - (before - target) ** 2
    repo = str(candidate.get("repo") or "")
    repo_count = sum(str(row.get("repo") or "") == repo for row in selected)
    return cost + 0.35 * repo_count


def select_representative(
    records: Iterable[Dict[str, Any]],
    limit: int = 150,
    seed: int = 20260807,
) -> List[Dict[str, Any]]:
    rows = enrich_rows(records)
    if limit < 3 or limit > len(rows):
        raise ValueError("limit must be between 3 and the source row count")

    task_type_counts = Counter(
        row["task_design"]["source_declared_task_type"] for row in rows
    )
    census_threshold = max(1, limit // 4)
    census_types = {
        task_type
        for task_type, count in task_type_counts.items()
        if count <= census_threshold
    }

    bin_targets = {
        name: limit // 3 + (1 if index < limit % 3 else 0)
        for index, name in enumerate(COMPLEXITY_BINS)
    }
    selected: List[Dict[str, Any]] = []
    for name in COMPLEXITY_BINS:
        pool = [
            row
            for row in rows
            if row["task_design"]["complexity"]["bin"] == name
        ]
        target = bin_targets[name]
        chosen = [
            row
            for row in pool
            if row["task_design"]["source_declared_task_type"] in census_types
        ]
        if len(chosen) > target:
            raise ValueError("Rare task-type census exceeds a complexity-bin target")
        chosen_ids = {str(row["instance_id"]) for row in chosen}
        remaining = [row for row in pool if str(row["instance_id"]) not in chosen_ids]
        while len(chosen) < target:
            candidate = min(
                remaining,
                key=lambda row: (
                    _selection_cost(row, chosen, pool, target),
                    stable_key(seed, str(row["instance_id"])),
                ),
            )
            chosen.append(candidate)
            remaining.remove(candidate)
        selected.extend(chosen)

    selected.sort(
        key=lambda row: (
            COMPLEXITY_BINS.index(row["task_design"]["complexity"]["bin"]),
            row["task_design"]["complexity"]["continuous_score"],
            stable_key(seed, str(row["instance_id"])),
        )
    )
    return selected


def selection_distribution(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    return {
        "source": dict(sorted(Counter(str(row.get("source") or "") for row in rows).items())),
        "language": dict(sorted(Counter(str(row.get("language") or "") for row in rows).items())),
        "repository": dict(sorted(Counter(str(row.get("repo") or "") for row in rows).items())),
        "complexity_bin": dict(
            sorted(
                Counter(
                    row["task_design"]["complexity"]["bin"] for row in rows
                ).items()
            )
        ),
        "solution_scope": dict(
            sorted(
                Counter(
                    row["task_design"]["solution_scope_rubric_v1"] for row in rows
                ).items()
            )
        ),
        "source_declared_task_type": dict(
            sorted(
                Counter(
                    row["task_design"]["source_declared_task_type"] for row in rows
                ).items()
            )
        ),
    }
