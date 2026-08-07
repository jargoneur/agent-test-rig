#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, Iterable

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


import pyarrow.parquet as parquet
from datasets import Dataset

from harness.contextbench_design import (
    COMPLEXITY_METRICS,
    declared_task_type,
    select_representative,
    selection_distribution,
)
from harness.upstreams import PROJECT_ROOT, component_spec, upstream_path


DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "contextbench"
DEFAULT_DATASET_REVISION = "c2855792b006af41c67202d33883fb9d46362853"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def selection_row(row: Dict[str, Any]) -> Dict[str, Any]:
    design = row["task_design"]
    complexity = design["complexity"]
    return {
        "instance_id": row["instance_id"],
        "original_inst_id": row.get("original_inst_id", ""),
        "source": row.get("source", ""),
        "language": row.get("language", ""),
        "repo": row.get("repo", ""),
        "source_declared_task_type": design["source_declared_task_type"],
        "solution_scope_rubric_v1": design["solution_scope_rubric_v1"],
        "complexity_bin": complexity["bin"],
        "complexity_score": complexity["continuous_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a representative 150-task sample from all 1,136 pinned "
            "ContextBench tasks."
        )
    )
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--dataset",
        default="Contextbench/ContextBench",
        help="Official dataset identifier recorded for provenance.",
    )
    parser.add_argument(
        "--dataset-revision",
        default=DEFAULT_DATASET_REVISION,
        help="Exact official Hugging Face dataset commit.",
    )
    args = parser.parse_args()

    contextbench_root = upstream_path("contextbench")
    source_parquet = contextbench_root / "data" / "full.parquet"
    if not source_parquet.is_file():
        raise FileNotFoundError("Pinned ContextBench full.parquet is missing")

    table = parquet.read_table(source_parquet)
    records = [jsonable(row) for row in table.to_pylist()]
    selected = select_representative(
        records,
        limit=args.limit,
        seed=args.seed,
    )

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks_path = output / "tasks.jsonl"
    selected_path = output / "frozen_150.csv"
    gold_path = output / "gold.parquet"
    manifest_path = output / "manifest.json"

    write_jsonl(tasks_path, selected)
    csv_rows = [selection_row(row) for row in selected]
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    Dataset.from_list(selected).to_parquet(str(gold_path))

    source_task_types = dict(
        Counter(
            declared_task_type(row.get("instance_id")) for row in records
        )
    )
    expected_task_types = {
        "maintenance_bugfix": 1098,
        "evolution_feature": 36,
        "evolution_refactor": 2,
    }
    if source_task_types != expected_task_types:
        raise RuntimeError(
            "Unexpected ContextBench task-type taxonomy: %s" % source_task_types
        )

    manifest = {
        "schema_version": 2,
        "contextbench_repository": component_spec("contextbench")["repository"],
        "contextbench_commit": component_spec("contextbench")["commit"],
        "source_file": "data/full.parquet",
        "source_file_sha256": sha256_file(source_parquet),
        "source_row_count": len(records),
        "huggingface_dataset": args.dataset,
        "huggingface_dataset_revision": args.dataset_revision,
        "selection": {
            "method": "rare_task_type_census_plus_representative_greedy_v1",
            "seed": args.seed,
            "count": len(selected),
            "complexity_rubric": "contextbench_complexity_v1",
            "complexity_metrics": list(COMPLEXITY_METRICS),
            "complexity_bins": "equal_population_tertiles_over_all_1136_rows",
            "distributions": selection_distribution(selected),
        },
        "scientific_scope": {
            "source_task_type_is_fixed": False,
            "source_task_types": source_task_types,
            "task_type_boundary_claims_allowed": True,
            "task_type_sampling": (
                "all feature and refactor tasks are included; bug-fix tasks "
                "are representative fills"
            ),
            "task_type_limitations": (
                "no source-labelled test-only or documentation-only category"
            ),
            "solution_scope_is_descriptive_not_a_task_type_substitute": True,
        },
        "tasks_sha256": sha256_file(tasks_path),
        "selection_sha256": sha256_file(selected_path),
        "gold_sha256": sha256_file(gold_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print("ContextBench source tasks:", len(records))
    print("Frozen tasks:", len(selected))
    print("Distribution:", json.dumps(manifest["selection"]["distributions"], sort_keys=True))
    print("Task cache:", tasks_path)
    print("Gold parquet:", gold_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
