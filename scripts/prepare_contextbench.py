#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset, load_dataset

from harness.upstreams import PROJECT_ROOT, component_spec, upstream_path


DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "contextbench"


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


def select_rows(csv_path: Path, limit: int) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("bench") or "").strip() != "Verified":
                continue
            if str(row.get("language") or "").strip().lower() != "python":
                continue
            selected.append(dict(row))
            if len(selected) >= limit:
                break
    if len(selected) != limit:
        raise RuntimeError(
            "Pinned ContextBench CSV yielded %d matching tasks, expected %d"
            % (len(selected), limit)
        )
    return selected


def index_dataset(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for raw in records:
        row = jsonable(dict(raw))
        for key in ("instance_id", "inst_id", "original_inst_id", "swebench_instance_id"):
            value = row.get(key)
            if value:
                index[str(value)] = row
    return index


def merge_selection(
    selected: List[Dict[str, str]],
    dataset_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    missing = []
    for selection in selected:
        instance_id = str(selection["instance_id"])
        original = str(selection.get("original_inst_id") or "")
        record = dataset_index.get(instance_id) or dataset_index.get(original)
        if record is None:
            missing.append("%s (%s)" % (instance_id, original))
            continue
        value = dict(record)
        value.update(selection)
        value["instance_id"] = instance_id
        value["original_inst_id"] = original
        value["bench"] = "Verified"
        value["language"] = "python"
        merged.append(value)
    if missing:
        raise RuntimeError(
            "ContextBench Hugging Face dataset is missing frozen instances: %s"
            % ", ".join(missing[:10])
        )
    return merged


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(jsonable(row), ensure_ascii=False, sort_keys=True) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen 150-task ContextBench Verified subset."
    )
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--dataset",
        default="Contextbench/ContextBench",
    )
    parser.add_argument("--config", default="contextbench_verified")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    contextbench_root = upstream_path("contextbench")
    source_csv = contextbench_root / "data" / "selected_500_instances.csv"
    if not source_csv.is_file():
        raise FileNotFoundError("Pinned ContextBench selection CSV is missing")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = select_rows(source_csv, args.limit)

    dataset = load_dataset(args.dataset, args.config, split=args.split)
    records = [dict(item) for item in dataset]
    merged = merge_selection(selected, index_dataset(records))

    tasks_path = output / "tasks.jsonl"
    selected_path = output / "frozen_150.csv"
    gold_path = output / "gold.parquet"
    manifest_path = output / "manifest.json"

    write_jsonl(tasks_path, merged)
    with selected_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    Dataset.from_list(merged).to_parquet(str(gold_path))

    manifest = {
        "schema_version": 1,
        "contextbench_repository": component_spec("contextbench")["repository"],
        "contextbench_commit": component_spec("contextbench")["commit"],
        "source_csv": "data/selected_500_instances.csv",
        "selection": {
            "bench": "Verified",
            "language": "python",
            "order": "pinned_csv_order",
            "count": args.limit,
        },
        "huggingface_dataset": args.dataset,
        "huggingface_config": args.config,
        "huggingface_split": args.split,
        "tasks_sha256": sha256_file(tasks_path),
        "selection_sha256": sha256_file(selected_path),
        "gold_sha256": sha256_file(gold_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print("ContextBench tasks:", len(merged))
    print("Task cache:", tasks_path)
    print("Gold parquet:", gold_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
