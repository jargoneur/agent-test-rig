#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

from harness.contextbench_trajectory import build_contextbench_trajectory
from harness.upstreams import upstream_path


ROOT = Path(__file__).resolve().parents[1]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def task_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(record.get("task") or {})
    config = dict(metadata.get("task_config") or {})
    return {
        "id": metadata.get("id") or config.get("id"),
        "instance_id": metadata.get("instance_id") or config.get("instance_id"),
        "original_inst_id": metadata.get("original_inst_id")
        or config.get("original_inst_id"),
        "bench": metadata.get("bench") or config.get("bench"),
        "repo": metadata.get("repo") or config.get("repo"),
        "base_commit": metadata.get("base_commit") or config.get("base_commit"),
    }


def condition_key(record: Dict[str, Any]) -> Tuple[str, str, int]:
    job = dict(record.get("job") or {})
    model = dict(job.get("model") or {})
    return (
        str(model.get("id") or model.get("name") or "unknown-model"),
        str(job.get("scaffold") or "unknown-scaffold"),
        int(job.get("repeat") or 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect completed runs and evaluate each ContextBench condition."
    )
    parser.add_argument("roots", nargs="+", help="Scheduler exports or worker result roots")
    parser.add_argument(
        "--gold",
        default="benchmarks/contextbench/gold.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="results/contextbench_evaluation",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/contextbench_evaluator",
    )
    parser.add_argument(
        "--evaluator-python",
        default=str(ROOT / ".venv-contextbench" / "bin" / "python"),
    )
    args = parser.parse_args()

    evaluator_python = Path(args.evaluator_python).expanduser().resolve()
    if not evaluator_python.is_file():
        raise FileNotFoundError(
            "ContextBench evaluator Python is missing: %s. Run scripts/bootstrap.sh"
            % evaluator_python
        )

    groups = defaultdict(dict)
    for root_value in args.roots:
        root = Path(root_value)
        candidates = root.rglob("run_*.json") if root.is_dir() else [root]
        for path in candidates:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if record.get("status") != "completed":
                continue
            task_metadata = dict(record.get("task") or {})
            if task_metadata.get("source") != "contextbench":
                continue
            task = task_from_record(record)
            if not task.get("instance_id"):
                continue
            trajectory = build_contextbench_trajectory(
                task,
                dict(record.get("result") or {}),
                str(record.get("model_patch") or ""),
                job=dict(record.get("job") or {}),
            )
            key = condition_key(record)
            instance_id = str(trajectory["instance_id"])
            existing = groups[key].get(instance_id)
            if existing is not None and existing != trajectory:
                raise RuntimeError(
                    "Conflicting completed trajectories for %s in %s"
                    % (instance_id, key)
                )
            groups[key][instance_id] = trajectory

    if not groups:
        raise RuntimeError("No completed ContextBench run records found")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contextbench = upstream_path("contextbench")
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(contextbench)
        if not existing_pythonpath
        else str(contextbench) + os.pathsep + existing_pythonpath
    )

    summary = []
    for (model_id, scaffold, repeat), by_instance in sorted(groups.items()):
        condition = "%s__%s__r%d" % (model_id, scaffold, repeat)
        stem = safe_name(condition)
        pred_path = output_dir / (stem + ".predictions.jsonl")
        metrics_path = output_dir / (stem + ".metrics.jsonl")
        with pred_path.open("w", encoding="utf-8") as handle:
            for instance_id in sorted(by_instance):
                handle.write(
                    json.dumps(
                        by_instance[instance_id],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        command = [
            str(evaluator_python),
            "-m",
            "contextbench.evaluate",
            "--gold",
            str(Path(args.gold).resolve()),
            "--pred",
            str(pred_path),
            "--out",
            str(metrics_path),
            "--cache",
            str(Path(args.cache_dir).resolve()),
        ]
        print("Condition:", condition, "instances=", len(by_instance))
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=contextbench, env=environment, check=True)
        summary.append(
            {
                "model_id": model_id,
                "scaffold": scaffold,
                "repeat": repeat,
                "instances": len(by_instance),
                "predictions": str(pred_path),
                "metrics": str(metrics_path),
            }
        )

    summary_path = output_dir / "conditions.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Condition summary:", summary_path)


if __name__ == "__main__":
    main()
