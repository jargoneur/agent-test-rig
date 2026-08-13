#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

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


def outcome_row(
    record: Dict[str, Any],
    trajectory: Dict[str, Any],
) -> Dict[str, Any]:
    outcome = dict(record.get("outcome") or {})
    return {
        "instance_id": trajectory["instance_id"],
        "run_id": record.get("run_id"),
        "outcome": outcome,
        "outcome_kind": str(outcome.get("kind") or "missing_outcome_metadata"),
        "usable_result": bool(outcome.get("usable_result", False)),
        "terminal_error": record.get("terminal_error"),
        "model_patch_present": bool(
            (trajectory.get("agent_rig") or {}).get("model_patch_present")
        ),
    }


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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

    # resolve() follows uv-managed symlinks to the base interpreter and drops
    # the isolated evaluator environment.
    evaluator_python = Path(os.path.abspath(Path(args.evaluator_python).expanduser()))
    if not evaluator_python.is_file():
        raise FileNotFoundError(
            "ContextBench evaluator Python is missing: %s. Run scripts/bootstrap.sh"
            % evaluator_python
        )

    groups = defaultdict(dict)
    outcome_groups = defaultdict(dict)
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
            row = outcome_row(record, trajectory)
            existing_outcome = outcome_groups[key].get(instance_id)
            if existing_outcome is not None and existing_outcome != row:
                raise RuntimeError(
                    "Conflicting completed outcomes for %s in %s"
                    % (instance_id, key)
                )
            outcome_groups[key][instance_id] = row

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
        outcomes_path = output_dir / (stem + ".outcomes.jsonl")
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

        by_outcome = outcome_groups[(model_id, scaffold, repeat)]
        with outcomes_path.open("w", encoding="utf-8") as handle:
            for instance_id in sorted(by_outcome):
                handle.write(
                    json.dumps(
                        by_outcome[instance_id],
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + chr(10)
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
        metric_rows = read_jsonl(metrics_path)
        outcome_counts = Counter(
            row["outcome_kind"] for row in by_outcome.values()
        )
        evaluator_errors = Counter(
            str(row["error"])
            for row in metric_rows
            if row.get("error")
        )
        summary.append(
            {
                "model_id": model_id,
                "scaffold": scaffold,
                "repeat": repeat,
                "instances": len(by_instance),
                "usable_outcomes": sum(
                    1
                    for row in by_outcome.values()
                    if row["usable_result"]
                ),
                "outcome_counts": dict(sorted(outcome_counts.items())),
                "evaluator_error_counts": dict(
                    sorted(evaluator_errors.items())
                ),
                "predictions": str(pred_path),
                "outcomes": str(outcomes_path),
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
