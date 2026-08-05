#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from harness.upstreams import upstream_path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect agent-rig trajectories and run the pinned ContextBench evaluator."
    )
    parser.add_argument("roots", nargs="+", help="Exported or worker result roots")
    parser.add_argument(
        "--gold",
        default="benchmarks/contextbench/gold.parquet",
    )
    parser.add_argument(
        "--pred",
        default="results/contextbench_predictions.jsonl",
    )
    parser.add_argument(
        "--out",
        default="results/contextbench_metrics.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/contextbench_evaluator",
    )
    args = parser.parse_args()

    trajectories = {}
    for root_value in args.roots:
        root = Path(root_value)
        candidates = root.rglob("*.context.json") if root.is_dir() else [root]
        for path in candidates:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            instance_id = value.get("instance_id")
            if not instance_id:
                continue
            key = (
                str(instance_id),
                str((value.get("agent_rig") or {}).get("task_id") or ""),
                str(path),
            )
            trajectories[key] = value

    if not trajectories:
        raise RuntimeError("No ContextBench trajectory files found")

    pred_path = Path(args.pred)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as handle:
        for key in sorted(trajectories):
            handle.write(
                json.dumps(
                    trajectories[key],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    contextbench = upstream_path("contextbench")
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(contextbench) if not existing else str(contextbench) + os.pathsep + existing
    )
    command = [
        sys.executable,
        "-m",
        "contextbench.evaluate",
        "--gold",
        str(Path(args.gold).resolve()),
        "--pred",
        str(pred_path.resolve()),
        "--out",
        str(Path(args.out).resolve()),
        "--cache-dir",
        str(Path(args.cache_dir).resolve()),
    ]
    print("Predictions:", pred_path)
    print("Trajectories:", len(trajectories))
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=contextbench, env=environment, check=True)
    print("Metrics:", args.out)


if __name__ == "__main__":
    main()
