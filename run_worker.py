from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

from agents.simple_agent import SimpleAgent
from harness.experiment_jobs import (
    atomic_write_json,
    load_jobs,
    result_file,
    result_is_complete,
    shard_matches,
)
from harness.logger import JsonlLogger
from harness.model_adapter import create_model
from harness.reproducibility import (
    environment_metadata,
    git_metadata,
    task_metadata,
    utc_now_iso,
)
from harness.task_loader import TaskLoader
from harness.tools import FileTools
from harness.workspace import prepare_workspace
from scaffolds import load_scaffold


def safe_name(value: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def worker_metadata(worker_id: str) -> dict:
    metadata = environment_metadata()
    metadata.update(
        {
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "repository": git_metadata(),
        }
    )
    return metadata


def execute_job(job: Dict[str, Any], results_root: Path, worker_id: str) -> dict:
    run_id = job["run_id"]
    task = TaskLoader().load(job["task_id"])
    model = create_model(
        job["model"],
        options=job["generation_options"],
    )
    model_metadata = model.runtime_metadata()
    scaffold = load_scaffold(job["scaffold"])

    workspace_name = safe_name(run_id)
    workspace_path = prepare_workspace(task, workspace_name)
    tools = FileTools(workspace_path)

    log_path = results_root / "logs" / (run_id + ".jsonl")
    logger = JsonlLogger(str(log_path))
    started_at = utc_now_iso()

    agent = SimpleAgent(
        model=model,
        tools=tools,
        logger=logger,
        scaffold=scaffold,
        max_steps=int(job["max_steps"]),
        expected_files=task.get("expected_files", []),
        run_seed=int(job["run_seed"]),
    )
    result = agent.run(task["issue"])

    return {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_id,
        "job": job,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "worker": worker_metadata(worker_id),
        "task": task_metadata(task),
        "model_runtime": model_metadata,
        "log_path": str(log_path),
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run resumable jobs from a distributed experiment manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", default="distributed_results")
    parser.add_argument("--worker-id", default=socket.gethostname())
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--resource-class", action="append")
    parser.add_argument("--wave", type=int, action="append")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--pause-file", default="PAUSE")
    parser.add_argument("--keep-workspaces", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args.manifest)
    completed = 0
    attempted = 0

    print("Worker:", args.worker_id)
    print("Jobs in manifest:", len(jobs))
    print("Results root:", results_root)

    for job in jobs:
        if Path(args.pause_file).exists():
            print("Pause file detected; stopping before next job.")
            break
        if args.max_jobs is not None and attempted >= args.max_jobs:
            break
        if not shard_matches(job["run_id"], args.shard_index, args.shard_count):
            continue
        if args.model_id and job["model"]["id"] not in args.model_id:
            continue
        resource_class = (job.get("requirements") or {}).get("resource_class")
        if args.resource_class and resource_class not in args.resource_class:
            continue
        if args.wave and int(job.get("wave", 0)) not in args.wave:
            continue

        output_path = result_file(results_root, job["run_id"])
        failed_path = results_root / "failed" / (job["run_id"] + ".json")
        if result_is_complete(output_path, job["run_id"]):
            continue
        if failed_path.exists() and not args.retry_failed:
            continue

        attempted += 1
        print(
            "[%d] %s task=%s model=%s scaffold=%s repeat=%s"
            % (
                attempted,
                job["run_id"],
                job["task_id"],
                job["model"]["id"],
                job["scaffold"],
                job["repeat"],
            )
        )

        try:
            record = execute_job(job, results_root, args.worker_id)
            atomic_write_json(output_path, record)
            if failed_path.exists():
                failed_path.unlink()
            completed += 1
            print("completed:", job["run_id"])
        except KeyboardInterrupt:
            print("Interrupted; current job was not marked complete.")
            raise
        except Exception as error:
            failure = {
                "schema_version": 1,
                "status": "failed",
                "run_id": job["run_id"],
                "job": job,
                "failed_at": utc_now_iso(),
                "worker": worker_metadata(args.worker_id),
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            atomic_write_json(failed_path, failure)
            print("failed:", error, file=sys.stderr)
        finally:
            if not args.keep_workspaces:
                workspace = Path(".workspaces") / safe_name(job["run_id"])
                if workspace.exists():
                    shutil.rmtree(workspace, ignore_errors=True)

    print("Attempted:", attempted)
    print("Completed now:", completed)


if __name__ == "__main__":
    main()
