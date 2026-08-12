from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from agents.simple_agent import SimpleAgent
from harness.contextbench_trajectory import build_contextbench_trajectory
from harness.experiment_jobs import (
    atomic_write_json,
    load_jobs,
    result_file,
    result_is_complete,
    shard_matches,
)
from harness.logger import JsonlLogger
from harness.model_adapter import create_model
from harness.outcomes import (
    TerminalRunError,
    completed_outcome,
    error_outcome,
)
from harness.reproducibility import (
    environment_metadata,
    git_metadata,
    task_metadata,
    utc_now_iso,
)
from harness.resource_policy import ResourcePolicy, ResourceYieldRequested
from harness.task_loader import TaskLoader
from harness.tools import FileTools
from harness.workspace import WORKSPACES_ROOT, prepare_workspace
from scaffolds import load_scaffold


def safe_name(value: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def cleanup_workspace(run_id: str) -> None:
    workspace = WORKSPACES_ROOT / safe_name(run_id)
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)


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


def load_worker_config(path: str) -> Dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Worker config must be a YAML mapping")
    return value


def execute_job(
    job: Dict[str, Any],
    results_root: Path,
    worker_id: str,
    resource_policy: Optional[ResourcePolicy] = None,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    if resource_policy is not None:
        resource_policy.checkpoint("before_job_setup")

    run_id = job["run_id"]
    task = TaskLoader().load(job["task_id"])
    runtime_model_spec = dict(job["model"])
    runtime_model_spec.update(dict(runtime_overrides or {}))
    model = create_model(
        runtime_model_spec,
        options=job["generation_options"],
        timeout=int(job["model_timeout_seconds"]),
    )
    model_metadata = model.runtime_metadata()
    workspace_name = safe_name(run_id)
    workspace_path = prepare_workspace(task, workspace_name)
    tools = FileTools(
        workspace_path,
        test_timeout=int(task.get("test_timeout", 300)),
    )

    log_path = results_root / "logs" / (run_id + ".jsonl")
    logger = JsonlLogger(str(log_path))
    started_at = utc_now_iso()
    scaffold_options = dict(job.get("scaffold_options") or {})
    scaffold_options["_run_seed"] = int(job["run_seed"])
    scaffold_options["_resource_policy"] = resource_policy
    scaffold = load_scaffold(
        job["scaffold"],
        model=model,
        task=task,
        options=scaffold_options,
    )

    model_patch = ""
    terminal_error = None
    try:
        agent = SimpleAgent(
            model=model,
            tools=tools,
            logger=logger,
            scaffold=scaffold,
            max_steps=int(job["max_steps"]),
            expected_files=task.get("expected_files", []),
            run_seed=int(job["run_seed"]),
            resource_policy=resource_policy,
            test_command=task.get("test_command", "pytest"),
            run_final_tests=task.get("run_final_tests", True),
        )
        result = agent.run(task["issue"])
        model_patch = tools.git_diff()
        outcome = completed_outcome(result, model_patch)
    except TerminalRunError as error:
        terminal_error = error
        try:
            model_patch = tools.git_diff()
        except Exception:
            model_patch = ""
        result = {
            "tests_passed": None,
            "final_tests_passed": None,
            "agent_self_verified": False,
            "termination_reason": error.kind,
            "terminal_error": error.as_dict(),
            "run_seed": int(job["run_seed"]),
            "steps": None,
            "history": [],
            "scaffold_state": {
                "strategy": job["scaffold"],
                "context_files": [],
                "context_spans": {},
            },
        }
        outcome = error_outcome(error, model_patch)
        logger.log("terminal_outcome", outcome)

    contextbench_path = None
    if task.get("source") == "contextbench":
        trajectory = build_contextbench_trajectory(
            task,
            result,
            model_patch,
            job=job,
        )
        contextbench_path = (
            results_root / "contextbench" / (run_id + ".context.json")
        )
        atomic_write_json(contextbench_path, trajectory)

    return {
        "schema_version": 3,
        "status": "completed",
        "run_id": run_id,
        "job": job,
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "worker": worker_metadata(worker_id),
        "resource_policy": (
            resource_policy.metadata() if resource_policy is not None else None
        ),
        "task": task_metadata(task),
        "model_runtime": model_metadata,
        "runtime_overrides": dict(runtime_overrides or {}),
        "log_path": str(log_path),
        "contextbench_trajectory_path": (
            str(contextbench_path) if contextbench_path is not None else None
        ),
        "model_patch": model_patch,
        "result": result,
        "outcome": outcome,
        "terminal_error": (
            terminal_error.as_dict() if terminal_error is not None else None
        ),
    }


def resolve_standalone_resource_policy(args) -> ResourcePolicy:
    if bool(args.worker_config) == bool(args.local_exclusive):
        raise ValueError(
            "Declare exactly one of --worker-config or --local-exclusive"
        )

    if args.worker_config:
        config = load_worker_config(args.worker_config)
        return ResourcePolicy.from_worker_config(
            config,
            pause_file=args.pause_file,
        )

    return ResourcePolicy(
        {
            "shared_resource": False,
            "mode": "local_exclusive",
            "pause_file": args.pause_file,
        }
    )


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
    parser.add_argument(
        "--worker-config",
        help="Worker YAML containing an explicit resource_policy declaration",
    )
    parser.add_argument(
        "--local-exclusive",
        action="store_true",
        help="Explicitly declare that this direct worker uses only local exclusive compute",
    )
    args = parser.parse_args()

    resource_policy = resolve_standalone_resource_policy(args)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args.manifest)
    completed = 0
    attempted = 0

    print("Worker:", args.worker_id)
    print("Jobs in manifest:", len(jobs))
    print("Results root:", results_root)
    print("Resource policy:", resource_policy.metadata())

    try:
        for job in jobs:
            try:
                resource_policy.checkpoint("before_job_selection")
            except ResourceYieldRequested as error:
                print("Yielding shared resource:", error)
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

            yielded = False
            try:
                record = execute_job(
                    job,
                    results_root,
                    args.worker_id,
                    resource_policy=resource_policy,
                )
                atomic_write_json(output_path, record)
                if failed_path.exists():
                    failed_path.unlink()
                completed += 1
                print("completed:", job["run_id"])
            except ResourceYieldRequested as error:
                yielded = True
                print("Yielding during job; incomplete run will be retried:", error)
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
                    "resource_policy": resource_policy.metadata(),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(failed_path, failure)
                print("failed:", error, file=sys.stderr)
            finally:
                if not args.keep_workspaces:
                    cleanup_workspace(job["run_id"])

            if yielded:
                break
    finally:
        release_code = resource_policy.release()
        if release_code not in (None, 0):
            print("Resource release command returned:", release_code)

    print("Attempted:", attempted)
    print("Completed now:", completed)


if __name__ == "__main__":
    main()
