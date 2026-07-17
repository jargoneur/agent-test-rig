import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from harness.logger import JsonlLogger
from harness.model_adapter import OllamaModel
from harness.reproducibility import (
    derive_run_seed,
    environment_metadata,
    git_metadata,
    task_metadata,
    utc_now_iso,
    write_json,
)
from harness.task_loader import TaskLoader
from harness.tools import FileTools
from harness.workspace import prepare_workspace

from agents.simple_agent import SimpleAgent
from scaffolds import load_scaffold


DEFAULT_BASE_SEED = 12345
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 2048


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def generation_options_from_args(args) -> dict:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }


def run_single(
    task_id,
    model_name,
    scaffold_name,
    repeat,
    max_steps,
    run_seed,
    generation_options,
    model_metadata,
    repository_metadata,
    manifest_path,
):
    task = TaskLoader().load(task_id)

    model = OllamaModel(
        model_name,
        options=generation_options,
    )
    scaffold = load_scaffold(scaffold_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workspace_name = safe_name(
        f"{task_id}_{model_name}_{scaffold_name}_r{repeat}_{timestamp}"
    )
    workspace_path = prepare_workspace(task, workspace_name)
    tools = FileTools(workspace_path)

    log_path = (
        f"logs/"
        f"{safe_name(task['id'])}_"
        f"{safe_name(scaffold_name)}_"
        f"{safe_name(model_name)}_"
        f"r{repeat}_"
        f"{timestamp}.jsonl"
    )

    logger = JsonlLogger(log_path)

    agent = SimpleAgent(
        model=model,
        tools=tools,
        logger=logger,
        scaffold=scaffold,
        max_steps=max_steps,
        expected_files=task.get("expected_files", []),
        run_seed=run_seed,
    )

    result = agent.run(task["issue"])
    state = result.get("scaffold_state", {})

    return {
        "task": task_id,
        "model": model_name,
        "model_digest": model_metadata.get("digest"),
        "ollama_version": model_metadata.get("ollama_version"),
        "scaffold": scaffold_name,
        "repeat": repeat,
        "run_seed": run_seed,
        "generation_options": json.dumps(
            generation_options,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "git_commit": repository_metadata.get("commit"),
        "git_dirty": repository_metadata.get("dirty"),
        "success": result["final_tests_passed"],
        "final_tests_passed": result["final_tests_passed"],
        "agent_self_verified": result["agent_self_verified"],
        "termination_reason": result["termination_reason"],
        "parse_errors": result["parse_errors"],
        "execution_errors": result["execution_errors"],
        "repeated_valid_actions": result["repeated_valid_actions"],
        "maximum_repeated_action_streak": result[
            "maximum_repeated_action_streak"
        ],
        "target_file_read": result["target_file_read"],
        "target_file_written": result["target_file_written"],
        "wrote_unread_file_attempts": result["wrote_unread_file_attempts"],
        "write_blocks": result["write_blocks"],
        "first_test_step": result["first_test_step"],
        "first_write_step": result["first_write_step"],
        "invalid_python_writes": result["invalid_python_writes"],
        "steps": result["steps"],
        "files_read": json.dumps(state.get("files_read", []), ensure_ascii=False),
        "files_written": json.dumps(
            state.get("files_written", []),
            ensure_ascii=False,
        ),
        "agent_tests_run": state.get("tests_run"),
        "agent_last_test_passed": state.get("last_test_passed"),
        "strategy": state.get("strategy"),
        "workspace_path": workspace_path,
        "log_path": log_path,
        "manifest_path": manifest_path,
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--scaffolds", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
    )

    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/batch_{timestamp}.csv"
    manifest_path = f"results/batch_{timestamp}.manifest.json"
    generation_options = generation_options_from_args(args)
    repository_metadata = git_metadata()

    loaded_tasks = {
        task_id: TaskLoader().load(task_id)
        for task_id in args.tasks
    }
    task_manifests = {
        task_id: task_metadata(task)
        for task_id, task in loaded_tasks.items()
    }

    model_manifests = {}
    for model_name in args.models:
        model = OllamaModel(
            model_name,
            options=generation_options,
        )
        model_manifests[model_name] = model.runtime_metadata()

    manifest = {
        "schema_version": 1,
        "started_at": utc_now_iso(),
        "command": sys.argv,
        "result_path": result_path,
        "manifest_path": manifest_path,
        "experiment": {
            "tasks": args.tasks,
            "models": args.models,
            "scaffolds": args.scaffolds,
            "repeats": args.repeats,
            "max_steps": args.max_steps,
            "base_seed": args.base_seed,
            "generation_options": generation_options,
            "seed_derivation": {
                "run_seed": (
                    "sha256(base_seed, task_id, model_name, repeat); "
                    "scaffold deliberately excluded"
                ),
                "step_seed": "run_seed + step - 1 modulo 2147483647",
            },
        },
        "repository": repository_metadata,
        "environment": environment_metadata(),
        "tasks": task_manifests,
        "models": model_manifests,
        "status": "running",
    }
    write_json(manifest_path, manifest)

    rows = []

    total = (
        len(args.tasks)
        * len(args.models)
        * len(args.scaffolds)
        * args.repeats
    )

    current = 0

    for task_id in args.tasks:
        for model_name in args.models:
            for scaffold_name in args.scaffolds:
                for repeat in range(1, args.repeats + 1):
                    current += 1
                    run_seed = derive_run_seed(
                        args.base_seed,
                        task_id,
                        model_name,
                        repeat,
                    )

                    print(
                        f"\n[{current}/{total}] "
                        f"task={task_id} "
                        f"model={model_name} "
                        f"scaffold={scaffold_name} "
                        f"repeat={repeat} "
                        f"seed={run_seed}"
                    )

                    try:
                        row = run_single(
                            task_id=task_id,
                            model_name=model_name,
                            scaffold_name=scaffold_name,
                            repeat=repeat,
                            max_steps=args.max_steps,
                            run_seed=run_seed,
                            generation_options=generation_options,
                            model_metadata=model_manifests[model_name],
                            repository_metadata=repository_metadata,
                            manifest_path=manifest_path,
                        )

                        print(
                            "success:",
                            row["success"],
                            "| verified:",
                            row["agent_self_verified"],
                            "| termination:",
                            row["termination_reason"],
                            "| parse errors:",
                            row["parse_errors"],
                            "| execution errors:",
                            row["execution_errors"],
                            "| repeated valid actions:",
                            row["repeated_valid_actions"],
                            "| write blocks:",
                            row["write_blocks"],
                            "| invalid Python writes:",
                            row["invalid_python_writes"],
                            "| steps:",
                            row["steps"],
                            "| log:",
                            row["log_path"],
                        )

                    except Exception as error:
                        model_metadata = model_manifests.get(model_name, {})
                        row = {
                            "task": task_id,
                            "model": model_name,
                            "model_digest": model_metadata.get("digest"),
                            "ollama_version": model_metadata.get("ollama_version"),
                            "scaffold": scaffold_name,
                            "repeat": repeat,
                            "run_seed": run_seed,
                            "generation_options": json.dumps(
                                generation_options,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "git_commit": repository_metadata.get("commit"),
                            "git_dirty": repository_metadata.get("dirty"),
                            "success": False,
                            "final_tests_passed": False,
                            "agent_self_verified": False,
                            "termination_reason": "error",
                            "parse_errors": None,
                            "execution_errors": None,
                            "repeated_valid_actions": None,
                            "maximum_repeated_action_streak": None,
                            "target_file_read": None,
                            "target_file_written": None,
                            "wrote_unread_file_attempts": None,
                            "write_blocks": None,
                            "first_test_step": None,
                            "first_write_step": None,
                            "invalid_python_writes": None,
                            "steps": None,
                            "files_read": "[]",
                            "files_written": "[]",
                            "agent_tests_run": None,
                            "agent_last_test_passed": None,
                            "strategy": None,
                            "workspace_path": None,
                            "log_path": None,
                            "manifest_path": manifest_path,
                            "error": str(error),
                        }

                        print("ERROR:", error)

                    rows.append(row)

    fieldnames = [
        "task",
        "model",
        "model_digest",
        "ollama_version",
        "scaffold",
        "repeat",
        "run_seed",
        "generation_options",
        "git_commit",
        "git_dirty",
        "success",
        "final_tests_passed",
        "agent_self_verified",
        "termination_reason",
        "parse_errors",
        "execution_errors",
        "repeated_valid_actions",
        "maximum_repeated_action_streak",
        "target_file_read",
        "target_file_written",
        "wrote_unread_file_attempts",
        "write_blocks",
        "first_test_step",
        "first_write_step",
        "invalid_python_writes",
        "steps",
        "files_read",
        "files_written",
        "agent_tests_run",
        "agent_last_test_passed",
        "strategy",
        "workspace_path",
        "log_path",
        "manifest_path",
        "error",
    ]

    with open(result_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    manifest["finished_at"] = utc_now_iso()
    manifest["status"] = "finished"
    manifest["row_count"] = len(rows)
    manifest["successful_runs"] = sum(bool(row["success"]) for row in rows)
    write_json(manifest_path, manifest)

    print("\nBATCH FINISHED")
    print("Results:", result_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
