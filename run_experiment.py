import argparse
import re
import sys
from datetime import datetime

from harness.logger import JsonlLogger
from harness.model_adapter import OllamaModel
from harness.reproducibility import (
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


DEFAULT_SEED = 12345
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 2048


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--scaffold", default="baseline")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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

    generation_options = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
    }

    task = TaskLoader().load(args.task)
    model = OllamaModel(
        args.model,
        options=generation_options,
    )
    model_metadata = model.runtime_metadata()
    scaffold = load_scaffold(args.scaffold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    workspace_name = safe_name(
        f"{task['id']}_{args.model}_{args.scaffold}_{timestamp}"
    )
    workspace_path = prepare_workspace(task, workspace_name)
    tools = FileTools(workspace_path)

    log_path = (
        f"logs/"
        f"{safe_name(task['id'])}_"
        f"{safe_name(args.scaffold)}_"
        f"{safe_name(args.model)}_"
        f"{timestamp}.jsonl"
    )
    manifest_path = log_path.removesuffix(".jsonl") + ".manifest.json"

    repository_metadata = git_metadata()
    manifest = {
        "schema_version": 1,
        "started_at": utc_now_iso(),
        "command": sys.argv,
        "status": "running",
        "task": task_metadata(task),
        "model": model_metadata,
        "scaffold": args.scaffold,
        "max_steps": args.max_steps,
        "run_seed": args.seed,
        "generation_options": generation_options,
        "repository": repository_metadata,
        "environment": environment_metadata(),
        "workspace_path": workspace_path,
        "log_path": log_path,
        "manifest_path": manifest_path,
    }
    write_json(manifest_path, manifest)

    logger = JsonlLogger(log_path)

    agent = SimpleAgent(
        model=model,
        tools=tools,
        logger=logger,
        scaffold=scaffold,
        max_steps=args.max_steps,
        expected_files=task.get("expected_files", []),
        run_seed=args.seed,
    )

    result = agent.run(task["issue"])

    manifest["finished_at"] = utc_now_iso()
    manifest["status"] = "finished"
    manifest["result"] = {
        "final_tests_passed": result["final_tests_passed"],
        "termination_reason": result["termination_reason"],
        "steps": result["steps"],
        "target_file_read": result["target_file_read"],
        "target_file_written": result["target_file_written"],
    }
    write_json(manifest_path, manifest)

    print("FINAL RESULT")
    print("Task:", task["id"])
    print("Model:", args.model)
    print("Model digest:", model_metadata.get("digest"))
    print("Ollama version:", model_metadata.get("ollama_version"))
    print("Scaffold:", args.scaffold)
    print("Seed:", args.seed)
    print("Generation options:", generation_options)
    print("Tests passed:", result["tests_passed"])
    print("Steps:", result["steps"])
    print("Target file read:", result["target_file_read"])
    print("Target file written:", result["target_file_written"])
    print("Write blocks:", result["write_blocks"])
    print("Invalid Python writes:", result["invalid_python_writes"])
    print("Workspace:", workspace_path)
    print("Log:", log_path)
    print("Manifest:", manifest_path)
    print("Final scaffold state:", result["scaffold_state"])


if __name__ == "__main__":
    main()
