import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

from harness.model_adapter import OllamaModel
from harness.tools import FileTools
from harness.logger import JsonlLogger
from harness.task_loader import TaskLoader
from harness.workspace import prepare_workspace

from agents.simple_agent import SimpleAgent
from scaffolds import load_scaffold


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def run_single(task_id, model_name, scaffold_name, repeat, max_steps):
    task = TaskLoader().load(task_id)

    model = OllamaModel(model_name)
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
    )

    result = agent.run(task["issue"])
    state = result.get("scaffold_state", {})

    return {
        "task": task_id,
        "model": model_name,
        "scaffold": scaffold_name,
        "repeat": repeat,
        "success": result["final_tests_passed"],
        "final_tests_passed": result["final_tests_passed"],
        "agent_self_verified": result["agent_self_verified"],
        "termination_reason": result["termination_reason"],
        "parse_errors": result["parse_errors"],
        "execution_errors": result["execution_errors"],
        "repeated_valid_actions": result["repeated_valid_actions"],
        "steps": result["steps"],
        "files_read": json.dumps(state.get("files_read", []), ensure_ascii=False),
        "files_written": json.dumps(state.get("files_written", []), ensure_ascii=False),
        "agent_tests_run": state.get("tests_run"),
        "agent_last_test_passed": state.get("last_test_passed"),
        "strategy": state.get("strategy"),
        "workspace_path": workspace_path,
        "log_path": log_path,
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--scaffolds", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)

    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/batch_{timestamp}.csv"

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

                    print(
                        f"\n[{current}/{total}] "
                        f"task={task_id} "
                        f"model={model_name} "
                        f"scaffold={scaffold_name} "
                        f"repeat={repeat}"
                    )

                    try:
                        row = run_single(
                            task_id=task_id,
                            model_name=model_name,
                            scaffold_name=scaffold_name,
                            repeat=repeat,
                            max_steps=args.max_steps,
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
                            "| steps:",
                            row["steps"],
                            "| log:",
                            row["log_path"],
                        )

                    except Exception as error:
                        row = {
                            "task": task_id,
                            "model": model_name,
                            "scaffold": scaffold_name,
                            "repeat": repeat,
                            "success": False,
                            "final_tests_passed": False,
                            "agent_self_verified": False,
                            "termination_reason": "error",
                            "parse_errors": None,
                            "execution_errors": None,
                            "repeated_valid_actions": None,
                            "steps": None,
                            "files_read": "[]",
                            "files_written": "[]",
                            "agent_tests_run": None,
                            "agent_last_test_passed": None,
                            "strategy": None,
                            "workspace_path": None,
                            "log_path": None,
                            "error": str(error),
                        }

                        print("ERROR:", error)

                    rows.append(row)

    fieldnames = [
        "task",
        "model",
        "scaffold",
        "repeat",
        "success",
        "final_tests_passed",
        "agent_self_verified",
        "termination_reason",
        "parse_errors",
        "execution_errors",
        "repeated_valid_actions",
        "steps",
        "files_read",
        "files_written",
        "agent_tests_run",
        "agent_last_test_passed",
        "strategy",
        "workspace_path",
        "log_path",
        "error",
    ]

    with open(result_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\nBATCH FINISHED")
    print("Results:", result_path)


if __name__ == "__main__":
    main()
