import argparse
import re
from datetime import datetime

from harness.model_adapter import OllamaModel
from harness.tools import FileTools
from harness.logger import JsonlLogger
from harness.task_loader import TaskLoader
from harness.workspace import prepare_workspace

from agents.simple_agent import SimpleAgent
from scaffolds import load_scaffold


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--scaffold", default="baseline")
    parser.add_argument("--max-steps", type=int, default=8)

    args = parser.parse_args()

    task = TaskLoader().load(args.task)
    model = OllamaModel(args.model)
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

    logger = JsonlLogger(log_path)

    agent = SimpleAgent(
        model=model,
        tools=tools,
        logger=logger,
        scaffold=scaffold,
        max_steps=args.max_steps,
        expected_files=task.get("expected_files", []),
    )

    result = agent.run(task["issue"])

    print("FINAL RESULT")
    print("Task:", task["id"])
    print("Model:", args.model)
    print("Scaffold:", args.scaffold)
    print("Tests passed:", result["tests_passed"])
    print("Steps:", result["steps"])
    print("Target file read:", result["target_file_read"])
    print("Target file written:", result["target_file_written"])
    print("Write blocks:", result["write_blocks"])
    print("Invalid Python writes:", result["invalid_python_writes"])
    print("Workspace:", workspace_path)
    print("Log:", log_path)
    print("Final scaffold state:", result["scaffold_state"])


if __name__ == "__main__":
    main()
