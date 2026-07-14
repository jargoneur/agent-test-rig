import argparse
import re
from datetime import datetime

from harness.model_adapter import OllamaModel
from harness.tools import FileTools
from harness.logger import JsonlLogger
from harness.task_loader import TaskLoader

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
    tools = FileTools(task["repo_path"])
    scaffold = load_scaffold(args.scaffold)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    )

    result = agent.run(task["issue"])

    print("FINAL RESULT")
    print("Task:", task["id"])
    print("Model:", args.model)
    print("Scaffold:", args.scaffold)
    print("Tests passed:", result["tests_passed"])
    print("Steps:", result["steps"])
    print("Log:", log_path)
    print("Final scaffold state:", result["scaffold_state"])


if __name__ == "__main__":
    main()