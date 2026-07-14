import os

from harness.model_adapter import OllamaModel
from harness.tools import FileTools
from harness.logger import JsonlLogger
from harness.task_loader import TaskLoader

from agents.simple_agent import SimpleAgent
from scaffolds.baseline import BaselineScaffold


def main():
    if "OLLAMA_HOST" not in os.environ:
        print("Warning: OLLAMA_HOST is not set.")

    task = TaskLoader().load("demo_add")

    model = OllamaModel("qwen2.5-coder:3b")
    tools = FileTools(task["repo_path"])
    logger = JsonlLogger(f"logs/{task['id']}_baseline_qwen3b.jsonl")
    scaffold = BaselineScaffold()

    agent = SimpleAgent(
        model=model,
        tools=tools,
        logger=logger,
        scaffold=scaffold,
        max_steps=8,
    )

    result = agent.run(task["issue"])

    print("FINAL RESULT:")
    print("Task:", task["id"])
    print("Tests passed:", result["tests_passed"])
    print("Steps:", result["steps"])

    print("\nExpected files:")
    print(task["expected_files"])

    print("\nFinal scaffold state:")
    print(result["scaffold_state"])


if __name__ == "__main__":
    main()