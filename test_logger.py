from harness.logger import JsonlLogger


def main():
    logger = JsonlLogger("logs/test_run.jsonl")

    logger.log("run_started", {
        "task": "demo_task",
        "model": "qwen2.5-coder:3b",
        "scaffold": "baseline"
    })

    logger.log("tool_call", {
        "tool": "read_file",
        "path": "src/calculator.py"
    })

    logger.log("run_finished", {
        "passed": False
    })

    print("Logger test finished. Check logs/test_run.jsonl")


if __name__ == "__main__":
    main()