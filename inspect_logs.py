import json
import glob


def inspect_log(path):
    steps = []
    final = None

    with open(path, encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)

            if entry["event_type"] == "step_finished":
                steps.append(entry["data"])

            if entry["event_type"] == "run_finished":
                final = entry["data"]

    last_state = steps[-1].get("scaffold_state", {}) if steps else {}

    return {
        "log": path,
        "steps": len(steps),
        "final_tests_passed": final["tests_passed"] if final else None,
        "agent_tests_run": last_state.get("tests_run"),
        "agent_last_test_passed": last_state.get("last_test_passed"),
        "files_read": last_state.get("files_read", []),
        "files_written": last_state.get("files_written", []),
        "strategy": last_state.get("strategy"),
    }


def main():
    paths = sorted(glob.glob("logs/*.jsonl"))

    for path in paths:
        result = inspect_log(path)

        print("\nLOG:", result["log"])
        print("final_tests_passed:", result["final_tests_passed"])
        print("steps:", result["steps"])
        print("strategy:", result["strategy"])
        print("files_read:", result["files_read"])
        print("files_written:", result["files_written"])
        print("agent_tests_run:", result["agent_tests_run"])
        print("agent_last_test_passed:", result["agent_last_test_passed"])


if __name__ == "__main__":
    main()