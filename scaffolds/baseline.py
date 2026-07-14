import json


class BaselineScaffold:
    def init_state(self, issue, tools):
        return {
            "files_read": [],
            "files_written": [],
            "tests_run": 0,
            "last_test_passed": None,
        }

    def build_context(self, issue, tools, history, state):
        files = tools.list_files()

        recent_history = history[-5:]
        history_text = json.dumps(recent_history, indent=2, ensure_ascii=False)
        state_text = json.dumps(state, indent=2, ensure_ascii=False)

        return f"""
You are a coding agent working in a small Python repository.

Your task:
{issue}

Repository files:
{files}

Current scaffold state:
{state_text}

Recent history:
{history_text}

You may use exactly one action per step.

Return ONLY valid JSON. No markdown. No explanation.

Allowed actions:

1. Read a file:
{{"action": "read_file", "path": "src/example.py"}}

2. Write a complete file:
{{"action": "write_file", "path": "src/example.py", "content": "complete file content here"}}

3. Run tests:
{{"action": "run_tests", "command": "pytest"}}

4. Finish only when tests pass:
{{"action": "finish", "reason": "tests pass"}}

Choose the next action.
"""

    def update_state(self, state, action, observation):
        action_name = action.get("action")

        if action_name == "read_file":
            path = action.get("path")
            if path and path not in state["files_read"]:
                state["files_read"].append(path)

        if action_name == "write_file":
            path = action.get("path")
            if path and path not in state["files_written"]:
                state["files_written"].append(path)

        if action_name == "run_tests":
            state["tests_run"] += 1
            state["last_test_passed"] = observation.get("passed")

        return state

    def export_state(self, state):
        return state
Scaffold = BaselineScaffold