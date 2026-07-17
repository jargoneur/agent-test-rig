import json


class BaseScaffold:
    """Shared prompt protocol and state handling for scaffold variants."""

    strategy_name = "baseline"
    scaffold_context = "No additional scaffold-specific guidance."
    history_limit = 5

    def init_state(self, issue, tools):
        return {
            "files_read": [],
            "files_written": [],
            "tests_run": 0,
            "last_test_passed": None,
            "strategy": self.strategy_name,
        }

    def build_context(self, issue, tools, history, state):
        files_text = json.dumps(tools.list_files(), indent=2, ensure_ascii=False)
        history_text = json.dumps(
            history[-self.history_limit :],
            indent=2,
            ensure_ascii=False,
        )
        state_text = json.dumps(state, indent=2, ensure_ascii=False)

        return f"""
You are a coding agent working in a small Python repository.

Task:
{issue}

Scaffold-specific context:
{self.scaffold_context}

Repository files:
{files_text}

Current scaffold state:
{state_text}

Recent history:
{history_text}

Protocol:
- Choose exactly one action per step.
- Return exactly one valid JSON object.
- Do not use markdown fences, comments, triple-quoted strings, or explanatory text.
- JSON string values must escape newlines as \\n.
- Read an existing file before attempting to overwrite it.
- A write to an unread existing file will be blocked.
- Preserve existing public interfaces unless the task explicitly requires changing them.
- After changing code, run the tests before finishing.
- Finish only after tests have passed.

Allowed actions:
{{"action": "read_file", "path": "src/example.py"}}
{{"action": "write_file", "path": "src/example.py", "content": "complete file content with escaped newlines"}}
{{"action": "run_tests", "command": "pytest"}}
{{"action": "finish", "reason": "tests pass"}}

Choose the next action.
"""

    def update_state(self, state, action, observation):
        action_name = action.get("action")

        if action_name == "read_file":
            path = action.get("path")
            if path and path not in state["files_read"]:
                state["files_read"].append(path)

        if action_name == "write_file" and observation.get("type") == "write_success":
            path = action.get("path")
            if path and path not in state["files_written"]:
                state["files_written"].append(path)

        if action_name == "run_tests":
            state["tests_run"] += 1
            state["last_test_passed"] = observation.get("passed")

        return state

    def export_state(self, state):
        return state
