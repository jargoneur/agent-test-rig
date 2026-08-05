from __future__ import annotations

import json
from typing import Any, Dict, List


class BaseScaffold:
    """Shared action protocol with overridable context-management hooks."""

    strategy_name = "baseline"
    scaffold_context = "No additional scaffold-specific context."
    history_limit = 5
    file_list_limit = 400

    def __init__(self, model=None, task=None, options=None):
        self.model = model
        self.task = dict(task or {})
        self.options = dict(options or {})
        self.auxiliary_calls: List[Dict[str, Any]] = []

    def bind(self, model=None, task=None, options=None):
        if model is not None:
            self.model = model
        if task is not None:
            self.task = dict(task)
        if options is not None:
            self.options = dict(options)
        return self

    def init_state(self, issue, tools):
        return {
            "files_read": [],
            "files_written": [],
            "searches": 0,
            "tests_run": 0,
            "last_test_passed": None,
            "strategy": self.strategy_name,
            "auxiliary_calls": [],
            "context_files": [],
            "context_spans": {},
        }

    def repository_context(self, issue, tools, history, state):
        files = tools.list_files(limit=self.file_list_limit)
        state["context_files"] = list(files)
        state["context_spans"] = {}
        return json.dumps(files, indent=2, ensure_ascii=False)

    def history_for_prompt(self, issue, tools, history, state):
        return history[-self.history_limit :]

    def scaffold_context_for_prompt(self, issue, tools, history, state):
        return self.scaffold_context

    def _test_command(self):
        return str(self.task.get("test_command") or "pytest")

    def build_context(self, issue, tools, history, state):
        files_text = self.repository_context(issue, tools, history, state)
        history_text = json.dumps(
            self.history_for_prompt(issue, tools, history, state),
            indent=2,
            ensure_ascii=False,
        )
        state_text = json.dumps(state, indent=2, ensure_ascii=False)
        scaffold_text = self.scaffold_context_for_prompt(
            issue,
            tools,
            history,
            state,
        )
        test_command = self._test_command()

        return f"""
You are a coding agent working in an existing software repository.

Task:
{issue}

Scaffold-specific context:
{scaffold_text}

Repository index:
{files_text}

Current scaffold state:
{state_text}

Managed interaction history:
{history_text}

Protocol:
- Choose exactly one action per step.
- Return exactly one valid JSON object.
- Do not use markdown fences, comments, or explanatory text outside the JSON object.
- JSON string values must escape literal newlines as \\n.
- Paths are relative to the repository root.
- Read an existing file before attempting to overwrite it.
- Preserve existing public interfaces unless the task explicitly requires changing them.
- Prefer narrow file ranges and targeted search over reading entire large files.
- Run the configured validation command after changing code when it is available.

Allowed actions:
{{"action": "list_files", "glob": "**/*.py", "limit": 200}}
{{"action": "search_text", "query": "symbol_or_text", "glob": "**/*.py", "limit": 50}}
{{"action": "read_file", "path": "src/example.py", "start_line": 1, "end_line": 200}}
{{"action": "write_file", "path": "src/example.py", "content": "complete file content with escaped newlines"}}
{{"action": "run_tests", "command": {json.dumps(test_command)}}}
{{"action": "finish", "reason": "implementation complete"}}

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

        if action_name == "search_text":
            state["searches"] += 1

        if action_name == "run_tests":
            state["tests_run"] += 1
            state["last_test_passed"] = observation.get("passed")

        state["auxiliary_calls"] = list(self.auxiliary_calls)
        return state

    def export_state(self, state):
        return state
