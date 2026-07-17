import ast
import json
import re
from pathlib import Path

from harness.reproducibility import derive_step_seed, sha256_text


class SimpleAgent:
    def __init__(
        self,
        model,
        tools,
        logger,
        scaffold,
        max_steps=8,
        expected_files=None,
        run_seed=None,
    ):
        self.model = model
        self.tools = tools
        self.logger = logger
        self.scaffold = scaffold
        self.max_steps = max_steps
        self.expected_files = set(expected_files or [])
        self.run_seed = run_seed
        self.files_read = set()

    def parse_action(self, response):
        response = response.strip()

        try:
            action = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON object found in response: {response}")
            action = json.loads(match.group(0))

        if isinstance(action, dict) and isinstance(action.get("action"), dict):
            action = action["action"]

        if not isinstance(action, dict):
            raise ValueError(f"Parsed action is not a dictionary: {action}")
        if "action" not in action:
            raise ValueError(f"Parsed JSON has no action field: {action}")

        return action

    def execute_action(self, action):
        name = action.get("action")

        if name == "read_file":
            path = action["path"]
            content = self.tools.read_file(path)
            self.files_read.add(path)
            return {"type": "file_content", "path": path, "content": content}

        if name == "write_file":
            path = action["path"]
            full_path = self.tools._safe_path(path)
            existed_before = full_path.exists()

            if existed_before and path not in self.files_read:
                return {
                    "type": "write_blocked",
                    "path": path,
                    "reason": "existing_file_not_read",
                }

            self.tools.write_file(path, action["content"])
            return {
                "type": "write_success",
                "path": path,
                "existed_before": existed_before,
            }

        if name == "run_tests":
            command = action.get("command", "pytest")
            result = self.tools.run_tests(command)
            return {
                "type": "test_result",
                "passed": result["passed"],
                "stdout": result["stdout"][-4000:],
                "stderr": result["stderr"][-4000:],
            }

        if name == "finish":
            return {"type": "finished", "reason": action.get("reason", "")}

        raise ValueError(f"Unknown action: {name}")

    @staticmethod
    def action_signature(action):
        return json.dumps(action, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def python_content_is_valid(path, content):
        if Path(path).suffix != ".py":
            return None

        try:
            ast.parse(content)
            return True
        except SyntaxError:
            return False

    def run(self, issue):
        reset_result = self.tools.reset_repo()
        if not reset_result.get("success"):
            raise RuntimeError(
                "Repository reset failed: "
                f"{reset_result.get('reset_stderr', '')} "
                f"{reset_result.get('clean_stderr', '')}"
            )

        history = []
        state = self.scaffold.init_state(issue, self.tools)
        termination_reason = "max_steps"
        agent_self_verified = False
        repeated_valid_actions = 0
        parse_errors = 0
        execution_errors = 0
        write_blocks = 0
        wrote_unread_file_attempts = 0
        invalid_python_writes = 0
        first_test_step = None
        first_write_step = None
        maximum_repeated_action_streak = 0
        repeated_action_streak = 0
        target_file_read = False
        target_file_written = False
        previous_valid_signature = None

        self.logger.log(
            "run_started",
            {
                "issue": issue,
                "max_steps": self.max_steps,
                "scaffold": self.scaffold.__class__.__name__,
                "expected_files": sorted(self.expected_files),
                "run_seed": self.run_seed,
                "model": getattr(self.model, "model_name", None),
                "generation_options": getattr(self.model, "options", {}),
            },
        )

        for step in range(1, self.max_steps + 1):
            context = self.scaffold.build_context(issue, self.tools, history, state)
            step_seed = (
                derive_step_seed(self.run_seed, step)
                if self.run_seed is not None
                else None
            )
            effective_options = self.model.effective_options(step_seed)

            self.logger.log(
                "prompt_built",
                {
                    "step": step,
                    "prompt": context,
                    "prompt_sha256": sha256_text(context),
                    "prompt_characters": len(context),
                    "step_seed": step_seed,
                    "generation_options": effective_options,
                },
            )

            response = self.model.generate(context, seed=step_seed)

            self.logger.log(
                "model_response",
                {
                    "step": step,
                    "response": response,
                    "step_seed": step_seed,
                    "generation_metadata": self.model.last_generation_metadata,
                },
            )

            parsed_successfully = False
            try:
                action = self.parse_action(response)
                parsed_successfully = True
            except Exception as error:
                parse_errors += 1
                action = {"action": "parse_error"}
                observation = {"type": "parse_error", "message": str(error)}

            if parsed_successfully:
                signature = self.action_signature(action)
                if signature == previous_valid_signature:
                    repeated_valid_actions += 1
                    repeated_action_streak += 1
                else:
                    repeated_action_streak = 0
                maximum_repeated_action_streak = max(
                    maximum_repeated_action_streak,
                    repeated_action_streak,
                )
                previous_valid_signature = signature

                action_name = action.get("action")
                path = action.get("path")

                if action_name == "read_file" and path in self.expected_files:
                    target_file_read = True

                if action_name == "write_file":
                    if first_write_step is None:
                        first_write_step = step

                    full_path = self.tools._safe_path(path)
                    if full_path.exists() and path not in self.files_read:
                        wrote_unread_file_attempts += 1

                    syntax_valid = self.python_content_is_valid(
                        path,
                        action.get("content", ""),
                    )
                    if syntax_valid is False:
                        invalid_python_writes += 1

                if action_name == "run_tests" and first_test_step is None:
                    first_test_step = step

                try:
                    observation = self.execute_action(action)
                except Exception as error:
                    execution_errors += 1
                    observation = {"type": "execution_error", "message": str(error)}

                if observation.get("type") == "write_blocked":
                    write_blocks += 1

                if (
                    observation.get("type") == "write_success"
                    and path in self.expected_files
                ):
                    target_file_written = True

            state = self.scaffold.update_state(state, action, observation)
            history.append({"step": step, "action": action, "observation": observation})

            self.logger.log(
                "step_finished",
                {
                    "step": step,
                    "step_seed": step_seed,
                    "action": action,
                    "observation": observation,
                    "scaffold_state": self.scaffold.export_state(state),
                    "parse_errors": parse_errors,
                    "execution_errors": execution_errors,
                    "repeated_valid_actions": repeated_valid_actions,
                    "write_blocks": write_blocks,
                    "wrote_unread_file_attempts": wrote_unread_file_attempts,
                    "invalid_python_writes": invalid_python_writes,
                    "maximum_repeated_action_streak": maximum_repeated_action_streak,
                },
            )

            if observation.get("type") == "finished":
                termination_reason = "agent_finish"
                break

            if observation.get("type") == "test_result" and observation.get("passed") is True:
                agent_self_verified = True
                termination_reason = "agent_tests_passed"
                self.logger.log("auto_finish", {"step": step, "reason": "tests passed"})
                break

        final_tests = self.tools.run_tests("pytest")

        diagnostics = {
            "target_file_read": target_file_read,
            "target_file_written": target_file_written,
            "wrote_unread_file_attempts": wrote_unread_file_attempts,
            "write_blocks": write_blocks,
            "first_test_step": first_test_step,
            "first_write_step": first_write_step,
            "invalid_python_writes": invalid_python_writes,
            "maximum_repeated_action_streak": maximum_repeated_action_streak,
        }

        self.logger.log(
            "run_finished",
            {
                "run_seed": self.run_seed,
                "final_tests_passed": final_tests["passed"],
                "agent_self_verified": agent_self_verified,
                "termination_reason": termination_reason,
                "parse_errors": parse_errors,
                "execution_errors": execution_errors,
                "repeated_valid_actions": repeated_valid_actions,
                **diagnostics,
                "stdout": final_tests["stdout"][-4000:],
                "stderr": final_tests["stderr"][-4000:],
            },
        )

        return {
            "tests_passed": final_tests["passed"],
            "final_tests_passed": final_tests["passed"],
            "agent_self_verified": agent_self_verified,
            "termination_reason": termination_reason,
            "parse_errors": parse_errors,
            "execution_errors": execution_errors,
            "repeated_valid_actions": repeated_valid_actions,
            "run_seed": self.run_seed,
            **diagnostics,
            "steps": len(history),
            "history": history,
            "scaffold_state": self.scaffold.export_state(state),
        }
