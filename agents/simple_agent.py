import json
import re


class SimpleAgent:
    def __init__(self, model, tools, logger, scaffold, max_steps=8):
        self.model = model
        self.tools = tools
        self.logger = logger
        self.scaffold = scaffold
        self.max_steps = max_steps

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
            content = self.tools.read_file(action["path"])
            return {"type": "file_content", "path": action["path"], "content": content}

        if name == "write_file":
            self.tools.write_file(action["path"], action["content"])
            return {"type": "write_success", "path": action["path"]}

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

        return {"type": "error", "message": f"Unknown action: {name}"}

    @staticmethod
    def action_signature(action):
        return json.dumps(action, sort_keys=True, ensure_ascii=False)

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
        repeated_actions = 0
        previous_signature = None

        self.logger.log("run_started", {
            "issue": issue,
            "max_steps": self.max_steps,
            "scaffold": self.scaffold.__class__.__name__,
        })

        for step in range(1, self.max_steps + 1):
            context = self.scaffold.build_context(issue, self.tools, history, state)
            response = self.model.generate(context)

            self.logger.log("model_response", {"step": step, "response": response})

            try:
                action = self.parse_action(response)
                observation = self.execute_action(action)
            except Exception as error:
                action = {"action": "parse_or_execution_error"}
                observation = {"type": "error", "message": str(error)}

            signature = self.action_signature(action)
            if signature == previous_signature:
                repeated_actions += 1
            previous_signature = signature

            state = self.scaffold.update_state(state, action, observation)
            history.append({"step": step, "action": action, "observation": observation})

            self.logger.log("step_finished", {
                "step": step,
                "action": action,
                "observation": observation,
                "scaffold_state": self.scaffold.export_state(state),
                "repeated_actions": repeated_actions,
            })

            if observation.get("type") == "finished":
                termination_reason = "agent_finish"
                break

            if observation.get("type") == "test_result" and observation.get("passed") is True:
                agent_self_verified = True
                termination_reason = "agent_tests_passed"
                self.logger.log("auto_finish", {"step": step, "reason": "tests passed"})
                break

        final_tests = self.tools.run_tests("pytest")

        self.logger.log("run_finished", {
            "final_tests_passed": final_tests["passed"],
            "agent_self_verified": agent_self_verified,
            "termination_reason": termination_reason,
            "repeated_actions": repeated_actions,
            "stdout": final_tests["stdout"][-4000:],
            "stderr": final_tests["stderr"][-4000:],
        })

        return {
            "tests_passed": final_tests["passed"],
            "final_tests_passed": final_tests["passed"],
            "agent_self_verified": agent_self_verified,
            "termination_reason": termination_reason,
            "repeated_actions": repeated_actions,
            "steps": len(history),
            "history": history,
            "scaffold_state": self.scaffold.export_state(state),
        }
