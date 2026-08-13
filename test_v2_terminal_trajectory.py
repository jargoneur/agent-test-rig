from __future__ import annotations

import json

import pytest

from agents.simple_agent import SimpleAgent
from harness.logger import JsonlLogger
from harness.outcomes import ModelTerminalError
from harness.tools import FileTools
from scaffolds.no_scaffold import NoScaffold


class LimitAfterOneActionModel:
    model_name = "bounded-model"

    def __init__(self):
        self.options = {"num_predict": 8192}
        self.last_generation_metadata = {}
        self.calls = 0

    def effective_options(self, seed=None):
        return {**self.options, "seed": seed}

    def generate_action(self, prompt, seed=None, timeout=None):
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "action": "read_file",
                    "path": "module.py",
                    "start_line": 1,
                    "end_line": 1,
                }
            )
        raise ModelTerminalError(
            "model_output_limit",
            "limit reached",
            stage="model_generation",
            details={"output_token_limit": 8192},
        )


def test_terminal_model_outcome_carries_completed_trajectory(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    # FileTools.reset_repo requires a real clean repository.
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    model = LimitAfterOneActionModel()
    scaffold = NoScaffold(model=model, task={"tests_enabled": False}, options={})
    agent = SimpleAgent(
        model=model,
        tools=FileTools(str(tmp_path)),
        logger=JsonlLogger(str(tmp_path / "run.jsonl")),
        scaffold=scaffold,
        max_steps=3,
        run_seed=123,
        run_final_tests=False,
    )

    with pytest.raises(ModelTerminalError) as captured:
        agent.run("inspect then stop at the output limit")

    partial = captured.value.partial_result
    assert partial["termination_reason"] == "model_output_limit"
    assert partial["steps"] == 1
    assert partial["target_file_read"] is False
    assert partial["history"][0]["observation"]["type"] == "file_content"
    assert partial["history"][0]["observation"]["path"] == "module.py"
