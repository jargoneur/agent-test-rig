from __future__ import annotations

import json

import pytest
import requests

from agents.simple_agent import SimpleAgent
from harness.model_adapter import ACTION_RESPONSE_SCHEMA, OpenAICompatibleModel
from harness.outcomes import ModelTerminalError
from harness.tools import FileTools


class DummyLogger:
    def log(self, *args, **kwargs):
        return None


def completion_response(content, finish_reason="stop", completion_tokens=7):
    response = requests.Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "id": "chatcmpl-test",
            "model": "model-a",
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"completion_tokens": completion_tokens},
        }
    ).encode()
    return response


def test_llama_cpp_action_call_freezes_schema_and_output_limit(monkeypatch):
    captured = {}

    def post(url, headers, json, timeout):
        captured.update(json)
        return completion_response('{"action":"finish"}')

    monkeypatch.setattr(requests, "post", post)
    model = OpenAICompatibleModel(
        "model-a",
        backend_name="llama_cpp",
        options={"num_predict": 8192},
        timeout=1800,
    )

    assert model.generate_action("choose one action", seed=11) == '{"action":"finish"}'
    assert captured["max_tokens"] == 8192
    assert captured["response_format"] == {
        "type": "json_object",
        "schema": ACTION_RESPONSE_SCHEMA,
    }


def test_length_finish_reason_is_a_usable_terminal_model_outcome(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: completion_response(
            '{"action":"read_file"',
            finish_reason="length",
            completion_tokens=8192,
        ),
    )
    model = OpenAICompatibleModel(
        "model-a",
        backend_name="llama_cpp",
        options={"num_predict": 8192},
        timeout=1800,
    )

    with pytest.raises(ModelTerminalError) as limit_error:
        model.generate_action("choose one action")

    assert limit_error.value.kind == "model_output_limit"
    assert limit_error.value.stage == "model_generation"
    assert limit_error.value.details["finish_reason"] == "length"
    assert limit_error.value.details["output_token_limit"] == 8192
    assert limit_error.value.details["usage"]["completion_tokens"] == 8192


def test_replace_text_requires_exact_match_and_preserves_unrelated_content(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("before\ntarget\nafter\n", encoding="utf-8")
    tools = FileTools(str(tmp_path))

    assert tools.replace_text("module.py", "target", "replacement") == 1
    assert source.read_text(encoding="utf-8") == "before\nreplacement\nafter\n"

    with pytest.raises(ValueError, match="found 0"):
        tools.replace_text("module.py", "missing", "unused")


def test_agent_blocks_unread_replace_then_applies_after_read(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    agent = SimpleAgent(
        model=object(),
        tools=FileTools(str(tmp_path)),
        logger=DummyLogger(),
        scaffold=object(),
    )
    action = {
        "action": "replace_text",
        "path": "module.py",
        "old_text": "value = 1",
        "new_text": "value = 2",
        "expected_replacements": 1,
    }

    assert agent.execute_action(action)["type"] == "write_blocked"
    agent.execute_action(
        {
            "action": "read_file",
            "path": "module.py",
            "start_line": 1,
            "end_line": 1,
        }
    )
    observation = agent.execute_action(action)

    assert observation == {
        "type": "write_success",
        "path": "module.py",
        "existed_before": True,
        "edit_kind": "replace_text",
        "replacements": 1,
    }
    assert source.read_text(encoding="utf-8") == "value = 2\n"
