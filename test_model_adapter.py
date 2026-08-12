import json

import pytest
import requests

from harness.model_adapter import (
    OllamaModel,
    OpenAICompatibleModel,
    _raise_generation_error,
)
from harness.outcomes import ModelTerminalError


def response(status, message):
    value = requests.Response()
    value.status_code = status
    value._content = json.dumps({"error": message}).encode()
    return value


def test_context_and_oom_are_terminal_capacity_outcomes():
    with pytest.raises(ModelTerminalError) as context_error:
        _raise_generation_error(
            response(400, "request exceeds the available context size")
        )
    assert context_error.value.kind == "context_limit_exceeded"

    with pytest.raises(ModelTerminalError) as oom_error:
        _raise_generation_error(response(500, "CUDA error: out of memory"))
    assert oom_error.value.kind == "capacity_oom"


def test_transient_server_error_remains_retryable_infrastructure_failure():
    with pytest.raises(requests.HTTPError):
        _raise_generation_error(response(503, "service temporarily unavailable"))


def test_generation_timeout_is_a_usable_terminal_model_outcome(monkeypatch):
    def time_out(*args, **kwargs):
        raise requests.Timeout("generation exceeded deadline")

    monkeypatch.setattr(requests, "post", time_out)
    model = OpenAICompatibleModel("model-a", timeout=21600)

    with pytest.raises(ModelTerminalError) as timeout_error:
        model.generate("hello")

    assert timeout_error.value.kind == "model_timeout"
    assert timeout_error.value.stage == "model_generation"
    assert timeout_error.value.details == {"timeout_seconds": 21600}


def test_openai_adapter_has_no_hidden_temperature_default():
    model = OpenAICompatibleModel("model-a", options={})
    payload = model.build_payload("hello", seed=7)

    assert payload["seed"] == 7
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_llama_cpp_payload_preserves_recommended_sampling_profile():
    model = OpenAICompatibleModel(
        "model-a",
        backend_name="llama_cpp",
        options={
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
            "num_predict": 2048,
            "num_ctx": 16384,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )

    payload = model.build_payload("hello", seed=11)

    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.05
    assert payload["repeat_penalty"] == 1.1
    assert payload["max_tokens"] == 2048
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["seed"] == 11
    assert "num_ctx" not in payload
    assert "do_sample" not in payload
    assert "repetition_penalty" not in payload


def test_greedy_profile_resolves_to_zero_temperature():
    model = OpenAICompatibleModel(
        "model-a",
        backend_name="llama_cpp",
        options={"do_sample": False, "temperature": 0.7},
    )

    payload = model.build_payload("hello")

    assert payload["temperature"] == 0.0


def test_generic_openai_adapter_does_not_send_llama_cpp_extensions():
    model = OpenAICompatibleModel(
        "model-a",
        backend_name="openai_compatible",
        options={
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.05,
            "repetition_penalty": 1.1,
        },
    )

    payload = model.build_payload("hello")

    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.95
    assert "top_k" not in payload
    assert "min_p" not in payload
    assert "repeat_penalty" not in payload


def test_ollama_adapter_translates_profile_field_names():
    model = OllamaModel(
        "model-a",
        options={
            "do_sample": False,
            "repetition_penalty": 1.05,
            "max_new_tokens": 512,
        },
    )

    options = model.effective_options(seed=19)

    assert options == {
        "temperature": 0.0,
        "repeat_penalty": 1.05,
        "num_predict": 512,
        "seed": 19,
    }
