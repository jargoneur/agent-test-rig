from harness.model_adapter import OllamaModel, OpenAICompatibleModel


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
        },
    )

    payload = model.build_payload("hello", seed=11)

    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.8
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.05
    assert payload["repeat_penalty"] == 1.1
    assert payload["max_tokens"] == 2048
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
