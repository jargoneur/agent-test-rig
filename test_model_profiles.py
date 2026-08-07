from pathlib import Path

import pytest
import yaml

from harness.model_profiles import resolve_models_from_lock


def _write_lock(path: Path, benchmark_runs_allowed: bool) -> None:
    value = {
        "common_protocol_limits": {
            "input_context_tokens": 16384,
            "max_new_tokens_per_action": 2048,
        },
        "runtime": {
            "backend": "llama.cpp_openai_compatible",
            "target_quantization": "Q4_K_M",
        },
        "models": {
            "model-a": {
                "model_id": "org/model-a",
                "model_revision": "abc123",
                "architecture": "dense",
                "input_context_tokens": 32768,
                "max_new_tokens_per_action": 8192,
                "thinking_mode": "enabled",
                "resolved_generation_options": {
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "repetition_penalty": 1.1,
                    "resolution_note": "not a runtime option",
                    "chat_template_kwargs": {"enable_thinking": True},
                },
                "stop_token_ids": [1, 2],
                "model_artifact_sha256": "pending_download",
                "tokenizer_sha256": "pending_download",
                "chat_template_sha256": "pending_download",
                "quantized_artifact_sha256": "pending_conversion",
                "profile_sha256": "pending_resolution",
            }
        },
        "validation_gate": {
            "benchmark_runs_allowed": benchmark_runs_allowed,
        },
    }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_pending_profiles_can_be_resolved_only_for_validation(tmp_path):
    lock_path = tmp_path / "models.yml"
    _write_lock(lock_path, benchmark_runs_allowed=False)

    models = resolve_models_from_lock(
        str(lock_path),
        allow_pending=True,
    )

    assert models == [
        {
            "id": "model-a",
            "name": "model-a",
            "backend": "llama_cpp",
            "upstream_model_id": "org/model-a",
            "model_revision": "abc123",
            "architecture": "dense",
            "quantization": "Q4_K_M",
            "model_artifact_sha256": "pending_download",
            "tokenizer_sha256": "pending_download",
            "chat_template_sha256": "pending_download",
            "quantized_artifact_sha256": "pending_conversion",
            "profile_sha256": "pending_resolution",
            "stop_token_ids": [1, 2],
            "generation_options": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.1,
                "chat_template_kwargs": {"enable_thinking": True},
                "num_ctx": 32768,
                "num_predict": 8192,
            },
        }
    ]


def test_validation_gate_blocks_benchmark_planning(tmp_path):
    lock_path = tmp_path / "models.yml"
    _write_lock(lock_path, benchmark_runs_allowed=False)

    with pytest.raises(ValueError, match="blocks benchmark runs"):
        resolve_models_from_lock(str(lock_path))


def test_resolved_gate_still_rejects_pending_hashes(tmp_path):
    lock_path = tmp_path / "models.yml"
    _write_lock(lock_path, benchmark_runs_allowed=True)

    with pytest.raises(ValueError, match="unresolved fields"):
        resolve_models_from_lock(str(lock_path))
