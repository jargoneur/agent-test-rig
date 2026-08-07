from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


_RUNTIME_OPTION_KEYS = {
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "stop",
    "chat_template_kwargs",
}


def load_model_profile_lock(path: str) -> Dict[str, Any]:
    lock_path = Path(path)
    value = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Model profile lock must be a YAML mapping")
    models = value.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("Model profile lock contains no models")
    return value


def _runtime_backend(lock: Dict[str, Any]) -> str:
    backend = str((lock.get("runtime") or {}).get("backend", "llama_cpp"))
    if backend == "llama.cpp_openai_compatible":
        return "llama_cpp"
    return backend


def _pending_fields(profile: Dict[str, Any]) -> List[str]:
    required = [
        "model_artifact_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "quantized_artifact_sha256",
        "profile_sha256",
    ]
    pending = []
    for field in required:
        value = profile.get(field)
        if not value or str(value).startswith("pending_"):
            pending.append(field)
    return pending


def profile_to_model_spec(
    profile_key: str,
    profile: Dict[str, Any],
    lock: Dict[str, Any],
    allow_pending: bool = False,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    if not allow_pending:
        gate = lock.get("validation_gate") or {}
        if not bool(gate.get("benchmark_runs_allowed", False)):
            raise ValueError(
                "Model profile lock blocks benchmark runs until artifact validation"
            )
        pending = _pending_fields(profile)
        if pending:
            raise ValueError(
                "Model profile %s has unresolved fields: %s"
                % (profile_key, ", ".join(pending))
            )

    raw_options = dict(profile.get("resolved_generation_options") or {})
    generation_options = {
        key: value
        for key, value in raw_options.items()
        if key in _RUNTIME_OPTION_KEYS
    }
    limits = lock.get("common_protocol_limits") or {}
    input_context_tokens = profile.get(
        "input_context_tokens", limits.get("input_context_tokens")
    )
    if input_context_tokens is not None:
        generation_options["num_ctx"] = int(input_context_tokens)
    max_new_tokens = profile.get(
        "max_new_tokens_per_action",
        limits.get("max_new_tokens_per_action"),
    )
    if max_new_tokens is not None:
        generation_options["num_predict"] = int(max_new_tokens)

    runtime = lock.get("runtime") or {}
    model_spec: Dict[str, Any] = {
        "id": profile_key,
        "name": profile_key,
        "backend": _runtime_backend(lock),
        "upstream_model_id": profile["model_id"],
        "model_revision": profile["model_revision"],
        "architecture": profile.get("architecture"),
        "quantization": runtime.get("target_quantization"),
        "model_artifact_sha256": profile.get("model_artifact_sha256"),
        "tokenizer_sha256": profile.get("tokenizer_sha256"),
        "chat_template_sha256": profile.get("chat_template_sha256"),
        "quantized_artifact_sha256": profile.get(
            "quantized_artifact_sha256"
        ),
        "profile_sha256": profile.get("profile_sha256"),
        "stop_token_ids": list(profile.get("stop_token_ids") or []),
        "generation_options": generation_options,
    }
    if base_url:
        model_spec["base_url"] = base_url
    if profile.get("resource_class"):
        model_spec["resource_class"] = profile["resource_class"]
    return model_spec


def resolve_models_from_lock(
    path: str,
    profile_keys: Optional[Sequence[str]] = None,
    allow_pending: bool = False,
    base_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    lock = load_model_profile_lock(path)
    profiles: Dict[str, Dict[str, Any]] = lock["models"]
    keys = list(profile_keys or profiles.keys())
    unknown = [key for key in keys if key not in profiles]
    if unknown:
        raise ValueError("Unknown model profile(s): " + ", ".join(unknown))
    return [
        profile_to_model_spec(
            key,
            profiles[key],
            lock,
            allow_pending=allow_pending,
            base_url=base_url,
        )
        for key in keys
    ]
