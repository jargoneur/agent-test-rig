from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


def _resolved_sampling_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve framework-neutral sampling controls without inventing defaults."""
    resolved = dict(options)
    do_sample = resolved.pop("do_sample", None)
    if do_sample is False:
        resolved["temperature"] = 0.0
    return resolved


class OllamaModel:
    def __init__(
        self,
        model_name: str,
        host: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: int = 600,
    ):
        self.model_name = model_name
        self.host = host or os.environ.get(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )
        self.host = self.host.rstrip("/")
        self.options = dict(options or {})
        self.timeout = timeout
        self.last_generation_metadata: Dict[str, Any] = {}

    def effective_options(self, seed=None) -> dict:
        options = _resolved_sampling_options(self.options)
        if "repetition_penalty" in options:
            options["repeat_penalty"] = options.pop("repetition_penalty")
        if "max_new_tokens" in options and "num_predict" not in options:
            options["num_predict"] = options.pop("max_new_tokens")
        if seed is not None:
            options["seed"] = int(seed)
        return options

    def generate(self, prompt, seed=None, timeout=None):
        url = "%s/api/generate" % self.host
        options = self.effective_options(seed)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        response = requests.post(
            url,
            json=payload,
            timeout=timeout or self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        metadata_fields = [
            "model",
            "created_at",
            "done",
            "done_reason",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        ]
        self.last_generation_metadata = {
            key: data.get(key)
            for key in metadata_fields
            if key in data
        }
        self.last_generation_metadata["options"] = options
        return data["response"]

    def runtime_metadata(self, timeout=60) -> dict:
        version_response = requests.get(
            "%s/api/version" % self.host,
            timeout=timeout,
        )
        version_response.raise_for_status()

        tags_response = requests.get(
            "%s/api/tags" % self.host,
            timeout=timeout,
        )
        tags_response.raise_for_status()
        models = tags_response.json().get("models", [])

        model_entry = next(
            (
                entry
                for entry in models
                if entry.get("name") == self.model_name
                or entry.get("model") == self.model_name
            ),
            None,
        )
        if model_entry is None:
            raise RuntimeError(
                "Model metadata not found in Ollama tags: %s" % self.model_name
            )

        show_response = requests.post(
            "%s/api/show" % self.host,
            json={"model": self.model_name, "verbose": False},
            timeout=timeout,
        )
        show_response.raise_for_status()
        show = show_response.json()
        model_info = show.get("model_info", {})
        context_lengths = {
            key: value
            for key, value in model_info.items()
            if key.endswith(".context_length")
        }

        return {
            "backend": "ollama",
            "host": self.host,
            "ollama_version": version_response.json().get("version"),
            "model_name": self.model_name,
            "digest": model_entry.get("digest"),
            "size": model_entry.get("size"),
            "modified_at": model_entry.get("modified_at"),
            "details": model_entry.get("details", show.get("details", {})),
            "capabilities": show.get("capabilities", []),
            "parameters": show.get("parameters", ""),
            "context_lengths": context_lengths,
            "model_info": model_info,
        }


class OpenAICompatibleModel:
    """Adapter for llama.cpp and standard OpenAI-compatible local servers."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: int = 600,
        backend_name: str = "openai_compatible",
    ):
        self.model_name = model_name
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:8080/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "local")
        self.options = dict(options or {})
        self.timeout = timeout
        self.backend_name = backend_name
        self.last_generation_metadata: Dict[str, Any] = {}

    def effective_options(self, seed=None) -> dict:
        options = _resolved_sampling_options(self.options)
        if seed is not None:
            options["seed"] = int(seed)
        return options

    def _headers(self) -> dict:
        return {
            "Authorization": "Bearer %s" % self.api_key,
            "Content-Type": "application/json",
        }

    def build_payload(self, prompt, seed=None) -> Dict[str, Any]:
        options = self.effective_options(seed)
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        standard_map = {
            "temperature": "temperature",
            "top_p": "top_p",
            "num_predict": "max_tokens",
            "max_new_tokens": "max_tokens",
            "seed": "seed",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "stop": "stop",
        }
        for source, target in standard_map.items():
            if source in options:
                payload[target] = options[source]

        if self.backend_name == "llama_cpp":
            llama_cpp_map = {
                "top_k": "top_k",
                "min_p": "min_p",
                "repetition_penalty": "repeat_penalty",
                "repeat_penalty": "repeat_penalty",
            }
            for source, target in llama_cpp_map.items():
                if source in options:
                    payload[target] = options[source]

        return payload

    def generate(self, prompt, seed=None, timeout=None):
        options = self.effective_options(seed)
        payload = self.build_payload(prompt, seed=seed)

        response = requests.post(
            "%s/chat/completions" % self.base_url,
            headers=self._headers(),
            json=payload,
            timeout=timeout or self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-compatible server returned no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError("OpenAI-compatible response has no message content")

        self.last_generation_metadata = {
            "id": data.get("id"),
            "created": data.get("created"),
            "model": data.get("model"),
            "finish_reason": choices[0].get("finish_reason"),
            "usage": data.get("usage", {}),
            "system_fingerprint": data.get("system_fingerprint"),
            "options": options,
            "request_payload": payload,
        }
        return content

    def runtime_metadata(self, timeout=60) -> dict:
        metadata = {
            "backend": self.backend_name,
            "base_url": self.base_url,
            "model_name": self.model_name,
        }
        try:
            response = requests.get(
                "%s/models" % self.base_url,
                headers=self._headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            entry = next(
                (item for item in models if item.get("id") == self.model_name),
                None,
            )
            metadata["models_endpoint"] = data
            metadata["model_entry"] = entry
        except requests.RequestException as error:
            metadata["metadata_warning"] = str(error)
        return metadata


def create_model(model_spec, options=None, timeout=600):
    if isinstance(model_spec, str):
        model_spec = {
            "name": model_spec,
            "backend": "ollama",
        }
    if not isinstance(model_spec, dict):
        raise TypeError("model_spec must be a string or mapping")

    backend = str(model_spec.get("backend", "ollama")).lower()
    model_name = model_spec.get("name") or model_spec.get("model")
    if not model_name:
        raise ValueError("model_spec requires a model name")

    if backend == "ollama":
        return OllamaModel(
            model_name,
            host=model_spec.get("host") or model_spec.get("base_url"),
            options=options,
            timeout=timeout,
        )
    if backend in {"openai", "openai_compatible", "llama_cpp", "vllm", "sglang"}:
        return OpenAICompatibleModel(
            model_name,
            base_url=model_spec.get("base_url") or model_spec.get("host"),
            api_key=model_spec.get("api_key"),
            options=options,
            timeout=timeout,
            backend_name=backend,
        )
    raise ValueError("Unsupported model backend: %s" % backend)
