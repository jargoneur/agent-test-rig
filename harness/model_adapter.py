from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import requests

from harness.outcomes import ModelTerminalError


_CONTEXT_MARKERS = (
    "context size",
    "context length",
    "context window",
    "exceeds the available context",
    "exceed_context",
    "n_ctx",
    "too many tokens",
    "prompt is too long",
)
_MEMORY_MARKERS = (
    "out of memory",
    "cuda oom",
    "cuda error: out of memory",
    "failed to allocate",
    "memory allocation failed",
    "kv cache allocation",
)


def _response_text(response: requests.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            value = data.get("error") or data.get("detail") or data.get("message")
            if value:
                return str(value)
    except ValueError:
        pass
    return str(response.text or "")[:8000]


def _raise_generation_error(response: requests.Response) -> None:
    if response.status_code < 400:
        return
    message = _response_text(response)
    normalized = message.lower()
    details = {
        "http_status": int(response.status_code),
        "response": message[:8000],
    }
    if any(marker in normalized for marker in _CONTEXT_MARKERS):
        raise ModelTerminalError(
            "context_limit_exceeded",
            message or "The request exceeded the deployed model context capacity",
            stage="model_generation",
            details=details,
        )
    if any(marker in normalized for marker in _MEMORY_MARKERS):
        raise ModelTerminalError(
            "capacity_oom",
            message or "The deployed model ran out of memory",
            stage="model_generation",
            details=details,
        )
    if response.status_code in {400, 413, 422}:
        raise ModelTerminalError(
            "model_request_rejected",
            message or "The model server rejected the generation request",
            stage="model_generation",
            details=details,
        )
    response.raise_for_status()


def _invalid_model_response(message: str, details=None) -> ModelTerminalError:
    return ModelTerminalError(
        "invalid_model_response",
        message,
        stage="model_generation",
        details=dict(details or {}),
    )


def _generation_timeout(timeout: int) -> ModelTerminalError:
    return ModelTerminalError(
        "model_timeout",
        "The model server did not finish generation within %d seconds" % timeout,
        stage="model_generation",
        details={"timeout_seconds": int(timeout)},
    )


def _resolved_sampling_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve framework-neutral sampling controls without inventing defaults."""
    resolved = dict(options)
    do_sample = resolved.pop("do_sample", None)
    if do_sample is False:
        resolved["temperature"] = 0.0
    return resolved


def _approximate_token_count(value: Any) -> int:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return max(1, int(math.ceil(len(text) / 4.0)))


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

    def _capture_metadata(self, data: Dict[str, Any], options: Dict[str, Any]) -> None:
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

    def generate(self, prompt, seed=None, timeout=None):
        options = self.effective_options(seed)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        request_timeout = int(timeout or self.timeout)
        try:
            response = requests.post(
                "%s/api/generate" % self.host,
                json=payload,
                timeout=request_timeout,
            )
        except requests.Timeout as error:
            raise _generation_timeout(request_timeout) from error
        _raise_generation_error(response)
        data = response.json()
        if data.get("error"):
            raise _invalid_model_response(str(data["error"]))
        self._capture_metadata(data, options)
        if data.get("response") is None:
            raise _invalid_model_response("Ollama response has no generated content")
        return data["response"]

    def generate_messages(self, messages, seed=None, timeout=None):
        options = self.effective_options(seed)
        payload = {
            "model": self.model_name,
            "messages": list(messages),
            "stream": False,
            "options": options,
        }
        request_timeout = int(timeout or self.timeout)
        try:
            response = requests.post(
                "%s/api/chat" % self.host,
                json=payload,
                timeout=request_timeout,
            )
        except requests.Timeout as error:
            raise _generation_timeout(request_timeout) from error
        _raise_generation_error(response)
        data = response.json()
        if data.get("error"):
            raise _invalid_model_response(str(data["error"]))
        self._capture_metadata(data, options)
        message = data.get("message") or {}
        content = message.get("content")
        if content is None:
            raise _invalid_model_response("Ollama chat response has no message content")
        return content

    def token_count(self, value: Any) -> int:
        return _approximate_token_count(value)

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

    def build_payload(
        self,
        prompt=None,
        seed=None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        options = self.effective_options(seed)
        if messages is None:
            messages = [{"role": "user", "content": str(prompt or "")}]
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": list(messages),
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
                "chat_template_kwargs": "chat_template_kwargs",
            }
            for source, target in llama_cpp_map.items():
                if source in options:
                    payload[target] = options[source]

        return payload

    def _request(self, payload, options, timeout=None):
        request_timeout = int(timeout or self.timeout)
        try:
            response = requests.post(
                "%s/chat/completions" % self.base_url,
                headers=self._headers(),
                json=payload,
                timeout=request_timeout,
            )
        except requests.Timeout as error:
            raise _generation_timeout(request_timeout) from error
        _raise_generation_error(response)
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise _invalid_model_response(
                "OpenAI-compatible server returned no choices",
                {"response_keys": sorted(data)},
            )
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise _invalid_model_response(
                "OpenAI-compatible response has no message content"
            )

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

    def generate(self, prompt, seed=None, timeout=None):
        options = self.effective_options(seed)
        payload = self.build_payload(prompt=prompt, seed=seed)
        return self._request(payload, options, timeout=timeout)

    def generate_messages(self, messages, seed=None, timeout=None):
        options = self.effective_options(seed)
        payload = self.build_payload(messages=list(messages), seed=seed)
        return self._request(payload, options, timeout=timeout)

    def token_count(self, value: Any) -> int:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if self.backend_name == "llama_cpp":
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            try:
                response = requests.post(
                    "%s/tokenize" % root,
                    headers=self._headers(),
                    json={"content": text},
                    timeout=min(30, self.timeout),
                )
                response.raise_for_status()
                data = response.json()
                tokens = data.get("tokens")
                if isinstance(tokens, list):
                    return max(1, len(tokens))
                if isinstance(tokens, int):
                    return max(1, tokens)
            except requests.RequestException:
                pass
        return _approximate_token_count(text)

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
            if entry is None and models:
                metadata["metadata_warning"] = (
                    "Requested model alias was not listed exactly by the server"
                )
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
    model_name = (
        model_spec.get("runtime_name")
        or model_spec.get("name")
        or model_spec.get("model")
    )
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
