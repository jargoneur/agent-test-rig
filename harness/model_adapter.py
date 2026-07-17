import os

import requests


class OllamaModel:
    def __init__(
        self,
        model_name,
        host=None,
        options=None,
        timeout=600,
    ):
        self.model_name = model_name
        self.host = host or os.environ.get(
            "OLLAMA_HOST",
            "http://localhost:11434",
        )
        self.options = dict(options or {"temperature": 0.2})
        self.timeout = timeout
        self.last_generation_metadata = {}

    def effective_options(self, seed=None) -> dict:
        options = dict(self.options)
        if seed is not None:
            options["seed"] = int(seed)
        return options

    def generate(self, prompt, seed=None, timeout=None):
        url = f"{self.host}/api/generate"
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
            f"{self.host}/api/version",
            timeout=timeout,
        )
        version_response.raise_for_status()

        tags_response = requests.get(
            f"{self.host}/api/tags",
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
                f"Model metadata not found in Ollama tags: {self.model_name}"
            )

        show_response = requests.post(
            f"{self.host}/api/show",
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
