from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class LlamaCppServerManager:
    """Keep exactly one llama-server model loaded for a scheduled worker."""

    def __init__(self, config: Dict[str, Any], worker_id: str):
        self.config = dict(config or {})
        self.worker_id = worker_id
        self.registry_path = _resolve_path(
            str(self.config.get("registry") or "model_artifacts/registry.yml")
        )
        self.binary = _resolve_path(
            str(
                self.config.get("binary")
                or ".upstreams/llama.cpp/build/bin/llama-server"
            )
        )
        self.host = str(self.config.get("host") or "127.0.0.1")
        self.port = int(self.config.get("port") or 8080)
        self.startup_timeout = int(
            self.config.get("startup_timeout_seconds") or 900
        )
        self.shutdown_timeout = int(
            self.config.get("shutdown_timeout_seconds") or 30
        )
        self.gpu = self.config.get("gpu")
        self.log_path = _resolve_path(
            str(
                self.config.get("log_path")
                or "logs/model_servers/%s.log" % worker_id
            )
        )
        self.process: Optional[subprocess.Popen] = None
        self.log_handle = None
        self.active_model_id: Optional[str] = None
        self.active_runtime: Optional[Dict[str, Any]] = None
        self.registry = self._load_registry()
        self._verified_entries: Dict[str, Dict[str, Any]] = {}

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if not self.registry_path.is_file():
            raise FileNotFoundError(
                "Model registry is missing: %s" % self.registry_path
            )
        value = yaml.safe_load(self.registry_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
            raise ValueError("Model registry must contain a models mapping")
        models = {}
        for model_id, raw in value["models"].items():
            if not isinstance(raw, dict):
                raise ValueError("Invalid model registry entry: %s" % model_id)
            entry = dict(raw)
            artifact = _resolve_path(str(entry["artifact"]))
            entry["artifact"] = str(artifact)
            models[str(model_id)] = entry
        return models

    def model_ids(self, verify: bool = False):
        model_ids = sorted(self.registry)
        if not verify:
            return model_ids
        verified = []
        failures = []
        for model_id in model_ids:
            try:
                self._entry(model_id)
                verified.append(model_id)
            except Exception as error:
                failures.append("%s: %s" % (model_id, error))
        if failures:
            raise RuntimeError(
                "Worker model registry contains unavailable artifacts:\n- "
                + "\n- ".join(failures)
            )
        return verified

    def _entry(self, model_id: str) -> Dict[str, Any]:
        cached = self._verified_entries.get(model_id)
        if cached is not None:
            return dict(cached)
        try:
            entry = dict(self.registry[model_id])
        except KeyError as error:
            raise KeyError("Model is not provisioned in registry: %s" % model_id) from error
        artifact = Path(entry["artifact"])
        if not artifact.is_file():
            raise FileNotFoundError(
                "GGUF artifact for %s is missing: %s" % (model_id, artifact)
            )
        expected = str(entry.get("sha256") or "").strip().lower()
        if not expected:
            raise RuntimeError("GGUF sha256 is missing for %s" % model_id)
        actual = sha256_file(artifact).lower()
        if actual != expected:
            raise RuntimeError(
                "GGUF hash mismatch for %s: %s != %s"
                % (model_id, actual, expected)
            )
        entry["sha256"] = actual
        self._verified_entries[model_id] = dict(entry)
        return entry

    def _base_url(self) -> str:
        return "http://%s:%d/v1" % (self.host, self.port)

    def _server_root(self) -> str:
        return "http://%s:%d" % (self.host, self.port)

    def _is_ready(self, alias: str) -> bool:
        try:
            response = requests.get(
                self._base_url() + "/models",
                timeout=5,
            )
            response.raise_for_status()
            models = response.json().get("data") or []
            return any(str(item.get("id")) == alias for item in models)
        except (requests.RequestException, ValueError):
            return False

    def _command(self, model_id: str, entry: Dict[str, Any]):
        alias = str(entry.get("alias") or model_id)
        command = [
            str(self.binary),
            "--model",
            str(entry["artifact"]),
            "--alias",
            alias,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(int(entry.get("context_size") or 32768)),
            "--n-gpu-layers",
            str(entry.get("gpu_layers") or "all"),
            "--split-mode",
            "none",
            "--parallel",
            str(int(entry.get("parallel") or 1)),
            "--cache-type-k",
            str(entry.get("cache_type_k") or "f16"),
            "--cache-type-v",
            str(entry.get("cache_type_v") or "f16"),
        ]
        flash_attention = entry.get("flash_attention", "auto")
        if flash_attention is not None:
            command.extend(["--flash-attn", str(flash_attention)])
        fit = entry.get("fit")
        if fit is not None:
            command.extend(["--fit", "on" if bool(fit) else "off"])
        fit_target = entry.get("fit_target_mib")
        if fit_target is not None:
            command.extend(["--fit-target", str(int(fit_target))])
        command.extend([str(value) for value in entry.get("extra_args") or []])
        return command

    def ensure_model(self, model_id: str) -> Dict[str, Any]:
        entry = self._entry(model_id)
        alias = str(entry.get("alias") or model_id)
        if (
            self.active_model_id == model_id
            and self.process is not None
            and self.process.poll() is None
            and self._is_ready(alias)
        ):
            return dict(self.active_runtime or {})

        self.stop()
        if not self.binary.is_file():
            raise FileNotFoundError(
                "llama-server binary is missing: %s. Run scripts/bootstrap.sh"
                % self.binary
            )

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("a", encoding="utf-8")
        environment = os.environ.copy()
        if self.gpu not in (None, ""):
            environment["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        environment.setdefault("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "0")

        command = self._command(model_id, entry)
        self.log_handle.write(
            "\n[%s] starting %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(command))
        )
        self.log_handle.flush()
        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited during startup for %s; see %s"
                    % (model_id, self.log_path)
                )
            if self._is_ready(alias):
                self.active_model_id = model_id
                self.active_runtime = {
                    "backend": "llama_cpp",
                    "base_url": self._base_url(),
                    "runtime_name": alias,
                    "artifact": str(entry["artifact"]),
                    "artifact_sha256": entry.get("sha256"),
                    "llama_server_binary": str(self.binary),
                    "llama_server_command": command,
                    "gpu": self.gpu,
                }
                return dict(self.active_runtime)
            time.sleep(2)

        self.stop()
        raise TimeoutError(
            "llama-server did not become ready for %s within %d seconds; see %s"
            % (model_id, self.startup_timeout, self.log_path)
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        self.active_model_id = None
        self.active_runtime = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.time() + self.shutdown_timeout
            while process.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
