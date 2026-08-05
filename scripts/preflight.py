#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from harness.contextbench_tasks import contextbench_task_ids
from harness.llama_cpp_server import LlamaCppServerManager
from harness.upstreams import upstream_path
from scaffolds import load_scaffold


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProbeModel:
    model_name = "preflight-probe"

    def __init__(self):
        self.options = {}
        self.last_generation_metadata = {}

    def effective_options(self, seed=None):
        return dict(self.options)

    def token_count(self, value):
        return max(1, len(str(value)) // 4)

    def generate(self, prompt, seed=None, timeout=None):
        return "{}"

    def generate_messages(self, messages, seed=None, timeout=None):
        return "summary"


def load_registry(path: Path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
        raise ValueError("Invalid model registry: %s" % path)
    return value["models"]


def verify_contextbench_evaluator(contextbench: Path) -> None:
    evaluator_python = ROOT / ".venv-contextbench" / "bin" / "python"
    if not evaluator_python.is_file():
        raise FileNotFoundError(
            "ContextBench evaluator environment is missing: %s" % evaluator_python
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(contextbench)
    result = subprocess.run(
        [
            str(evaluator_python),
            "-c",
            (
                "from contextbench.extractors.treesitter import available; "
                "assert available(), 'tree-sitter unavailable'; "
                "print('available')"
            ),
        ],
        cwd=contextbench,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ContextBench evaluator preflight failed: %s"
            % (result.stderr or result.stdout).strip()
        )
    print("ContextBench evaluator:", result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the complete agent-rig stack.")
    parser.add_argument(
        "--registry",
        default="model_artifacts/registry.yml",
    )
    parser.add_argument("--require-model", action="append", default=[])
    parser.add_argument("--start-model")
    parser.add_argument("--gpu")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--skip-contextbench-count", action="store_true")
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        raise RuntimeError("Full upstream stack requires Python 3.11 or newer")

    print("Python:", sys.version.split()[0])
    contextbench = None
    for component in (
        "contextbench",
        "sweagent_last5",
        "aider_repomap",
        "agentless_localization",
        "aider_chat_summary",
        "llama_cpp_runtime",
    ):
        path = upstream_path(component)
        print("Upstream:", component, path)
        if component == "contextbench":
            contextbench = path
    assert contextbench is not None
    verify_contextbench_evaluator(contextbench)

    probe = ProbeModel()
    for scaffold_name in (
        "sweagent_last5",
        "aider_repomap",
        "agentless_localization",
        "aider_chat_summary",
    ):
        scaffold = load_scaffold(
            scaffold_name,
            model=probe,
            task={"id": "preflight", "test_command": "true"},
            options={},
        )
        print("Scaffold:", scaffold_name, scaffold.__class__.__module__)

    if not args.skip_contextbench_count:
        task_ids = contextbench_task_ids(
            str(ROOT / "benchmarks" / "contextbench" / "tasks.jsonl")
        )
        if len(task_ids) != 150:
            raise RuntimeError(
                "Frozen ContextBench cache contains %d tasks, expected 150"
                % len(task_ids)
            )
        print("ContextBench tasks:", len(task_ids))

    registry_path = Path(args.registry).expanduser()
    if not registry_path.is_absolute():
        registry_path = ROOT / registry_path
    if not registry_path.is_file():
        raise FileNotFoundError("Model registry is missing: %s" % registry_path)
    models = load_registry(registry_path)
    required = set(args.require_model or models.keys())
    missing = required.difference(models)
    if missing:
        raise RuntimeError(
            "Required models are absent from registry: %s"
            % ", ".join(sorted(missing))
        )

    for model_id in sorted(required):
        entry = models[model_id]
        artifact = Path(str(entry["artifact"])).expanduser()
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        if not artifact.is_file():
            raise FileNotFoundError(
                "Model artifact is missing for %s: %s" % (model_id, artifact)
            )
        expected = str(entry.get("sha256") or "")
        actual = sha256_file(artifact)
        if expected and actual != expected:
            raise RuntimeError(
                "Model hash mismatch for %s: %s != %s"
                % (model_id, actual, expected)
            )
        print("Model:", model_id, artifact, actual)

    binary = ROOT / ".upstreams" / "llama.cpp" / "build" / "bin" / "llama-server"
    if not binary.is_file():
        raise FileNotFoundError("llama-server is missing: %s" % binary)
    version = subprocess.run(
        [str(binary), "--version"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    if version.returncode != 0:
        raise RuntimeError("llama-server --version failed: %s" % version.stderr)
    print("llama-server:", (version.stdout or version.stderr).strip())

    if args.start_model:
        manager = LlamaCppServerManager(
            {
                "registry": str(registry_path),
                "binary": str(binary),
                "host": "127.0.0.1",
                "port": args.port,
                "gpu": args.gpu,
                "startup_timeout_seconds": 1800,
                "log_path": "logs/preflight-llama-server.log",
            },
            worker_id="preflight",
        )
        try:
            runtime = manager.ensure_model(args.start_model)
            print("Started model:", json.dumps(runtime, indent=2))
        finally:
            manager.stop()

    print("Preflight passed.")


if __name__ == "__main__":
    main()
