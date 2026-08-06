from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT / ".upstreams"
LOCK_PATH = PROJECT_ROOT / "docs" / "upstream_components.lock.yml"

_DIRECTORY_NAMES = {
    "contextbench": "contextbench",
    "sweagent_last5": "swe-agent",
    "aider_repomap": "aider",
    "aider_chat_summary": "aider",
    "agentless_localization": "agentless",
    "llama_cpp_runtime": "llama.cpp",
}


def load_upstream_lock() -> Dict[str, Any]:
    value = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Upstream component lock must be a YAML mapping")
    components = value.get("components")
    if not isinstance(components, dict):
        raise ValueError("Upstream component lock contains no components")
    return value


def component_spec(name: str) -> Dict[str, Any]:
    lock = load_upstream_lock()
    try:
        value = lock["components"][name]
    except KeyError as error:
        raise KeyError("Unknown locked upstream component: %s" % name) from error
    if not isinstance(value, dict):
        raise ValueError("Invalid upstream component specification: %s" % name)
    return value


def upstream_path(name: str, verify: bool = True) -> Path:
    try:
        directory = _DIRECTORY_NAMES[name]
    except KeyError as error:
        raise KeyError("No checkout directory registered for %s" % name) from error

    path = (UPSTREAM_ROOT / directory).resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            "Locked upstream checkout is missing: %s. Run scripts/fetch_upstreams.sh"
            % path
        )

    if verify:
        expected = str(component_spec(name)["commit"])
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Cannot verify upstream checkout %s: %s"
                % (path, result.stderr.strip())
            )
        actual = result.stdout.strip()
        if actual != expected:
            raise RuntimeError(
                "Upstream revision mismatch for %s: %s != %s"
                % (name, actual, expected)
            )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if dirty.returncode != 0 or dirty.stdout.strip():
            raise RuntimeError("Locked upstream checkout is not clean: %s" % path)

    return path


def add_upstream_to_path(name: str, verify: bool = True) -> Path:
    path = upstream_path(name, verify=verify)
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    return path


def import_upstream(name: str, module: str, verify: bool = True):
    add_upstream_to_path(name, verify=verify)
    return importlib.import_module(module)
