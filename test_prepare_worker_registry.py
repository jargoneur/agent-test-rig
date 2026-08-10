from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "scripts" / "prepare_worker_registry.py"


def write_inputs(tmp_path: Path, context_size: int = 4096):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"test-gguf")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    revision = "b" * 40
    deployment = {
        "context_size": 4096,
        "gpu_layers": "all",
        "parallel": 1,
        "cache_type_k": "q8_0",
        "cache_type_v": "q8_0",
        "flash_attention": "auto",
        "fit": True,
        "fit_target_mib": 1536,
        "gpu_count": 1,
        "split_mode": "none",
        "tensor_split": None,
        "validation_status": "validated",
    }
    registry = {
        "schema_version": 1,
        "models": {
            "demo": {
                "artifact": str(artifact),
                "sha256": digest,
                "model_revision": revision,
                **{
                    key: value
                    for key, value in deployment.items()
                    if key != "validation_status"
                },
            }
        },
    }
    registry["models"]["demo"]["context_size"] = context_size
    lock = {
        "models": {
            "demo": {
                "quantized_artifact_sha256": digest,
                "profile_sha256": "a" * 64,
                "model_revision": revision,
                "deployment_status": "validated",
                "deployment_profile": deployment,
            }
        }
    }
    registry_path = tmp_path / "registry.yml"
    lock_path = tmp_path / "profiles.yml"
    output_path = tmp_path / "verified.yml"
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    return registry_path, lock_path, output_path


def run_prepare(registry: Path, lock: Path, output: Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--registry",
            str(registry),
            "--profile-lock",
            str(lock),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_prepare_worker_registry_accepts_exact_deployment(tmp_path):
    registry, lock, output = write_inputs(tmp_path)

    result = run_prepare(registry, lock, output)

    assert result.returncode == 0, result.stderr
    verified = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert verified["models"]["demo"]["context_size"] == 4096
    assert verified["models"]["demo"]["profile_sha256"] == "a" * 64
    assert verified["models"]["demo"]["gpu_count"] == 1


def test_prepare_worker_registry_rejects_deployment_mismatch(tmp_path):
    registry, lock, output = write_inputs(tmp_path, context_size=2048)

    result = run_prepare(registry, lock, output)

    assert result.returncode != 0
    assert "registry deployment fields do not match profile lock" in (
        result.stdout + result.stderr
    )
    assert not output.exists()