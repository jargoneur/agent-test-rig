import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plato_model_slot_requires_explicit_cleanup(tmp_path):
    first = tmp_path / "first.gguf"
    second = tmp_path / "second.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    registry = tmp_path / "registry.yml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "models": {
                    "model-a": {
                        "artifact": str(first),
                        "sha256": digest(first),
                    },
                    "model-b": {
                        "artifact": str(second),
                        "sha256": digest(second),
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    script = Path(__file__).parent / "scripts" / "plato_model_slot.py"
    dry_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--registry",
            str(registry),
            "--keep-model-id",
            "model-a",
        ],
        text=True,
        capture_output=True,
    )
    assert dry_run.returncode != 0
    assert first.exists()
    assert second.exists()

    cleanup = subprocess.run(
        [
            sys.executable,
            str(script),
            "--registry",
            str(registry),
            "--keep-model-id",
            "model-a",
            "--remove-other-artifacts",
        ],
        text=True,
        capture_output=True,
    )
    assert cleanup.returncode == 0, cleanup.stderr
    assert first.exists()
    assert not second.exists()
    assert '"policy_ok": true' in cleanup.stdout
