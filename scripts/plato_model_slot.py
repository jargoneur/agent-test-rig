from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(value: str, registry_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    registry_relative = (registry_path.parent / path).resolve()
    if registry_relative.exists():
        return registry_relative
    return (ROOT / path).resolve()


def load_registry(path: Path) -> Dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
        raise ValueError("Registry must contain a models mapping")
    return value


def inspect_slot(registry_path: Path, keep_model_id: str) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    if keep_model_id not in registry["models"]:
        raise ValueError("Requested model is not present in registry: %s" % keep_model_id)

    present: List[Dict[str, Any]] = []
    for model_id, raw in sorted(registry["models"].items()):
        if not isinstance(raw, dict):
            continue
        artifact = resolve_artifact(str(raw.get("artifact") or ""), registry_path)
        if artifact.is_file():
            present.append(
                {
                    "model_id": str(model_id),
                    "artifact": str(artifact),
                    "size_bytes": artifact.stat().st_size,
                    "keep": str(model_id) == keep_model_id,
                }
            )

    keep_entries = [entry for entry in present if entry["keep"]]
    other_entries = [entry for entry in present if not entry["keep"]]
    return {
        "registry": str(registry_path),
        "keep_model_id": keep_model_id,
        "keep_present": len(keep_entries) == 1,
        "present_artifacts": present,
        "other_artifacts": other_entries,
    }


def validate_keep_hash(registry_path: Path, keep_model_id: str) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    raw = registry["models"][keep_model_id]
    artifact = resolve_artifact(str(raw.get("artifact") or ""), registry_path)
    if not artifact.is_file():
        raise FileNotFoundError(
            "Selected Plato model artifact is missing: %s" % artifact
        )
    expected = str(raw.get("sha256") or "").strip().lower()
    if not expected:
        raise RuntimeError("Selected model has no recorded SHA-256")
    actual = sha256_file(artifact).lower()
    if actual != expected:
        raise RuntimeError(
            "Selected model artifact hash mismatch: expected %s, got %s"
            % (expected, actual)
        )
    return {
        "model_id": keep_model_id,
        "artifact": str(artifact),
        "sha256": actual,
        "size_bytes": artifact.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce the Plato policy that only one registered GGUF artifact "
            "is stored on Plato at a time."
        )
    )
    parser.add_argument("--registry", default="model_artifacts/registry.yml")
    parser.add_argument("--keep-model-id", required=True)
    parser.add_argument(
        "--remove-other-artifacts",
        action="store_true",
        help=(
            "Delete registered GGUF files for all other model IDs. Registry and "
            "provenance records are preserved. Without this flag the command is "
            "a validation-only dry run."
        ),
    )
    args = parser.parse_args()

    registry_path = Path(args.registry).expanduser().resolve()
    if not registry_path.is_file():
        raise FileNotFoundError("Model registry is missing: %s" % registry_path)

    state = inspect_slot(registry_path, args.keep_model_id)
    selected = validate_keep_hash(registry_path, args.keep_model_id)
    if args.remove_other_artifacts:
        for entry in state["other_artifacts"]:
            Path(entry["artifact"]).unlink()
            print("Removed:", entry["model_id"], entry["artifact"])
        state = inspect_slot(registry_path, args.keep_model_id)

    state["selected"] = selected
    state["policy_ok"] = state["keep_present"] and not state["other_artifacts"]

    print(json.dumps(state, indent=2, ensure_ascii=False))
    if not state["policy_ok"]:
        raise SystemExit(
            "Plato single-model storage policy is not satisfied. Re-run with "
            "--remove-other-artifacts after confirming that completed results "
            "and scheduler backups are preserved."
        )


if __name__ == "__main__":
    main()
