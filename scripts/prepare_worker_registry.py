from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_FIELDS = (
    "context_size",
    "gpu_layers",
    "parallel",
    "cache_type_k",
    "cache_type_v",
    "flash_attention",
    "fit",
    "fit_target_mib",
)


def resolve_path(value: str, registry_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    registry_relative = (registry_path.parent / path).resolve()
    if registry_relative.exists():
        return registry_relative
    return (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_ids(values: Iterable[str]) -> set[str]:
    return {value.strip() for value in values if value.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a worker-local registry containing only verified GGUF artifacts."
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile-lock",
        default="experiments/model_profiles.lock.yml",
        help="Frozen profile lock that must match every advertised artifact.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Restrict to one or more model IDs. Repeat the option as needed.",
    )
    args = parser.parse_args()

    source = Path(args.registry).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    profile_lock_path = Path(args.profile_lock).expanduser().resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
        raise ValueError("Registry must contain a models mapping")
    profile_lock = yaml.safe_load(
        profile_lock_path.read_text(encoding="utf-8")
    )
    if (
        not isinstance(profile_lock, dict)
        or not isinstance(profile_lock.get("models"), dict)
    ):
        raise ValueError("Profile lock must contain a models mapping")
    locked_profiles = profile_lock["models"]

    requested = selected_ids(args.model_id)
    verified: Dict[str, Dict[str, Any]] = {}
    failures = []

    for model_id, raw_entry in value["models"].items():
        model_id = str(model_id)
        if requested and model_id not in requested:
            continue
        if not isinstance(raw_entry, dict):
            failures.append("%s: registry entry is not a mapping" % model_id)
            continue
        entry = dict(raw_entry)
        artifact = resolve_path(str(entry.get("artifact") or ""), source)
        if not artifact.is_file():
            failures.append("%s: artifact missing: %s" % (model_id, artifact))
            continue
        expected = str(entry.get("sha256") or "").strip().lower()
        if not expected:
            failures.append("%s: sha256 is required" % model_id)
            continue
        actual = sha256_file(artifact)
        if actual.lower() != expected:
            failures.append(
                "%s: sha256 mismatch: expected %s, got %s"
                % (model_id, expected, actual)
            )
            continue
        entry["artifact"] = str(artifact)
        entry["sha256"] = actual

        profile = locked_profiles.get(model_id)
        if not isinstance(profile, dict):
            failures.append("%s: absent from profile lock" % model_id)
            continue
        expected_artifact = str(
            profile.get("quantized_artifact_sha256") or ""
        ).lower()
        profile_sha256 = str(profile.get("profile_sha256") or "").lower()
        expected_revision = str(profile.get("model_revision") or "")
        registry_revision = str(
            entry.get("model_revision")
            or entry.get("upstream_revision")
            or ""
        )
        if not expected_artifact or expected_artifact.startswith("pending_"):
            failures.append("%s: profile lock has no frozen GGUF hash" % model_id)
            continue
        if actual.lower() != expected_artifact:
            failures.append("%s: artifact does not match profile lock" % model_id)
            continue
        if not profile_sha256 or profile_sha256.startswith("pending_"):
            failures.append("%s: profile lock has no frozen profile hash" % model_id)
            continue
        if not expected_revision or registry_revision != expected_revision:
            failures.append("%s: model revision does not match profile lock" % model_id)
            continue
        expected_deployment = profile.get("deployment_profile")
        if (
            profile.get("deployment_status") != "validated"
            or not isinstance(expected_deployment, dict)
            or expected_deployment.get("validation_status") != "validated"
        ):
            failures.append("%s: deployment profile is not validated" % model_id)
            continue
        locked_deployment = {
            field: expected_deployment.get(field)
            for field in DEPLOYMENT_FIELDS
        }
        registry_deployment = {
            field: entry.get(field)
            for field in DEPLOYMENT_FIELDS
        }
        if registry_deployment != locked_deployment:
            failures.append(
                "%s: registry deployment fields do not match profile lock"
                % model_id
            )
            continue
        entry["profile_sha256"] = profile_sha256
        entry["model_revision"] = expected_revision
        verified[model_id] = entry

    missing_requested = requested.difference(verified)
    if missing_requested:
        failures.append(
            "requested models not verified: %s" % ", ".join(sorted(missing_requested))
        )
    if failures:
        raise SystemExit("Worker registry validation failed:\n- " + "\n- ".join(failures))
    if not verified:
        raise SystemExit("Worker registry contains no verified model artifacts")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": value.get("schema_version", 1),
        "source_registry": str(source),
        "source_profile_lock": str(profile_lock_path),
        "models": verified,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    print("Verified worker models:")
    for model_id in sorted(verified):
        print("-", model_id)
    print("Worker registry:", output)


if __name__ == "__main__":
    main()
