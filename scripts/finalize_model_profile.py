#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_FIELDS = (
    "context_size",
    "gpu_layers",
    "parallel",
    "cache_type_k",
    "cache_type_v",
    "flash_attention",
    "fit",
    "fit_target_mib",
    "gpu_count",
    "split_mode",
    "tensor_split",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_artifact(value: str, registry_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    registry_relative = (registry_path.parent / path).resolve()
    if registry_relative.exists():
        return registry_relative
    return (ROOT / path).resolve()


def profile_payload(
    key: str,
    profile: Dict[str, Any],
    lock: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "profile_key": key,
        "model_id": profile["model_id"],
        "model_revision": profile["model_revision"],
        "family": profile["family"],
        "architecture": profile["architecture"],
        "model_artifact_sha256": profile["model_artifact_sha256"],
        "tokenizer_sha256": profile["tokenizer_sha256"],
        "chat_template_sha256": profile["chat_template_sha256"],
        "quantized_artifact_sha256": profile["quantized_artifact_sha256"],
        "thinking_mode": profile["thinking_mode"],
        "resolved_generation_options": profile["resolved_generation_options"],
        "input_context_tokens": profile["input_context_tokens"],
        "max_new_tokens_per_action": profile["max_new_tokens_per_action"],
        "stop_token_ids": profile["stop_token_ids"],
        "deployment_profile": profile["deployment_profile"],
        "runtime": lock["runtime"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize one frozen model profile from artifact and validation evidence."
    )
    parser.add_argument("profile_key")
    parser.add_argument(
        "--lock",
        default=str(ROOT / "experiments" / "model_profiles.lock.yml"),
    )
    parser.add_argument(
        "--registry",
        default=str(ROOT / "model_artifacts" / "registry.yml"),
    )
    parser.add_argument(
        "--provenance",
        help="Defaults to model_artifacts/<profile>/provenance.json.",
    )
    parser.add_argument(
        "--validation-evidence",
        required=True,
        help="Measured full-context prefill/runtime validation JSON.",
    )
    args = parser.parse_args()

    lock_path = Path(args.lock).expanduser().resolve()
    registry_path = Path(args.registry).expanduser().resolve()
    provenance_path = (
        Path(args.provenance).expanduser().resolve()
        if args.provenance
        else ROOT
        / "model_artifacts"
        / args.profile_key
        / "provenance.json"
    )
    evidence_path = Path(args.validation_evidence).expanduser().resolve()

    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    profile = lock["models"][args.profile_key]
    entry = registry["models"][args.profile_key]
    deployment = profile["deployment_profile"]

    artifact = resolve_artifact(str(entry["artifact"]), registry_path)
    actual_artifact_hash = sha256_file(artifact)
    if actual_artifact_hash != str(entry.get("sha256") or ""):
        raise RuntimeError("GGUF registry hash does not match the artifact")
    if actual_artifact_hash != str(provenance.get("artifact_sha256") or ""):
        raise RuntimeError("GGUF provenance hash does not match the artifact")
    if provenance.get("upstream_revision") != profile.get("model_revision"):
        raise RuntimeError("source revision differs between provenance and lock")
    if provenance.get("llama_cpp_commit") != lock["runtime"]["backend_revision"]:
        raise RuntimeError("llama.cpp conversion commit differs from the runtime lock")

    expected_deployment = {
        field: deployment.get(field) for field in DEPLOYMENT_FIELDS
    }
    actual_deployment = {field: entry.get(field) for field in expected_deployment}
    if actual_deployment != expected_deployment:
        raise RuntimeError("registry deployment fields differ from the frozen profile")

    if evidence.get("status") != "validated":
        raise RuntimeError("validation evidence status must be validated")
    evidence_schema = int(evidence.get("schema_version") or 0)
    if evidence_schema not in (2, 3):
        raise RuntimeError("validation evidence schema must be version 2 or 3")
    if evidence.get("profile_key") != args.profile_key:
        raise RuntimeError("validation evidence profile key mismatch")
    if evidence.get("artifact_sha256") != actual_artifact_hash:
        raise RuntimeError("validation evidence artifact hash mismatch")
    if evidence.get("llama_cpp_commit") != lock["runtime"]["backend_revision"]:
        raise RuntimeError("validation evidence llama.cpp commit mismatch")
    if not evidence.get("gpu_uuid"):
        raise RuntimeError("validation evidence lacks an exact GPU UUID")
    evidence_gpu_uuids = list(
        evidence.get("gpu_uuids") or [evidence.get("gpu_uuid")]
    )
    expected_gpu_count = int(expected_deployment.get("gpu_count") or 1)
    if len(evidence_gpu_uuids) != expected_gpu_count:
        raise RuntimeError("validation evidence GPU count mismatch")
    if len(evidence_gpu_uuids) != len(set(evidence_gpu_uuids)):
        raise RuntimeError("validation evidence contains duplicate GPU UUIDs")
    if expected_gpu_count > 1 and evidence_schema != 3:
        raise RuntimeError("multi-GPU validation requires evidence schema 3")
    if evidence.get("deployment_profile") != expected_deployment:
        raise RuntimeError("validation evidence deployment profile mismatch")
    if not evidence.get("chat_template_validated"):
        raise RuntimeError("chat template was not validated")
    target_tokens = int(expected_deployment["context_size"]) - 1
    if int(evidence.get("tokens_requested") or 0) != target_tokens:
        raise RuntimeError("validation did not request the full native context")
    if int(evidence.get("tokens_evaluated") or 0) != target_tokens:
        raise RuntimeError("validation did not evaluate the full native context")
    if evidence.get("prompt_tokens_truncated") is not False:
        raise RuntimeError("validation dropped prompt tokens")
    if evidence.get("prompt_prefill_complete") is not True:
        raise RuntimeError("validation prompt prefill was incomplete")
    if evidence.get("context_shift_disabled") is not True:
        raise RuntimeError("validation did not disable context shifting")
    if (
        evidence.get("server_reported_truncated") is True
        and evidence.get("capacity_boundary_stop") is not True
    ):
        raise RuntimeError("validation reported unexplained server truncation")
    if not evidence.get("safety_margin_validated"):
        raise RuntimeError("validation did not preserve the frozen VRAM margin")
    peak = evidence.get("measured_peak_vram_gib")
    if not isinstance(peak, (int, float)) or peak <= 0:
        raise RuntimeError("validation evidence lacks measured_peak_vram_gib")
    if not evidence.get("full_context_prefill_completed"):
        raise RuntimeError("full native-context prefill was not validated")

    profile["model_artifact_sha256"] = provenance[
        "source_weight_manifest_sha256"
    ]
    profile["quantized_artifact_sha256"] = actual_artifact_hash
    profile["measured_peak_vram_gib"] = float(peak)
    profile["deployment_status"] = "validated"
    profile["deployment_profile"]["validation_status"] = "validated"
    profile["deployment_profile"]["validation_evidence_sha256"] = sha256_file(
        evidence_path
    )
    profile["profile_sha256"] = canonical_hash(
        profile_payload(args.profile_key, profile, lock)
    )
    lock["status"] = "profiles_partially_validated"

    temporary = lock_path.with_suffix(lock_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(lock, sort_keys=False, width=100),
        encoding="utf-8",
    )
    temporary.replace(lock_path)
    print("Finalized profile:", args.profile_key)
    print("Profile SHA-256:", profile["profile_sha256"])


if __name__ == "__main__":
    main()
