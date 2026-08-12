#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))


import yaml

from harness.experiment_jobs import group_jobs_by_block, load_jobs
from harness.reproducibility import (
    PROJECT_ROOT,
    experiment_source_sha256,
    git_metadata,
)


EXPECTED_SCAFFOLDS = {
    "no_scaffold",
    "sweagent_last5",
    "aider_repomap",
    "agentless_localization",
    "aider_chat_summary",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, relative_to: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    root_path = (PROJECT_ROOT / path).resolve()
    if root_path.exists():
        return root_path
    return (relative_to.parent / path).resolve()


def pending_fields(profile: Dict[str, Any]) -> List[str]:
    required = (
        "model_revision",
        "model_artifact_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "quantized_artifact_sha256",
        "profile_sha256",
        "measured_peak_vram_gib",
        "deployment_status",
    )
    unresolved = []
    for field in required:
        value = profile.get(field)
        if value in (None, "") or str(value).startswith("pending_"):
            unresolved.append(field)
    deployment = profile.get("deployment_profile")
    if not isinstance(deployment, dict):
        unresolved.append("deployment_profile")
    elif deployment.get("validation_status") != "validated":
        unresolved.append("deployment_profile.validation_status")
    return unresolved


def check_task_cache(
    config: Dict[str, Any],
    config_path: Path,
    errors: List[str],
) -> Dict[str, Any]:
    source = config.get("task_source") or {}
    cache = resolve_path(
        str(source.get("cache") or "benchmarks/contextbench/tasks.jsonl"),
        config_path,
    )
    manifest_path = cache.parent / "manifest.json"
    if not cache.is_file() or not manifest_path.is_file():
        errors.append("frozen ContextBench cache or manifest is missing")
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tasks_sha256") != sha256_file(cache):
        errors.append("ContextBench task cache hash does not match its manifest")
    selection = manifest.get("selection") or {}
    distributions = selection.get("distributions") or {}
    if int(selection.get("count") or 0) != 150:
        errors.append("ContextBench selection is not exactly 150 tasks")
    if distributions.get("complexity_bin") != {
        "easy": 50,
        "hard": 50,
        "intermediate": 50,
    }:
        errors.append("ContextBench complexity bins are not 50/50/50")
    if distributions.get("source_declared_task_type") != {
        "evolution_feature": 36,
        "evolution_refactor": 2,
        "maintenance_bugfix": 112,
    }:
        errors.append("ContextBench task-type census/fill distribution changed")
    if manifest.get("huggingface_dataset_revision") != (
        "c2855792b006af41c67202d33883fb9d46362853"
    ):
        errors.append("ContextBench Hugging Face revision is not frozen")
    return manifest


def check_profiles(
    config: Dict[str, Any],
    config_path: Path,
    errors: List[str],
) -> Dict[str, Any]:
    lock_path = resolve_path(str(config.get("model_profile_lock") or ""), config_path)
    if not lock_path.is_file():
        errors.append("model profile lock is missing")
        return {}
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    models = lock.get("models") if isinstance(lock, dict) else None
    if not isinstance(models, dict) or len(models) != 11:
        errors.append("model profile lock must contain exactly eleven models")
        return {}
    gate = lock.get("validation_gate") or {}
    if not bool(gate.get("benchmark_runs_allowed")):
        errors.append("model profile validation gate still blocks benchmark runs")
    else:
        source_hash = experiment_source_sha256()
        if gate.get("harness_source_sha256") != source_hash:
            errors.append(
                "tracked experiment source differs from recovery evidence"
            )
        evidence_reference = str(
            gate.get("distributed_recovery_evidence") or ""
        )
        evidence_path = resolve_path(evidence_reference, lock_path)
        if not evidence_path.is_file():
            errors.append("distributed recovery evidence is missing")
        else:
            evidence_hash = sha256_file(evidence_path)
            if evidence_hash != gate.get(
                "distributed_recovery_evidence_sha256"
            ):
                errors.append("distributed recovery evidence hash mismatch")
            evidence = json.loads(
                evidence_path.read_text(encoding="utf-8")
            )
            required = {
                "schema_version": 1,
                "status": "validated",
                "harness_source_sha256": source_hash,
                "five_condition_block_completed": True,
                "whole_block_restart_validated": True,
                "manual_stop_resume_validated": True,
                "scheduler_backup_restore_validated": True,
                "provenance_mismatch_rejected": True,
            }
            for field, expected in required.items():
                if evidence.get(field) != expected:
                    errors.append(
                        "distributed recovery evidence %s mismatch"
                        % field
                    )
    for model_id, profile in models.items():
        unresolved = pending_fields(profile)
        if unresolved:
            errors.append(
                "%s unresolved: %s" % (model_id, ", ".join(unresolved))
            )
        deployment = profile.get("deployment_profile") or {}
        if deployment.get("context_size") != profile.get("input_context_tokens"):
            errors.append("%s deployment context differs from its frozen profile" % model_id)
        options = profile.get("resolved_generation_options") or {}
        if (options.get("chat_template_kwargs") or {}).get("enable_thinking") is not True:
            errors.append("%s does not enforce its frozen thinking mode" % model_id)
    return lock


def check_jobs(
    config: Dict[str, Any],
    config_path: Path,
    profile_lock: Dict[str, Any],
    errors: List[str],
) -> Dict[str, Any]:
    manifest_path = resolve_path(str(config.get("manifest") or ""), config_path)
    if not manifest_path.is_file():
        errors.append("final job manifest is missing")
        return {}
    try:
        jobs = load_jobs(str(manifest_path))
        blocks = group_jobs_by_block(jobs)
    except Exception as error:
        errors.append("job manifest is invalid: %s" % error)
        return {}

    if len(jobs) != 24750:
        errors.append("job manifest does not contain exactly 24,750 runs")
    if len(blocks) != 4950:
        errors.append("job manifest does not contain exactly 4,950 paired blocks")
    models = profile_lock.get("models") or {}
    current = git_metadata()
    if not current.get("commit") or current.get("dirty"):
        errors.append("repository must be at a clean, exact commit")
    for block in blocks:
        scaffold_names = {job["scaffold"] for job in block["jobs"]}
        if scaffold_names != EXPECTED_SCAFFOLDS:
            errors.append("block %s does not contain the exact five conditions" % block["block_id"])
            break
        if block.get("harness_commit") != current.get("commit"):
            errors.append("block %s was planned from a different harness commit" % block["block_id"])
            break
        if int(block.get("model_timeout_seconds") or 0) != 21600:
            errors.append("block %s has the wrong model timeout" % block["block_id"])
            break
        recovery = block.get("recovery") or {}
        if recovery != {
            "whole_block_restart": True,
            "reuse_partial_results": False,
            "max_infrastructure_retries": 2,
        }:
            errors.append("block %s has the wrong recovery contract" % block["block_id"])
            break
        model_id = str(block["model_id"])
        profile = models.get(model_id) or {}
        model = block.get("model") or {}
        if (
            model.get("profile_sha256") != profile.get("profile_sha256")
            or model.get("quantized_artifact_sha256")
            != profile.get("quantized_artifact_sha256")
            or model.get("model_revision") != profile.get("model_revision")
        ):
            errors.append("block %s model provenance differs from the lock" % block["block_id"])
            break

    return {
        "manifest": str(manifest_path),
        "jobs": len(jobs),
        "blocks": len(blocks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refuse a core launch unless every scientific and recovery invariant holds."
    )
    parser.add_argument(
        "--config",
        default="experiments/contextbench_core.yml",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a mapping")

    errors: List[str] = []
    if set(config.get("scaffolds") or []) != EXPECTED_SCAFFOLDS:
        errors.append("experiment config does not define the exact five conditions")
    if int(config.get("repeats") or 0) != 3:
        errors.append("experiment config must use three paired repeats")
    if int(config.get("max_infrastructure_retries", -1)) != 2:
        errors.append("experiment config must cap infrastructure retries at two")
    if int(config.get("model_timeout_seconds", 0)) != 21600:
        errors.append("experiment config must use the frozen six-hour model timeout")
    design = config.get("design") or {}
    if design.get("cost_is_research_factor") is not False:
        errors.append("cost must not be encoded as a research factor")
    if design.get("reuse_partial_results") is not False:
        errors.append("partial paired-block results must never be reused")

    task_manifest = check_task_cache(config, config_path, errors)
    profile_lock = check_profiles(config, config_path, errors)
    jobs = check_jobs(config, config_path, profile_lock, errors)
    report = {
        "ok": not errors,
        "config": str(config_path),
        "task_manifest_sha256": (
            sha256_file(
                resolve_path(
                    str((config.get("task_source") or {}).get("cache")),
                    config_path,
                ).parent
                / "manifest.json"
            )
            if task_manifest
            else None
        ),
        "profile_status": profile_lock.get("status") if profile_lock else None,
        "jobs": jobs,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
