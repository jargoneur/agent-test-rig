from __future__ import annotations

import json
from pathlib import Path

import yaml

from harness.reproducibility import experiment_source_sha256
from scripts.launch_gate import (
    check_profiles,
    check_task_cache,
    pending_fields,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent


def complete_profile(context_size: int = 4096):
    return {
        "model_revision": "b" * 40,
        "model_artifact_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
        "chat_template_sha256": "3" * 64,
        "quantized_artifact_sha256": "4" * 64,
        "profile_sha256": "5" * 64,
        "measured_peak_vram_gib": 10.5,
        "deployment_status": "validated",
        "input_context_tokens": context_size,
        "resolved_generation_options": {
            "chat_template_kwargs": {"enable_thinking": True}
        },
        "deployment_profile": {
            "context_size": context_size,
            "validation_status": "validated",
        },
    }


def test_pending_fields_requires_validated_deployment():
    profile = complete_profile()

    assert pending_fields(profile) == []

    profile["deployment_profile"]["validation_status"] = "pending"
    assert pending_fields(profile) == [
        "deployment_profile.validation_status"
    ]


def test_check_profiles_accepts_eleven_complete_locked_profiles(tmp_path):
    source_hash = experiment_source_sha256()
    evidence_path = tmp_path / "recovery.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "validated",
                "harness_source_sha256": source_hash,
                "five_condition_block_completed": True,
                "whole_block_restart_validated": True,
                "manual_stop_resume_validated": True,
                "scheduler_backup_restore_validated": True,
                "provenance_mismatch_rejected": True,
            }
        ),
        encoding="utf-8",
    )
    lock_path = tmp_path / "profiles.yml"
    lock_path.write_text(
        yaml.safe_dump(
            {
                "validation_gate": {
                    "benchmark_runs_allowed": True,
                    "harness_source_sha256": source_hash,
                    "distributed_recovery_evidence": str(evidence_path),
                    "distributed_recovery_evidence_sha256": sha256_file(
                        evidence_path
                    ),
                },
                "models": {
                    "model_%02d" % index: complete_profile()
                    for index in range(11)
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    errors = []

    lock = check_profiles(
        {"model_profile_lock": str(lock_path)},
        tmp_path / "experiment.yml",
        errors,
    )

    assert len(lock["models"]) == 11
    assert errors == []


def test_check_profiles_rejects_closed_gate_and_context_mismatch(tmp_path):
    profiles = {
        "model_%02d" % index: complete_profile()
        for index in range(11)
    }
    profiles["model_00"]["deployment_profile"]["context_size"] = 2048
    lock_path = tmp_path / "profiles.yml"
    lock_path.write_text(
        yaml.safe_dump(
            {
                "validation_gate": {"benchmark_runs_allowed": False},
                "models": profiles,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    errors = []

    check_profiles(
        {"model_profile_lock": str(lock_path)},
        tmp_path / "experiment.yml",
        errors,
    )

    assert "model profile validation gate still blocks benchmark runs" in errors
    assert any("deployment context differs" in error for error in errors)


def test_frozen_contextbench_cache_passes_launch_gate_checks():
    errors = []
    config_path = ROOT / "experiments" / "contextbench_core.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    manifest = check_task_cache(config, config_path, errors)

    assert manifest["selection"]["count"] == 150
    assert errors == []