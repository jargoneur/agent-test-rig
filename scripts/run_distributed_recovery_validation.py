#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from harness.distributed_scheduler_store import DistributedSchedulerStore
from harness.experiment_jobs import build_jobs, write_jobs
from harness.reproducibility import (
    experiment_source_sha256,
    git_metadata,
    utc_now_iso,
)


ROOT = Path(__file__).resolve().parents[1]
SCAFFOLDS = [
    "no_scaffold",
    "sweagent_last5",
    "aider_repomap",
    "agentless_localization",
    "aider_chat_summary",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capabilities(harness_commit: str) -> Dict[str, Any]:
    return {
        "model_ids": ["validation-model"],
        "resource_classes": ["small"],
        "harness_commit": harness_commit,
        "model_profiles": {
            "validation-model": {
                "profile_sha256": "profile-validation",
                "quantized_artifact_sha256": "artifact-validation",
            }
        },
    }


def completed_results(claim: Dict[str, Any]):
    return [
        {
            "schema_version": 3,
            "status": "completed",
            "run_id": job["run_id"],
            "job": job,
            "result": {"termination_reason": "validation"},
            "outcome": {
                "terminal": True,
                "usable_result": True,
                "kind": "validation",
            },
        }
        for job in claim["block"]["jobs"]
    ]


def run_validation(work_root: Path, harness_commit: str) -> Dict[str, Any]:
    work_root.mkdir(parents=True, exist_ok=True)
    manifest = work_root / "jobs.jsonl"
    jobs = build_jobs(
        tasks=["recovery-validation-task"],
        models=[
            {
                "id": "validation-model",
                "name": "validation-model",
                "backend": "llama_cpp",
                "resource_class": "small",
                "profile_sha256": "profile-validation",
                "quantized_artifact_sha256": "artifact-validation",
                "model_revision": "b" * 40,
            }
        ],
        scaffolds=SCAFFOLDS,
        repeats=1,
        max_steps=1,
        base_seed=20260807,
        generation_options={},
        experiment_id="distributed-recovery-validation",
        harness_commit=harness_commit,
        max_infrastructure_retries=2,
    )
    write_jobs(str(manifest), jobs)

    database = work_root / "scheduler.sqlite3"
    store = DistributedSchedulerStore(str(database))
    store.import_manifest(str(manifest))
    exact_capabilities = capabilities(harness_commit)

    wrong_harness = capabilities("wrong-commit")
    wrong_profile = capabilities(harness_commit)
    wrong_profile["model_profiles"]["validation-model"][
        "profile_sha256"
    ] = "wrong-profile"
    wrong_artifact = capabilities(harness_commit)
    wrong_artifact["model_profiles"]["validation-model"][
        "quantized_artifact_sha256"
    ] = "wrong-artifact"
    mismatch_rejected = all(
        store.claim_block(name, candidate, 120) is None
        for name, candidate in (
            ("wrong-harness", wrong_harness),
            ("wrong-profile", wrong_profile),
            ("wrong-artifact", wrong_artifact),
        )
    )

    first_claim = store.claim_block(
        "validation-worker",
        exact_capabilities,
        120,
    )
    if first_claim is None or len(first_claim["block"]["jobs"]) != 5:
        raise RuntimeError("could not claim the exact five-condition block")
    stopped = store.fail_block(
        "validation-worker",
        first_claim["block"]["block_id"],
        first_claim["lease_token"],
        "manual validation stop",
        retry=True,
        failure_kind="operator_stop",
    )
    manual_stop_ok = (
        stopped["state"] == "pending"
        and stopped["infrastructure_failures"] == 0
    )

    store.set_paused(True, "recovery validation pause")
    reopened = DistributedSchedulerStore(str(database))
    pause_persisted = (
        reopened.pause_state().get("paused") is True
        and reopened.claim_block(
            "blocked-worker",
            exact_capabilities,
            120,
        )
        is None
    )
    reopened.set_paused(False)

    second_claim = reopened.claim_block(
        "validation-worker",
        exact_capabilities,
        120,
    )
    if second_claim is None:
        raise RuntimeError("block did not requeue after manual stop")
    whole_block_restart = (
        second_claim["attempt"] == 2
        and len(second_claim["block"]["jobs"]) == 5
        and reopened.export_results(str(work_root / "pre-results")) == 0
    )
    results = completed_results(second_claim)
    reopened.complete_block(
        "validation-worker",
        second_claim["block"]["block_id"],
        second_claim["lease_token"],
        results,
    )
    completion_status = reopened.status()
    completion_status["results"] = reopened.export_results(
        str(work_root / "completed-results")
    )
    five_completed = (
        completion_status["states"] == {"completed": 1}
        and completion_status["results"] == 5
    )

    backup = work_root / "backups" / "scheduler.sqlite3"
    backup_metadata = reopened.backup_database(str(backup))
    restored_database = work_root / "restored" / "scheduler.sqlite3"
    DistributedSchedulerStore.restore_database(
        str(backup),
        str(restored_database),
    )
    restored = DistributedSchedulerStore(str(restored_database))
    restored_status = restored.status()
    restored_status["results"] = restored.export_results(
        str(work_root / "restored-results")
    )
    backup_restore_ok = (
        restored.integrity_check(thorough=True)["ok"] is True
        and restored_status["states"] == {"completed": 1}
        and restored_status["results"] == 5
    )

    checks = {
        "five_condition_block_completed": five_completed,
        "whole_block_restart_validated": whole_block_restart,
        "manual_stop_resume_validated": (
            manual_stop_ok and pause_persisted
        ),
        "scheduler_backup_restore_validated": backup_restore_ok,
        "provenance_mismatch_rejected": mismatch_rejected,
    }
    status = "validated" if all(checks.values()) else "failed"
    return {
        "schema_version": 1,
        "status": status,
        "harness_commit": harness_commit,
        "harness_source_sha256": experiment_source_sha256(),
        **checks,
        "scaffolds": list(SCAFFOLDS),
        "first_attempt": first_claim["attempt"],
        "restart_attempt": second_claim["attempt"],
        "restored_status": restored_status,
        "backup_integrity": backup_metadata["integrity"],
        "validated_at": utc_now_iso(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate machine-checked distributed recovery evidence."
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "model_artifacts"
            / "validation"
            / "distributed-recovery.json"
        ),
    )
    args = parser.parse_args()

    repository = git_metadata()
    if not repository.get("commit") or repository.get("dirty"):
        raise SystemExit(
            "Distributed recovery validation requires a clean exact commit."
        )
    output = Path(args.output).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="agent-rig-recovery-") as temporary:
        evidence = run_validation(
            Path(temporary),
            str(repository["commit"]),
        )
    if evidence["status"] != "validated":
        raise SystemExit(json.dumps(evidence, indent=2, sort_keys=True))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print("Evidence:", output)
    print("SHA-256:", sha256_file(output))


if __name__ == "__main__":
    main()
