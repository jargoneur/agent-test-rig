#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys
from typing import Any

import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from harness.reproducibility import experiment_source_sha256


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def unresolved(profile: dict[str, Any]) -> list[str]:
    fields = (
        "model_artifact_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "quantized_artifact_sha256",
        "profile_sha256",
        "measured_peak_vram_gib",
    )
    result = [
        field
        for field in fields
        if profile.get(field) in (None, "")
        or str(profile.get(field)).startswith("pending_")
    ]
    if profile.get("deployment_status") != "validated":
        result.append("deployment_status")
    if (
        (profile.get("deployment_profile") or {}).get("validation_status")
        != "validated"
    ):
        result.append("deployment_profile.validation_status")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the benchmark gate only after all profiles and machine-checked recovery evidence pass."
    )
    parser.add_argument(
        "--lock",
        default=str(ROOT / "experiments" / "model_profiles.lock.yml"),
    )
    parser.add_argument(
        "--distributed-recovery-evidence",
        "--distributed-smoke-evidence",
        dest="distributed_recovery_evidence",
        required=True,
    )
    args = parser.parse_args()

    lock_path = Path(args.lock).expanduser().resolve()
    evidence_path = Path(args.distributed_recovery_evidence).expanduser().resolve()
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    failures = []
    for key, profile in lock["models"].items():
        pending = unresolved(profile)
        if pending:
            failures.append("%s: %s" % (key, ", ".join(pending)))
    required_evidence = {
        "schema_version": 1,
        "status": "validated",
        "harness_commit": current_commit(),
        "harness_source_sha256": experiment_source_sha256(),
        "five_condition_block_completed": True,
        "whole_block_restart_validated": True,
        "manual_stop_resume_validated": True,
        "scheduler_backup_restore_validated": True,
        "provenance_mismatch_rejected": True,
    }
    for field, expected in required_evidence.items():
        if evidence.get(field) != expected:
            failures.append(
                "distributed smoke %s is %r, expected %r"
                % (field, evidence.get(field), expected)
            )
    if failures:
        raise SystemExit(
            "Model profile gate remains closed:\n- " + "\n- ".join(failures)
        )

    lock["validation_gate"]["benchmark_runs_allowed"] = True
    try:
        evidence_reference = str(evidence_path.relative_to(ROOT))
    except ValueError:
        evidence_reference = str(evidence_path)
    lock["validation_gate"]["distributed_recovery_evidence"] = (
        evidence_reference
    )
    lock["validation_gate"]["distributed_recovery_evidence_sha256"] = (
        sha256_file(evidence_path)
    )
    lock["validation_gate"]["distributed_recovery_harness_commit"] = (
        evidence["harness_commit"]
    )
    lock["validation_gate"]["harness_source_sha256"] = evidence[
        "harness_source_sha256"
    ]
    lock["status"] = "benchmark_ready"
    temporary = lock_path.with_suffix(lock_path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(lock, sort_keys=False, width=100),
        encoding="utf-8",
    )
    temporary.replace(lock_path)
    print("Benchmark profile gate opened for all eleven models.")


if __name__ == "__main__":
    main()
