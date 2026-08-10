#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision one model using only its frozen lock profile."
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
        "--output-root",
        default=str(ROOT / "model_artifacts"),
    )
    parser.add_argument(
        "--work-root",
        help="Scratch root for download, conversion and quantization intermediates.",
    )
    parser.add_argument("--existing-gguf")
    parser.add_argument("--storage-reserve-gib", type=float, default=5.0)
    parser.add_argument("--skip-storage-check", action="store_true")
    parser.add_argument("--keep-source-snapshot", action="store_true")
    parser.add_argument("--keep-failed-intermediates", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    lock = yaml.safe_load(Path(args.lock).read_text(encoding="utf-8"))
    profiles = lock.get("models") if isinstance(lock, dict) else None
    if not isinstance(profiles, dict) or args.profile_key not in profiles:
        raise ValueError("Unknown frozen profile: %s" % args.profile_key)
    profile = profiles[args.profile_key]
    deployment = profile.get("deployment_profile")
    if not isinstance(deployment, dict):
        raise ValueError("Frozen profile has no deployment_profile")

    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "provision_gguf.py"),
        "--profile-key",
        args.profile_key,
        "--model-id",
        str(profile["model_id"]),
        "--revision",
        str(profile["model_revision"]),
        "--registry",
        str(Path(args.registry).expanduser().resolve()),
        "--output-root",
        str(Path(args.output_root).expanduser().resolve()),
        "--context-size",
        str(int(deployment["context_size"])),
        "--parallel",
        str(int(deployment["parallel"])),
        "--cache-type-k",
        str(deployment["cache_type_k"]),
        "--cache-type-v",
        str(deployment["cache_type_v"]),
        "--fit-target-mib",
        str(int(deployment["fit_target_mib"])),
        "--storage-reserve-gib",
        str(args.storage_reserve_gib),
    ]
    if args.work_root:
        command.extend(
            ["--work-root", str(Path(args.work_root).expanduser().resolve())]
        )
    if args.existing_gguf:
        command.extend(["--existing-gguf", args.existing_gguf])
    if args.skip_storage_check:
        command.append("--skip-storage-check")
    if args.keep_source_snapshot:
        command.append("--keep-source-snapshot")
    if args.keep_failed_intermediates:
        command.append("--keep-failed-intermediates")
    if args.token:
        command.extend(["--token", args.token])

    print("Provisioning locked profile:", args.profile_key)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
