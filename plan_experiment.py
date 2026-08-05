from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from harness.experiment_jobs import (
    build_jobs,
    load_jobs,
    merge_jobs,
    plan_metadata,
    write_jobs,
)
from harness.model_profiles import resolve_models_from_lock
from harness.reproducibility import write_json


def _resolve_path(value: str, config_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return config_path.parent / candidate


def _resolve_models(config, config_path: Path):
    lock_value = config.get("model_profile_lock")
    if not lock_value:
        return list(config["models"])
    if config.get("models"):
        raise ValueError(
            "Use either models or model_profile_lock, not both"
        )

    lock_path = _resolve_path(str(lock_value), config_path)
    profile_keys = config.get("model_profile_keys")
    if profile_keys is not None and not isinstance(profile_keys, list):
        raise TypeError("model_profile_keys must be a list")
    return resolve_models_from_lock(
        str(lock_path),
        profile_keys=profile_keys,
        allow_pending=bool(config.get("allow_pending_model_profiles", False)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create or extend a deterministic distributed job manifest."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a YAML mapping")

    experiment_id = config.get("experiment_id")
    if not experiment_id:
        experiment_id = datetime.now(timezone.utc).strftime("experiment_%Y%m%d_%H%M%S")

    output = args.output or config.get("manifest") or (
        "jobs/%s.jsonl" % experiment_id
    )
    generation_options = dict(config.get("generation_options") or {})
    models = _resolve_models(config, config_path)
    jobs = build_jobs(
        tasks=list(config["tasks"]),
        models=models,
        scaffolds=list(config["scaffolds"]),
        repeats=int(config.get("repeats", 1)),
        max_steps=int(config.get("max_steps", 8)),
        base_seed=int(config.get("base_seed", 12345)),
        generation_options=generation_options,
        experiment_id=str(experiment_id),
        wave_size=int(config.get("wave_size", 25)),
        default_backend=str(config.get("default_backend", "ollama")),
    )

    output_path = Path(output)
    if args.extend and output_path.exists():
        jobs = merge_jobs(load_jobs(str(output_path)), jobs)

    write_jobs(str(output_path), jobs)
    metadata_path = output_path.with_suffix(".manifest.json")
    write_json(
        metadata_path,
        plan_metadata(str(experiment_id), config, jobs, str(output_path)),
    )

    print("Experiment:", experiment_id)
    print("Jobs:", len(jobs))
    print("Manifest:", output_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
