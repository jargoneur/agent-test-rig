from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from harness.contextbench_tasks import contextbench_task_ids
from harness.experiment_jobs import (
    build_jobs,
    load_jobs,
    merge_jobs,
    plan_metadata,
    write_jobs,
)
from harness.model_profiles import resolve_models_from_lock
from harness.reproducibility import PROJECT_ROOT, write_json


def _resolve_path(value: str, config_path: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    root_candidate = PROJECT_ROOT / candidate
    if root_candidate.exists():
        return root_candidate
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


def _resolve_tasks(config, config_path: Path):
    source = config.get("task_source")
    if not source:
        return list(config["tasks"])
    if config.get("tasks"):
        raise ValueError("Use either tasks or task_source, not both")
    if not isinstance(source, dict):
        raise TypeError("task_source must be a mapping")
    source_type = str(source.get("type") or "")
    if source_type != "contextbench":
        raise ValueError("Unsupported task_source.type: %s" % source_type)
    cache_value = source.get("cache", "benchmarks/contextbench/tasks.jsonl")
    cache_path = _resolve_path(str(cache_value), config_path)
    limit = source.get("limit")
    return contextbench_task_ids(
        str(cache_path),
        limit=int(limit) if limit is not None else None,
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
    tasks = _resolve_tasks(config, config_path)
    jobs = build_jobs(
        tasks=tasks,
        models=models,
        scaffolds=list(config["scaffolds"]),
        repeats=int(config.get("repeats", 1)),
        max_steps=int(config.get("max_steps", 8)),
        base_seed=int(config.get("base_seed", 12345)),
        generation_options=generation_options,
        experiment_id=str(experiment_id),
        wave_size=int(config.get("wave_size", 25)),
        default_backend=str(config.get("default_backend", "ollama")),
        max_infrastructure_retries=int(
            config.get("max_infrastructure_retries", 2)
        ),
        model_timeout_seconds=int(config.get("model_timeout_seconds", 600)),
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
    print("Tasks:", len(tasks))
    print("Jobs:", len(jobs))
    print("Manifest:", output_path)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()
