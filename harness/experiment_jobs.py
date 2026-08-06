from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from harness.reproducibility import derive_run_seed, git_metadata, utc_now_iso


JOB_SCHEMA_VERSION = 2


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_model_spec(model: Any, default_backend: str = "ollama") -> Dict[str, Any]:
    if isinstance(model, str):
        return {
            "id": model,
            "name": model,
            "backend": default_backend,
        }

    if not isinstance(model, dict):
        raise TypeError("Each model must be a string or mapping")

    normalized = dict(model)
    name = normalized.get("name") or normalized.get("model")
    model_id = normalized.get("id") or name
    if not name or not model_id:
        raise ValueError("Model mapping must define 'name' or 'model'")

    model_options = normalized.get("generation_options")
    if model_options is not None and not isinstance(model_options, dict):
        raise TypeError("Model generation_options must be a mapping")

    normalized["id"] = str(model_id)
    normalized["name"] = str(name)
    normalized.setdefault("backend", default_backend)
    return normalized


def model_identity(model_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return fields that define the scientific model/deployment condition.

    Scheduling hints and endpoint addresses are intentionally excluded.
    """
    excluded = {
        "resource_class",
        "preferred_worker",
        "base_url",
        "host",
        "api_key",
        "notes",
    }
    return {
        key: value
        for key, value in sorted(model_spec.items())
        if key not in excluded
    }


def block_identity(job: Dict[str, Any]) -> Dict[str, Any]:
    """Identity shared by all scaffold conditions of one paired block."""
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "task_id": job["task_id"],
        "model": model_identity(job["model"]),
        "repeat": int(job["repeat"]),
        "run_seed": int(job["run_seed"]),
        "max_steps": int(job["max_steps"]),
        "generation_options": job["generation_options"],
        "harness_commit": job.get("harness_commit", ""),
    }


def job_identity(job: Dict[str, Any]) -> Dict[str, Any]:
    identity = block_identity(job)
    identity["scaffold"] = job["scaffold"]
    return identity


def make_block_id(job: Dict[str, Any]) -> str:
    digest = sha256_json(block_identity(job))
    return "block_" + digest[:24]


def make_run_id(job: Dict[str, Any]) -> str:
    digest = sha256_json(job_identity(job))
    return "run_" + digest[:24]


def validate_job(job: Dict[str, Any]) -> None:
    required = {
        "schema_version",
        "block_id",
        "run_id",
        "task_id",
        "model",
        "scaffold",
        "repeat",
        "run_seed",
        "max_steps",
        "generation_options",
    }
    missing = sorted(required.difference(job))
    if missing:
        raise ValueError("Job is missing required fields: " + ", ".join(missing))
    if job["schema_version"] != JOB_SCHEMA_VERSION:
        raise ValueError("Unsupported job schema version: %r" % job["schema_version"])
    expected_block = make_block_id(job)
    if job["block_id"] != expected_block:
        raise ValueError(
            "Block ID does not match block identity: %s != %s"
            % (job["block_id"], expected_block)
        )
    expected_run = make_run_id(job)
    if job["run_id"] != expected_run:
        raise ValueError(
            "Run ID does not match job identity: %s != %s"
            % (job["run_id"], expected_run)
        )


def build_jobs(
    tasks: Sequence[str],
    models: Sequence[Any],
    scaffolds: Sequence[str],
    repeats: int,
    max_steps: int,
    base_seed: int,
    generation_options: Dict[str, Any],
    experiment_id: str,
    wave_size: int = 25,
    harness_commit: Optional[str] = None,
    default_backend: str = "ollama",
) -> List[Dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if wave_size < 1:
        raise ValueError("wave_size must be at least 1")
    if not scaffolds:
        raise ValueError("At least one scaffold is required")
    if len(set(scaffolds)) != len(scaffolds):
        raise ValueError("Scaffold names must be unique")

    normalized_models = [
        normalize_model_spec(model, default_backend=default_backend)
        for model in models
    ]
    if harness_commit is None:
        harness_commit = git_metadata().get("commit", "")

    jobs: List[Dict[str, Any]] = []
    for task_index, task_id in enumerate(tasks):
        wave = task_index // wave_size + 1
        for model_index, model in enumerate(normalized_models):
            model_generation_options = dict(generation_options)
            model_generation_options.update(
                dict(model.get("generation_options") or {})
            )
            job_model = dict(model)
            job_model.pop("generation_options", None)

            for repeat in range(1, repeats + 1):
                run_seed = derive_run_seed(
                    base_seed,
                    str(task_id),
                    model["id"],
                    repeat,
                )
                for scaffold_index, scaffold in enumerate(scaffolds):
                    job: Dict[str, Any] = {
                        "schema_version": JOB_SCHEMA_VERSION,
                        "experiment_id": experiment_id,
                        "wave": wave,
                        "task_index": task_index,
                        "task_id": str(task_id),
                        "model_index": model_index,
                        "model": dict(job_model),
                        "scaffold_index": scaffold_index,
                        "scaffold": str(scaffold),
                        "repeat": repeat,
                        "run_seed": run_seed,
                        "max_steps": int(max_steps),
                        "generation_options": dict(model_generation_options),
                        "harness_commit": harness_commit or "",
                        "requirements": {
                            "resource_class": model.get("resource_class"),
                        },
                    }
                    job["block_id"] = make_block_id(job)
                    job["run_id"] = make_run_id(job)
                    jobs.append(job)

    jobs.sort(
        key=lambda job: (
            int(job.get("wave", 0)),
            int(job.get("model_index", 0)),
            int(job.get("task_index", 0)),
            int(job["repeat"]),
            int(job.get("scaffold_index", 0)),
        )
    )
    return jobs


def group_jobs_by_block(jobs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for job in jobs:
        validate_job(job)
        grouped.setdefault(job["block_id"], []).append(job)

    blocks: List[Dict[str, Any]] = []
    for block_id, block_jobs in grouped.items():
        ordered_jobs = sorted(
            block_jobs,
            key=lambda item: int(item.get("scaffold_index", 0)),
        )
        first = ordered_jobs[0]
        scaffold_names = [item["scaffold"] for item in ordered_jobs]
        if len(scaffold_names) != len(set(scaffold_names)):
            raise ValueError("Duplicate scaffold in block %s" % block_id)
        for job in ordered_jobs[1:]:
            if make_block_id(job) != block_id:
                raise ValueError("Inconsistent block identity in %s" % block_id)
        blocks.append(
            {
                "block_id": block_id,
                "experiment_id": first.get("experiment_id"),
                "wave": int(first.get("wave", 0)),
                "task_id": first["task_id"],
                "model_id": first["model"]["id"],
                "resource_class": (first.get("requirements") or {}).get(
                    "resource_class"
                ),
                "repeat": int(first["repeat"]),
                "jobs": ordered_jobs,
            }
        )

    return sorted(
        blocks,
        key=lambda block: (
            int(block.get("wave", 0)),
            str(block["model_id"]),
            str(block["task_id"]),
            int(block["repeat"]),
        ),
    )


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON in %s line %d: %s"
                    % (path, line_number, error)
                )
            if not isinstance(value, dict):
                raise ValueError(
                    "Expected JSON object in %s line %d" % (path, line_number)
                )
            yield value


def load_jobs(path: str) -> List[Dict[str, Any]]:
    jobs = list(read_jsonl(Path(path)))
    seen = set()
    for job in jobs:
        validate_job(job)
        if job["run_id"] in seen:
            raise ValueError("Duplicate run_id in manifest: %s" % job["run_id"])
        seen.add(job["run_id"])
    return jobs


def write_jobs(path: str, jobs: Iterable[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for job in jobs:
            validate_job(job)
            handle.write(canonical_json(job) + "\n")


def merge_jobs(
    existing: Sequence[Dict[str, Any]],
    new: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for job in list(existing) + list(new):
        validate_job(job)
        previous = by_id.get(job["run_id"])
        if previous is not None:
            if canonical_json(job_identity(previous)) != canonical_json(
                job_identity(job)
            ):
                raise ValueError(
                    "Conflicting definitions for run_id %s" % job["run_id"]
                )
            continue
        by_id[job["run_id"]] = job
    return sorted(
        by_id.values(),
        key=lambda job: (
            int(job.get("wave", 0)),
            int(job.get("model_index", 0)),
            int(job.get("task_index", 0)),
            int(job["repeat"]),
            int(job.get("scaffold_index", 0)),
        ),
    )


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def result_file(results_root: Path, run_id: str) -> Path:
    return results_root / "runs" / (run_id + ".json")


def result_is_complete(path: Path, run_id: Optional[str] = None) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if data.get("status") != "completed":
        return False
    if run_id is not None and data.get("run_id") != run_id:
        return False
    return True


def shard_matches(block_id: str, shard_index: int, shard_count: int) -> bool:
    """Keep every scaffold condition of a block on the same shard."""
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")
    value = int(hashlib.sha256(block_id.encode("utf-8")).hexdigest()[:16], 16)
    return value % shard_count == shard_index


def plan_metadata(
    experiment_id: str,
    config: Dict[str, Any],
    jobs: Sequence[Dict[str, Any]],
    manifest_path: str,
) -> Dict[str, Any]:
    blocks = group_jobs_by_block(jobs)
    return {
        "schema_version": 2,
        "created_at": utc_now_iso(),
        "experiment_id": experiment_id,
        "manifest_path": manifest_path,
        "job_count": len(jobs),
        "block_count": len(blocks),
        "run_ids_sha256": sha256_json([job["run_id"] for job in jobs]),
        "block_ids_sha256": sha256_json(
            [block["block_id"] for block in blocks]
        ),
        "config_sha256": sha256_json(config),
        "config": config,
        "repository": git_metadata(),
    }
