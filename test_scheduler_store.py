from harness.experiment_jobs import build_jobs, write_jobs
from harness.scheduler_store import SchedulerStore


def test_scheduler_claims_and_completes_full_scaffold_block(tmp_path):
    jobs = build_jobs(
        tasks=["task-a"],
        models=[
            {
                "id": "model-a-q4",
                "name": "model-a",
                "backend": "ollama",
                "resource_class": "small",
            }
        ],
        scaffolds=["baseline", "repo_map"],
        repeats=1,
        max_steps=8,
        base_seed=1,
        generation_options={"temperature": 0.2},
        experiment_id="test",
        harness_commit="abc123",
    )
    manifest = tmp_path / "jobs.jsonl"
    write_jobs(str(manifest), jobs)

    store = SchedulerStore(str(tmp_path / "scheduler.sqlite3"))
    imported = store.import_manifest(str(manifest))
    assert imported == {"inserted": 1, "unchanged": 0}

    capabilities = {"resource_classes": ["small"]}
    store.register_worker("worker-a", capabilities)
    claim = store.claim_block(
        "worker-a",
        capabilities,
        lease_seconds=120,
    )
    assert claim is not None
    assert len(claim["block"]["jobs"]) == 2

    results = [
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": job["run_id"],
            "job": job,
            "result": {"final_tests_passed": True},
        }
        for job in claim["block"]["jobs"]
    ]
    store.complete_block(
        "worker-a",
        claim["block"]["block_id"],
        claim["lease_token"],
        results,
    )

    status = store.status()
    assert status["states"] == {"completed": 1}
    assert store.claim_block(
        "worker-a",
        capabilities,
        lease_seconds=120,
    ) is None
