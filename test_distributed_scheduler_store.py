import threading

import pytest

from harness.distributed_scheduler_store import DistributedSchedulerStore
from harness.experiment_jobs import build_jobs, write_jobs


def make_manifest(tmp_path, tasks=None, resource_class="small"):
    jobs = build_jobs(
        tasks=tasks or ["task-a"],
        models=[
            {
                "id": "model-a-q8",
                "name": "model-a",
                "backend": "llama_cpp",
                "resource_class": resource_class,
            }
        ],
        scaffolds=["baseline", "repo_map"],
        repeats=1,
        max_steps=8,
        base_seed=1,
        generation_options={"temperature": 0.2},
        experiment_id="distributed-test",
        harness_commit="abc123",
    )
    manifest = tmp_path / "jobs.jsonl"
    write_jobs(str(manifest), jobs)
    return manifest


def completed_results(claim):
    return [
        {
            "schema_version": 1,
            "status": "completed",
            "run_id": job["run_id"],
            "job": job,
            "result": {"final_tests_passed": True},
        }
        for job in claim["block"]["jobs"]
    ]


def test_model_and_resource_class_must_both_match(tmp_path):
    manifest = make_manifest(tmp_path, resource_class="small")
    store = DistributedSchedulerStore(str(tmp_path / "scheduler.sqlite3"))
    store.import_manifest(str(manifest))

    assert store.claim_block(
        "wrong-model",
        {"model_ids": ["another-model"], "resource_classes": ["small"]},
        lease_seconds=120,
    ) is None
    assert store.claim_block(
        "wrong-class",
        {"model_ids": ["model-a-q8"], "resource_classes": ["large"]},
        lease_seconds=120,
    ) is None

    claim = store.claim_block(
        "matching-worker",
        {"model_ids": ["model-a-q8"], "resource_classes": ["small"]},
        lease_seconds=120,
    )
    assert claim is not None


def test_pause_persists_and_blocks_new_claims(tmp_path):
    manifest = make_manifest(tmp_path)
    database = tmp_path / "scheduler.sqlite3"
    store = DistributedSchedulerStore(str(database))
    store.import_manifest(str(manifest))
    paused = store.set_paused(True, "operator test")
    assert paused["paused"] is True

    reopened = DistributedSchedulerStore(str(database))
    assert reopened.pause_state()["reason"] == "operator test"
    assert reopened.claim_block(
        "worker-a",
        {"model_ids": ["model-a-q8"], "resource_classes": ["small"]},
        lease_seconds=120,
    ) is None

    reopened.set_paused(False)
    assert reopened.claim_block(
        "worker-a",
        {"model_ids": ["model-a-q8"], "resource_classes": ["small"]},
        lease_seconds=120,
    ) is not None


def test_completion_is_idempotent_but_rejects_conflicting_payload(tmp_path):
    manifest = make_manifest(tmp_path)
    store = DistributedSchedulerStore(str(tmp_path / "scheduler.sqlite3"))
    store.import_manifest(str(manifest))
    capabilities = {"model_ids": ["model-a-q8"], "resource_classes": ["small"]}
    claim = store.claim_block("worker-a", capabilities, lease_seconds=120)
    assert claim is not None
    results = completed_results(claim)

    store.complete_block(
        "worker-a", claim["block"]["block_id"], claim["lease_token"], results
    )
    store.complete_block(
        "worker-a", claim["block"]["block_id"], claim["lease_token"], results
    )

    conflicting = [dict(value) for value in results]
    conflicting[0] = dict(conflicting[0])
    conflicting[0]["result"] = {"final_tests_passed": False}
    with pytest.raises(ValueError, match="different result payloads"):
        store.complete_block(
            "worker-a",
            claim["block"]["block_id"],
            claim["lease_token"],
            conflicting,
        )


def test_online_backup_restores_completed_queue(tmp_path):
    manifest = make_manifest(tmp_path)
    database = tmp_path / "scheduler.sqlite3"
    store = DistributedSchedulerStore(str(database))
    store.import_manifest(str(manifest))
    capabilities = {"model_ids": ["model-a-q8"], "resource_classes": ["small"]}
    claim = store.claim_block("worker-a", capabilities, lease_seconds=120)
    assert claim is not None
    store.complete_block(
        "worker-a",
        claim["block"]["block_id"],
        claim["lease_token"],
        completed_results(claim),
    )

    backup = tmp_path / "backups" / "scheduler.sqlite3"
    metadata = store.backup_database(str(backup))
    assert metadata["integrity"] == "ok"

    restored_path = tmp_path / "restored" / "scheduler.sqlite3"
    DistributedSchedulerStore.restore_database(str(backup), str(restored_path))
    restored = DistributedSchedulerStore(str(restored_path))
    assert restored.integrity_check(thorough=True)["ok"] is True
    assert restored.status()["states"] == {"completed": 1}


def test_concurrent_workers_never_receive_same_block(tmp_path):
    manifest = make_manifest(
        tmp_path,
        tasks=["task-%02d" % index for index in range(12)],
    )
    store = DistributedSchedulerStore(str(tmp_path / "scheduler.sqlite3"))
    store.import_manifest(str(manifest))
    capabilities = {"model_ids": ["model-a-q8"], "resource_classes": ["small"]}
    claimed = []
    lock = threading.Lock()

    def claim(worker_index):
        value = store.claim_block(
            "worker-%02d" % worker_index,
            capabilities,
            lease_seconds=120,
        )
        with lock:
            claimed.append(value["block"]["block_id"] if value else None)

    threads = [threading.Thread(target=claim, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert None not in claimed
    assert len(claimed) == len(set(claimed)) == 12
