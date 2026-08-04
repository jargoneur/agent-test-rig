from harness.experiment_jobs import (
    build_jobs,
    make_run_id,
    merge_jobs,
    shard_matches,
)


def test_jobs_are_paired_across_scaffolds():
    jobs = build_jobs(
        tasks=["task-a"],
        models=["model-a"],
        scaffolds=["baseline", "repo_map"],
        repeats=2,
        max_steps=10,
        base_seed=12345,
        generation_options={"temperature": 0.2},
        experiment_id="test",
        harness_commit="abc123",
    )

    assert len(jobs) == 4
    by_repeat = {}
    for job in jobs:
        by_repeat.setdefault(job["repeat"], set()).add(job["run_seed"])
        assert job["run_id"] == make_run_id(job)

    assert by_repeat == {1: {jobs[0]["run_seed"]}, 2: {jobs[2]["run_seed"]}}
    assert len({job["run_id"] for job in jobs}) == 4


def test_manifest_extension_is_idempotent():
    first = build_jobs(
        tasks=["task-a"],
        models=["model-a"],
        scaffolds=["baseline"],
        repeats=1,
        max_steps=8,
        base_seed=1,
        generation_options={},
        experiment_id="first",
        harness_commit="abc123",
    )
    second = build_jobs(
        tasks=["task-a", "task-b"],
        models=["model-a"],
        scaffolds=["baseline"],
        repeats=1,
        max_steps=8,
        base_seed=1,
        generation_options={},
        experiment_id="expanded",
        harness_commit="abc123",
    )

    merged = merge_jobs(first, second)
    assert len(merged) == 2
    assert merged[0]["experiment_id"] == "first"


def test_shards_partition_run_ids():
    run_ids = ["run_%03d" % index for index in range(100)]
    assignments = []
    for run_id in run_ids:
        matches = [
            index
            for index in range(4)
            if shard_matches(run_id, index, 4)
        ]
        assert len(matches) == 1
        assignments.extend(matches)

    assert set(assignments) == {0, 1, 2, 3}
