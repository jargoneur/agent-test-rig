from harness.experiment_jobs import (
    build_jobs,
    group_jobs_by_block,
    make_block_id,
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
        model_timeout_seconds=21600,
    )

    assert len(jobs) == 4
    blocks = group_jobs_by_block(jobs)
    assert len(blocks) == 2
    assert all(len(block["jobs"]) == 2 for block in blocks)

    by_repeat = {}
    for job in jobs:
        by_repeat.setdefault(job["repeat"], set()).add(job["run_seed"])
        assert job["run_id"] == make_run_id(job)
        assert job["block_id"] == make_block_id(job)

    assert by_repeat == {1: {jobs[0]["run_seed"]}, 2: {jobs[2]["run_seed"]}}
    assert len({job["run_id"] for job in jobs}) == 4
    assert len({job["block_id"] for job in jobs}) == 2
    assert all(job["model_timeout_seconds"] == 21600 for job in jobs)


def test_model_timeout_is_part_of_paired_block_identity():
    arguments = {
        "tasks": ["task-a"],
        "models": ["model-a"],
        "scaffolds": ["no_scaffold", "repo_map"],
        "repeats": 1,
        "max_steps": 8,
        "base_seed": 1,
        "generation_options": {},
        "experiment_id": "test",
        "harness_commit": "abc123",
    }
    short = build_jobs(model_timeout_seconds=600, **arguments)
    long = build_jobs(model_timeout_seconds=21600, **arguments)

    assert short[0]["block_id"] != long[0]["block_id"]
    assert short[0]["run_id"] != long[0]["run_id"]


def test_model_generation_options_override_global_defaults():
    jobs = build_jobs(
        tasks=["task-a"],
        models=[
            {
                "id": "model-a",
                "name": "model-a",
                "backend": "llama_cpp",
                "profile_sha256": "profile-a",
                "generation_options": {
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "repetition_penalty": 1.1,
                },
            },
            {
                "id": "model-b",
                "name": "model-b",
                "backend": "llama_cpp",
                "profile_sha256": "profile-b",
                "generation_options": {
                    "do_sample": False,
                },
            },
        ],
        scaffolds=["baseline", "repo_map"],
        repeats=1,
        max_steps=10,
        base_seed=12345,
        generation_options={
            "temperature": 0.2,
            "num_ctx": 16384,
            "num_predict": 2048,
        },
        experiment_id="test",
        harness_commit="abc123",
    )

    model_a_jobs = [job for job in jobs if job["model"]["id"] == "model-a"]
    model_b_jobs = [job for job in jobs if job["model"]["id"] == "model-b"]

    assert all(
        job["generation_options"]
        == {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "num_ctx": 16384,
            "num_predict": 2048,
        }
        for job in model_a_jobs
    )
    assert all(
        job["generation_options"]
        == {
            "temperature": 0.2,
            "do_sample": False,
            "num_ctx": 16384,
            "num_predict": 2048,
        }
        for job in model_b_jobs
    )
    assert all("generation_options" not in job["model"] for job in jobs)
    assert len({job["block_id"] for job in model_a_jobs}) == 1
    assert len({job["block_id"] for job in model_b_jobs}) == 1
    assert model_a_jobs[0]["block_id"] != model_b_jobs[0]["block_id"]


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
        experiment_id="first",
        harness_commit="abc123",
    )

    merged = merge_jobs(first, second)
    assert len(merged) == 2
    assert merged[0]["experiment_id"] == "first"


def test_experiment_id_separates_otherwise_identical_runs():
    arguments = {
        "tasks": ["task-a"],
        "models": ["model-a"],
        "scaffolds": ["no_scaffold"],
        "repeats": 1,
        "max_steps": 8,
        "base_seed": 1,
        "generation_options": {},
        "harness_commit": "abc123",
    }
    first = build_jobs(
        experiment_id="first",
        **arguments,
    )
    second = build_jobs(
        experiment_id="second",
        **arguments,
    )

    assert first[0]["run_id"] != second[0]["run_id"]
    assert first[0]["block_id"] != second[0]["block_id"]


def test_shards_keep_whole_blocks_together():
    jobs = build_jobs(
        tasks=["task-a", "task-b"],
        models=["model-a"],
        scaffolds=["baseline", "repo_map", "test_first"],
        repeats=2,
        max_steps=8,
        base_seed=1,
        generation_options={},
        experiment_id="test",
        harness_commit="abc123",
    )
    for block in group_jobs_by_block(jobs):
        matches = [
            index
            for index in range(4)
            if shard_matches(block["block_id"], index, 4)
        ]
        assert len(matches) == 1
