import json
from pathlib import Path

import yaml

from harness.contextbench_tasks import (
    TASK_PREFIX,
    contextbench_task_ids,
    load_contextbench_task,
)
from harness.contextbench_trajectory import build_contextbench_trajectory
from harness.llama_cpp_server import LlamaCppServerManager
from harness.reproducibility import task_metadata


def _write_cache(path: Path):
    row = {
        "instance_id": "Verified-python-0001",
        "original_inst_id": "psf__requests-1234",
        "bench": "Verified",
        "language": "python",
        "repo": "psf/requests",
        "base_commit": "a" * 40,
        "problem_statement": "Fix the request behavior.",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return row


def test_contextbench_cache_loads_as_a_repository_task(tmp_path):
    cache = tmp_path / "tasks.jsonl"
    row = _write_cache(cache)

    ids = contextbench_task_ids(str(cache))
    assert ids == [TASK_PREFIX + row["instance_id"]]

    task = load_contextbench_task(ids[0], str(cache))
    assert task["repo"] == "psf/requests"
    assert task["base_commit"] == "a" * 40
    assert task["workspace_kind"] == "git_checkout"
    assert task["run_final_tests"] is False

    metadata = task_metadata(task)
    assert metadata["source"] == "contextbench"
    assert metadata["instance_id"] == row["instance_id"]
    assert len(metadata["contextbench_record_sha256"]) == 64


def test_contextbench_trajectory_preserves_exact_read_and_search_spans():
    task = {
        "id": "contextbench::example",
        "instance_id": "example",
        "original_inst_id": "org__repo-1",
        "bench": "Verified",
        "repo": "org/repo",
        "base_commit": "b" * 40,
    }
    result = {
        "history": [
            {
                "observation": {
                    "type": "file_content",
                    "path": "src/a.py",
                    "start_line": 10,
                    "end_line": 20,
                }
            },
            {
                "observation": {
                    "type": "search_result",
                    "results": [
                        {"path": "src/a.py", "line": 30},
                        {"path": "src/b.py", "line": 7},
                    ],
                }
            },
        ]
    }

    trajectory = build_contextbench_trajectory(task, result, "diff --git")
    assert trajectory["instance_id"] == "example"
    assert trajectory["model_patch"] == "diff --git"
    assert trajectory["traj_data"]["pred_files"] == ["src/a.py", "src/b.py"]
    assert trajectory["traj_data"]["pred_spans"]["src/a.py"] == [
        {"start": 10, "end": 20},
        {"start": 30, "end": 30},
    ]


def test_llama_cpp_registry_builds_a_single_gpu_server_command(tmp_path):
    artifact = tmp_path / "model.gguf"
    artifact.write_bytes(b"gguf")
    registry = tmp_path / "registry.yml"
    registry.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "model-a": {
                        "artifact": str(artifact),
                        "alias": "model-a",
                        "context_size": 8192,
                        "gpu_layers": "all",
                        "cache_type_k": "f16",
                        "cache_type_v": "f16",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    manager = LlamaCppServerManager(
        {
            "registry": str(registry),
            "binary": "/bin/true",
            "host": "127.0.0.1",
            "port": 9090,
            "gpu": "GPU-test",
            "log_path": str(tmp_path / "server.log"),
        },
        worker_id="test-worker",
    )
    entry = manager._entry("model-a")
    command = manager._command("model-a", entry)

    assert manager.model_ids() == ["model-a"]
    assert command[0] == "/bin/true"
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert command[command.index("--split-mode") + 1] == "none"
    assert command[command.index("--alias") + 1] == "model-a"
