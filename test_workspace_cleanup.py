import run_worker


def test_cleanup_workspace_removes_only_the_exact_run_directory(
    tmp_path,
    monkeypatch,
):
    workspace_root = tmp_path / "workspaces"
    target = workspace_root / "run_exact"
    sibling = workspace_root / "run_exact_other"
    target.mkdir(parents=True)
    sibling.mkdir()
    (target / "artifact.txt").write_text("disposable", encoding="utf-8")
    (sibling / "result.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(run_worker, "WORKSPACES_ROOT", workspace_root)

    run_worker.cleanup_workspace("run/exact")

    assert not target.exists()
    assert sibling.is_dir()
