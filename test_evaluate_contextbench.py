from scripts.evaluate_contextbench import outcome_row


def test_outcome_row_preserves_terminal_result_and_patch_flag():
    record = {
        "run_id": "run_1",
        "outcome": {
            "kind": "context_limit_rejection",
            "usable_result": True,
        },
        "terminal_error": {
            "kind": "context_limit_rejection",
            "stage": "model_call",
        },
    }
    trajectory = {
        "instance_id": "task_1",
        "agent_rig": {"model_patch_present": False},
    }

    row = outcome_row(record, trajectory)

    assert row["outcome_kind"] == "context_limit_rejection"
    assert row["usable_result"] is True
    assert row["model_patch_present"] is False
    assert row["terminal_error"]["stage"] == "model_call"


def test_outcome_row_marks_legacy_missing_metadata():
    row = outcome_row(
        {"run_id": "legacy"},
        {
            "instance_id": "task_2",
            "agent_rig": {"model_patch_present": True},
        },
    )

    assert row["outcome_kind"] == "missing_outcome_metadata"
    assert row["usable_result"] is False
    assert row["model_patch_present"] is True