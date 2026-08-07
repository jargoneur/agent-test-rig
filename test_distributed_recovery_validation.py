from scripts.run_distributed_recovery_validation import run_validation


def test_distributed_recovery_validation_generates_all_evidence(tmp_path):
    evidence = run_validation(tmp_path, "abc123")

    assert evidence["status"] == "validated"
    assert len(evidence["harness_source_sha256"]) == 64
    assert evidence["five_condition_block_completed"] is True
    assert evidence["whole_block_restart_validated"] is True
    assert evidence["manual_stop_resume_validated"] is True
    assert evidence["scheduler_backup_restore_validated"] is True
    assert evidence["provenance_mismatch_rejected"] is True
    assert evidence["restart_attempt"] == 2
    assert evidence["restored_status"]["results"] == 5
