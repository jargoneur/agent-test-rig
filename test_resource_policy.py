import sys

import pytest

from harness.resource_policy import ResourcePolicy, ResourceYieldRequested


def test_local_resource_policy_is_allowed_without_external_verification(tmp_path):
    policy = ResourcePolicy(
        {
            "shared_resource": False,
            "mode": "local_exclusive",
            "pause_file": str(tmp_path / "PAUSE"),
        }
    )

    policy.checkpoint("test")
    assert policy.metadata()["shared_resource"] is False


def test_worker_config_must_explicitly_declare_resource_policy():
    with pytest.raises(ValueError, match="explicitly declare"):
        ResourcePolicy.from_worker_config({})


def test_shared_worker_is_blocked_until_other_user_priority_is_verified():
    with pytest.raises(ValueError, match="blocked until"):
        ResourcePolicy(
            {
                "shared_resource": True,
                "other_users_priority": True,
                "mode": "scheduler_preemptible",
                "priority_mechanism_verified": False,
                "verification_reference": "pending",
                "release_managed_externally": True,
            }
        )


def test_shared_worker_requires_written_verification_reference():
    with pytest.raises(ValueError, match="verification_reference"):
        ResourcePolicy(
            {
                "shared_resource": True,
                "other_users_priority": True,
                "mode": "scheduler_preemptible",
                "priority_mechanism_verified": True,
                "release_managed_externally": True,
            }
        )


def test_shared_worker_requires_a_release_mechanism():
    with pytest.raises(ValueError, match="release_command"):
        ResourcePolicy(
            {
                "shared_resource": True,
                "other_users_priority": True,
                "mode": "scheduler_preemptible",
                "priority_mechanism_verified": True,
                "verification_reference": "queue-policy-approval",
            }
        )


def test_pause_file_requests_yield_at_safe_checkpoint(tmp_path):
    pause_file = tmp_path / "PAUSE"
    policy = ResourcePolicy(
        {
            "shared_resource": False,
            "mode": "local_exclusive",
            "pause_file": str(pause_file),
        }
    )
    pause_file.write_text("yield\n", encoding="utf-8")

    with pytest.raises(ResourceYieldRequested, match="pause_file"):
        policy.checkpoint("before_scaffold")


def test_resource_yield_is_not_swallowed_by_generic_exception_handlers(tmp_path):
    pause_file = tmp_path / "PAUSE"
    pause_file.write_text("yield\n", encoding="utf-8")
    policy = ResourcePolicy(
        {
            "shared_resource": False,
            "mode": "local_exclusive",
            "pause_file": str(pause_file),
        }
    )

    swallowed = False
    try:
        policy.checkpoint("before_model_call")
    except Exception:
        swallowed = True
    except ResourceYieldRequested:
        pass

    assert swallowed is False


def test_external_availability_command_fails_closed(tmp_path):
    command = '"%s" -c "import sys; sys.exit(1)"' % sys.executable
    policy = ResourcePolicy(
        {
            "shared_resource": True,
            "other_users_priority": True,
            "mode": "external_yield_signal",
            "priority_mechanism_verified": True,
            "verification_reference": "resource-owner-test-procedure",
            "availability_command": command,
            "release_managed_externally": True,
            "pause_file": str(tmp_path / "PAUSE"),
        }
    )

    with pytest.raises(ResourceYieldRequested, match="availability_command"):
        policy.checkpoint("before_block_claim")
