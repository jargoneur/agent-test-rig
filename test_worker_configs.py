from pathlib import Path

import pytest
import yaml

from harness.resource_policy import ResourcePolicy


WORKER_ROOT = Path(__file__).resolve().parent / "workers"


def load_worker(path: Path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def test_every_worker_explicitly_declares_resource_policy():
    paths = sorted(WORKER_ROOT.glob("*.yml")) + sorted(WORKER_ROOT.glob("*.yaml"))
    assert paths
    for path in paths:
        config = load_worker(path)
        assert "resource_policy" in config, path
        assert isinstance(config["resource_policy"], dict), path


def test_every_shared_worker_has_an_explicit_operation_mode():
    paths = sorted(WORKER_ROOT.glob("*.yml")) + sorted(WORKER_ROOT.glob("*.yaml"))
    for path in paths:
        config = load_worker(path)
        policy = config["resource_policy"]
        if not bool(policy.get("shared_resource", False)):
            ResourcePolicy.from_worker_config(config)
            continue

        assert policy.get("other_users_priority") is True, path
        mode = policy.get("mode")
        assert mode in {
            "scheduler_preemptible",
            "external_yield_signal",
            "manual_operator",
        }, path

        if mode == "manual_operator":
            assert policy.get("operator_acknowledgement"), path
            ResourcePolicy.from_worker_config(config)
            continue

        assert policy.get("verification_reference"), path
        if policy.get("priority_mechanism_verified") is True:
            ResourcePolicy.from_worker_config(config)
        else:
            with pytest.raises(ValueError, match="blocked until"):
                ResourcePolicy.from_worker_config(config)
