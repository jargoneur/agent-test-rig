from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class ResourceYieldRequested(BaseException):
    """Raised at a safe checkpoint when a shared resource must be yielded.

    This deliberately inherits from BaseException so generic application-level
    ``except Exception`` blocks cannot accidentally convert a required resource
    yield into an ordinary model or tool error.
    """


class ResourcePolicy:
    """Fail-closed policy for opportunistic use of shared compute.

    The harness cannot prove a cluster administrator's scheduling policy. It can,
    however, refuse to start shared-resource work until that policy has been
    explicitly verified, react to scheduler signals or an external availability
    command, and release the current scheduler lease without losing completed runs.
    """

    VALID_SHARED_MODES = {
        "scheduler_preemptible",
        "external_yield_signal",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        pause_file: Optional[str] = None,
        signal_event: Any = None,
    ) -> None:
        value = dict(config or {})
        self.shared_resource = bool(value.get("shared_resource", False))
        self.other_users_priority = bool(
            value.get("other_users_priority", not self.shared_resource)
        )
        self.mode = str(
            value.get(
                "mode",
                "local_exclusive" if not self.shared_resource else "",
            )
        )
        self.priority_mechanism_verified = bool(
            value.get("priority_mechanism_verified", False)
        )
        self.verification_reference = str(
            value.get("verification_reference") or ""
        ).strip()
        self.pause_file = Path(
            value.get("pause_file") or pause_file or "PAUSE"
        )
        self.availability_command = value.get("availability_command")
        self.release_command = value.get("release_command")
        self.release_managed_externally = bool(
            value.get("release_managed_externally", False)
        )
        self.command_timeout_seconds = max(
            1,
            int(value.get("command_timeout_seconds", 30)),
        )
        self.signal_event = signal_event
        self._validate()

    def _validate(self) -> None:
        if not self.shared_resource:
            if self.mode != "local_exclusive":
                raise ValueError(
                    "Non-shared workers must use resource_policy.mode=local_exclusive"
                )
            return

        if not self.other_users_priority:
            raise ValueError(
                "Shared resources require other_users_priority=true"
            )
        if self.mode not in self.VALID_SHARED_MODES:
            raise ValueError(
                "Shared resources require mode scheduler_preemptible or "
                "external_yield_signal"
            )
        if not self.priority_mechanism_verified:
            raise ValueError(
                "Shared-resource worker is blocked until the external priority "
                "mechanism is verified"
            )
        if not self.verification_reference:
            raise ValueError(
                "Shared-resource verification_reference is required"
            )
        if self.mode == "external_yield_signal" and not self.availability_command:
            raise ValueError(
                "external_yield_signal mode requires availability_command"
            )
        if not self.release_command and not self.release_managed_externally:
            raise ValueError(
                "Shared resources require release_command or "
                "release_managed_externally=true"
            )

    @classmethod
    def from_worker_config(
        cls,
        worker_config: Dict[str, Any],
        pause_file: Optional[str] = None,
        signal_event: Any = None,
    ) -> "ResourcePolicy":
        if "resource_policy" not in worker_config:
            raise ValueError(
                "Worker config must explicitly declare resource_policy"
            )
        policy = worker_config.get("resource_policy")
        if not isinstance(policy, dict):
            raise ValueError("resource_policy must be a YAML mapping")
        return cls(
            config=policy,
            pause_file=pause_file,
            signal_event=signal_event,
        )

    def _command_available(self) -> Optional[str]:
        if not self.availability_command:
            return None
        try:
            result = subprocess.run(
                str(self.availability_command),
                shell=True,
                timeout=self.command_timeout_seconds,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return "availability_command_error:%s" % error.__class__.__name__
        if result.returncode != 0:
            return "availability_command_requested_yield"
        return None

    def yield_reason(self) -> Optional[str]:
        if self.signal_event is not None and self.signal_event.is_set():
            return "scheduler_or_operator_signal"
        if self.pause_file.exists():
            return "pause_file:%s" % self.pause_file
        return self._command_available()

    def checkpoint(self, label: str = "checkpoint") -> None:
        reason = self.yield_reason()
        if reason:
            raise ResourceYieldRequested("%s at %s" % (reason, label))

    def release(self) -> Optional[int]:
        """Run the configured backend/resource release command, if any."""
        if not self.release_command:
            return None
        result = subprocess.run(
            str(self.release_command),
            shell=True,
            timeout=self.command_timeout_seconds,
        )
        return int(result.returncode)

    def metadata(self) -> Dict[str, Any]:
        return {
            "shared_resource": self.shared_resource,
            "other_users_priority": self.other_users_priority,
            "mode": self.mode,
            "priority_mechanism_verified": self.priority_mechanism_verified,
            "verification_reference": self.verification_reference,
            "pause_file": str(self.pause_file),
            "availability_command_configured": bool(self.availability_command),
            "release_command_configured": bool(self.release_command),
            "release_managed_externally": self.release_managed_externally,
        }
