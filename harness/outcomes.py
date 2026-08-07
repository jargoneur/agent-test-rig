from __future__ import annotations

from typing import Any, Dict, Optional


class TerminalRunError(RuntimeError):
    """A usable terminal experimental outcome, not an infrastructure failure."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        stage: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.kind = str(kind)
        self.stage = str(stage)
        self.details = dict(details or {})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "stage": self.stage,
            "message": str(self),
            "details": dict(self.details),
        }


class ModelTerminalError(TerminalRunError):
    pass


class ScaffoldTerminalError(TerminalRunError):
    def __init__(self, stage: str, error: Exception):
        super().__init__(
            "scaffold_failure",
            "%s: %s" % (error.__class__.__name__, error),
            stage=stage,
            details={"exception_type": error.__class__.__name__},
        )


def completed_outcome(result: Dict[str, Any], model_patch: str) -> Dict[str, Any]:
    """Classify a normally terminated run without collapsing its raw fields."""
    tests_passed = result.get("final_tests_passed")
    patch_present = bool(str(model_patch or "").strip())
    if tests_passed is True:
        kind = "tests_passed"
    elif tests_passed is False:
        kind = "tests_failed"
    elif patch_present:
        kind = "patch_produced_unevaluated"
    else:
        kind = "no_patch"
    return {
        "terminal": True,
        "usable_result": True,
        "kind": kind,
        "tests_passed": tests_passed,
        "patch_present": patch_present,
    }


def error_outcome(
    error: TerminalRunError,
    model_patch: str = "",
) -> Dict[str, Any]:
    return {
        "terminal": True,
        "usable_result": True,
        "kind": error.kind,
        "stage": error.stage,
        "tests_passed": None,
        "patch_present": bool(str(model_patch or "").strip()),
        "error": error.as_dict(),
    }
