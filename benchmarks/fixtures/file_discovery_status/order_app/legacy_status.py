LEGACY_STATUS_LABELS = {
    "P": "Processing",
    "C": "Complete",
}


def legacy_status_text(code: str) -> str:
    """Translate status codes used by an unrelated legacy export."""
    return LEGACY_STATUS_LABELS.get(code, code)
