STATUS_LABELS = {
    "new": "New",
    "in_progress": "In progress",
    "done": "Done",
}


def display_status(status: str) -> str:
    """Return the customer-facing label for an internal status key."""
    return STATUS_LABELS.get(status, status.replace("_", " ").capitalize())
