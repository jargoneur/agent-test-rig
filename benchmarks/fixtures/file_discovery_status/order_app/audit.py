def build_audit_entry(order: dict) -> str:
    """Keep internal status keys unchanged in machine-readable audit output."""
    return f"{order['id']}:{order['status']}:{order['owner']}"
