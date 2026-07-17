from .status_labels import display_status


def format_order_summary(order: dict) -> str:
    """Render the customer-facing order summary."""
    return "\n".join(
        [
            f"Order #{order['id']}",
            f"Status: {order['status']}",
            f"Owner: {order['owner']}",
        ]
    )
