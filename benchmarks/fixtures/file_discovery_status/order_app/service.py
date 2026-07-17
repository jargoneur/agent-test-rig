from .formatter import format_order_summary


def build_order_summary(order: dict) -> str:
    """Public service entry point for rendering one order summary."""
    return format_order_summary(order)
