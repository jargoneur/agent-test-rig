from order_app.service import build_order_summary


def test_in_progress_status_uses_customer_facing_label():
    order = {"id": 42, "status": "in_progress", "owner": "Mina"}

    assert build_order_summary(order) == (
        "Order #42\n"
        "Status: In progress\n"
        "Owner: Mina"
    )


def test_unknown_status_falls_back_to_readable_text():
    order = {"id": 7, "status": "waiting_for_stock", "owner": "Lee"}

    assert build_order_summary(order) == (
        "Order #7\n"
        "Status: Waiting for stock\n"
        "Owner: Lee"
    )


def test_summary_preserves_order_number_and_owner():
    order = {"id": 99, "status": "done", "owner": "Sam"}

    summary = build_order_summary(order)

    assert summary.splitlines()[0] == "Order #99"
    assert summary.splitlines()[2] == "Owner: Sam"
