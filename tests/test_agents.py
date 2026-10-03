from datetime import date
from pathlib import Path

from app.agents.order_agent.tools.order_tool import OrderTool
from app.persistence.database import Database
from app.persistence.orders import OrderRepository


def test_order_id_extraction_normalizes_spaces_and_case():
    assert OrderTool.extract_order_id("Please track ord 1001") == "ORD-1001"
    assert OrderTool.extract_order_id("I need a return") is None


def test_database_seeds_demo_orders_and_checks_customer_ownership(tmp_path: Path):
    database = Database(str(tmp_path / "shop.db"))
    database.initialize()
    repository = OrderRepository(database)

    assert repository.get_owned_order("ORD-1001", "customer-1001") is not None
    assert repository.get_owned_order("ORD-1001", "customer-2001") is None


def test_return_window_rejects_future_delivery_and_accepts_in_window():
    base = {
        "status": "delivered",
        "refund_status": "none",
        "delivered_at": "2026-09-28",
        "return_window_days": 30,
        "item_unused": 1,
        "tags_attached": 1,
        "final_sale": 0,
        "return_received": 1,
        "inspection_passed": 1,
    }
    assert OrderRepository.is_eligible(base, today=date(2026, 10, 3))
    assert not OrderRepository.is_eligible(
        {**base, "delivered_at": "2026-10-04"},
        today=date(2026, 10, 3),
    )
    assert not OrderRepository.is_eligible(
        {**base, "inspection_passed": 0},
        today=date(2026, 10, 3),
    )


def test_approval_is_idempotent_and_persists_refund(tmp_path: Path):
    database = Database(str(tmp_path / "shop.db"))
    database.initialize()
    repository = OrderRepository(database)
    approval_id = repository.create_approval("ORD-1001", "customer-1001")

    first = repository.process_approval(approval_id, approved=True)
    second = repository.process_approval(approval_id, approved=True)

    assert first["status"] == "approved"
    assert second["status"] == "already_processed"
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM refunds").fetchone()[0] == 1
