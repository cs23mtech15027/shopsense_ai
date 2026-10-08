import sqlite3
from datetime import date
from pathlib import Path

from app.agents.order_agent.tools.order_tool import OrderTool
from app.persistence.database import Database
from app.persistence.orders import OrderRepository


def test_database_seeds_twelve_distinct_demo_customers_and_products(tmp_path: Path):
    database = Database(str(tmp_path / "shop.db"))
    database.initialize()

    with database.connect() as connection:
        orders = connection.execute(
            "SELECT order_id, customer_id, item_name FROM orders"
        ).fetchall()

    assert len(orders) >= 12
    assert len({row["customer_id"] for row in orders}) == len(orders)
    assert len({row["item_name"] for row in orders}) == len(orders)


def test_order_id_extraction_normalizes_spaces_and_case():
    assert OrderTool.extract_order_id("Please track ord 1001") == "ORD-1001"
    assert OrderTool.extract_order_id("I need a return") is None


def test_database_seeds_demo_orders_and_checks_customer_ownership(tmp_path: Path):
    database = Database(str(tmp_path / "shop.db"))
    database.initialize()
    repository = OrderRepository(database)

    assert repository.get_owned_order("ORD-1001", "customer-1001") is not None
    assert repository.get_owned_order("ORD-1001", "customer-2001") is None


def test_database_migrates_legacy_refund_status_to_initiated_flags(tmp_path: Path):
    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            item_name TEXT NOT NULL,
            status TEXT NOT NULL,
            delivered_at TEXT,
            return_window_days INTEGER NOT NULL,
            amount_minor INTEGER NOT NULL,
            currency TEXT NOT NULL,
            refund_status TEXT NOT NULL DEFAULT 'none',
            item_unused INTEGER NOT NULL DEFAULT 0,
            tags_attached INTEGER NOT NULL DEFAULT 0,
            final_sale INTEGER NOT NULL DEFAULT 0,
            return_received INTEGER NOT NULL DEFAULT 0,
            inspection_passed INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO orders
        (order_id, customer_id, item_name, status, delivered_at,
         return_window_days, amount_minor, currency, refund_status)
        VALUES ('ORD-LEGACY', 'customer-legacy', 'Sample legacy item', 'delivered',
                '2026-10-01', 30, 10000, 'INR', 'refunded');
        """
    )
    connection.commit()
    connection.close()

    Database(str(database_path)).initialize()

    with sqlite3.connect(database_path) as connection:
        migrated = connection.execute(
            "SELECT refund_status, refund_initiated, refund_initiated_date "
            "FROM orders WHERE order_id = 'ORD-LEGACY'"
        ).fetchone()
    assert migrated[0] == "initiated"
    assert migrated[1] == 1
    assert migrated[2] is not None


def test_return_window_rejects_future_delivery_and_accepts_in_window():
    base = {
        "status": "delivered",
        "refund_status": "none",
        "delivered_at": "2026-10-01",
        "return_window_days": 30,
        "refund_initiated": 0,
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
    assert not OrderRepository.is_eligible(
        {**base, "refund_initiated": 1, "refund_status": "initiated"},
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
    assert first["refund_initiated_date"] == date.today().isoformat()
    assert second["status"] == "already_processed"
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM refunds WHERE order_id = 'ORD-1001'"
        ).fetchone()[0] == 1
        order = connection.execute(
            "SELECT refund_status, refund_initiated, refund_initiated_date "
            "FROM orders WHERE order_id = 'ORD-1001'"
        ).fetchone()
    assert order["refund_status"] == "initiated"
    assert order["refund_initiated"] == 1
    assert order["refund_initiated_date"] == date.today().isoformat()


def test_repeated_eligible_chat_reuses_pending_approval(tmp_path: Path):
    database = Database(str(tmp_path / "shop.db"))
    database.initialize()
    repository = OrderRepository(database)

    first_id = repository.create_approval("ORD-1001", "customer-1001")
    second_id = repository.create_approval("ORD-1001", "customer-1001")

    assert first_id == second_id
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM approvals WHERE order_id = 'ORD-1001'"
        ).fetchone()[0] == 1


def test_declined_approval_does_not_mark_refund_initiated(tmp_path: Path):
    database = Database(str(tmp_path / "shop.db"))
    database.initialize()
    repository = OrderRepository(database)
    approval_id = repository.create_approval("ORD-1001", "customer-1001")

    result = repository.process_approval(approval_id, approved=False)

    assert result["status"] == "declined"
    with database.connect() as connection:
        order = connection.execute(
            "SELECT refund_initiated, refund_initiated_date FROM orders "
            "WHERE order_id = 'ORD-1001'"
        ).fetchone()
    assert order["refund_initiated"] == 0
    assert order["refund_initiated_date"] is None
