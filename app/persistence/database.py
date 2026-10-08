import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator


class Database:
    def __init__(self, database_path: str):
        self.database_path = database_path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        path = Path(self.database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    item_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    delivered_at TEXT,
                    estimated_delivery_at TEXT,
                    return_window_days INTEGER NOT NULL,
                    amount_minor INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    refund_status TEXT NOT NULL DEFAULT 'none',
                    item_unused INTEGER NOT NULL DEFAULT 0,
                    tags_attached INTEGER NOT NULL DEFAULT 0,
                    final_sale INTEGER NOT NULL DEFAULT 0,
                    return_received INTEGER NOT NULL DEFAULT 0,
                    inspection_passed INTEGER NOT NULL DEFAULT 0,
                    refund_initiated INTEGER NOT NULL DEFAULT 0,
                    refund_initiated_date TEXT
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL REFERENCES orders(order_id),
                    customer_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'approved', 'declined')),
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS refunds (
                    idempotency_key TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
                    amount_minor INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            from app.persistence.users import seed_demo_customers

            seed_demo_customers(connection)
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(orders)")
            }
            for column in (
                "item_unused",
                "tags_attached",
                "final_sale",
                "return_received",
                "inspection_passed",
                "refund_initiated",
                "refund_initiated_date",
                "estimated_delivery_at",
            ):
                if column not in existing_columns:
                    if column in {"refund_initiated_date", "estimated_delivery_at"}:
                        connection.execute(
                            f"ALTER TABLE orders ADD COLUMN {column} TEXT"
                        )
                    else:
                        connection.execute(
                            f"ALTER TABLE orders ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                        )

            connection.execute(
                """
                UPDATE orders
                SET refund_status = 'initiated',
                    refund_initiated = 1,
                    refund_initiated_date = COALESCE(
                        refund_initiated_date,
                        (
                            SELECT substr(refunds.created_at, 1, 10)
                            FROM refunds
                            WHERE refunds.order_id = orders.order_id
                            LIMIT 1
                        ),
                        date('now')
                    )
                WHERE refund_status = 'refunded'
                   OR EXISTS (
                       SELECT 1 FROM refunds WHERE refunds.order_id = orders.order_id
                   )
                """
            )

            connection.executemany(
                """
                INSERT OR IGNORE INTO orders
                (order_id, customer_id, item_name, status, delivered_at,
                 return_window_days, amount_minor, currency, item_unused,
                 tags_attached, final_sale, return_received, inspection_passed,
                 refund_initiated, refund_initiated_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [*_demo_orders(), *_multi_customer_orders()],
            )
            connection.execute(
                """
                UPDATE orders
                SET estimated_delivery_at = date('now', '+3 days')
                WHERE status IN ('shipped', 'in_transit')
                  AND estimated_delivery_at IS NULL
                """
            )
            connection.execute(
                "UPDATE orders SET customer_id = 'customer-1002' WHERE order_id = 'ORD-1002'"
            )
            connection.execute(
                """
                UPDATE orders
                SET item_name = 'Linen trousers (legacy sample)'
                WHERE order_id = 'ORD-2001' AND customer_id = 'customer-2001'
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO refunds
                (idempotency_key, order_id, amount_minor, currency)
                SELECT 'demo-refund:ORD-1006', order_id, amount_minor, currency
                FROM orders
                WHERE order_id = 'ORD-1006' AND refund_initiated = 1
                """
            )


def _demo_orders() -> list[tuple]:
    today = date.today()
    return [
        (
            "ORD-1001",
            "customer-1001",
            "Cotton T-shirt",
            "delivered",
            (today - timedelta(days=5)).isoformat(),
            30,
            249900,
            "INR",
            1,
            1,
            0,
            1,
            1,
            0,
            None,
        ),
        (
            "ORD-1002",
            "customer-1002",
            "Denim jacket",
            "shipped",
            None,
            30,
            399900,
            "INR",
            0,
            0,
            0,
            0,
            0,
            0,
            None,
        ),
        (
            "ORD-1003",
            "customer-1003",
            "Linen trousers",
            "delivered",
            (today - timedelta(days=8)).isoformat(),
            30,
            329900,
            "INR",
            1,
            1,
            0,
            0,
            0,
            0,
            None,
        ),
        (
            "ORD-1004",
            "customer-1004",
            "Printed maxi dress",
            "delivered",
            (today - timedelta(days=42)).isoformat(),
            30,
            459900,
            "INR",
            1,
            1,
            0,
            1,
            1,
            0,
            None,
        ),
        (
            "ORD-1005",
            "customer-1005",
            "Limited-edition sneakers",
            "delivered",
            (today - timedelta(days=4)).isoformat(),
            30,
            799900,
            "INR",
            1,
            1,
            1,
            1,
            1,
            0,
            None,
        ),
        (
            "ORD-1006",
            "customer-1006",
            "Classic white shirt",
            "delivered",
            (today - timedelta(days=12)).isoformat(),
            30,
            199900,
            "INR",
            1,
            1,
            0,
            1,
            1,
            1,
            today.isoformat(),
        ),
        (
            "ORD-1007",
            "customer-1007",
            "Wool blend cardigan",
            "delivered",
            (today - timedelta(days=6)).isoformat(),
            30,
            549900,
            "INR",
            1,
            1,
            0,
            1,
            0,
            0,
            None,
        ),
        (
            "ORD-1008",
            "customer-1008",
            "Leather-look belt",
            "delivered",
            (today - timedelta(days=9)).isoformat(),
            30,
            149900,
            "INR",
            1,
            0,
            0,
            1,
            1,
            0,
            None,
        ),
        (
            "ORD-1009",
            "customer-1009",
            "Running shorts",
            "in_transit",
            None,
            30,
            189900,
            "INR",
            0,
            0,
            0,
            0,
            0,
            0,
            None,
        ),
        (
            "ORD-1010",
            "customer-1010",
            "Striped knit top",
            "delivered",
            (today - timedelta(days=3)).isoformat(),
            30,
            219900,
            "INR",
            1,
            1,
            0,
            0,
            0,
            0,
            None,
        ),
        (
            "ORD-1011",
            "customer-1011",
            "Water-resistant parka",
            "delivered",
            (today - timedelta(days=35)).isoformat(),
            30,
            1299900,
            "INR",
            1,
            1,
            0,
            1,
            1,
            0,
            None,
        ),
        (
            "ORD-1012",
            "customer-1012",
            "Pleated midi skirt",
            "delivered",
            (today - timedelta(days=2)).isoformat(),
            30,
            379900,
            "INR",
            1,
            1,
            0,
            1,
            1,
            0,
            None,
        ),
    ]


def _multi_customer_orders() -> list[tuple]:
    today = date.today()
    catalog = (
        "Cotton shirt",
        "Denim jeans",
        "Linen trousers",
        "Printed dress",
        "Running shoes",
        "Wool cardigan",
        "Leather belt",
        "Running shorts",
        "Knit top",
        "Pleated skirt",
    )
    orders: list[tuple] = []
    for customer_number, first_order_number in ((1, 1101), (2, 2101), (3, 3101)):
        for order_index, item_name in enumerate(catalog):
            status = "delivered"
            delivered_at = (today - timedelta(days=5)).isoformat()
            item_unused = tags_attached = return_received = inspection_passed = 1
            final_sale = refund_initiated = 0
            initiated_at = None

            if order_index == 1:
                final_sale = 1
            elif order_index == 2:
                status, delivered_at = "shipped", None
                item_unused = tags_attached = return_received = inspection_passed = 0
            elif order_index == 3:
                delivered_at = (today - timedelta(days=45)).isoformat()
            elif order_index == 4:
                tags_attached = 0
            elif order_index == 5:
                inspection_passed = 0
            elif order_index == 6:
                return_received = 0
            elif order_index == 7:
                status, delivered_at = "in_transit", None
                item_unused = tags_attached = return_received = inspection_passed = 0
            elif order_index == 8:
                delivered_at = (today - timedelta(days=3)).isoformat()
                return_received = 0
            elif order_index == 9:
                delivered_at = (today - timedelta(days=10)).isoformat()
                return_received = inspection_passed = 0

            orders.append(
                (
                    f"ORD-{first_order_number + order_index}",
                    f"customer-{customer_number}",
                    f"{item_name} (Customer {customer_number})",
                    status,
                    delivered_at,
                    30,
                    (249900 + customer_number * 1000 + order_index * 2500),
                    "INR",
                    item_unused,
                    tags_attached,
                    final_sale,
                    return_received,
                    inspection_passed,
                    refund_initiated,
                    initiated_at,
                )
            )
    return orders
