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
                CREATE TABLE IF NOT EXISTS orders (
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
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(orders)")
            }
            for column in (
                "item_unused",
                "tags_attached",
                "final_sale",
                "return_received",
                "inspection_passed",
            ):
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE orders ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                    )
            connection.executemany(
                """
                INSERT OR IGNORE INTO orders
                (order_id, customer_id, item_name, status, delivered_at,
                 return_window_days, amount_minor, currency, item_unused,
                 tags_attached, final_sale, return_received, inspection_passed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "ORD-1001",
                        "customer-1001",
                        "Cotton T-shirt",
                        "delivered",
                        (date.today() - timedelta(days=5)).isoformat(),
                        30,
                        249900,
                        "INR",
                        1,
                        1,
                        0,
                        1,
                        1,
                    ),
                    (
                        "ORD-1002",
                        "customer-1001",
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
                    ),
                    (
                        "ORD-2001",
                        "customer-2001",
                        "Linen trousers",
                        "delivered",
                        "2026-08-01",
                        14,
                        299900,
                        "INR",
                        0,
                        0,
                        0,
                        0,
                        0,
                    ),
                ],
            )
