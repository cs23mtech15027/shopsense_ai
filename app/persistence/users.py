import hashlib
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta

def seed_demo_customers(connection: sqlite3.Connection) -> None:
    for customer_id in ("customer-1", "customer-2", "customer-3"):
        existing = connection.execute(
            "SELECT 1 FROM customers WHERE customer_id = ?",
            (customer_id,),
        ).fetchone()
        if existing is not None:
            continue
        connection.execute(
            """
            INSERT INTO customers (customer_id, username, password_salt, password_hash)
            VALUES (?, ?, ?, ?)
            """,
            (customer_id, customer_id, "", ""),
        )


class UserRepository:
    def __init__(self, database):
        self.database = database

    def create_demo_session(self, customer_id: str) -> str | None:
        if customer_id not in {"customer-1", "customer-2", "customer-3"}:
            return None
        with self.database.connect() as connection:
            user = connection.execute(
                "SELECT customer_id FROM customers WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
            if user is None:
                return None

            token = secrets.token_urlsafe(32)
            expires_at = (datetime.now(UTC) + timedelta(hours=12)).isoformat()
            connection.execute(
                """
                INSERT INTO auth_sessions (token_hash, customer_id, expires_at)
                VALUES (?, ?, ?)
                """,
                (
                    hashlib.sha256(token.encode("ascii")).hexdigest(),
                    user["customer_id"],
                    expires_at,
                ),
            )
            return token

    def get_session_customer(self, token: str) -> str | None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT customer_id FROM auth_sessions
                WHERE token_hash = ? AND expires_at > ?
                """,
                (token_hash, datetime.now(UTC).isoformat()),
            ).fetchone()
        return row["customer_id"] if row is not None else None
