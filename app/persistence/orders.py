from datetime import date

from app.persistence.database import Database


class OrderRepository:
    def __init__(self, database: Database):
        self.database = database

    def get_owned_order(self, order_id: str, customer_id: str):
        with self.database.connect() as connection:
            return connection.execute(
                "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
                (order_id.upper(), customer_id),
            ).fetchone()

    def create_approval(self, order_id: str, customer_id: str) -> str:
        from uuid import uuid4

        approval_id = str(uuid4())
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO approvals
                (approval_id, order_id, customer_id, status, idempotency_key)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (approval_id, order_id, customer_id, f"refund:{approval_id}"),
            )
        return approval_id

    def process_approval(self, approval_id: str, approved: bool):
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            approval = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if approval is None:
                return None
            if approval["status"] != "pending":
                return {"status": "already_processed", "approval": approval}
            if not approved:
                connection.execute(
                    "UPDATE approvals SET status = 'declined' WHERE approval_id = ?",
                    (approval_id,),
                )
                return {"status": "declined", "approval": approval}

            order = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (approval["order_id"],),
            ).fetchone()
            if order is None or not self.is_eligible(order):
                connection.execute(
                    "UPDATE approvals SET status = 'declined' WHERE approval_id = ?",
                    (approval_id,),
                )
                return {"status": "declined", "approval": approval}
            connection.execute(
                """
                INSERT INTO refunds (idempotency_key, order_id, amount_minor, currency)
                VALUES (?, ?, ?, ?)
                """,
                (
                    approval["idempotency_key"],
                    order["order_id"],
                    order["amount_minor"],
                    order["currency"],
                ),
            )
            connection.execute(
                "UPDATE orders SET refund_status = 'refunded' WHERE order_id = ?",
                (order["order_id"],),
            )
            connection.execute(
                "UPDATE approvals SET status = 'approved' WHERE approval_id = ?",
                (approval_id,),
            )
            return {
                "status": "approved",
                "approval": approval,
                "amount_minor": order["amount_minor"],
                "currency": order["currency"],
            }

    @staticmethod
    def is_eligible(order, today: date | None = None) -> bool:
        if (
            order["status"] != "delivered"
            or order["refund_status"] != "none"
            or not order["item_unused"]
            or not order["tags_attached"]
            or order["final_sale"]
            or not order["return_received"]
            or not order["inspection_passed"]
        ):
            return False
        if not order["delivered_at"]:
            return False
        delivery_date = date.fromisoformat(order["delivered_at"])
        today = today or date.today()
        days_since_delivery = (today - delivery_date).days
        return 0 <= days_since_delivery <= order["return_window_days"]
