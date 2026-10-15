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

        with self.database.connect() as connection:
            pending = connection.execute(
                """
                SELECT approval_id FROM approvals
                WHERE order_id = ? AND customer_id = ? AND status = 'pending'
                """,
                (order_id.upper(), customer_id),
            ).fetchone()
            if pending is not None:
                return pending["approval_id"]

            approval_id = str(uuid4())
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
                """
                UPDATE orders
                SET refund_status = 'initiated',
                    refund_initiated = 1,
                    refund_initiated_date = ?
                WHERE order_id = ?
                """,
                (date.today().isoformat(), order["order_id"]),
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
                "refund_initiated_date": date.today().isoformat(),
            }

    @staticmethod
    def is_eligible(order, today: date | None = None) -> bool:
        if (
            order["status"] != "delivered"
            or order["refund_status"] != "none"
            or (
                "refund_initiated" in order.keys()
                and order["refund_initiated"]
            )
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

    @staticmethod
    def return_eligibility(order, today: date | None = None) -> dict[str, object]:
        reasons: list[str] = []
        if order["status"] != "delivered":
            reasons.append("The order has not been delivered.")
        if order["final_sale"]:
            reasons.append("The item is marked as final sale.")
        if not order["item_unused"]:
            reasons.append("The item is not recorded as unused.")
        if not order["tags_attached"]:
            reasons.append("The original tags are not recorded as attached.")
        if order["refund_status"] != "none" or (
            "refund_initiated" in order.keys() and order["refund_initiated"]
        ):
            reasons.append("A refund has already been initiated for this order.")

        days_since_delivery = None
        if order["status"] == "delivered":
            if not order["delivered_at"]:
                reasons.append("The delivery date is not available.")
            else:
                delivery_date = date.fromisoformat(order["delivered_at"])
                current_date = today or date.today()
                days_since_delivery = (current_date - delivery_date).days
                if days_since_delivery < 0:
                    reasons.append("The delivery date is in the future.")
                elif days_since_delivery > order["return_window_days"]:
                    reasons.append(
                        f"The {order['return_window_days']}-day return window has passed."
                    )

        return {
            "eligible": not reasons,
            "reasons": reasons,
            "days_since_delivery": days_since_delivery,
            "return_window_days": order["return_window_days"],
        }
