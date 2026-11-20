from datetime import date
from uuid import uuid4

from app.persistence.database import Database

_HYGIENE_RESTRICTED = ("innerwear", "underwear", "lingerie", "cosmetic", "makeup")


class OrderRepository:
    def __init__(self, database: Database):
        self.database = database

    def get_owned_order(self, order_id: str, customer_id: str):
        with self.database.connect() as connection:
            return connection.execute(
                "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
                (order_id.upper(), customer_id),
            ).fetchone()

    def list_customer_orders(self, customer_id: str):
        with self.database.connect() as connection:
            return connection.execute(
                """
                SELECT order_id, item_name, status, delivered_at,
                       estimated_delivery_at, amount_minor, currency,
                       refund_status, final_sale
                FROM orders
                WHERE customer_id = ?
                ORDER BY order_id
                """,
                (customer_id,),
            ).fetchall()

    def get_approval_for_customer(self, approval_id: str, customer_id: str):
        with self.database.connect() as connection:
            return connection.execute(
                """
                SELECT a.*, o.amount_minor, o.currency
                FROM approvals AS a
                JOIN orders AS o
                  ON o.order_id = a.order_id AND o.customer_id = a.customer_id
                WHERE a.approval_id = ? AND a.customer_id = ?
                """,
                (approval_id, customer_id),
            ).fetchone()

    def get_approval_for_staff(self, approval_id: str):
        with self.database.connect() as connection:
            return connection.execute(
                """
                SELECT a.*, o.amount_minor, o.currency
                FROM approvals AS a
                JOIN orders AS o
                  ON o.order_id = a.order_id AND o.customer_id = a.customer_id
                WHERE a.approval_id = ?
                """,
                (approval_id,),
            ).fetchone()

    def list_pending_approvals(self):
        with self.database.connect() as connection:
            return connection.execute(
                """
                SELECT a.approval_id, a.order_id, o.item_name,
                       o.amount_minor, o.currency
                FROM approvals AS a
                JOIN orders AS o
                  ON o.order_id = a.order_id AND o.customer_id = a.customer_id
                WHERE a.status = 'pending'
                ORDER BY a.created_at
                """
            ).fetchall()

    def cancel_owned_order(
        self,
        order_id: str,
        customer_id: str,
        confirmed_order_id: str,
    ) -> str | None:
        normalized = order_id.strip().upper()
        if confirmed_order_id.strip().upper() != normalized:
            return "confirmation_mismatch"
        with self.database.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE orders SET status = 'cancelled'
                WHERE order_id = ? AND customer_id = ?
                  AND status IN ('processing', 'pending')
                """,
                (normalized, customer_id),
            )
            if cursor.rowcount:
                return "cancelled"
            exists = connection.execute(
                "SELECT 1 FROM orders WHERE order_id = ? AND customer_id = ?",
                (normalized, customer_id),
            ).fetchone()
            return "not_cancellable" if exists else None

    def create_approval(self, order_id: str, customer_id: str) -> str:
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

    def create_return_request(
        self,
        order_id: str,
        customer_id: str,
        confirmed_order_id: str,
    ) -> dict[str, str] | None:
        normalized = order_id.strip().upper()
        if confirmed_order_id.strip().upper() != normalized:
            return {"status": "confirmation_mismatch"}
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            order = connection.execute(
                "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
                (normalized, customer_id),
            ).fetchone()
            if order is None:
                return None
            if not self.is_eligible(order):
                return {"status": "ineligible"}
            existing = connection.execute(
                """
                SELECT request_id FROM return_requests
                WHERE order_id = ? AND customer_id = ? AND status = 'requested'
                """,
                (normalized, customer_id),
            ).fetchone()
            if existing is not None:
                return {
                    "status": "already_requested",
                    "request_id": existing["request_id"],
                }
            request_id = str(uuid4())
            connection.execute(
                """
                INSERT INTO return_requests
                (request_id, order_id, customer_id, status)
                VALUES (?, ?, ?, 'requested')
                """,
                (request_id, normalized, customer_id),
            )
        return {"status": "requested", "request_id": request_id}

    def initiate_refund(self, order_id: str, customer_id: str) -> dict | None:
        normalized = order_id.strip().upper()
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            order = connection.execute(
                "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
                (normalized, customer_id),
            ).fetchone()
            if order is None:
                return None
            if not self.is_eligible(order):
                return {"status": "ineligible"}
            existing = connection.execute(
                "SELECT 1 FROM refunds WHERE order_id = ?",
                (normalized,),
            ).fetchone()
            if existing is not None:
                return {"status": "already_initiated"}

            connection.execute(
                """
                INSERT INTO refunds (idempotency_key, order_id, amount_minor, currency)
                VALUES (?, ?, ?, ?)
                """,
                (
                    f"mcp-refund:{normalized}",
                    normalized,
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
                WHERE order_id = ? AND customer_id = ?
                """,
                (date.today().isoformat(), normalized, customer_id),
            )
            return {
                "status": "initiated",
                "amount_minor": order["amount_minor"],
                "currency": order["currency"],
            }

    def process_approval(
        self,
        approval_id: str,
        approved: bool,
        customer_id: str | None = None,
        max_auto_refund_minor: int | None = None,
        allow_over_limit: bool = False,
    ):
        with self.database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            query = "SELECT * FROM approvals WHERE approval_id = ?"
            parameters: tuple = (approval_id,)
            if customer_id is not None:
                query += " AND customer_id = ?"
                parameters = (approval_id, customer_id)
            approval = connection.execute(query, parameters).fetchone()
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
                "SELECT * FROM orders WHERE order_id = ? AND customer_id = ?",
                (approval["order_id"], approval["customer_id"]),
            ).fetchone()
            if order is None or not self.is_eligible(order):
                connection.execute(
                    "UPDATE approvals SET status = 'declined' WHERE approval_id = ?",
                    (approval_id,),
                )
                return {"status": "declined", "approval": approval}
            if (
                max_auto_refund_minor is not None
                and (
                    order["currency"] != "INR"
                    or order["amount_minor"] > max_auto_refund_minor
                )
                and not allow_over_limit
            ):
                return {
                    "status": "requires_escalation",
                    "approval": approval,
                    "amount_minor": order["amount_minor"],
                    "currency": order["currency"],
                }
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
            any(term in order["item_name"].casefold() for term in _HYGIENE_RESTRICTED)
            or order["status"] != "delivered"
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
        if any(term in order["item_name"].casefold() for term in _HYGIENE_RESTRICTED):
            reasons.append("Hygiene-restricted innerwear and cosmetics are non-returnable.")
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
