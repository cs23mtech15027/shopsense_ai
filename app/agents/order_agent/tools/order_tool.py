import re

from app.persistence.orders import OrderRepository


_ORDER_ID = re.compile(r"\bORD[-_ ]?\d{4,}\b", re.IGNORECASE)


class OrderTool:
    def __init__(self, repository: OrderRepository):
        self.repository = repository

    @staticmethod
    def extract_order_id(message: str) -> str | None:
        match = _ORDER_ID.search(message)
        if match is None:
            return None
        return match.group().upper().replace("_", "-").replace(" ", "-")

    def get_owned_order(self, order_id: str, customer_id: str):
        return self.repository.get_owned_order(order_id, customer_id)
