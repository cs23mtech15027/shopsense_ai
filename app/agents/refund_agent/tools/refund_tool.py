from app.persistence.orders import OrderRepository


class RefundTool:
    def __init__(self, repository: OrderRepository):
        self.repository = repository

    def eligible(self, order) -> bool:
        return self.repository.is_eligible(order)

    def create_approval(self, order_id: str, customer_id: str) -> str:
        return self.repository.create_approval(order_id, customer_id)
