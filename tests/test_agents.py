import sqlite3
from datetime import date
from pathlib import Path

import pytest

from app.agents.order_agent.tools.order_tool import OrderTool
from app.agents.care_agent.agent import CareAgent
from app.persistence.database import Database
from app.persistence.orders import OrderRepository


def test_database_seeds_ten_distinct_orders_for_each_demo_customer(tmp_path: Path):
    database = Database(str(tmp_path / "shop.db"))
    database.initialize()

    with database.connect() as connection:
        orders = connection.execute(
            """
            SELECT order_id, customer_id, item_name, status, final_sale,
                   item_unused, tags_attached, return_received, inspection_passed,
                   delivered_at, return_window_days, refund_status, refund_initiated
            FROM orders
            WHERE customer_id IN ('customer-1', 'customer-2', 'customer-3')
            """
        ).fetchall()

    assert len(orders) == 30
    assert len({row["item_name"] for row in orders}) == len(orders)
    for customer_id in ("customer-1", "customer-2", "customer-3"):
        customer_orders = [row for row in orders if row["customer_id"] == customer_id]
        assert len(customer_orders) == 10
        eligible = [
            row
            for row in customer_orders
            if OrderRepository.is_eligible(row, today=date.today())
        ]
        ineligible = [
            row for row in customer_orders
            if row["status"] == "delivered" and row["final_sale"]
        ]
        assert len(eligible) == 1
        assert len(ineligible) == 1


def test_order_id_extraction_normalizes_spaces_underscores_and_case():
    assert OrderTool.extract_order_id("Please track ord 1001") == "ORD-1001"
    assert OrderTool.extract_order_id("Track ORD_1001") == "ORD-1001"
    assert OrderTool.extract_order_id("I need a return") is None


@pytest.mark.asyncio
async def test_care_agent_answers_without_passing_assistant_history_to_model():
    class FakePolicyTool:
        def search_policy(self, query):
            assert "shipping" in query
            return [
                {
                    "source": "shipping.txt:1",
                    "text": "Shipping updates are provided by the carrier.",
                }
            ]

    class FakeLLM:
        system_prompt = ""
        user_prompt = ""

        async def generate(self, system_prompt, user_prompt):
            self.system_prompt = system_prompt
            self.user_prompt = user_prompt
            return (
                "Under the policy, an approved refund is sent to your original payment "
                "method after the returned item is received and inspected. The policy "
                "does not specify an exact amount."
            )

    llm = FakeLLM()
    agent = CareAgent(FakePolicyTool(), llm)
    result = await agent.handle(
        {
            "message": "How are shipping updates provided?",
            "conversation": [
                {"role": "user", "content": "ORD_1101"},
                {
                    "role": "assistant",
                    "content": "Let me analyze the customer question and policy passages.",
                },
            ],
        }
    )

    assert result["sources"] == ["shipping.txt:1"]
    assert result["answer"].startswith("Under the policy")
    assert "Let me analyze" not in llm.user_prompt
    assert "Recent conversation" not in llm.user_prompt
    assert "Never show analysis" in llm.system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "expected_phrases"),
    [
        (
            "i want to return my order. What is policy for return",
            (
                "within 30 calendar days after delivery",
                "unused, in its original condition",
                "Final-sale items",
                "provide the order number to customer support",
            ),
        ),
        (
            "what would be refudn for this product",
            (
                "original payment method",
                "policy does not specify an exact refund amount",
            ),
        ),
    ],
)
async def test_care_agent_answers_return_and_refund_questions_from_policy(
    question,
    expected_phrases,
):
    class FakePolicyTool:
        def search_policy(self, query):
            return [
                {
                    "source": "returns.txt:1",
                    "text": (
                        "Eligible items may be returned within 30 calendar days after "
                        "delivery. An item must be unused, in its original condition, "
                        "and include the original tags and proof of purchase. Final-sale "
                        "items, personalized items, and items marked non-returnable at "
                        "purchase are not eligible for return unless required by "
                        "applicable law. After a returned item is received and inspected, "
                        "an approved refund is issued to the original payment method. "
                        "Processing time depends on the payment provider. Submitting a "
                        "return request does not mean a refund has been approved or issued. "
                        "To request a return, provide the order number to customer support. "
                        "Refunds require approval and must not be promised before the return "
                        "has been reviewed."
                    ),
                }
            ]

    class FailIfCalled:
        async def generate(self, system_prompt, user_prompt):
            raise AssertionError("Common return/refund answers should not use the LLM.")

    agent = CareAgent(FakePolicyTool(), FailIfCalled())
    result = await agent.handle({"message": question})

    for phrase in expected_phrases:
        assert phrase.lower() in result["answer"].lower()
    assert len(result["answer"].split(".")) <= 4
    assert result["sources"] == ["returns.txt:1"]


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
