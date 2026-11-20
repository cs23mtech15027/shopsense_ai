import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATABASE_PATH
from app.persistence.database import Database
from app.persistence.users import hash_password

_DEMO_PASSWORD = "ShopSenseDemo123!"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or reset the five local ShopSense demo shoppers and orders."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing OMS and session rows before recreating demo data.",
    )
    arguments = parser.parse_args()

    database = Database(str(DATABASE_PATH))
    database.initialize()
    if arguments.reset:
        with database.connect() as connection:
            for table in (
                "return_requests",
                "refunds",
                "approvals",
                "orders",
                "auth_sessions",
                "customers",
            ):
                connection.execute(f"DELETE FROM {table}")
        database.initialize()
    with database.connect() as connection:
        for number in range(1, 6):
            user_id = f"customer-{number}"
            salt, password_hash = hash_password(_DEMO_PASSWORD)
            connection.execute(
                """
                UPDATE customers
                SET username = ?, password_salt = ?, password_hash = ?, role = 'customer'
                WHERE customer_id = ?
                """,
                (user_id, salt, password_hash, user_id),
            )

    print("Created five demo shoppers and seeded OMS scenarios.")
    print(f"Username: customer-1 through customer-5; password: {_DEMO_PASSWORD}")
    print("Demo scenarios for customer-1 include ORD-1101 (INR 750),")
    print("ORD-1113 (> INR 5,000, Tier-2), ORD-1110/ORD-1112 (Open-Box OTP),")
    print("and ORD-1111 (innerwear).")


if __name__ == "__main__":
    main()
