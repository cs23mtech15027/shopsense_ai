import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.config import DATABASE_PATH, JWT_AUDIENCE, JWT_ISSUER
from app.persistence.database import Database
from app.persistence.users import UserRepository


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue a signed local MCP JWT for a seeded demo shopper."
    )
    parser.add_argument("--user-id", default="customer-1")
    arguments = parser.parse_args()

    secret = os.getenv("JWT_SECRET", "")
    if len(secret) < 32:
        parser.error("Set JWT_SECRET to at least 32 characters in .env.")
    if arguments.user_id not in {f"customer-{number}" for number in range(1, 6)}:
        parser.error("--user-id must be customer-1 through customer-5.")

    database = Database(str(DATABASE_PATH))
    database.initialize()
    token = UserRepository(database).issue_token(
        arguments.user_id,
        "customer",
        secret,
        JWT_ISSUER,
        JWT_AUDIENCE,
    )
    print(token)


if __name__ == "__main__":
    main()
