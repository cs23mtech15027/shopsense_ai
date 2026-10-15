import argparse

from app.config import get_settings
from app.persistence.database import Database


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Restore seeded ShopSense demo orders to their initial state."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the confirmation prompt",
    )
    args = parser.parse_args()

    if not args.yes:
        confirmation = input(
            f"Reset seeded demo orders and their approvals/refunds in "
            f"'{settings.database_path}'? [y/N] "
        )
        if confirmation.strip().lower() not in {"y", "yes"}:
            print("Reset cancelled.")
            return 1

    count = Database(settings.database_path).reset_demo_orders()
    print(
        f"Reset {count} seeded demo orders in '{settings.database_path}'. "
        "Auth sessions and conversation checkpoints were not changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
