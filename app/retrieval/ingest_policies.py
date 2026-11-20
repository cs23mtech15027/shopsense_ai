import argparse
import logging
from pathlib import Path

from app.config import POLICY_DIRECTORY, get_settings
from app.retrieval.policies import PolicyStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed policy files with Qdrant Cloud Inference and sync them."
    )
    parser.add_argument(
        "--policy-dir",
        default=POLICY_DIRECTORY,
        type=Path,
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the configured policy collection before ingesting.",
    )
    args = parser.parse_args()
    settings = get_settings()

    logging.basicConfig(level=logging.INFO)
    store = PolicyStore(
        qdrant_url=settings.qdrant_url,
        api_key=settings.qdrant_api_key.get_secret_value(),
        collection_name=settings.qdrant_collection,
        policy_directory=args.policy_dir,
    )
    if args.recreate and store.client.collection_exists(store.collection_name):
        store.client.delete_collection(store.collection_name)
    count = store.ingest()
    print(
        f"Synced {count} policy chunks to Qdrant collection "
        f"{store.collection_name!r}."
    )


if __name__ == "__main__":
    main()
