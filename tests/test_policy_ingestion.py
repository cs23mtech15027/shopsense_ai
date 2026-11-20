from types import SimpleNamespace

from qdrant_client import models
import pytest

import app.retrieval.policies as policies
from app.config import EMBEDDING_MODEL
from app.retrieval.policies import PolicyStore


def test_policy_store_uses_qdrant_cloud_inference_for_upsert_and_query(
    tmp_path,
    monkeypatch,
):
    policy_directory = tmp_path / "policies"
    policy_directory.mkdir()
    (policy_directory / "shipping.txt").write_text(
        "Check the verified delivery estimate before replying.",
        encoding="utf-8",
    )
    (policy_directory / "return_rules.csv").write_text(
        "category,window\nclothing,30 days\n",
        encoding="utf-8",
    )
    (policy_directory / "manual.pdf").touch()
    monkeypatch.setattr(
        policies,
        "_extract_pdf_documents",
        lambda path: [("manual.pdf", "manual.pdf:p1", "Policy PDF text.")],
    )
    created = {}

    class FakeQdrant:
        def __init__(self, **kwargs):
            created["client_kwargs"] = kwargs
            self.points = []
            self.collection_created = False

        def collection_exists(self, collection_name):
            return self.collection_created

        def create_collection(self, collection_name, vectors_config):
            created["collection"] = collection_name
            created["vectors_config"] = vectors_config
            self.collection_created = True

        def upsert(self, collection_name, points, wait):
            self.points.extend(points)

        def get_collection(self, collection_name):
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(vectors=created["vectors_config"])
                ),
                points_count=len(self.points),
            )

        def query_points(self, **kwargs):
            created["query"] = kwargs["query"]
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        payload={
                            "source": "shipping.txt:1",
                            "text": "Check the verified delivery estimate before replying.",
                        }
                    )
                ]
            )

    monkeypatch.setattr(policies, "QdrantClient", FakeQdrant)
    store = PolicyStore(
        qdrant_url="https://qdrant.example",
        api_key="qdrant-test-key",
        collection_name="test-policies",
        policy_directory=policy_directory,
    )

    assert store.ingest() == 4
    assert created["client_kwargs"]["cloud_inference"] is True
    assert created["collection"] == "test-policies"
    assert created["vectors_config"].size == 384
    assert created["vectors_config"].distance == models.Distance.COSINE
    assert len(store.client.points) == 4
    assert all(
        isinstance(point.vector, models.Document) for point in store.client.points
    )
    assert all(point.vector.model == EMBEDDING_MODEL for point in store.client.points)
    assert not hasattr(store, "embedding")

    passages = store.search("delivery delay", limit=2)
    assert passages[0]["source"] == "shipping.txt:1"
    assert isinstance(created["query"], models.Document)
    assert created["query"].model == EMBEDDING_MODEL


def test_policy_point_ids_are_stable():
    store = PolicyStore.__new__(PolicyStore)
    store.embedding_model = EMBEDDING_MODEL
    policy = ("returns.txt", "returns.txt:1", "30-day return window")

    assert store._point_id(policy) == store._point_id(policy)


def test_runtime_retrieval_requires_ingested_cloud_collection():
    store = PolicyStore.__new__(PolicyStore)
    store.collection_name = "missing-policy-collection"
    store.client = SimpleNamespace(
        collection_exists=lambda _: False,
    )
    store._load_documents = lambda: pytest.fail(
        "Runtime retrieval must not fall back to local policy files."
    )

    with pytest.raises(RuntimeError, match="Run python scripts/ingest_policies.py"):
        store.search("return policy", limit=3)
