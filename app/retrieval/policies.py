import hashlib
import threading
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from pypdf import PdfReader
from qdrant_client import QdrantClient, models


class PolicyStore:
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        embedding_model: str,
        policy_directory: Path,
    ):
        self.client = QdrantClient(url=qdrant_url, timeout=10)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding = TextEmbedding(model_name=embedding_model)
        self.policy_directory = policy_directory
        self._initialized = False
        self._initialization_lock = threading.Lock()

    def search(self, query: str, limit: int) -> list[dict[str, str]]:
        self._ensure_index()
        vector = list(self.embedding.query_embed(query))[0].tolist()
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "text": point.payload["text"],
                "source": point.payload["source"],
            }
            for point in result.points
            if point.payload and "text" in point.payload and "source" in point.payload
        ]

    def _ensure_index(self) -> None:
        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            documents = self._load_documents()
            expected_ids = {self._point_id(document) for document in documents}
            if self.client.collection_exists(self.collection_name):
                indexed_ids = self._indexed_ids()
                if indexed_ids != expected_ids:
                    self.client.delete_collection(self.collection_name)
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
                )
            if expected_ids:
                self._index_documents(documents)
            self._initialized = True

    def _load_documents(self) -> list[tuple[str, str, str]]:
        documents: list[tuple[str, str, str]] = []
        paths = sorted(
            (*self.policy_directory.glob("*.txt"), *self.policy_directory.glob("*.pdf"))
        )
        for path in paths:
            if path.suffix.lower() == ".pdf":
                reader = PdfReader(path)
                for page_number, page in enumerate(reader.pages, start=1):
                    text = page.extract_text() or ""
                    for chunk_number, chunk in enumerate(_chunks(text), start=1):
                        source = f"{path.name}:p{page_number}:c{chunk_number}"
                        documents.append((path.name, source, chunk))
            else:
                text = path.read_text(encoding="utf-8")
                for index, chunk in enumerate(_chunks(text), start=1):
                    documents.append((path.name, f"{path.name}:{index}", chunk))
        if not documents:
            raise RuntimeError(f"No policy documents found in {self.policy_directory}.")
        return documents

    def _index_documents(self, documents: list[tuple[str, str, str]]) -> None:
        vectors = list(self.embedding.passage_embed([document[2] for document in documents]))
        points = [
            models.PointStruct(
                id=self._point_id(document),
                vector=vector.tolist(),
                payload={"source": document[1], "text": document[2]},
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def _indexed_ids(self) -> set[str]:
        point_ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=256,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.update(str(point.id) for point in points)
            if offset is None:
                return point_ids

    def _point_id(self, document: tuple[str, str, str]) -> str:
        value = f"{self.embedding_model}\0" + "\0".join(document)
        digest = hashlib.sha256(value.encode()).hexdigest()[:32]
        return str(uuid.UUID(hex=digest))


def _chunks(text: str, max_words: int = 180) -> list[str]:
    words = text.split()
    return [
        " ".join(words[index : index + max_words])
        for index in range(0, len(words), max_words)
    ]
