import csv
import hashlib
import io
import re
import threading
import uuid
from pathlib import Path

from fastembed import TextEmbedding
import pdfplumber
from qdrant_client import QdrantClient, models

_RETURN_POLICY_QUERY = re.compile(r"\b(return|refund|refudn|refun|exchange)\b", re.IGNORECASE)
_REFUND_TYPO = re.compile(r"\b(?:refudn|refun)\b", re.IGNORECASE)


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
        normalized_query = _REFUND_TYPO.sub("refund", query)
        vector = list(self.embedding.query_embed(normalized_query))[0].tolist()
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        passages = [
            {
                "text": point.payload["text"],
                "source": point.payload["source"],
            }
            for point in result.points
            if point.payload and "text" in point.payload and "source" in point.payload
        ]
        if _RETURN_POLICY_QUERY.search(query):
            passages = _ensure_return_policy_passage(
                passages,
                self.policy_directory / "returns.txt",
                limit,
            )
        return passages

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
            (
                *self.policy_directory.glob("*.txt"),
                *self.policy_directory.glob("*.pdf"),
                *self.policy_directory.glob("*.csv"),
            )
        )
        for path in paths:
            if path.suffix.lower() == ".pdf":
                documents.extend(_extract_pdf_documents(path))
            elif path.suffix.lower() == ".csv":
                documents.extend(_extract_csv_documents(path))
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


def _extract_csv_documents(path: Path) -> list[tuple[str, str, str]]:
    content = path.read_text(encoding="utf-8-sig")
    sample = content[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(content), dialect=dialect)
    headers = [header.strip() for header in reader.fieldnames or [] if header]
    if not headers:
        raise ValueError(f"CSV policy file {path.name} is missing a header row.")
    reader.fieldnames = headers

    documents = [
        (
            path.name,
            f"{path.name}:schema",
            f"CSV policy table from {path.name}. Columns: {', '.join(headers)}.",
        )
    ]
    for row_number, row in enumerate(reader, start=1):
        fields = [
            f"{header}: {_clean_cell(row.get(header))}"
            for header in headers
            if _clean_cell(row.get(header))
        ]
        if fields:
            documents.append(
                (
                    path.name,
                    f"{path.name}:row{row_number}",
                    f"Policy table {path.stem}, record {row_number}. "
                    + "; ".join(fields),
                )
            )
    return documents


def _ensure_return_policy_passage(
    passages: list[dict[str, str]],
    policy_path: Path,
    limit: int,
) -> list[dict[str, str]]:
    if any(item["source"].startswith("returns.txt:") for item in passages):
        return passages[:1]
    chunks = _chunks(policy_path.read_text(encoding="utf-8"))
    if not chunks:
        return passages[:limit]
    return [{"source": "returns.txt:1", "text": chunks[0]}]


def _extract_pdf_documents(path: Path) -> list[tuple[str, str, str]]:
    documents: list[tuple[str, str, str]] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            page_text = _page_text_without_tables(page, [table.bbox for table in tables])
            for chunk_number, chunk in enumerate(_chunks(page_text), start=1):
                documents.append(
                    (
                        path.name,
                        f"{path.name}:p{page_number}:text{chunk_number}",
                        f"Document: {path.stem}. Page: {page_number}.\n{chunk}",
                    )
                )
            for table_number, table in enumerate(tables, start=1):
                documents.extend(
                    _table_documents(
                        path.name,
                        page_number,
                        table_number,
                        table.extract(),
                    )
                )
    return documents


def _page_text_without_tables(page, table_bboxes: list[tuple[float, float, float, float]]) -> str:
    words_by_line: dict[int, list[dict]] = {}
    for word in page.extract_words():
        center_x = (word["x0"] + word["x1"]) / 2
        center_y = (word["top"] + word["bottom"]) / 2
        if any(
            left <= center_x <= right and top <= center_y <= bottom
            for left, top, right, bottom in table_bboxes
        ):
            continue
        line_number = round(word["top"] / 3)
        words_by_line.setdefault(line_number, []).append(word)
    return "\n".join(
        " ".join(word["text"] for word in sorted(line, key=lambda item: item["x0"]))
        for _, line in sorted(words_by_line.items())
    )


def _table_documents(
    filename: str,
    page_number: int,
    table_number: int,
    table: list[list[str | None]],
) -> list[tuple[str, str, str]]:
    rows = [
        [_clean_cell(cell) for cell in row]
        for row in table
        if any(_clean_cell(cell) for cell in row)
    ]
    if not rows:
        return []

    headers = [value or f"Column {index + 1}" for index, value in enumerate(rows[0])]
    prefix = f"{filename}:p{page_number}:table{table_number}"
    table_title = f"Table from {filename}, page {page_number}"
    documents = []
    for row_number, values in enumerate(rows[1:], start=1):
        fields = [
            f"{headers[index] if index < len(headers) else f'Column {index + 1}'}: {value}"
            for index, value in enumerate(values)
            if value
        ]
        if fields:
            documents.append(
                (
                    filename,
                    f"{prefix}:row{row_number}",
                    f"{table_title}, record {row_number}. " + "; ".join(fields),
                )
            )
    return documents


def _clean_cell(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _chunks(
    text: str,
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[str]:
    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_word_count = 0

    def flush() -> None:
        nonlocal current, current_word_count
        if current:
            chunks.append(" ".join(current))
            current = []
            current_word_count = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_words:
            flush()
            step = max_words - overlap_words
            for start in range(0, len(words), step):
                chunk_words = words[start : start + max_words]
                if chunk_words:
                    chunks.append(" ".join(chunk_words))
            continue

        if current and current_word_count + len(words) > max_words:
            previous_words = " ".join(current).split()
            flush()
            overlap = previous_words[-overlap_words:] if overlap_words else []
            if overlap and len(overlap) + len(words) <= max_words:
                current = [" ".join(overlap)]
                current_word_count = len(overlap)
        current.append(paragraph)
        current_word_count += len(words)

    flush()
    return chunks
