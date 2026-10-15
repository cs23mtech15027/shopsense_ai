from pathlib import Path

import pytest

from app.retrieval.policies import (
    _ensure_return_policy_passage,
    _chunks,
    _extract_csv_documents,
    _extract_pdf_documents,
    _table_documents,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_DIRECTORY = ROOT / "data" / "policies"


def test_csv_rows_keep_column_names_attached_to_values():
    documents = _extract_csv_documents(
        POLICY_DIRECTORY / "return_eligibility.csv"
    )

    footwear_record = next(text for _, source, text in documents if source.endswith("row5"))
    assert "Category: Footwear" in footwear_record
    assert "Return window (calendar days): 30" in footwear_record
    assert "Original box and tags required" in footwear_record


def test_pdf_text_and_tables_are_extracted_with_page_and_row_sources():
    documents = _extract_pdf_documents(
        POLICY_DIRECTORY / "shop_return_and_refund_handbook.pdf"
    )

    text = "\n".join(document[2] for document in documents)
    sources = [document[1] for document in documents]
    footwear_record = next(
        document[2]
        for document in documents
        if "Category: Footwear" in document[2]
    )
    assert "Policy scope and decision principles" in text
    assert "Delivery and window" in text
    assert "Original box" in footwear_record
    assert any(":p" in source and ":table" in source for source in sources)


def test_chunking_respects_maximum_size_and_overlaps_paragraph_boundaries():
    text = "\n\n".join(
        " ".join(f"term{paragraph}_{word}" for word in range(80))
        for paragraph in range(4)
    )

    chunks = _chunks(text, max_words=100, overlap_words=20)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 100 for chunk in chunks)
    assert "term0_79" in chunks[0]
    assert any("term0_79" in chunk and "term1_0" in chunk for chunk in chunks[1:])


def test_generic_return_search_uses_canonical_return_policy_source():
    passages = _ensure_return_policy_passage(
        [
            {"source": "shop_return_and_refund_handbook.pdf:p1:table1:row4", "text": "Outerwear"},
            {"source": "return_eligibility.csv:row5", "text": "Footwear"},
        ],
        POLICY_DIRECTORY / "returns.txt",
        limit=3,
    )

    assert len(passages) == 1
    assert passages[0]["source"] == "returns.txt:1"
    assert "original payment method" in passages[0]["text"]
    assert "Final-sale items" in passages[0]["text"]


def test_canonical_return_policy_passage_is_selected_when_not_first():
    passages = _ensure_return_policy_passage(
        [
            {"source": "shop_return_and_refund_handbook.pdf:p1", "text": "Unrelated result"},
            {"source": "returns.txt:2", "text": "Canonical return policy"},
        ],
        POLICY_DIRECTORY / "returns.txt",
        limit=3,
    )

    assert passages == [{"source": "returns.txt:2", "text": "Canonical return policy"}]


def test_chunking_splits_a_single_long_paragraph_with_word_overlap():
    chunks = _chunks(
        "one two three four five six seven eight nine",
        max_words=5,
        overlap_words=2,
    )

    assert chunks == [
        "one two three four five",
        "four five six seven eight",
        "seven eight nine",
    ]


def test_chunking_rejects_invalid_size_and_overlap():
    with pytest.raises(ValueError, match="Chunk size"):
        _chunks("some policy text", max_words=0)
    with pytest.raises(ValueError, match="Chunk size"):
        _chunks("some policy text", max_words=5, overlap_words=5)
    with pytest.raises(ValueError, match="Chunk size"):
        _chunks("some policy text", max_words=5, overlap_words=-1)


def test_csv_extraction_handles_bom_semicolon_delimiter_and_blank_rows(tmp_path):
    csv_path = tmp_path / "policy.csv"
    csv_path.write_text(
        "\ufeffCategory ; Return window ; Notes\n"
        " Footwear ; 30 ; Original box required\n"
        " ; ; \n",
        encoding="utf-8",
    )

    documents = _extract_csv_documents(csv_path)

    assert len(documents) == 2
    assert documents[0][1] == "policy.csv:schema"
    assert "Columns: Category, Return window, Notes" in documents[0][2]
    assert documents[1][1] == "policy.csv:row1"
    assert "Category: Footwear" in documents[1][2]
    assert "Return window: 30" in documents[1][2]


def test_csv_extraction_rejects_a_missing_header_row(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(" \n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing a header row"):
        _extract_csv_documents(csv_path)


def test_pdf_table_rows_preserve_headers_for_ragged_rows():
    documents = _table_documents(
        "policy.pdf",
        page_number=2,
        table_number=1,
        table=[
            ["Category", None, "Return window"],
            ["Footwear", "Original box required", "30 days"],
        ],
    )

    assert len(documents) == 1
    assert documents[0][1] == "policy.pdf:p2:table1:row1"
    assert "Category: Footwear" in documents[0][2]
    assert "Column 2: Original box required" in documents[0][2]
    assert "Return window: 30 days" in documents[0][2]
