from pathlib import Path

from app.retrieval.policies import (
    _chunks,
    _extract_csv_documents,
    _extract_pdf_documents,
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
