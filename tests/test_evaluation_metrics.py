import pytest

from app.evaluation_metrics import compute_rag_metrics


def test_rag_precision_uses_exact_labeled_sources_not_keyword_hits():
    chunks = [
        {"source": "unrelated.txt", "text": "refund policy and tracking"},
        {"source": "policies/shipping.txt", "text": "shipping status"},
        {"source": "other.csv", "text": "refund approval"},
    ]

    precision, recall, matched_sources, matched_keywords = compute_rag_metrics(
        chunks,
        "shipping.txt",
        "refund,tracking",
        k=3,
    )

    assert precision == pytest.approx(0.3333)
    assert recall == 1.0
    assert matched_sources == ["shipping.txt"]
    assert matched_keywords == ["refund", "tracking"]


def test_rag_metrics_count_missing_top_k_slots_as_not_relevant():
    precision, recall, matched_sources, matched_keywords = compute_rag_metrics(
        [{"source": "returns.txt", "text": "eligible return"}],
        "shipping.txt",
        "refund",
        k=3,
    )

    assert precision == 0.0
    assert recall == 0.0
    assert matched_sources == []
    assert matched_keywords == []


def test_rag_metrics_normalize_chunk_ids_in_source_names():
    precision, recall, matched_sources, _ = compute_rag_metrics(
        [
            {"source": "peak_sale_delivery.txt:1", "text": "delivery delay"},
            {"source": "policies/shipping.txt:2", "text": "shipping status"},
            {"source": "handbook.pdf:p1:text1", "text": "policy"},
        ],
        "peak_sale_delivery.txt",
        "",
        k=3,
    )

    assert precision == pytest.approx(0.3333)
    assert recall == 1.0
    assert matched_sources == ["peak_sale_delivery.txt"]


def test_rag_metrics_reject_nonpositive_k():
    with pytest.raises(ValueError, match="k must be greater than zero"):
        compute_rag_metrics([], "", "", k=0)
