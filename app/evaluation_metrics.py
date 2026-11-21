from typing import Any


def _source_name(source: str) -> str:
    filename = source.strip().split(":", 1)[0]
    return filename.replace("\\", "/").rsplit("/", 1)[-1].casefold()


def compute_rag_metrics(
    retrieved_chunks: list[dict[str, Any]],
    expected_sources_csv: str,
    expected_keywords_csv: str,
    k: int = 3,
) -> tuple[float, float, list[str], list[str]]:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    top_chunks = retrieved_chunks[:k]
    expected_sources = [
        _source_name(source)
        for source in expected_sources_csv.split(",")
        if source.strip()
    ]
    expected_keywords = [
        keyword.strip().casefold()
        for keyword in expected_keywords_csv.split(",")
        if keyword.strip()
    ]
    retrieved_sources = [
        _source_name(str(chunk.get("source", "")))
        for chunk in top_chunks
    ]
    combined_text = " ".join(
        str(chunk.get("text", "")).casefold()
        for chunk in top_chunks
    )

    relevant_chunks = sum(
        source in expected_sources
        for source in retrieved_sources
    )
    matched_sources = [
        source for source in expected_sources
        if source in retrieved_sources
    ]
    matched_keywords = [
        keyword for keyword in expected_keywords
        if keyword in combined_text
    ]

    precision_at_k = relevant_chunks / k
    recall_at_k = (
        len(matched_sources) / len(expected_sources)
        if expected_sources
        else 1.0
    )
    return (
        round(precision_at_k, 4),
        round(recall_at_k, 4),
        matched_sources,
        matched_keywords,
    )
